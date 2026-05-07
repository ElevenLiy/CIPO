
import json
import copy
import logging
import os
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import asdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    RL_DATASET_PATH, AUGMENTED_TOOLS_PATH, SKILL_LIBRARY_PATH,
    SFT_DATA_PATH, CHECKPOINT_DIR, ADAMACRO_OUTPUT_DIR,
    SFTConfig, get_model_path, DEFAULT_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_skill_matcher(skill_library_path: str) -> Dict[str, Dict]:
    import re as _re

    def _normalize(name: str) -> str:
        return _re.sub(r'_v\d+\w*$', '', name)

    with open(skill_library_path, "r", encoding="utf-8") as f:
        skill_lib = json.load(f)

    macros = skill_lib.get("macros", {})
    matchers = {}

    for macro_id, macro in macros.items():
        skill_name = f"skill_{macro_id}"
        raw_chain = macro.get("tool_names", [])
        tool_chain = [_normalize(t) for t in raw_chain]
        if len(tool_chain) >= 2:
            matchers[skill_name] = {
                "tool_chain": tool_chain,
                "length": len(tool_chain),
                "macro": macro,
            }

    return matchers


def match_skills_in_sequence(
    tool_names: List[str],
    tool_args: List[str],
    output_texts: List[str],
    skill_matchers: Dict[str, Dict],
) -> List[Dict]:
    n = len(tool_names)
    actions = []
    
    sorted_matchers = sorted(
        skill_matchers.items(),
        key=lambda x: x[1]["length"],
        reverse=True,
    )
    
    i = 0
    while i < n:
        matched = False
        
        for skill_name, matcher in sorted_matchers:
            chain = matcher["tool_chain"]
            chain_len = matcher["length"]
            
            if i + chain_len > n:
                continue
            
            if tool_names[i:i + chain_len] == chain:
                sub_steps = []
                sub_args = {}
                sub_output = ""
                
                for j in range(chain_len):
                    idx = i + j
                    step_args_str = tool_args[idx] if idx < len(tool_args) else "{}"
                    step_output = output_texts[idx] if idx < len(output_texts) else ""
                    
                    try:
                        step_args = json.loads(step_args_str) if isinstance(step_args_str, str) else step_args_str
                    except:
                        step_args = {}
                    
                    sub_steps.append({
                        "tool_name": tool_names[idx],
                        "args": step_args,
                        "output": step_output,
                    })
                    
                    if j == 0:
                        sub_args = step_args
                    
                    if j == chain_len - 1:
                        sub_output = step_output
                
                actions.append({
                    "type": "skill",
                    "skill_name": skill_name,
                    "args": sub_args,
                    "sub_steps": sub_steps,
                    "output": sub_output,
                })
                
                i += chain_len
                matched = True
                break
        
        if not matched:
            args_str = tool_args[i] if i < len(tool_args) else "{}"
            output = output_texts[i] if i < len(output_texts) else ""
            
            actions.append({
                "type": "atomic",
                "tool_name": tool_names[i],
                "args": args_str,
                "output": output,
            })
            i += 1
    
    return actions


def format_tool_call_message(tool_name: str, args: Any) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except:
            pass
    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
    return json.dumps({
        "name": tool_name,
        "arguments": args if isinstance(args, dict) else {}
    }, ensure_ascii=False)


def generate_sft_data(
    rl_dataset_path: str,
    augmented_tools_path: str,
    skill_library_path: str,
    output_path: str,
) -> List[Dict]:
    import random

    with open(rl_dataset_path, "r", encoding="utf-8") as f:
        rl_data = json.load(f)

    with open(augmented_tools_path, "r", encoding="utf-8") as f:
        augmented_tools = json.load(f)

    tool_desc_map = {}
    skill_names = set()

    def _normalize_tool_name(name: str) -> str:
        import re as _re
        return _re.sub(r'_v\d+\w*$', '', name)

    for t in augmented_tools:
        orig_name = t["name"]
        norm_name = _normalize_tool_name(orig_name)
        tag = "[SKILL]" if t.get("is_skill") else ""
        params = t.get("parameters_schema", {})
        param_str = ""
        if params:
            props = params.get("properties", {})
            if props:
                param_names = list(props.keys())[:5]
                param_str = f" params: {param_names}"

        chain_str = ""
        if t.get("is_skill"):
            chain = t.get("tool_chain", [])
            if not chain:
                chain = [s.get("tool_name", "") for s in t.get("execution_plan", [])]
            if chain:
                norm_chain = [_normalize_tool_name(c) for c in chain[:5]]
                chain_str = f" [chain: {' → '.join(norm_chain)}]"

        tool_desc_map[norm_name] = f"- {tag}{norm_name}: {t.get('description','')[:100]}{chain_str}{param_str}"
        if t.get("is_skill"):
            skill_names.add(norm_name)

    skill_matchers = build_skill_matcher(skill_library_path)

    def build_per_task_tool_list(gt_tools: List[str]) -> str:
        lines = []
        seen = set()

        for tn in skill_names:
            if tn in tool_desc_map:
                lines.append(tool_desc_map[tn])
                seen.add(tn)
            if len(seen) >= 15:
                break

        for tn in gt_tools:
            if tn not in seen and tn in tool_desc_map:
                lines.append(tool_desc_map[tn])
                seen.add(tn)

        gt_prefixes = set()
        for tn in gt_tools:
            parts = tn.replace("-", "_").split("_")
            if parts:
                gt_prefixes.add(parts[0])
        for tn in tool_desc_map:
            if tn in seen:
                continue
            tn_prefix = tn.replace("-", "_").split("_")[0] if tn else ""
            if tn_prefix in gt_prefixes:
                lines.append(tool_desc_map[tn])
                seen.add(tn)
            if len(seen) >= 35:
                break

        remaining = [tn for tn in tool_desc_map if tn not in seen]
        random.shuffle(remaining)
        for tn in remaining:
            lines.append(tool_desc_map[tn])
            seen.add(tn)
            if len(seen) >= 50:
                break

        return "\n".join(lines), len([tn for tn in seen if tn in skill_names])

    def build_system_prompt(tool_list_str: str, n_tools: int, n_skills: int) -> str:
        return (
            "You are a tool-calling agent. You MUST use tools to complete tasks. "
            "Do NOT answer directly — always call at least one tool first.\n\n"
            "You have access to both atomic tools and composite skills. "
            "Skills are pre-composed tool chains that execute multiple tools in sequence. "
            "Choose whichever tools (atomic or skill) best fit the task.\n\n"
            f"Available tools ({n_tools} total, including {n_skills} skills):\n"
            f"{tool_list_str}\n\n"
            "To call a tool, respond ONLY with:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"param": "value"}}\n'
            "</tool_call>\n\n"
            "After receiving all tool responses, provide a brief text summary to finish."
        )

    def build_messages(system_prompt, user_prompt, actions) -> List[Dict]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for action in actions:
            if action["type"] == "skill":
                tool_name = action["skill_name"]
                args = action["args"] if isinstance(action["args"], dict) else {}
                output = action["output"][:1500] if isinstance(action["output"], str) else ""
            else:
                tool_name = action["tool_name"]
                try:
                    args = json.loads(action["args"]) if isinstance(action["args"], str) else action["args"]
                except:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                output = action["output"][:1500] if isinstance(action["output"], str) else ""

            tc_json = json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False)
            messages.append({
                "role": "assistant",
                "content": f"<tool_call>\n{tc_json}\n</tool_call>",
            })
            messages.append({
                "role": "user",
                "content": f"<tool_response name=\"{tool_name}\">\n{output}\n</tool_response>",
            })
        messages.append({"role": "assistant", "content": "Task completed successfully."})
        return messages

    episodes = rl_data.get("episodes", [])
    sft_examples = []
    skill_used_count = 0
    atomic_only_count = 0
    mixed_count = 0
    stepwise_count = 0

    for ep in episodes:
        if ep.get("success", 0) != 1:
            continue

        tool_names = [_normalize_tool_name(t) for t in ep.get("tool_names", [])]
        tool_args = ep.get("tool_args", [])
        output_texts = ep.get("output_texts", [])
        user_prompt = ep.get("user_prompt", "")
        gt_tools = [_normalize_tool_name(t) for t in ep.get("tool_names", [])]

        if not tool_names or not user_prompt:
            continue

        tool_list_str, n_skills = build_per_task_tool_list(gt_tools)
        system_prompt = build_system_prompt(tool_list_str, 50, n_skills)

        atomic_actions = []
        for i, tn in enumerate(tool_names):
            atomic_actions.append({
                "type": "atomic",
                "tool_name": tn,
                "args": tool_args[i] if i < len(tool_args) else "{}",
                "output": output_texts[i] if i < len(output_texts) else "",
            })

        msgs_atomic = build_messages(system_prompt, user_prompt, atomic_actions)
        sft_examples.append({
            "messages": msgs_atomic,
            "task_name": ep.get("task_name", ""),
            "variant": "atomic",
            "has_skill": False,
            "num_actions": len(atomic_actions),
            "num_skill_actions": 0,
            "num_atomic_actions": len(atomic_actions),
        })
        atomic_only_count += 1

        skill_actions = match_skills_in_sequence(
            tool_names, tool_args, output_texts, skill_matchers
        )
        has_skill = any(a["type"] == "skill" for a in skill_actions)

        if has_skill:
            msgs_skill = build_messages(system_prompt, user_prompt, skill_actions)
            sft_examples.append({
                "messages": msgs_skill,
                "task_name": ep.get("task_name", ""),
                "variant": "skill",
                "has_skill": True,
                "num_actions": len(skill_actions),
                "num_skill_actions": sum(1 for a in skill_actions if a["type"] == "skill"),
                "num_atomic_actions": sum(1 for a in skill_actions if a["type"] == "atomic"),
            })
            skill_used_count += 1

        if has_skill and len(skill_actions) >= 2:
            partial_actions = []
            first_skill_done = False
            for action in skill_actions:
                if action["type"] == "skill" and not first_skill_done:
                    partial_actions.append(action)
                    first_skill_done = True
                elif action["type"] == "skill" and first_skill_done:
                    for sub in action.get("sub_steps", []):
                        partial_actions.append({
                            "type": "atomic",
                            "tool_name": sub["tool_name"],
                            "args": sub.get("args", {}),
                            "output": sub.get("output", ""),
                        })
                else:
                    partial_actions.append(action)

            msgs_partial = build_messages(system_prompt, user_prompt, partial_actions)
            sft_examples.append({
                "messages": msgs_partial,
                "task_name": ep.get("task_name", ""),
                "variant": "partial_skill",
                "has_skill": True,
                "num_actions": len(partial_actions),
                "num_skill_actions": sum(1 for a in partial_actions if a["type"] == "skill"),
                "num_atomic_actions": sum(1 for a in partial_actions if a["type"] == "atomic"),
            })
            mixed_count += 1

        if len(atomic_actions) >= 3:
            start_indices = list(range(1, len(atomic_actions)))
            if len(start_indices) > 3:
                n = len(start_indices)
                start_indices = [start_indices[0], start_indices[n//2], start_indices[-1]]

            for step_idx in start_indices:
                prefix = atomic_actions[:step_idx]
                remaining = atomic_actions[step_idx:]

                step_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                for prev in prefix:
                    prev_name = prev.get("skill_name") if prev["type"] == "skill" else prev["tool_name"]
                    try:
                        prev_args = json.loads(prev["args"]) if isinstance(prev["args"], str) else prev["args"]
                    except:
                        prev_args = {}
                    if not isinstance(prev_args, dict):
                        prev_args = {}
                    tc_json = json.dumps({"name": prev_name, "arguments": prev_args}, ensure_ascii=False)
                    step_messages.append({
                        "role": "assistant",
                        "content": f"<tool_call>\n{tc_json}\n</tool_call>",
                    })
                    prev_output = prev.get("output", "")
                    prev_output = prev_output[:1500] if isinstance(prev_output, str) else ""
                    step_messages.append({
                        "role": "user",
                        "content": f"<tool_response name=\"{prev_name}\">\n{prev_output}\n</tool_response>",
                    })

                for rem in remaining:
                    rem_name = rem.get("skill_name") if rem["type"] == "skill" else rem["tool_name"]
                    try:
                        rem_args = json.loads(rem["args"]) if isinstance(rem["args"], str) else rem["args"]
                    except:
                        rem_args = {}
                    if not isinstance(rem_args, dict):
                        rem_args = {}
                    tc_json = json.dumps({"name": rem_name, "arguments": rem_args}, ensure_ascii=False)
                    step_messages.append({
                        "role": "assistant",
                        "content": f"<tool_call>\n{tc_json}\n</tool_call>",
                    })
                    rem_output = rem.get("output", "")
                    rem_output = rem_output[:1500] if isinstance(rem_output, str) else ""
                    step_messages.append({
                        "role": "user",
                        "content": f"<tool_response name=\"{rem_name}\">\n{rem_output}\n</tool_response>",
                    })

                step_messages.append({"role": "assistant", "content": "Task completed successfully."})

                sft_examples.append({
                    "messages": step_messages,
                    "task_name": ep.get("task_name", ""),
                    "variant": f"continuation_from_step{step_idx}",
                    "has_skill": False,
                    "num_actions": len(remaining),
                    "num_skill_actions": 0,
                    "num_atomic_actions": len(remaining),
                })
                stepwise_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "total_examples": len(sft_examples),
                "atomic_only": atomic_only_count,
                "with_skills": skill_used_count,
                "mixed_skill_atomic": mixed_count,
                "stepwise_next_tool": stepwise_count,
                "num_skills_available": len(skill_matchers),
            },
            "examples": sft_examples,
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Generated {len(sft_examples)} SFT examples")
    logger.info(f"  Atomic only: {atomic_only_count}")
    logger.info(f"  Full skill: {skill_used_count}")
    logger.info(f"  Mixed (partial skill): {mixed_count}")
    logger.info(f"  Stepwise next-tool: {stepwise_count}")

    return sft_examples


def _tokenize_with_assistant_mask(messages, tokenizer, max_length=4096):
    import re as _re
    import torch

    formatted = [{"role": m["role"], "content": m.get("content", "") or ""} for m in messages]
    try:
        text = tokenizer.apply_chat_template(formatted, tokenize=False, add_generation_prompt=False)
    except Exception:
        parts = [f"<|{m['role']}|>\n{m.get('content', '')}" for m in formatted]
        text = "\n".join(parts)

    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    labels = torch.full_like(input_ids, -100)

    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    patterns = [
        r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>)',
        r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>)',
    ]
    regions = []
    for pat in patterns:
        for m in _re.finditer(pat, decoded, _re.DOTALL):
            regions.append((m.start(1), m.end(1)))

    if not regions:
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    enc2 = tokenizer(decoded, return_offsets_mapping=True, add_special_tokens=False,
                     truncation=True, max_length=max_length)
    offsets = enc2.get("offset_mapping", [])

    if offsets and len(offsets) >= len(input_ids):
        for cs, ce in regions:
            for ti in range(len(input_ids)):
                if ti < len(offsets):
                    ts, te = offsets[ti]
                    if te > cs and ts < ce:
                        labels[ti] = input_ids[ti]
    else:
        toks = [tokenizer.decode([t]) for t in input_ids.tolist()]
        in_asst = False
        for idx, tok_text in enumerate(toks):
            if 'assistant' in tok_text.lower() and not in_asst:
                in_asst = True
                continue
            if in_asst and any(x in tok_text for x in ['im_end', 'eot_id', 'im_start', 'start_header']):
                in_asst = False
                continue
            if in_asst:
                labels[idx] = input_ids[idx]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_sft(
    model_name: str,
    sft_data_path: str,
    output_dir: str,
    sft_config: SFTConfig,
):
    import torch
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    model_path = get_model_path(model_name)
    logger.info(f"Loading model: {model_name} from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=sft_config.lora_rank,
        lora_alpha=sft_config.lora_alpha,
        lora_dropout=sft_config.lora_dropout,
        target_modules=sft_config.lora_target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    with open(sft_data_path, "r", encoding="utf-8") as f:
        sft_data = json.load(f)

    examples = sft_data.get("examples", [])
    logger.info(f"Loaded {len(examples)} SFT examples")

    max_seq_len = sft_config.max_seq_length
    tokenized_examples = []
    n_skipped = 0
    n_asst_tokens_total = 0

    for ex in examples:
        item = _tokenize_with_assistant_mask(ex["messages"], tokenizer, max_length=max_seq_len)
        n_asst = (item["labels"] != -100).sum().item()
        if n_asst == 0:
            n_skipped += 1
            continue
        tokenized_examples.append(item)
        n_asst_tokens_total += n_asst

    logger.info(f"Tokenized {len(tokenized_examples)} examples "
                f"({n_skipped} skipped with 0 assistant tokens)")
    if tokenized_examples:
        avg_asst = n_asst_tokens_total / len(tokenized_examples)
        avg_total = sum(len(ex["input_ids"]) for ex in tokenized_examples) / len(tokenized_examples)
        logger.info(f"Avg tokens per example: {avg_total:.0f} total, {avg_asst:.0f} assistant-only "
                    f"({avg_asst/max(avg_total,1)*100:.1f}% supervised)")

    class PreTokenizedDataset(TorchDataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]

    dataset = PreTokenizedDataset(tokenized_examples)

    def collate_fn(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), tokenizer.pad_token_id)]))
            attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

    logger.info(f"Dataset size: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=sft_config.num_epochs,
        per_device_train_batch_size=sft_config.per_device_batch_size,
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        learning_rate=sft_config.learning_rate,
        warmup_ratio=sft_config.warmup_ratio,
        weight_decay=sft_config.weight_decay,
        logging_steps=sft_config.logging_steps,
        save_steps=sft_config.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    logger.info("Starting SFT training (assistant-only loss mask)...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"SFT training complete. Model saved to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AdaMacro Step 3: SFT Training")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--rl-dataset", type=str, default=RL_DATASET_PATH)
    parser.add_argument("--augmented-tools", type=str, default=AUGMENTED_TOOLS_PATH)
    parser.add_argument("--skill-library", type=str, default=SKILL_LIBRARY_PATH)
    parser.add_argument("--sft-data", type=str, default=SFT_DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=os.path.join(CHECKPOINT_DIR, "sft"))
    parser.add_argument("--generate-only", action="store_true",
                       help="Only generate SFT data, skip training")
    parser.add_argument("--train-only", action="store_true",
                       help="Only train, skip data generation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    sft_config = SFTConfig()
    if args.epochs: sft_config.num_epochs = args.epochs
    if args.lr: sft_config.learning_rate = args.lr
    if args.batch_size: sft_config.per_device_batch_size = args.batch_size
    if args.lora_rank: sft_config.lora_rank = args.lora_rank
    
    logger.info("=" * 70)
    logger.info("AdaMacro Step 3: SFT Data Generation & Training")
    logger.info("=" * 70)
    
    if not args.train_only:
        logger.info("\n[Phase 1] Generating SFT training data...")
        generate_sft_data(
            args.rl_dataset,
            args.augmented_tools,
            args.skill_library,
            args.sft_data,
        )
    
    if not args.generate_only:
        logger.info(f"\n[Phase 2] Training with model: {args.model}")
        train_sft(
            args.model,
            args.sft_data,
            args.output_dir,
            sft_config,
        )


if __name__ == "__main__":
    main()
