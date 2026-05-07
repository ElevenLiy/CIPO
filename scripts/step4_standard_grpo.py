
import argparse
import bisect
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    RL_DATASET_PATH, AUGMENTED_TOOLS_PATH,
    TOOL_SIMULATOR_DB_PATH, CHECKPOINT_DIR,
    GRPOConfig, get_model_path, DEFAULT_MODEL,
)
from step4_grpo_training import (
    ToolEnvironment, normalize_tool_name, run_rollout,
    tokenize_with_assistant_mask, ExecutionLogger,
)

logging.basicConfig(
    level=logging.INFO,
    format="[std_grpo] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class TaskReward:

    @staticmethod
    def _tokenize_tool(name: str) -> set:
        n = name.lower().replace("-", "_").replace(".", "_")
        n = re.sub(r'_v\d+$', '', n)
        return set(t for t in n.split("_") if len(t) >= 2)

    @staticmethod
    def _fuzzy_match_score(a: str, b: str) -> float:
        a_n = a.lower().replace("-", "_").replace(".", "_")
        b_n = b.lower().replace("-", "_").replace(".", "_")
        if a_n == b_n or a == b:
            return 1.0
        if a_n in b_n or b_n in a_n:
            return 0.8
        a_tokens = TaskReward._tokenize_tool(a)
        b_tokens = TaskReward._tokenize_tool(b)
        if a_tokens and b_tokens:
            inter = len(a_tokens & b_tokens)
            union = len(a_tokens | b_tokens)
            jaccard = inter / union if union > 0 else 0
            if jaccard >= 0.5:
                return jaccard
        return 0.0

    def compute(
        self,
        used_tools: List[str],
        gt_tools: List[str],
        skill_traces: List[List[Tuple[str, str]]],
        num_decision_steps: int,
        completed: bool,
    ) -> Tuple[float, Dict]:
        all_used = set(normalize_tool_name(t) for t in used_tools)
        for trace in skill_traces:
            for tool_name, _ in trace:
                all_used.add(normalize_tool_name(tool_name))

        gt_set = set(normalize_tool_name(t) for t in gt_tools) if gt_tools else set()
        gt_list = [normalize_tool_name(t) for t in gt_tools] if gt_tools else []
        used_list = [normalize_tool_name(t) for t in used_tools]
        for trace in skill_traces:
            for tool_name, _ in trace:
                used_list.append(normalize_tool_name(tool_name))

        if num_decision_steps == 0:
            return 0.0, {"r_task": 0.0, "note": "0-step penalty"}

        if gt_set and all_used:
            total_credit = 0.0
            for gt in gt_set:
                best_score = 0.0
                for ut in all_used:
                    score = self._fuzzy_match_score(gt, ut)
                    if score > best_score:
                        best_score = score
                    if best_score == 1.0:
                        break
                total_credit += best_score

            base_coverage = min(total_credit / max(len(gt_set), 1), 1.0)

            gt_match_positions = []
            for gi, gt in enumerate(gt_list):
                best_score = 0.0
                best_ui = -1
                for ui, ut in enumerate(used_list):
                    score = self._fuzzy_match_score(gt, ut)
                    if score > best_score:
                        best_score = score
                        best_ui = ui
                    if best_score == 1.0:
                        break
                if best_score > 0:
                    gt_match_positions.append((gi, best_ui, best_score))

            if len(gt_match_positions) >= 2:
                pos_seq = [ui for _, ui, _ in gt_match_positions]
                tails = []
                for p in pos_seq:
                    idx = bisect.bisect_left(tails, p)
                    if idx == len(tails):
                        tails.append(p)
                    else:
                        tails[idx] = p
                order_bonus = len(tails) / len(gt_match_positions)
            elif len(gt_match_positions) == 1:
                order_bonus = 1.0
            else:
                order_bonus = 0.0

            r_task = base_coverage * (0.7 + 0.3 * order_bonus)

            if r_task == 0.0 and completed:
                r_task = 0.1
        elif completed and all_used:
            r_task = 0.15
        elif completed:
            r_task = 0.0
        else:
            r_task = 0.0

        total = max(r_task, 0.0)
        return total, {"r_task": round(r_task, 4)}


def train_standard_grpo(
    model_name: str,
    sft_checkpoint_dir: str,
    output_dir: str,
    grpo_config: GRPOConfig,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType

    model_path = get_model_path(model_name)
    logger.info(f"Loading base model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(os.path.join(sft_checkpoint_dir, "adapter_config.json")):
        logger.info(f"Loading SFT LoRA from {sft_checkpoint_dir}")
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, sft_checkpoint_dir)
        model = model.merge_and_unload()
    else:
        logger.info("No SFT checkpoint; starting from base model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=grpo_config.lora_rank, lora_alpha=grpo_config.lora_alpha,
        lora_dropout=grpo_config.lora_dropout,
        target_modules=grpo_config.lora_target_modules, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    device = next(model.parameters()).device

    env = ToolEnvironment(AUGMENTED_TOOLS_PATH, TOOL_SIMULATOR_DB_PATH, RL_DATASET_PATH)
    reward_fn = TaskReward()

    with open(AUGMENTED_TOOLS_PATH, "r") as f:
        all_augmented_tools = json.load(f)

    tool_desc_map = {}
    for t in all_augmented_tools:
        norm_name = normalize_tool_name(t["name"])
        params = t.get("parameters_schema", {})
        param_str = ""
        if params:
            props = params.get("properties", {})
            if props:
                param_names = list(props.keys())[:5]
                param_str = f" params: {param_names}"
        tool_desc_map[norm_name] = (
            f"- {norm_name}: {t.get('description', '')[:100]}{param_str}"
        )

    def build_system_prompt(task_name: str) -> str:
        lines = []
        seen = set()

        task_lower = task_name.lower().replace("-", "_")
        category_keywords = set()
        for part in task_lower.split("_"):
            if len(part) >= 3:
                category_keywords.add(part)

        for tn, desc in tool_desc_map.items():
            if tn in seen:
                continue
            tn_lower = tn.lower().replace("-", "_")
            if any(kw in tn_lower for kw in category_keywords):
                lines.append(desc)
                seen.add(tn)
            if len(seen) >= 40:
                break

        other_tools = [tn for tn in tool_desc_map if tn not in seen]
        random.shuffle(other_tools)
        for tn in other_tools:
            lines.append(tool_desc_map[tn])
            seen.add(tn)
            if len(seen) >= 50:
                break

        tool_list = "\n".join(lines)
        return (
            "You are a tool-calling agent. You MUST use tools to complete tasks. "
            "Do NOT answer directly — always call at least one tool first.\n\n"
            f"Available tools ({len(seen)} total):\n"
            f"{tool_list}\n\n"
            "To call a tool, respond ONLY with:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"param": "value"}}\n'
            "</tool_call>\n\n"
            "After receiving all tool responses, provide a brief text summary to finish."
        )

    with open(RL_DATASET_PATH, "r") as f:
        rl_data = json.load(f)

    train_prompts = []
    for ep in rl_data.get("episodes", []):
        if ep.get("success", 0) == 1 and ep.get("user_prompt") and ep.get("tool_names"):
            raw_gt = ep.get("tool_names", [])
            norm_gt = list(dict.fromkeys(normalize_tool_name(t) for t in raw_gt))
            train_prompts.append({
                "user_prompt": ep["user_prompt"],
                "task_name": ep.get("task_name", ""),
                "gt_tools": norm_gt,
            })
    logger.info(f"Training prompts: {len(train_prompts)}")

    G = grpo_config.group_size
    num_epochs = grpo_config.num_epochs
    lr = grpo_config.learning_rate
    grad_accum = grpo_config.gradient_accumulation_steps
    max_turns = 10
    max_gen_tok = grpo_config.max_gen_length
    base_temp = grpo_config.temperature

    total_prompts = len(train_prompts) * num_epochs
    total_steps = total_prompts // grad_accum
    warmup_steps = int(total_steps * 0.1)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, max(total_steps, 1))

    log_path = os.path.join(output_dir, "execution_log.jsonl")
    exec_logger = ExecutionLogger(log_path)

    logger.info(
        f"Standard GRPO  G={G}  epochs={num_epochs}  lr={lr}  "
        f"grad_accum={grad_accum}  total_steps≈{total_steps}"
    )

    global_step = 0
    acc_loss = acc_reward = 0.0
    acc_cnt = 0
    best_reward = -1e9
    save_steps = getattr(grpo_config, "save_steps", 100)

    for epoch in range(num_epochs):
        random.shuffle(train_prompts)
        logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")

        for pidx, pdata in enumerate(train_prompts):
            user_prompt = pdata["user_prompt"]
            gt_tools = pdata["gt_tools"]
            task_name = pdata["task_name"]
            system_prompt = build_system_prompt(task_name)

            rollouts = []
            for g in range(G):
                t = base_temp * (0.6 + g * 0.3)
                t = max(0.1, min(t, 1.5))
                if g == G - 1:
                    t = min(base_temp * 1.8, 2.0)

                rollout = run_rollout(
                    model, tokenizer, env,
                    system_prompt, user_prompt,
                    max_turns=max_turns, max_new_tokens=max_gen_tok,
                    temperature=t, device=device,
                    oracle_first_tool=None,
                    gt_tools_len=len(gt_tools),
                )
                used_tools = [name for name, _ in rollout["actions"]]
                reward, breakdown = reward_fn.compute(
                    used_tools=used_tools,
                    gt_tools=gt_tools,
                    skill_traces=rollout["skill_traces"],
                    num_decision_steps=rollout["num_steps"],
                    completed=rollout["completed"],
                )
                rollout["reward"] = reward
                rollout["reward_breakdown"] = breakdown
                rollouts.append(rollout)

            has_any_action = any(r["num_steps"] > 0 for r in rollouts)
            resample_attempts = 0
            while not has_any_action and resample_attempts < 3:
                resample_attempts += 1
                t = base_temp * (1.5 + resample_attempts * 0.5)
                t = min(t, 2.0)
                rollout = run_rollout(
                    model, tokenizer, env,
                    system_prompt, user_prompt,
                    max_turns=max_turns, max_new_tokens=max_gen_tok,
                    temperature=t, device=device,
                    oracle_first_tool=None,
                    gt_tools_len=len(gt_tools),
                )
                used_tools = [name for name, _ in rollout["actions"]]
                reward, breakdown = reward_fn.compute(
                    used_tools=used_tools,
                    gt_tools=gt_tools,
                    skill_traces=rollout["skill_traces"],
                    num_decision_steps=rollout["num_steps"],
                    completed=rollout["completed"],
                )
                rollout["reward"] = reward
                rollout["reward_breakdown"] = breakdown
                if rollout["num_steps"] > 0:
                    for ri in range(len(rollouts)):
                        if rollouts[ri]["num_steps"] == 0:
                            rollouts[ri] = rollout
                            break
                    has_any_action = True


            rewards = [r["reward"] for r in rollouts]
            mu = sum(rewards) / len(rewards)
            raw_std = math.sqrt(sum((r - mu)**2 for r in rewards) / len(rewards))
            if raw_std < 1e-6:
                for r in rollouts:
                    r["advantage"] = 0.0
            else:
                std = max(raw_std, 0.05)
                for i, r in enumerate(rollouts):
                    r["advantage"] = (rewards[i] - mu) / std
                    r["advantage"] = max(-3.0, min(3.0, r["advantage"]))

            exec_logger.log_prompt(
                epoch, pidx, global_step, user_prompt, gt_tools, rollouts)

            if pidx < 5 or pidx % 50 == 0:
                r_str = " ".join(f"{r['reward']:.2f}" for r in rollouts)
                logger.info(
                    f"[{pidx+1}/{len(train_prompts)}] rewards=[{r_str}] "
                    f"mu={mu:.3f} std={raw_std:.3f}"
                )

            model.train()
            for rollout in rollouts:
                adv = rollout["advantage"]
                if rollout["num_steps"] == 0:
                    continue
                try:
                    input_ids, attn_mask, labels = tokenize_with_assistant_mask(
                        rollout["messages"], tokenizer,
                        max_length=grpo_config.max_seq_length,
                    )
                except Exception:
                    continue
                n_asst = (labels != -100).sum().item()
                if n_asst == 0:
                    continue

                input_ids = input_ids.unsqueeze(0).to(device)
                attn_mask = attn_mask.unsqueeze(0).to(device)
                labels = labels.unsqueeze(0).to(device)

                out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                pg_loss = adv * out.loss / (G * grad_accum)
                if torch.isfinite(pg_loss):
                    pg_loss.backward()
                    acc_loss += pg_loss.item()
                    acc_reward += rollout["reward"]
                    acc_cnt += 1

            if (pidx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if acc_cnt > 0:
                    avg_loss = acc_loss / acc_cnt
                    avg_reward = acc_reward / acc_cnt
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                    logger.info(
                        f"  step {global_step}  loss={avg_loss:.4f}  "
                        f"avg_r={avg_reward:.3f}  best_r={best_reward:.3f}  "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                acc_loss = acc_reward = 0.0
                acc_cnt = 0

                if global_step % save_steps == 0:
                    ckpt = os.path.join(output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    logger.info(f"  Saved checkpoint: {ckpt}")

        remaining = len(train_prompts) % grad_accum
        if remaining > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    exec_logger.save_summary()
    logger.info(f"Training complete. Final checkpoint: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Standard GRPO Training (R=R_task only)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=["qwen2.5-1.5b", "qwen2.5-7b", "llama3.1-8b", "llama3.2-3b"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    sft_dir = os.path.join(CHECKPOINT_DIR, "sft", args.model)
    output_dir = os.path.join(CHECKPOINT_DIR, "standard_grpo", args.model)
    os.makedirs(output_dir, exist_ok=True)

    config = GRPOConfig()
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.group_size = args.group_size
    config.gradient_accumulation_steps = args.grad_accum

    logger.info(f"Model: {args.model}")
    logger.info(f"SFT checkpoint: {sft_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Epochs: {config.num_epochs}, G: {config.group_size}, LR: {config.learning_rate}")

    train_standard_grpo(args.model, sft_dir, output_dir, config)


if __name__ == "__main__":
    main()
