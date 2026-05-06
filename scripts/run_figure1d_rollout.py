#!/usr/bin/env python3
"""
Figure 1 Panel (d): State-dependent granularity — Counterfactual Rollout
=========================================================================

Runs real counterfactual rollouts to compare opportunity distribution and
skill invocation rate across different base policies (SFT, Standard GRPO).

Usage (on server):
  cd /path/to/CIPO
  conda activate tool

  # SFT policy
  python scripts/run_figure1d_rollout.py \
      --model qwen2.5-7b \
      --policy_name sft \
      --lora_path outputs/TOOLATHLON/checkpoints/sft/qwen2.5-7b \
      --n-prompts 200

  # Standard GRPO policy (replay same prompts)
  python scripts/run_figure1d_rollout.py \
      --model qwen2.5-7b \
      --policy_name standard_grpo \
      --lora_path outputs/TOOLATHLON/checkpoints/grpo/qwen2.5-7b \
      --replay_from ../figure1/outputs/exp_c_full_sft.json

Output:
  ../figure1/outputs/exp_c_{policy_name}.json
  ../figure1/outputs/exp_c_full_{policy_name}.json
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[fig1d] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Match project convention: add code/ to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent          # .../code/scripts/
CODE_DIR = SCRIPT_DIR.parent                          # .../code/
sys.path.insert(0, str(CODE_DIR))

from configs.config import (
    RL_DATASET_PATH, TOOL_SIMULATOR_DB_PATH,
    AUGMENTED_TOOLS_PATH, ADAMACRO_OUTPUT_DIR,
    GRPOConfig, get_model_path,
)
from step4_gipo_training import (
    ToolEnvironment, AdaMacroReward, run_rollout,
    find_counterfactual_action, run_imagination_branch,
    normalize_tool_name,
)

OUTPUT_DIR = CODE_DIR.parent / "figure1" / "outputs"


def load_prompts(n_prompts: int, seed: int):
    """Load n_prompts from the RL dataset (successful episodes only)."""
    with open(RL_DATASET_PATH, "r") as f:
        rl_data = json.load(f)

    prompts = []
    for ep in rl_data.get("episodes", []):
        if ep.get("success", 0) != 1:
            continue
        if not ep.get("user_prompt") or not ep.get("tool_names"):
            continue
        raw_gt = ep.get("tool_names", [])
        norm_gt = list(dict.fromkeys(normalize_tool_name(t) for t in raw_gt))
        prompts.append({
            "user_prompt": ep["user_prompt"],
            "task_name": ep.get("task_name", ""),
            "gt_tools": norm_gt,
        })

    random.seed(seed)
    random.shuffle(prompts)
    selected = prompts[:n_prompts]
    logger.info(f"Selected {len(selected)}/{len(prompts)} prompts (seed={seed})")
    return selected


def load_prompts_from_replay(replay_path: str, all_prompts_seed: int):
    """
    Reproduce the exact prompt list from a previous run by replaying the
    same seed-based shuffle, then filter to only the prompt_indices that
    produced divergence states.
    """
    with open(replay_path, "r") as f:
        prev_data = json.load(f)

    prev_records = prev_data.get("records", [])
    # Get the unique prompt_indices and original n_prompts
    used_indices = sorted(set(r["prompt_idx"] for r in prev_records))
    # Infer n_prompts from the max index + 1
    n_prompts = max(used_indices) + 1

    # Reproduce the same prompt list
    all_prompts = load_prompts(n_prompts, seed=all_prompts_seed)

    logger.info(f"Replay: {len(used_indices)} prompts with divergence states "
                f"(out of {n_prompts} total)")
    return all_prompts, used_indices


def build_system_prompt(env, tool_desc_map, norm_skill_names, task_name):
    """Build per-prompt system prompt (same logic as train_grpo)."""
    lines = []
    seen = set()

    task_lower = task_name.lower().replace("-", "_")
    category_keywords = set()
    for part in task_lower.split("_"):
        if len(part) >= 3:
            category_keywords.add(part)

    for tn, desc in tool_desc_map.items():
        if tn in norm_skill_names:
            lines.append(desc)
            seen.add(tn)
        if len(seen) >= 15:
            break

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
    n_skills = sum(1 for tn in seen if tn in norm_skill_names)

    return (
        "You are a tool-calling agent. You MUST use tools to complete tasks. "
        "Do NOT answer directly — always call at least one tool first.\n\n"
        "You have access to both atomic tools and composite skills. "
        "Skills are pre-composed tool chains that execute multiple tools in sequence. "
        "Choose whichever tools (atomic or skill) best fit the task.\n\n"
        f"Available tools ({len(seen)} total, including {n_skills} skills):\n"
        f"{tool_list}\n\n"
        "To call a tool, respond ONLY with:\n"
        "<tool_call>\n"
        '{"name": "tool_name", "arguments": {"param": "value"}}\n'
        "</tool_call>\n\n"
        "After receiving all tool responses, provide a brief text summary to finish."
    )


def run_experiment(model_name: str, policy_name: str, lora_path: str,
                   n_prompts: int, seed: int = 42,
                   replay_from: str = None):
    """
    Main experiment: for each prompt, generate base rollout, find divergence
    state, fork counterfactual branch, compare rewards.
    """
    import torch

    random.seed(seed)
    torch.manual_seed(seed)

    # ── Load model ──
    model_path = get_model_path(model_name)
    logger.info(f"Loading model: {model_path}")
    logger.info(f"Policy: {policy_name}, LoRA: {lora_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA checkpoint (fallback: check checkpoint-* subdirs)
    resolved_lora = None
    if lora_path:
        if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            resolved_lora = lora_path
        else:
            # Try latest checkpoint-N subdirectory
            import glob
            ckpt_dirs = sorted(
                glob.glob(os.path.join(lora_path, "checkpoint-*")),
                key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else 0,
            )
            for d in reversed(ckpt_dirs):
                if os.path.exists(os.path.join(d, "adapter_config.json")):
                    resolved_lora = d
                    break

    if resolved_lora:
        logger.info(f"Loading LoRA from {resolved_lora}")
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, resolved_lora)
        model = model.merge_and_unload()
    else:
        logger.info("No LoRA checkpoint found; using base model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )

    model.eval()
    device = next(model.parameters()).device
    logger.info(f"Model loaded on {device}")

    # ── Environment ──
    env = ToolEnvironment(AUGMENTED_TOOLS_PATH, TOOL_SIMULATOR_DB_PATH, RL_DATASET_PATH)
    config = GRPOConfig()
    reward_fn = AdaMacroReward(config, skill_definitions=env.skills)

    # Build tool descriptions
    with open(AUGMENTED_TOOLS_PATH, "r") as f:
        all_tools = json.load(f)

    tool_desc_map = {}
    norm_skill_names = set(normalize_tool_name(s) for s in env.skills)
    for t in all_tools:
        orig_name = t["name"]
        norm_name = normalize_tool_name(orig_name)
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
                norm_chain = [normalize_tool_name(c) for c in chain[:5]]
                chain_str = f" [chain: {' -> '.join(norm_chain)}]"
        tool_desc_map[norm_name] = (
            f"- {tag}{norm_name}: {t.get('description','')[:100]}{chain_str}{param_str}"
        )

    # Build skill chains for counterfactual lookup
    skill_chains = {}
    for sname, sdef in env.skills.items():
        chain = sdef.get("tool_chain", [])
        if not chain:
            chain = [s.get("tool_name", "") for s in sdef.get("execution_plan", [])]
        skill_chains[sname] = [normalize_tool_name(t) for t in chain]

    # ── Load prompts ──
    prompt_filter = None
    if replay_from:
        prompts, prompt_filter = load_prompts_from_replay(replay_from, seed)
    else:
        prompts = load_prompts(n_prompts, seed)

    # ── Compute reward helper ──
    def compute_reward(ro, gt_tools):
        ut = [name for name, _ in ro["actions"]]
        sn = list(ro.get("skill_names_used", []))
        rw, bd = reward_fn.compute(
            used_tools=ut, gt_tools=gt_tools,
            skill_traces=ro.get("skill_traces", []),
            skill_names=sn,
            num_decision_steps=ro["num_steps"],
            num_skill_calls=ro.get("num_skill_calls", 0),
            total_atomic_cost=ro.get("total_atomic", len(ut)),
            completed=ro.get("completed", False),
            max_steps=10,
        )
        return rw, bd

    # ── Run rollouts ──
    max_turns = 10
    max_gen_tok = 512
    temperature = 0.7
    THRESHOLD = 0.05

    n_skill_better = 0
    n_comparable = 0
    n_atomic_better = 0
    n_no_branch = 0
    n_total_branches = 0
    records = []

    def classify_branch(base_bd, branch_bd, base_ro, branch, cf_action,
                        actions, branch_step):
        base_rtask = base_bd.get("r_task", 0)
        branch_rtask = branch_bd.get("r_task", 0)

        original_tool = actions[branch_step][0]
        original_is_skill = original_tool in env.skills or any(
            normalize_tool_name(s) == normalize_tool_name(original_tool)
            for s in env.skills
        )

        delta = branch_rtask - base_rtask

        if abs(delta) <= THRESHOLD:
            category = "comparable"
        elif delta > THRESHOLD:
            category = "skill_better" if cf_action["is_skill"] else "atomic_better"
        else:
            category = "skill_better" if original_is_skill else "atomic_better"

        return category, {
            "original_tool": original_tool,
            "original_is_skill": original_is_skill,
            "cf_tool": cf_action["name"],
            "cf_is_skill": cf_action["is_skill"],
            "base_rtask": round(base_rtask, 4),
            "branch_rtask": round(branch_rtask, 4),
            "delta_rtask": round(delta, 4),
            "category": category,
        }

    def compute_prefix_info(actions, step_idx):
        prefix_atomic = 0
        prefix_skill_traces = []
        prefix_skill_names = []
        for pa_name, _ in actions[:step_idx]:
            pa_norm = normalize_tool_name(pa_name)
            is_ps = (pa_name in env.skills or any(
                normalize_tool_name(s) == pa_norm for s in env.skills
            ))
            if is_ps:
                sk_def = env.skills.get(pa_name)
                if not sk_def:
                    for sn, sd in env.skills.items():
                        if normalize_tool_name(sn) == pa_norm:
                            sk_def = sd
                            break
                if sk_def:
                    chain = sk_def.get("tool_chain", [])
                    if not chain:
                        chain = [s.get("tool_name", "") for s in sk_def.get("execution_plan", [])]
                    prefix_atomic += max(len(chain), 1)
                else:
                    prefix_atomic += 1
            else:
                prefix_atomic += 1
        return prefix_atomic, prefix_skill_traces, prefix_skill_names

    def process_base_rollout(base_ro, gt_tools, pi, rollout_tag=""):
        nonlocal n_skill_better, n_comparable, n_atomic_better, n_no_branch, n_total_branches

        if base_ro["num_steps"] == 0:
            n_no_branch += 1
            return

        base_rw, base_bd = compute_reward(base_ro, gt_tools)
        actions = base_ro["actions"]
        offsets = base_ro.get("action_msg_offsets", [])
        found_any = False

        for si, (tool_name, tool_args) in enumerate(actions):
            if si >= len(offsets):
                break

            is_skill = tool_name in env.skills or any(
                normalize_tool_name(s) == normalize_tool_name(tool_name)
                for s in env.skills
            )
            cf = find_counterfactual_action(
                chosen_tool=tool_name,
                is_skill=is_skill,
                skills=env.skills,
                skill_chains=skill_chains,
                original_arguments=tool_args,
            )
            if cf is None:
                continue

            found_any = True
            msg_offset = offsets[si]
            prefix_messages = base_ro["messages"][:msg_offset]
            prefix_actions = actions[:si]
            prefix_atomic, prefix_st, prefix_sn = compute_prefix_info(actions, si)

            with torch.no_grad():
                branch = run_imagination_branch(
                    model, tokenizer, env,
                    prefix_messages=prefix_messages,
                    prefix_actions=prefix_actions,
                    cf_tool_name=cf["name"],
                    cf_arguments=cf["arguments"],
                    max_turns=max_turns,
                    max_new_tokens=max_gen_tok,
                    temperature=temperature,
                    device=device,
                    gt_tools_len=len(gt_tools),
                    prefix_total_atomic=prefix_atomic,
                    prefix_skill_traces=prefix_st,
                    prefix_skill_names=prefix_sn,
                )

            branch_rw, branch_bd = compute_reward(branch, gt_tools)
            n_total_branches += 1

            category, record = classify_branch(
                base_bd, branch_bd, base_ro, branch, cf, actions, si
            )
            record["prompt_idx"] = pi
            record["branch_step"] = si
            record["rollout_tag"] = rollout_tag
            records.append(record)

            if category == "skill_better":
                n_skill_better += 1
            elif category == "atomic_better":
                n_atomic_better += 1
            else:
                n_comparable += 1

        if not found_any:
            n_no_branch += 1

    # ── Main loop ──
    for pi, pdata in enumerate(prompts):
        # If replaying, only process prompts that had divergence states
        if prompt_filter is not None and pi not in prompt_filter:
            continue

        user_prompt = pdata["user_prompt"]
        gt_tools = pdata["gt_tools"]
        task_name = pdata["task_name"]

        sys_prompt = build_system_prompt(env, tool_desc_map, norm_skill_names, task_name)

        skill_biased_prompt = sys_prompt + (
            "\n\nIMPORTANT: You SHOULD prefer [SKILL] tools over atomic tools when possible. "
            "Skills chain multiple steps and are more efficient. "
            "Check the [SKILL] entries in the tool list first."
        )

        # Base rollout 1: normal
        with torch.no_grad():
            base_0 = run_rollout(
                model, tokenizer, env,
                sys_prompt, user_prompt,
                max_turns=max_turns, max_new_tokens=max_gen_tok,
                temperature=temperature, device=device,
                gt_tools_len=len(gt_tools),
            )
        process_base_rollout(base_0, gt_tools, pi, rollout_tag="normal")

        # Base rollout 2: skill-biased
        with torch.no_grad():
            base_1 = run_rollout(
                model, tokenizer, env,
                skill_biased_prompt, user_prompt,
                max_turns=max_turns, max_new_tokens=max_gen_tok,
                temperature=max(0.1, temperature * 0.8), device=device,
                gt_tools_len=len(gt_tools),
            )
        process_base_rollout(base_1, gt_tools, pi, rollout_tag="skill_biased")

        if (pi + 1) % 10 == 0 or pi < 5:
            logger.info(
                f"[{pi+1}/{len(prompts)}] branches_so_far={n_total_branches}  "
                f"S={n_skill_better} C={n_comparable} A={n_atomic_better}  "
                f"no_branch={n_no_branch}"
            )

    # ── Compute results ──
    n_total = n_skill_better + n_comparable + n_atomic_better
    if n_total == 0:
        logger.warning("No branches generated!")
        return _mock_result(policy_name)

    pct_skill = round(n_skill_better / n_total * 100, 1)
    pct_comp = round(n_comparable / n_total * 100, 1)
    pct_atomic = round(n_atomic_better / n_total * 100, 1)

    # Skill invocation rate per category
    sir = {}
    for cat in ["skill_better", "comparable", "atomic_better"]:
        cat_records = [r for r in records if r["category"] == cat]
        if cat_records:
            chosen_skill = sum(1 for r in cat_records if r["original_is_skill"])
            sir[cat] = round(chosen_skill / len(cat_records) * 100, 1)
        else:
            sir[cat] = 0.0
    all_skill = sum(1 for r in records if r["original_is_skill"])
    sir["overall"] = round(all_skill / max(len(records), 1) * 100, 1)

    logger.info(f"\n{'='*60}")
    logger.info(f"Results: {policy_name} ({n_total} divergence states, {n_no_branch} no-branch)")
    logger.info(f"  Distribution:  Skill={pct_skill}% ({n_skill_better})  "
                f"Comp={pct_comp}% ({n_comparable})  Atomic={pct_atomic}% ({n_atomic_better})")
    logger.info(f"  Skill invoc:   Skill-better={sir['skill_better']:.1f}%  "
                f"Comp={sir['comparable']:.1f}%  Atomic-better={sir['atomic_better']:.1f}%  "
                f"Overall={sir['overall']:.1f}%")
    logger.info(f"{'='*60}")

    result = {
        "policy_name": policy_name,
        "model": model_name,
        "lora_path": lora_path or "none",
        "n_states": n_total,
        "n_no_branch": n_no_branch,
        "threshold": "r_task delta > 0.05",
        "categories": ["Skill\nbetter", "Comparable", "Atomic\nbetter"],
        "percentages": [pct_skill, pct_comp, pct_atomic],
        "raw_counts": {
            "skill_better": n_skill_better,
            "comparable": n_comparable,
            "atomic_better": n_atomic_better,
        },
        "skill_invocation_rate": sir,
        "records": records,
    }
    return result


def _mock_result(policy_name="unknown"):
    return {
        "policy_name": policy_name,
        "n_states": 0,
        "categories": ["Skill\nbetter", "Comparable", "Atomic\nbetter"],
        "percentages": [38, 30, 32],
        "skill_invocation_rate": {
            "skill_better": 50, "comparable": 50, "atomic_better": 50, "overall": 50
        },
        "note": "mock data — rollout failed",
    }


def save_result(result, policy_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    compact_path = OUTPUT_DIR / f"exp_c_{policy_name}.json"
    full_path = OUTPUT_DIR / f"exp_c_full_{policy_name}.json"

    # Full result (with records)
    with open(full_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Full results saved to {full_path}")

    # Compact result for plotting (no records)
    compact = {k: v for k, v in result.items() if k != "records"}
    with open(compact_path, "w") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)
    logger.info(f"Compact results saved to {compact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Figure 1(d): Counterfactual rollout experiment")
    parser.add_argument("--model", default="qwen2.5-7b", help="Base model name")
    parser.add_argument("--policy_name", required=True,
                        help="Policy identifier (e.g., sft, standard_grpo)")
    parser.add_argument("--lora_path", default=None,
                        help="Path to LoRA checkpoint for this policy")
    parser.add_argument("--n-prompts", type=int, default=200,
                        help="Number of prompts to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--replay_from", default=None,
                        help="Path to previous exp_c_full_*.json to reuse same prompts")
    args = parser.parse_args()

    result = run_experiment(
        model_name=args.model,
        policy_name=args.policy_name,
        lora_path=args.lora_path,
        n_prompts=args.n_prompts,
        seed=args.seed,
        replay_from=args.replay_from,
    )
    save_result(result, args.policy_name)
