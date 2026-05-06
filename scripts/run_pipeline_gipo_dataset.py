"""
AdaMacro: Universal Multi-Dataset GIPO Pipeline Runner
=======================================================

A single pipeline script that works with ANY dataset (TOOLATHLON, Toucan,
Traject-bench, etc.) by accepting dataset paths via command-line arguments.

Supports both single-GPU (1.5B/3B) and 2-GPU (7B/8B) training.

Usage:
  # Toucan + qwen2.5-1.5b (single GPU)
  python run_pipeline_gipo_dataset.py \
      --dataset toucan \
      --rl-dataset /path/to/rl_dataset_toucan.json \
      --all-tools /path/to/all_tools_toucan.json \
      --tool-simulator-db /path/to/tool_simulator_database_toucan.json \
      --output-dir /path/to/outputs/toucan \
      --model qwen2.5-1.5b --steps 1,2,3,4,5

  # Traject-bench + llama3.1-8b (2 GPU)
  python run_pipeline_gipo_dataset.py \
      --dataset traject \
      --rl-dataset /path/to/rl_dataset_llm.json \
      --all-tools /path/to/all_tools.json \
      --tool-simulator-db /path/to/tool_simulator_database.json \
      --output-dir /path/to/outputs/traject \
      --model llama3.1-8b --steps 1,2,3,4,5 --gpu-mode 2gpu
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    BPEConfig, SkillConfig, SFTConfig, GRPOConfig, GIPO3BConfig, GIPO7BConfig,
    EvalConfig, get_model_path,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Model → Config mapping
# ============================================================================

# Models that should use 2-GPU config (GIPO7BConfig)
_2GPU_MODELS = {"qwen2.5-7b", "llama3.1-8b"}

# Models that should use 3B config (GIPO3BConfig)
_3B_MODELS = {"llama3.2-3b"}


def _model_short(model_name: str) -> str:
    """Convert model name to directory-safe short name.
    e.g. 'qwen2.5-1.5b' -> 'qwen25_15b', 'llama3.2-3b' -> 'llama32_3b'
    """
    return model_name.replace(".", "").replace("-", "_")


def _get_gipo_config(model_name: str, gpu_mode: str):
    """Select the right GIPO config based on model size and GPU mode."""
    if gpu_mode == "2gpu" or model_name in _2GPU_MODELS:
        return GIPO7BConfig()
    elif model_name in _3B_MODELS:
        return GIPO3BConfig()
    else:
        return GRPOConfig()  # default for 1.5B


# ============================================================================
# Path helpers
# ============================================================================

class DatasetPaths:
    """Holds all resolved paths for a dataset + model combination."""

    def __init__(self, output_dir: str, model_name: str, gpu_mode: str,
                 rl_dataset: str, all_tools: str, tool_simulator_db: str):
        self.rl_dataset = rl_dataset
        self.all_tools = all_tools
        self.tool_simulator_db = tool_simulator_db
        self.output_dir = output_dir

        # Intermediate outputs (per-dataset, shared across models)
        self.skill_library = os.path.join(output_dir, "skill_library.json")
        self.augmented_tools = os.path.join(output_dir, "augmented_tools.json")
        self.sft_data = os.path.join(output_dir, "sft_data.json")
        self.grpo_data = os.path.join(output_dir, "grpo_data.json")

        # Checkpoints (per-dataset, per-model)
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.sft_dir = os.path.join(self.checkpoint_dir, "sft", model_name)

        # GIPO checkpoint: e.g. checkpoints/gipo_qwen25_15b/ or checkpoints/gipo_2gpu_llama31_8b/
        prefix = "gipo_2gpu" if gpu_mode == "2gpu" else "gipo"
        self.gipo_dir = os.path.join(self.checkpoint_dir, f"{prefix}_{_model_short(model_name)}")

        # Eval results
        self.eval_dir = os.path.join(output_dir, f"eval_{prefix}_{_model_short(model_name)}_results")

    def ensure_dirs(self):
        """Create all necessary directories."""
        for d in [self.output_dir, self.checkpoint_dir, self.sft_dir,
                  self.gipo_dir, self.eval_dir]:
            os.makedirs(d, exist_ok=True)


# ============================================================================
# Pipeline Steps
# ============================================================================

def run_step1(args, paths: DatasetPaths):
    """BPE Macro Mining."""
    from step1_bpe_mining import load_successful_sequences, BPEMacroMiner, load_tool_schemas
    import json

    bpe_config = BPEConfig(
        max_merges=args.max_merges,
        min_freq=args.min_freq,
        max_macro_len=args.max_macro_len,
    )

    sequences = load_successful_sequences(paths.rl_dataset, bpe_config.success_only)
    if not sequences:
        logger.error("No sequences found!")
        return

    miner = BPEMacroMiner(bpe_config)
    macros = miner.mine(sequences)

    tool_schemas = load_tool_schemas(paths.all_tools)
    for mid, macro in macros.items():
        enriched = []
        for tname in macro["tool_names"]:
            if tname in tool_schemas:
                enriched.append({
                    "name": tname,
                    "description": tool_schemas[tname]["description"][:200],
                    "params": tool_schemas[tname]["actual_keys"],
                })
            else:
                enriched.append({"name": tname, "description": "", "params": []})
        macro["tool_details"] = enriched

    output = {
        "metadata": {
            "algorithm": "BPE", "max_merges": bpe_config.max_merges,
            "min_freq": bpe_config.min_freq, "num_macros": len(macros),
            "num_sequences": len(sequences),
        },
        "macros": macros,
        "merge_history": miner.merge_history,
    }
    Path(paths.skill_library).parent.mkdir(parents=True, exist_ok=True)
    with open(paths.skill_library, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Step 1 done: {len(macros)} macros -> {paths.skill_library}")


def run_step2(args, paths: DatasetPaths):
    """Skill Instantiation."""
    from step2_skill_instantiation import build_augmented_tools
    import json

    tool_schemas = {}
    with open(paths.all_tools, "r") as f:
        tools = json.load(f)
    for t in tools:
        tool_schemas[t.get("name", "")] = t

    augmented = build_augmented_tools(
        paths.all_tools, paths.skill_library, tool_schemas, SkillConfig()
    )
    Path(paths.augmented_tools).parent.mkdir(parents=True, exist_ok=True)
    with open(paths.augmented_tools, "w") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)
    logger.info(f"Step 2 done: {len(augmented)} tools -> {paths.augmented_tools}")


def run_step3(args, paths: DatasetPaths):
    """SFT Training."""
    from step3_sft_training import generate_sft_data, train_sft

    sft_config = SFTConfig()
    if args.epochs: sft_config.num_epochs = args.epochs
    if args.lr: sft_config.learning_rate = args.lr
    if args.batch_size: sft_config.per_device_batch_size = args.batch_size
    if args.lora_rank: sft_config.lora_rank = args.lora_rank

    os.makedirs(paths.sft_dir, exist_ok=True)

    logger.info("[Step 3.1] Generating SFT data...")
    generate_sft_data(paths.rl_dataset, paths.augmented_tools, paths.skill_library, paths.sft_data)

    logger.info(f"[Step 3.2] Training SFT with {args.model}...")
    train_sft(args.model, paths.sft_data, paths.sft_dir, sft_config)
    logger.info(f"Step 3 done: SFT checkpoint -> {paths.sft_dir}")


def run_step4(args, paths: DatasetPaths):
    """GIPO Training (single-GPU or 2-GPU based on gpu_mode)."""

    gipo_config = _get_gipo_config(args.model, args.gpu_mode)
    if args.epochs: gipo_config.num_epochs = args.epochs
    if args.lr: gipo_config.learning_rate = args.lr
    if args.group_size: gipo_config.group_size = args.group_size

    os.makedirs(paths.gipo_dir, exist_ok=True)

    if args.gpu_mode == "2gpu":
        from step4_gipo_training_2gpu import generate_grpo_rollouts, train_grpo
    else:
        from step4_gipo_training import generate_grpo_rollouts, train_grpo

    generate_grpo_rollouts()

    logger.info(f"[Step 4] GIPO training with {args.model} ({args.gpu_mode})...")
    logger.info(f"  SFT checkpoint: {paths.sft_dir}")
    logger.info(f"  GIPO output:    {paths.gipo_dir}")
    logger.info(f"  Config: lr={gipo_config.learning_rate} lora_rank={gipo_config.lora_rank}")
    train_grpo(args.model, paths.sft_dir, paths.grpo_data, paths.gipo_dir, gipo_config)
    logger.info(f"Step 4 done: GIPO checkpoint -> {paths.gipo_dir}")


def run_step5(args, paths: DatasetPaths):
    """Evaluation."""
    from step5_evaluation import evaluate

    eval_config = EvalConfig(
        max_turns=args.max_turns,
        max_atomic_calls=args.max_atomic_calls,
    )

    if args.stage == "base":
        lora_path = None
    elif args.stage == "sft":
        lora_path = paths.sft_dir
    else:
        lora_path = paths.gipo_dir

    os.makedirs(paths.eval_dir, exist_ok=True)
    output_path = os.path.join(
        paths.eval_dir,
        f"eval_{args.model}_{args.stage}_{int(time.time())}.json"
    )

    logger.info(f"[Step 5] Evaluating {args.model} ({args.stage})...")
    logger.info(f"  LoRA path: {lora_path}")
    logger.info(f"  Output:    {output_path}")
    evaluate(
        model_name=args.model, lora_path=lora_path,
        rl_dataset_path=paths.rl_dataset, augmented_tools_path=paths.augmented_tools,
        tool_simulator_db_path=paths.tool_simulator_db,
        output_path=output_path, eval_config=eval_config,
        max_episodes=args.max_episodes,
    )
    logger.info(f"Step 5 done: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AdaMacro: Universal Multi-Dataset GIPO Pipeline")

    # --- Dataset paths (REQUIRED) ---
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (e.g., toucan, traject)")
    parser.add_argument("--rl-dataset", type=str, required=True,
                       help="Path to rl_dataset JSON")
    parser.add_argument("--all-tools", type=str, required=True,
                       help="Path to all_tools JSON")
    parser.add_argument("--tool-simulator-db", type=str, required=True,
                       help="Path to tool_simulator_database JSON")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for this dataset")

    # --- Model & steps ---
    parser.add_argument("--model", type=str, default="qwen2.5-1.5b",
                       choices=["qwen2.5-1.5b", "qwen2.5-7b", "llama3.1-8b", "llama3.2-3b"])
    parser.add_argument("--steps", type=str, default="1,2,3,4,5",
                       help="Comma-separated steps to run (1-5)")
    parser.add_argument("--stage", type=str, default="grpo",
                       choices=["base", "sft", "grpo"])
    parser.add_argument("--gpu-mode", type=str, default="1gpu",
                       choices=["1gpu", "2gpu"],
                       help="GPU mode: 1gpu (single GPU) or 2gpu (model parallel)")

    # BPE params
    parser.add_argument("--max-merges", type=int, default=50)
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--max-macro-len", type=int, default=6)

    # Training params
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)

    # Eval params
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-atomic-calls", type=int, default=50)
    parser.add_argument("--max-episodes", type=int, default=100)

    args = parser.parse_args()

    # --- Setup logging ---
    log_dir = os.path.join(Path(__file__).resolve().parent.parent, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(log_dir, f"pipeline_{args.dataset}_{_model_short(args.model)}.log"),
                mode="a",
            ),
        ]
    )

    # --- Resolve paths ---
    paths = DatasetPaths(
        output_dir=args.output_dir,
        model_name=args.model,
        gpu_mode=args.gpu_mode,
        rl_dataset=args.rl_dataset,
        all_tools=args.all_tools,
        tool_simulator_db=args.tool_simulator_db,
    )
    paths.ensure_dirs()

    logger.info("=" * 70)
    logger.info(f"AdaMacro GIPO Pipeline — Dataset: {args.dataset}")
    logger.info("=" * 70)
    logger.info(f"  Model:          {args.model} ({args.gpu_mode})")
    logger.info(f"  RL dataset:     {paths.rl_dataset}")
    logger.info(f"  All tools:      {paths.all_tools}")
    logger.info(f"  Tool sim DB:    {paths.tool_simulator_db}")
    logger.info(f"  Output dir:     {paths.output_dir}")
    logger.info(f"  SFT dir:        {paths.sft_dir}")
    logger.info(f"  GIPO dir:       {paths.gipo_dir}")
    logger.info(f"  Eval dir:       {paths.eval_dir}")
    logger.info("=" * 70)

    steps = [int(s.strip()) for s in args.steps.split(",")]
    logger.info(f"Running steps: {steps}")

    step_fns = {
        1: run_step1,
        2: run_step2,
        3: run_step3,
        4: run_step4,
        5: run_step5,
    }

    for step in steps:
        if step in step_fns:
            logger.info(f"\n{'='*70}\nStep {step}\n{'='*70}")
            start = time.time()
            step_fns[step](args, paths)
            logger.info(f"Step {step} completed in {time.time()-start:.1f}s")
        else:
            logger.warning(f"Unknown step: {step}")


if __name__ == "__main__":
    main()
