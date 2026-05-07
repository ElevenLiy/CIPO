
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


PROJECT_ROOT = "/path/to/your/data"

DATASET_NAME = "TOOLATHLON"

TRAJECTORIES_DIR = os.path.join(PROJECT_ROOT, "Toolathlon-Trajectories-merge")

ALL_TOOLS_PATH = os.path.join(PROJECT_ROOT, "json_file", "all_tools_v2.json")

MCP_GRAPH_PATH = os.path.join(PROJECT_ROOT, "json_file", "mcp_rl_graph_v2.json")

ALL_MESSAGES_PATH = os.path.join(PROJECT_ROOT, "json_file", "all_messages.json")

RL_DATASET_PATH = os.path.join(PROJECT_ROOT, "GRPO-ACO", "data", "rl_dataset_llm_v3.json")

TOOL_SIMULATOR_DB_PATH = os.path.join(PROJECT_ROOT, "GRPO-ACO", "data", "tool_simulator_database.json")

ADAMACRO_OUTPUT_DIR = os.path.join("/path/to/CIPO", "outputs", DATASET_NAME)

SKILL_LIBRARY_PATH = os.path.join(ADAMACRO_OUTPUT_DIR, "skill_library.json")

AUGMENTED_TOOLS_PATH = os.path.join(ADAMACRO_OUTPUT_DIR, "augmented_tools.json")

SFT_DATA_PATH = os.path.join(ADAMACRO_OUTPUT_DIR, "sft_data.json")

GRPO_DATA_PATH = os.path.join(ADAMACRO_OUTPUT_DIR, "grpo_data.json")

EVAL_RESULTS_DIR = os.path.join(ADAMACRO_OUTPUT_DIR, "eval_results")

CHECKPOINT_DIR = os.path.join(ADAMACRO_OUTPUT_DIR, "checkpoints")

MODEL_PATHS = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

DEFAULT_MODEL = "qwen2.5-7b"


@dataclass
class BPEConfig:
    max_merges: int = 50
    min_freq: int = 3
    max_macro_len: int = 6
    min_macro_len: int = 2
    success_only: bool = True
    min_usage_ratio: float = 0.01


@dataclass
class SkillConfig:
    templates: List[str] = field(default_factory=lambda: [
        "sequential",
        "select",
        "conditional",
    ])
    select_strategies: List[str] = field(default_factory=lambda: [
        "rank-0",
        "rank-1",
        "random",
        "filter",
    ])
    enable_trace: bool = True
    enable_soft_interrupt: bool = True


@dataclass
class SFTConfig:
    num_epochs: int = 3
    learning_rate: float = 1e-5
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    warmup_ratio: float = 0.05
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    weight_decay: float = 0.01
    save_steps: int = 200
    logging_steps: int = 10


@dataclass
class GRPOConfig:
    num_epochs: int = 3
    learning_rate: float = 5e-6
    group_size: int = 4
    kl_coeff: float = 0.05
    max_gen_length: int = 512
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lambda_skill: float = 0.3
    r_complete: float = 1.0
    r_pass: float = 0.2
    r_fail: float = -0.5
    temperature: float = 0.7
    min_forced_lower: int = 3
    min_forced_upper: int = 6
    r_eff_gate: float = 0.4
    under_explore_penalty: float = 0.05
    under_explore_threshold: int = 3
    save_steps: int = 100
    logging_steps: int = 1
    gipo_step_reward_scale: float = 0.15
    gipo_step_reward_cap: float = 0.1
    gipo_total_reward_cap: float = 0.3


@dataclass
class GIPOAPIConfig(GRPOConfig):

    api_key: str = "YOUR_API_KEY_HERE"
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_model: str = "qwen3.5-plus"

    api_timeout: int = 60
    api_max_retries: int = 3
    api_temperature: float = 0.7


@dataclass
class GIPO3BConfig(GRPOConfig):
    learning_rate: float = 3e-6
    lora_rank: int = 48
    lora_alpha: int = 96
    lora_dropout: float = 0.05
    max_seq_length: int = 4096
    max_gen_length: int = 512
    gradient_accumulation_steps: int = 8
    save_steps: int = 100
    gradient_checkpointing: bool = False


@dataclass
class GIPO7BConfig(GRPOConfig):
    learning_rate: float = 2e-6
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    max_seq_length: int = 3072
    max_gen_length: int = 384
    gradient_accumulation_steps: int = 4
    save_steps: int = 50
    gradient_checkpointing: bool = True


@dataclass
class EvalConfig:
    max_turns: int = 30
    max_atomic_calls: int = 50
    temperature: float = 0.0
    top_p: float = 0.95
    batch_size: int = 1
    enable_continuation: bool = True


def get_model_path(model_name: str) -> str:
    if model_name in MODEL_PATHS:
        return MODEL_PATHS[model_name]
    raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_PATHS.keys())}")


def ensure_dirs():
    dirs = [
        ADAMACRO_OUTPUT_DIR,
        EVAL_RESULTS_DIR,
        CHECKPOINT_DIR,
        os.path.join(CHECKPOINT_DIR, "sft"),
        os.path.join(CHECKPOINT_DIR, "grpo"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def print_config():
    print("=" * 70)
    print("AdaMacro Configuration")
    print("=" * 70)
    print(f"  Dataset:          {DATASET_NAME}")
    print(f"  Project root:     {PROJECT_ROOT}")
    print(f"  Trajectories:     {TRAJECTORIES_DIR}")
    print(f"  All tools:        {ALL_TOOLS_PATH}")
    print(f"  RL dataset:       {RL_DATASET_PATH}")
    print(f"  Tool simulator:   {TOOL_SIMULATOR_DB_PATH}")
    print(f"  Output dir:       {ADAMACRO_OUTPUT_DIR}")
    print(f"  Default model:    {DEFAULT_MODEL}")
    print("=" * 70)
