import argparse
import json
import math
import os
import random
import time
from typing import List, Dict, Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizerBase
from peft import PeftModel
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from datasets import load_dataset, Dataset
import pandas as pd

# --- MODIFICATION: Import Accelerator ---
from accelerate import Accelerator
from accelerate.utils import gather_object

# ======================================================================================
# 1. Utility Functions (Unchanged)
# ======================================================================================

def init_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def add_gumbel_noise(logits, temperature, dtype=torch.float64):
    """Adds Gumbel noise to logits for sampling."""
    if temperature <= 0.0:
        return logits
    logits = logits.to(dtype)
    noise = torch.rand_like(logits, dtype=dtype)
    gumbel_noise = (-torch.log(torch.clamp(noise, min=1e-9))) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Computes the number of tokens to unmask at each step for a block."""
    if steps <= 0:
        bsz = mask_index.size(0)
        return torch.zeros(bsz, 0, device=mask_index.device, dtype=torch.int64)
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if (remainder > 0).any():
        idx = torch.arange(steps, device=mask_index.device)
        front_mask = idx.unsqueeze(0) < remainder
        num_transfer_tokens[front_mask] += 1
    return num_transfer_tokens.to(torch.int64)

# ======================================================================================
# 2. Core Generation Logic 
# ======================================================================================

@torch.no_grad()
def generate_rollout(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
) -> torch.Tensor:
    model.eval()
    gen_length = args.gen_length
    total_diffusion_steps = args.diffusion_steps
    block_length = args.block_length
    cfg_scale = args.cfg_scale
    temperature = args.temperature
    mask_id = args.mask_id
    remasking = args.remasking
    device = model.device
    dtype = model.dtype
    batch_size, prompt_length = prompt_ids.shape
    x = torch.full(
        (batch_size, prompt_length + gen_length), mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_length] = prompt_ids.clone()
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    prompt_index[:, :prompt_length] = True
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = max(1, total_diffusion_steps // num_blocks)
    for block_idx in range(num_blocks):
        start_idx = prompt_length + block_idx * block_length
        end_idx = prompt_length + (block_idx + 1) * block_length
        block_mask_index_now = x[:, start_idx:end_idx] == mask_id
        num_transfer_tokens_schedule = get_num_transfer_tokens(block_mask_index_now, steps_per_block)
        for step_i in range(steps_per_block):
            mask_index_full = x == mask_id
            with torch.cuda.amp.autocast(enabled=True):
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits
                logits_with_noise = add_gumbel_noise(logits, temperature, dtype)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                if remasking == "low_confidence":
                    p = F.softmax(logits.to(dtype), dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand_like(x0, dtype=torch.float32)
                else:
                    raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
                x0_p[:, :start_idx] = -float('inf')
                x0_p[:, end_idx:] = -float('inf')
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
                per_step_budget = num_transfer_tokens_schedule[:, step_i]
                for j in range(batch_size):
                    k_plan = int(per_step_budget[j].item())
                    if k_plan <= 0: continue
                    block_mask_index_step = x[j, start_idx:end_idx] == mask_id
                    k_avail = int(block_mask_index_step.sum().item())
                    if k_avail <= 0: continue
                    k = min(k_plan, k_avail)
                    block_conf = confidence[j, start_idx:end_idx]
                    _, candidate_indices_in_block = torch.topk(block_conf, k=k)
                    final_indices_in_block = candidate_indices_in_block
                    select_indices_global = final_indices_in_block + start_idx
                    transfer_index[j, select_indices_global] = True
                x[transfer_index] = x0[transfer_index]
    return x

# ======================================================================================
# 3. Data Loading Functions
# ======================================================================================
SYSTEM_PROMPT = "Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""

GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""

def extract_hash_answer(text: str) -> str:
    match = text.split("####"); return match[1].strip() if len(match) > 1 else ""
def get_gsm8k_questions(split="test") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]; return data.map(lambda x: {"prompt_text": GSM_SYSTEM_PROMPT + "\n\n" + x["question"], "ground_truth": extract_hash_answer(x["answer"]),"question": x["question"]})
def get_math_questions(split="test") -> Dataset:
    try: data = load_dataset("HuggingFaceH4/MATH-500", split=split)
    except FileNotFoundError: print("Warning: MATH dataset not found at 'HuggingFaceH4/MATH-500'. Skipping."); return None
    return data.map(lambda x: {"prompt_text": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}", "ground_truth": x["answer"], "question": x["problem"]})
def get_sudoku_questions() -> Dataset:
    sudoku_file_path = "../dataset/4x4_test_sudoku.csv"
    try: df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    except FileNotFoundError: print(f"Error: Sudoku dataset not found at '{sudoku_file_path}'."); return None
    data = Dataset.from_pandas(df); return data.map(lambda x: {"prompt_text": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n", "ground_truth": x["Solution"], "question": x["Puzzle"]})
def get_countdown_questions() -> Dataset:
    countdown_file_path = "../dataset/countdown_cd3_test.jsonl"
    try:
        examples = []
        with open(countdown_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                nums_str = obj.get("input", "")
                target = int(obj.get("output"))
                nums = [int(x) for x in nums_str.split(",")] if isinstance(nums_str, str) else list(nums_str)
                if len(nums) != 3:
                    continue

                content = (
                    f"{SYSTEM_PROMPT}\n"
                    f"Using only the numbers {nums}, create an arithmetic expression that evaluates to exactly {target}. "
                    f"You must use all numbers from the list, and each number must be used exactly once. "
                    f"You may use the operations +, -, *, and / as needed. "
                    f"After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. "
                    f"For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
                )

                examples.append({
                    "prompt_text": content,
                    "target": target,
                    "numbers": nums,
                    "question": f"Numbers: {nums}\nTarget: {target}",
                    "ground_truth": str(target),
                })

        return Dataset.from_list(examples)
    except FileNotFoundError:
        print(f"Error: Countdown dataset not found at '{countdown_file_path}'.")
        return None




DATASET_MAP = {"gsm8k": get_gsm8k_questions, "math": get_math_questions, "sudoku": get_sudoku_questions, "countdown": get_countdown_questions}

# ======================================================================================
# 4. Main DDP-enabled Evaluation Script
# ======================================================================================

def main():
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description="DDP-enabled evaluation for DTreeRLGRPO models.")
    parser.add_argument("--model_path", type=str, required=True); 
    parser.add_argument("--checkpoint_path", type=str, default=""); 
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "math", "sudoku", "countdown"]); 
    parser.add_argument("--output_dir", type=str, default="results/"); 
    parser.add_argument("--output_file", type=str, default=""); 
    parser.add_argument("--gen_length", type=int, default=256); 
    parser.add_argument("--diffusion_steps", type=int, default=128); 
    parser.add_argument("--block_length", type=int, default=32); 
    parser.add_argument("--temperature", type=float, default=0.7); 
    parser.add_argument("--cfg_scale", type=float, default=0.0); 
    parser.add_argument("--mask_id", type=int, default=126336); 
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"]); 
    parser.add_argument("--seed", type=int, default=42); 
    parser.add_argument("--batch_size", type=int, default=4); 
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    
    init_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    args.diffusion_steps = args.gen_length // 2

    # --- Load Model and Tokenizer (on main process first) ---
    if accelerator.is_main_process:
        print(f"Loading model from {args.model_path}...")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.checkpoint_path:
        if accelerator.is_main_process:
            print(f"Loading PEFT adapter from {args.checkpoint_path}...")
        model = PeftModel.from_pretrained(model, args.checkpoint_path)

    # --- Load Dataset ---
    if accelerator.is_main_process:
        print(f"Loading dataset '{args.dataset}'...")
    dataset_func = DATASET_MAP.get(args.dataset)
    if not dataset_func:
        raise ValueError(f"Dataset '{args.dataset}' not supported.")
    
    dataset = dataset_func()
    if dataset is None:
        return
    if args.limit > 0:
        dataset = dataset.select(range(args.limit))

    # We need a collate function to handle batching of raw data
    def collate_fn(batch):
        # batch is a list of dicts, e.g., [{'prompt_text': '...', 'question': '...', ...}]
        collated = {}
        for key in batch[0].keys():
            collated[key] = [d[key] for d in batch]
        return collated

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False, # Evaluation should not be shuffled
    )

    model, dataloader = accelerator.prepare(model, dataloader)
    
    all_results = []
    start_time = time.time()
    
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        # Loop through the dataloader prepared by Accelerator
        for batch_data in tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Evaluating on {args.dataset}"):
            
            prompts_with_template = []
            for text in batch_data["prompt_text"]:
                example_dict = {"prompt": [{"role": "user", "content": text}]}
                formatted_prompt = maybe_apply_chat_template(example_dict, tokenizer)["prompt"]
                prompts_with_template.append(formatted_prompt)
            
            inputs = tokenizer(
                prompts_with_template, return_tensors="pt", padding=True
            ).to(accelerator.device)
            
            generated_ids = generate_rollout(unwrapped_model, inputs.input_ids, tokenizer, args)

            generated_completions = tokenizer.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            
            for j, completion in enumerate(generated_completions):
                # Each process collects its own results
                all_results.append({
                    "question": batch_data["question"][j],
                    "prompt_with_template": prompts_with_template[j],
                    "generated_answer": completion,
                    "ground_truth": batch_data["ground_truth"][j],
                })

    # Gather results from all GPUs
    all_results_gathered = gather_object(all_results)
    
    # Only the main process saves the file
    if accelerator.is_main_process:
        end_time = time.time()
        print(f"Evaluation finished in {end_time - start_time:.2f} seconds.")
        print(f"Gathered {len(all_results_gathered)} results from all processes.")

        if not args.output_file:
            model_name = os.path.basename(args.checkpoint_path) if args.checkpoint_path else "base"
            output_filename = f"{args.dataset}_{model_name}_seed{args.seed}.json"
        else:
            output_filename = args.output_file
        
        output_path = os.path.join(args.output_dir, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results_gathered, f, indent=2, ensure_ascii=False)
            
        print(f"All results saved to {output_path}")


if __name__ == "__main__":
    main()
