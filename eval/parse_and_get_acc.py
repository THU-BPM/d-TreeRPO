import json
import re
import os
import glob
import tiktoken
from collections import defaultdict
from parser_helper import is_equiv, remove_boxed, last_boxed_only_string, is_num_equiv


def count_effective_tokens(text):
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)

def parse_answer_from_generation_gsm(raw_generation):
    parsed_answer = None
    answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        try:
            boxed_in_answer = last_boxed_only_string(answer_content)
            if boxed_in_answer:
                parsed_answer = remove_boxed(boxed_in_answer)
        except:
            pass 
        
        if parsed_answer is None:
            parsed_answer = answer_content
    
    if parsed_answer is None:
        try:
            parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
        except:
            parsed_answer = None
    
    if parsed_answer is not None:
        cleaned_answer = re.sub(r'\\text(?:normal)?\{.*?\}', '', str(parsed_answer))
        cleaned_answer = cleaned_answer.replace(',', '').strip()
        match = re.search(r"[-+]?\d*\.?\d+", cleaned_answer)
        
        if match:
            parsed_answer = match.group(0)
        else:
            parsed_answer = None   
    
    return parsed_answer

def parse_answer_from_generation_math(raw_generation):
    parsed_answer = None

    # Step 1: Look for the <answer> tag
    answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        # Step 1a: Search for the last \boxed{...} inside the <answer> tag
        boxed_in_answer = last_boxed_only_string(answer_content)
        if boxed_in_answer:
            parsed_answer = remove_boxed(boxed_in_answer)

    # Step 2: If no <answer> or no valid boxed content, search the entire text for the last \boxed{...}
    if parsed_answer is None:
        boxed_in_raw = last_boxed_only_string(raw_generation)
        if boxed_in_raw:
            parsed_answer = remove_boxed(boxed_in_raw)

    return parsed_answer

def parse_gsm_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data:
        total_processed += 1
        ground_truth = item.get("ground_truth")
        raw_generation = item.get("generated_answer", "")
        question = item.get("question", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        parsed_answer = parse_answer_from_generation_gsm(raw_generation)     

        is_correct = parsed_answer is not None and is_num_equiv(parsed_answer, ground_truth) 
        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )

def parse_math_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = json_data if isinstance(json_data, list) else [json_data]

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data:
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generated_answer", "")

        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        parsed_answer = parse_answer_from_generation_math(raw_generation)

        is_correct = False
        if parsed_answer is not None:
            is_correct = is_equiv(parsed_answer, ground_truth)

        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )

def parse_countdown_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data
    
    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    def numbers_in_expression(expr):
        return [int(n) for n in re.findall(r"\d+", expr)]

    def uses_exact_numbers_once(expr, available_numbers):
        try:
            nums_expr = sorted(numbers_in_expression(expr))
            nums_avail = sorted(available_numbers)
            return nums_expr == nums_avail
        except Exception:
            return False

    def safe_eval(expr):
        # Evaluate arithmetic expression safely using AST
        import ast
        import operator as op

        allowed_ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.USub: op.neg,
            ast.UAdd: op.pos,
            ast.FloorDiv: op.floordiv,
        }

        def _eval(node):
            if isinstance(node, ast.Num): 
                return node.n
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Invalid constant")
            if isinstance(node, ast.BinOp):
                if type(node.op) not in allowed_ops:
                    raise ValueError("Invalid operator")
                return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp):
                if type(node.op) not in allowed_ops:
                    raise ValueError("Invalid unary operator")
                return allowed_ops[type(node.op)](_eval(node.operand))
            if isinstance(node, ast.Expr):
                return _eval(node.value)
            raise ValueError("Invalid expression node")

        tree = ast.parse(expr, mode="eval")
        return _eval(tree.body)

    for item in data:
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", [])
        generated_text = item.get("generated_answer", "")

        # Tokens
        effective_tokens = count_effective_tokens(generated_text)
        total_effective_tokens += effective_tokens

        # Extract numbers & target
        numbers = []
        target = None
        if isinstance(ground_truth, list) and len(ground_truth) == 2:
            numbers = ground_truth[0]
            target = ground_truth[1]
        else:
            numbers_match = re.search(r"Numbers:\s*\[(.*?)\]", question, re.IGNORECASE)
            if numbers_match:
                numbers = [int(num.strip()) for num in numbers_match.group(1).split(",") if num.strip()]

            target_match = re.search(r"Target:\s*(\d+)", question, re.IGNORECASE)
            if target_match:
                target = int(target_match.group(1))

        equation = None
        answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        if answer_match:
            equation = answer_match.group(1)
        else:
            try:
                equation = remove_boxed(last_boxed_only_string(generated_text))
            except Exception:
                equation = generated_text

        equation = (equation or "").strip()

        fence = re.search(r"```(.*?)```", equation, re.DOTALL)
        if fence and fence.group(1).strip():
            equation = fence.group(1).strip()

        candidate_line = None
        for line in equation.splitlines():
            line = line.strip()
            if re.search(r"[+\-*/]", line) and re.fullmatch(r"[0-9+\-*/().=\s]+", line):
                candidate_line = line
                break
        if candidate_line:
            equation = candidate_line

        equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")

        if "=" in equation:
            parts = [p.strip() for p in equation.split("=")]
            parts = [p for p in parts if p and re.fullmatch(r"[0-9+\-*/().\s]+", p)]
            chosen = None
            if numbers:
                for p in parts:
                    if uses_exact_numbers_once(p, numbers):
                        chosen = p
                        break
            if chosen is None:
                chosen = parts[0] if parts else ""
            equation = chosen

        if not re.fullmatch(r"[0-9+\-*/().\s]+", equation or ""):
            equation = ""

        is_correct = False
        result = None

        if equation and numbers:
            if uses_exact_numbers_once(equation, numbers):
                try:
                    result = safe_eval(equation)
                    # print(f"Evaluation Result: {result}")
                except Exception:
                    result = float("Inf")
                if target is not None and isinstance(result, (int, float)) and abs(result - target) < 1e-5:
                    is_correct = True
                    total_correct += 1
    

        processed_items.append(
            {
                "question": question,
                "extracted_answer": equation,
                "evaluation_result": result,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )

def parse_sudoku_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct_cells = total_empty_cells = total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data:
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generated_answer", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        # Extract puzzle
        puzzle_str = ""
        if len(question) >= 16 and all(c.isdigit() or c == "0" for c in question[:16]):
            puzzle_str = question[:16]
        else:
            match = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
            if match:
                puzzle_str = match.group(1)
        assert len(puzzle_str) == 16, f"Invalid puzzle string: {puzzle_str}"

        empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
        empty_cells = len(empty_indices)

        # Extract solution using regex patterns
        solution_str = ""
        patterns = [
            r"<answer>.*?```\s*([\d\s]+)```",
            r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
            r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
            r".*?(\d{16})\s*</answer>",
            r"\b(\d{16})\b",
        ]

        for pattern in patterns:
            if solution_str:
                break
            match = re.search(pattern, raw_generation, re.DOTALL)
            if match and match.group(1).strip():
                solution_str = match.group(1).strip()

        solution_str = re.sub(r"\s", "", solution_str)

        # Handle solution length
        if not solution_str:
            correct_cells = 0
        else:
            if len(solution_str) < 16:
                solution_str = solution_str + "0" * (16 - len(solution_str))
            elif len(solution_str) > 16:
                solution_str = solution_str[:16]
            correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])

        accuracy = correct_cells / empty_cells if empty_cells > 0 else 0.0
        total_correct_cells += correct_cells
        total_empty_cells += empty_cells

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": solution_str,
                "ground_truth": ground_truth,
                "empty_cells": empty_cells,
                "correct_cells": correct_cells,
                "accuracy": accuracy,
                "effective_tokens": effective_tokens,
            }
        )
    return (
        total_correct_cells,
        total_empty_cells,
        processed_items,
        total_effective_tokens * 8,
    )


def aggregate_results(directory="."):
    """Aggregate results from all JSON files and save detailed results."""
    # Find all JSON files matching the pattern
    json_files = glob.glob(os.path.join(directory, "*.json"))



    # Dictionary to store aggregated results by setup
    setups = defaultdict(
        lambda: {
            "correct": 0,
            "processed": 0,
            "accuracy": 0.0,
            "questions": [],
            "total_effective_tokens": 0,
        }
    )

    for json_file in json_files:
        filename = os.path.basename(json_file)

        if "gsm" in filename:
            (
                correct,
                processed,
                detailed_results,
                total_effective_tokens,
            ) = parse_gsm_answers(json_path=json_file)
        elif "math" in filename:
            (
                correct,
                processed,
                detailed_results,
                total_effective_tokens,
            ) = parse_math_answers(json_path=json_file)
        elif "countdown" in filename:
            (
                correct,
                processed,
                detailed_results,
                total_effective_tokens,
            ) = parse_countdown_answers(json_path=json_file)
        elif "sudoku" in filename:
            (
                correct,
                processed,
                detailed_results,
                total_effective_tokens,
            ) = parse_sudoku_answers(json_path=json_file)

        setups[filename]["correct"] += correct
        setups[filename]["processed"] += processed
        setups[filename]["total_effective_tokens"] += total_effective_tokens
        setups[filename]["questions"].extend(detailed_results)

    # Calculate final accuracy and save results
    for setup, results in sorted(setups.items()):
        results["accuracy"] = (
            results["correct"] / results["processed"] * 100 if results["processed"] > 0 else 0
        )
        results["avg_effective_tokens"] = (
            results["total_effective_tokens"] / results["processed"] if len(results["questions"]) > 0 else 0
        )
    # Header
    header_format = "{:<40} {:>12} {:>25}"
    print(header_format.format("Setup (task_model_seqlen_diffusteps)", "Accuracy", "Avg Effective Tokens"))
    print("-" * 80)

    # Data rows
    row_format = "{:<40} {:>11.2f}% {:>25.2f}"
    for setup, results in sorted(setups.items()):
        print(row_format.format(setup, results["accuracy"], results["avg_effective_tokens"]))

    print("=" * 80)


if __name__ == "__main__":
    for i in range(0, 130, 120):
        print(f"Processing checkpoint-{i}_gen256...")
        aggregate_results(directory=f"eval_results/dtreerpo/sudoku/checkpoint-{i}_gen256")
