import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Union, Optional, List, Dict, Tuple
from collections import defaultdict
import uuid

from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from datasets import Dataset, IterableDataset
from trl.models import unwrap_model_for_generation
from trl.data_utils import maybe_apply_chat_template, is_conversational

from diffu_grpo_trainer import DiffuGRPOTrainer
from dtreerpo_config import DTreeRPOConfig
import random
import os
import time
import contextlib
import math
import torch.distributions as dists


class DTreeRPOTrainer(DiffuGRPOTrainer):
    """
    DTreeRPO Trainer
    """
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[Any, list[Any]],
        args: Optional[DTreeRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        peft_config: Optional[Any] = None,
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        if not isinstance(args, DTreeRPOConfig):
            raise ValueError("args must be an instance of DTreeRPOConfig")

        self._buffered_segments = []
        self._current_segment_for_reuse = None

    # -------- Reward --------
    def _get_weighted_rewards(self, inputs, prompts, completions, device):
        num_samples = len(prompts)
        rewards_per_func = torch.zeros(num_samples, len(self.reward_funcs), device=device)

        keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        for i, reward_func in enumerate(self.reward_funcs):
            output_reward_func = reward_func(
                prompts=prompts,
                completions=completions,
                run_name=self.args.output_dir,
                step=self._step,
                **reward_kwargs,
            )
            output_reward_func = [r if r is not None else torch.nan for r in output_reward_func]
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        weighted_rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        return weighted_rewards

    # -------- Visualization --------
    def _print_tree_structure(
        self,
        all_nodes: Dict[str, Dict],
        branch_points: List[int],
        prompt_length: int,
        prompt_idx: int = 0
    ):
        print("\n" + "="*100)
        print(f"🌳 SEARCH TREE VISUALIZATION (Step {self.state.global_step}, Prompt Index {prompt_idx})")
        print("="*100)

        nodes_by_step = defaultdict(list)
        for node_id, node in all_nodes.items():
            nodes_by_step[node["step"]].append(node)

        for level_idx, step in enumerate(sorted(nodes_by_step.keys(), reverse=True)):
            nodes_at_level = nodes_by_step[step]
            print(f"\n{'─'*100}")
            print(f"📍 Level {level_idx} | Diffusion Step: {step} | Nodes: {len(nodes_at_level)}")
            print(f"{'─'*100}")

            for node_idx, node in enumerate(nodes_at_level):
                if prompt_idx >= node["generation"].shape[0]:
                    continue

                generation = node["generation"][prompt_idx]
                prompt_ids = generation[:prompt_length]
                completion_ids = generation[prompt_length:]

                prompt_text = self.processing_class.decode(prompt_ids, skip_special_tokens=True)
                completion_text = self.processing_class.decode(completion_ids, skip_special_tokens=False)

                mask_count = (completion_ids == self.args.mask_id).sum().item()
                total_tokens = completion_ids.shape[0] if completion_ids.shape[0] > 0 else 1
                unmask_ratio = (total_tokens - mask_count) / total_tokens * 100

                node_type = "🌱 ROOT" if node["id"] == "root" else ("🍃 LEAF" if node["is_leaf"] else "🌿 NODE")
                print(f"\n  {node_type} #{node_idx+1} [ID: {node['id'][:8]}...]")
                print(f"  ├─ Parent: {node['parent_id'][:8] if node['parent_id'] else 'None'}...")
                print(f"  ├─ Value: {node['value_vec'][prompt_idx]:.4f}")
                print(f"  ├─ Children: {len(node['children'])}")
                print(f"  ├─ Unmask Progress: {unmask_ratio:.1f}%")
                print(f"  │")
                print(f"  ├─ 📝 Prompt: {prompt_text}")
                print(f"  └─ 💬 Completion: {completion_text}")

                if node["is_leaf"]:
                    print(f"     └─ ⭐ Final Reward: {node['value_vec'][prompt_idx]:.4f}")

        print(f"\n{'='*100}")
        print(f"📊 TREE STATISTICS")
        print(f"{'='*100}")
        print(f"  • Total Nodes: {len(all_nodes)}")
        print(f"  • Levels: {len(nodes_by_step)}")
        print(f"  • Leaf Nodes: {sum(1 for n in all_nodes.values() if n['is_leaf'])}")
        print(f"  • Root Value: {all_nodes['root']['value_vec'][prompt_idx]:.4f}")

        print(f"\n  Branch Factors by Level:")
        for level_idx, step in enumerate(sorted(nodes_by_step.keys(), reverse=True)[:-1]):
            nodes = nodes_by_step[step]
            avg_children = np.mean([len(n["children"]) for n in nodes])
            print(f"    Level {level_idx} → Level {level_idx+1}: {avg_children:.1f} children/node")

        print("="*100 + "\n")

    def _print_tree_to_file(
        self,
        all_nodes: Dict[str, Dict],
        branch_points: List[int],
        prompt_length: int,
        prompt_idx: int,
        file_path: str,
    ):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            with contextlib.redirect_stdout(f):
                self._print_tree_structure(
                    all_nodes=all_nodes,
                    branch_points=branch_points,
                    prompt_length=prompt_length,
                    prompt_idx=prompt_idx,
                )

    # -------- Local conditional logps --------
    def _get_local_transition_logps(
        self,
        model: PreTrainedModel,
        parent_ids: torch.Tensor,
        child_ids: torch.Tensor,
        prompt_length: int,
    ) -> torch.Tensor:
        device = parent_ids.device
        batch_size, seq_len = parent_ids.size()
        logits_to_keep = seq_len - prompt_length
        logits = self.get_logits(
            model, parent_ids, None, self.args.cfg_scale, self.args.mask_id
        )

        completion_logits = logits[:, prompt_length:]  # [B, Lc, V]
        completion_child_ids = child_ids[:, prompt_length:]  # [B, Lc]
        completion_parent_ids = parent_ids[:, prompt_length:]  # [B, Lc]
        changed_mask = (completion_parent_ids == self.args.mask_id) & (completion_child_ids != self.args.mask_id)
        eos_mask = self._build_completion_mask(completion_child_ids)
        active_mask = (changed_mask & (eos_mask > 0)).to(completion_logits.dtype)

        loss_flat = F.cross_entropy(
            completion_logits.reshape(-1, completion_logits.size(-1)),
            completion_child_ids.reshape(-1),
            reduction="none"
        ).view(batch_size, logits_to_keep)
        per_token_logps = -loss_flat

        step_logps = per_token_logps * active_mask  # [B, Lc]
        step_logps = torch.nan_to_num(step_logps, nan=0.0, posinf=0.0, neginf=0.0)
        return step_logps

    def _build_completion_mask(self, completion_ids: torch.Tensor) -> torch.Tensor:
        device = completion_ids.device
        is_unmasked = (completion_ids != self.args.mask_id).float()
        eos_id = getattr(self.processing_class, "eos_token_id", None)
        if eos_id is None:
            return is_unmasked

        is_eos = (completion_ids == eos_id)
        B, L = is_eos.size()
        eos_idx = torch.full((B,), L, dtype=torch.long, device=device)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos[has_eos].int().argmax(dim=1)

        seq_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        eos_pos_mask = (seq_idx <= eos_idx.unsqueeze(1)).float()

        completion_mask = is_unmasked * eos_pos_mask
        return completion_mask

    # -------- Value Aggregation --------
    @staticmethod
    def _mean_aggregate_value_vecs(child_value_vecs: List[torch.Tensor]) -> torch.Tensor:
        stack = torch.stack(child_value_vecs, dim=0)  # [C, B]
        aggregated = torch.nanmean(stack, dim=0)      # [B]
        aggregated = torch.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0)
        return aggregated

    def _get_used_mask_for_block(self, parent_node_id: str, block_idx: int, batch_size: int, block_len: int, device: torch.device):
        reg = self._soft_dedup_registry.setdefault(parent_node_id, {})
        if block_idx not in reg:
            reg[block_idx] = torch.zeros((batch_size, block_len), dtype=torch.bool, device=device)
        return reg[block_idx]

    def _update_used_mask_for_block(self, parent_node_id: str, block_idx: int, selected_block_mask: torch.Tensor):
        reg = self._soft_dedup_registry.setdefault(parent_node_id, {})
        if block_idx not in reg:
            reg[block_idx] = selected_block_mask.clone()
        else:
            reg[block_idx] = reg[block_idx] | selected_block_mask

    # -------- Tree search, scoring, and segment building --------
    def _perform_tree_search_and_compute_advantages(
        self, initial_inputs: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        device = self.accelerator.device

        if isinstance(initial_inputs, dict):
            initial_inputs = [initial_inputs]

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in initial_inputs]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False
        ).to(device)
        prompt_ids = prompt_inputs["input_ids"]
        batch_size = prompt_ids.size(0)
        prompt_length = prompt_ids.shape[1]

        gen_length = self.args.max_completion_length
        mask_id = self.args.mask_id

        initial_generation = torch.full(
            (batch_size, prompt_ids.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=device
        )
        initial_generation[:, :prompt_ids.shape[1]] = prompt_ids.clone()

        prompt_index = torch.zeros_like(initial_generation, dtype=torch.bool)
        prompt_index[:, :prompt_ids.shape[1]] = True

        total_steps = self.args.diffusion_steps
        s = self.args.tree_contraction_factor
        branch_points = [total_steps] + [total_steps - int(k * total_steps / s) for k in range(1, s + 1)]
        branch_points = sorted(list(set(branch_points)), reverse=True)

        root_node = {
            "id": "root", "parent_id": None, "generation": initial_generation, "step": total_steps,
            "children": [], "value": 0.0, "value_vec": torch.full((batch_size,), 0.0, device=device),
            "is_leaf": False, "prompt_idx": list(range(batch_size))
        }
        group_key_to_id = {}  # For self-distillation grouping: the same (parent node, batch index b) shares the same group id
        next_gid = 0

        all_nodes = {"root": root_node}
        current_level_nodes = [root_node]

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            for i in range(len(branch_points) - 1):
                start_step, end_step = branch_points[i], branch_points[i+1]
                steps_to_run = start_step - end_step
                next_level_nodes = []

                if self.accelerator.is_local_main_process:
                    self.accelerator.print(f"\n[Tree Search] Expanding Level {i} -> {i+1} (Steps {start_step} -> {end_step}). Processing {len(current_level_nodes)} nodes...")

                for node in current_level_nodes:
                    branch_factor = self.args.num_tree_samples if node["id"] == "root" else self.args.tree_branch_factor

                    for k in range(branch_factor):
                        new_generation = self._generate_partial(
                            model=unwrapped_model, start_generation=node["generation"], current_step=start_step,
                            steps_to_run=steps_to_run, prompt_index=prompt_index, tree_level_idx=i, node_id=node["id"],
                            branch_idx=k, branch_factor=branch_factor
                        )

                        child_id = str(uuid.uuid4())
                        child_node = {
                            "id": child_id, "parent_id": node["id"], "generation": new_generation, "step": end_step,
                            "children": [], "value": 0.0, "value_vec": None, "is_leaf": (end_step == 0), "prompt_idx": node["prompt_idx"]
                        }
                        all_nodes[child_id] = child_node
                        node["children"].append(child_id)
                        next_level_nodes.append(child_node)
                current_level_nodes = next_level_nodes

        leaf_nodes = [n for n in all_nodes.values() if n["is_leaf"]]
        if not leaf_nodes:
            self.accelerator.print("Warning: No leaf nodes were generated. Skipping batch.")
            return []

        # Calculate leaf rewards
        leaf_completions_ids_list, leaf_inputs_data = [], []
        for leaf_node in leaf_nodes:
            for prompt_idx in range(leaf_node["generation"].shape[0]):
                completion_ids = leaf_node["generation"][prompt_idx:prompt_idx+1, prompt_ids.size(1):]
                leaf_completions_ids_list.append(completion_ids)
                leaf_inputs_data.append(initial_inputs[prompt_idx])

        leaf_completions_ids = torch.cat(leaf_completions_ids_list, dim=0)
        leaf_completions_text = self.processing_class.batch_decode(leaf_completions_ids, skip_special_tokens=True)

        leaf_prompts = [item["prompt"] for item in leaf_inputs_data]
        if is_conversational(leaf_inputs_data[0]):
            completions = [[{"role": "assistant", "content": t}] for t in leaf_completions_text]
        else:
            completions = leaf_completions_text

        leaf_rewards = self._get_weighted_rewards(leaf_inputs_data, leaf_prompts, completions, device)

        if leaf_rewards.numel() > 0:
            mode = "eval" if self.control.should_evaluate else "train"
            mean_reward_local = torch.nanmean(leaf_rewards)
            mean_reward_global = self.accelerator.gather_for_metrics(mean_reward_local).mean().item()
            self._metrics[mode]["reward"].append(mean_reward_global)

        assert len(leaf_inputs_data) == len(leaf_completions_text) == leaf_rewards.shape[0], \
            f"Leaf data mismatch: inputs={len(leaf_inputs_data)}, comps={len(leaf_completions_text)}, rewards={leaf_rewards.shape[0]}"

        reward_idx = 0
        for leaf_node in leaf_nodes:
            num_prompts_in_batch = leaf_node["generation"].shape[0]
            node_rewards = leaf_rewards[reward_idx: reward_idx + num_prompts_in_batch]  # [B]
            leaf_node["value_vec"] = node_rewards.clone()
            leaf_node["value"] = torch.nanmean(node_rewards).item()
            reward_idx += num_prompts_in_batch

        # Reward backpropagation: bottom-up
        for step in reversed(branch_points[:-1]):
            nodes_at_level = [n for n in all_nodes.values() if n["step"] == step and n["id"] != "root"]
            for node in nodes_at_level:
                if node["children"]:
                    child_value_vecs = [all_nodes[child_id]["value_vec"] for child_id in node["children"]]
                    node["value_vec"] = self._mean_aggregate_value_vecs(child_value_vecs)
                    node["value"] = torch.nanmean(node["value_vec"]).item()
        if all_nodes["root"]["children"]:
            child_value_vecs = [all_nodes[child_id]["value_vec"] for child_id in all_nodes["root"]["children"]]
            all_nodes["root"]["value_vec"] = self._mean_aggregate_value_vecs(child_value_vecs)
            all_nodes["root"]["value"] = torch.nanmean(all_nodes["root"]["value_vec"]).item()

        # Print tree structure
        try:
            run_name = getattr(self.args, "run_name", os.path.basename(self.args.output_dir.rstrip("/")))
            out_dir = os.path.join("logs", "llada", run_name)
            ts = int(time.time() * 1000)
            rank = int(self.accelerator.process_index)
            vis_prompt_idx = min(self.args.print_tree_prompt_idx, batch_size - 1)
            file_name = f"tree_gstep{self.state.global_step}rank{rank}{ts}.txt"
            file_path = os.path.join(out_dir, file_name)
            self._print_tree_to_file(
                all_nodes=all_nodes,
                branch_points=branch_points,
                prompt_length=prompt_length,
                prompt_idx=vis_prompt_idx,
                file_path=file_path,
            )
        except Exception as e:
            self.accelerator.print(f"[DTreeRPO] Warning: failed to write tree log: {e}")

        if self.accelerator.is_local_main_process:
            vis_prompt_idx = min(self.args.print_tree_prompt_idx, batch_size - 1)
            self._print_tree_structure(all_nodes, branch_points, prompt_length, vis_prompt_idx)

        # Calculate local advantages
        for node_id, node in all_nodes.items():
            if node_id == "root":
                node["local_advantages_vec"] = torch.zeros_like(node["value_vec"])
                node["local_advantages"] = 0.0
                continue
            parent = all_nodes[node["parent_id"]]
            siblings = [cid for cid in parent["children"] if cid != node_id]
            if len(siblings) == 0:
                sibling_mean_vec = torch.zeros_like(node["value_vec"])
            else:
                sib_vecs = [all_nodes[cid]["value_vec"] for cid in siblings]
                sibling_mean_vec = torch.nanmean(torch.stack(sib_vecs, dim=0), dim=0)
                sibling_mean_vec = torch.nan_to_num(sibling_mean_vec, nan=0.0, posinf=0.0, neginf=0.0)
            node["local_advantages_vec"] = node["value_vec"] - sibling_mean_vec
            node["local_advantages"] = torch.nanmean(node["local_advantages_vec"]).item()

        # Construct training segments, then pack them by parent
        child_segments: List[Dict[str, Any]] = []

        for node_id, node in all_nodes.items():
            if node_id == "root":
                continue

            parent = all_nodes[node["parent_id"]]
            keep_idx = torch.arange(node["generation"].size(0), device=device)

            prompt_seg_ids = node["generation"][keep_idx, :prompt_length]
            completion_seg_ids = node["generation"][keep_idx, prompt_length:]
            parent_ids = parent["generation"][keep_idx]
            child_ids = node["generation"][keep_idx]

            logits_to_keep = completion_seg_ids.size(1)
            if logits_to_keep == 0:
                continue

            local_advantages_vec_keep = node["local_advantages_vec"][keep_idx]

            B_local = node["generation"].size(0)
            group_ids_local = torch.empty(B_local, dtype=torch.long, device=device)
            parent_node_id = node["parent_id"]
            for b in range(B_local):
                key = (parent_node_id, int(b))
                gid = group_key_to_id.get(key)
                if gid is None:
                    gid = next_gid
                    next_gid += 1
                    group_key_to_id[key] = gid
                group_ids_local[b] = gid

            segment_data = {
                "prompt_ids": prompt_seg_ids,
                "completion_ids": completion_seg_ids,
                "parent_ids": parent_ids,
                "child_ids": child_ids,
                "local_advantages": local_advantages_vec_keep,  # [B]
                "prompt_length": prompt_length,
                "group_ids": group_ids_local,                 
                "parent_node_id": parent_node_id,              
            }

            with torch.no_grad():
                segment_data["old_local_logps"] = self._get_local_transition_logps(
                    self.model, parent_ids, child_ids, prompt_length
                )
                if self.beta != 0.0:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    with unwrapped.disable_adapter():
                        segment_data["ref_local_logps"] = self._get_local_transition_logps(
                            unwrapped, parent_ids, child_ids, prompt_length
                        )
                else:
                    segment_data["ref_local_logps"] = None

            child_segments.append(segment_data)

        training_segments = self._pack_segments_by_parent(child_segments)
        return training_segments

    def _pack_segments_by_parent(self, segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Bucket the child segments by parent_node_id, and within each bucket vertically concatenate them into a "parent segment" using _concat_segments.
        """
        buckets = defaultdict(list)
        for s in segs:
            buckets[s["parent_node_id"]].append(s)
        packed = []
        for _, lst in buckets.items():
            if len(lst) == 1:
                packed.append(lst[0])
            else:
                packed.append(self._concat_segments(lst))
        return packed

    def get_num_transfer_tokens(self, mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        if steps <= 0:
            bsz = mask_index.size(0)
            return torch.zeros(bsz, 0, device=mask_index.device, dtype=torch.int64)
        mask_num = mask_index.sum(dim=1, keepdim=True)  # [B, 1]
        base = mask_num // steps                        # [B, 1]
        remainder = mask_num % steps                    # [B, 1]

        num_transfer_tokens = base.expand(-1, steps).clone()  # [B, steps]
        if (remainder > 0).any():
            idx = torch.arange(steps, device=mask_index.device)  # [steps]
            front_mask = idx.unsqueeze(0) < remainder
            num_transfer_tokens[front_mask] += 1

        return num_transfer_tokens.to(torch.int64)

    # -------- Generation --------
    def _generate_partial(
        self,
        model,
        start_generation,
        current_step: int,
        steps_to_run: int,
        prompt_index,
        block_length=None,
        tree_level_idx: int = -1,
        node_id: str = "N/A",
        branch_idx: int = -1,
        branch_factor: int = 1
    ):
        cfg_scale = self.args.cfg_scale
        temperature = self.args.temperature or 0.0
        mask_id = self.args.mask_id
        remasking = self.args.remasking
        dtype = model.dtype
        device = start_generation.device
        if block_length is None:
            block_length = self.args.block_length

        x = start_generation.clone().detach()
        x.requires_grad_(False)

        if prompt_index.dim() > 1:
            prompt_length = prompt_index.sum(dim=1)[0].item()
        else:
            prompt_length = prompt_index.sum().item()

        gen_length = x.shape[1] - prompt_length
        total_blocks = gen_length // block_length
        total_diffusion_steps = self.args.diffusion_steps
        steps_per_block = total_diffusion_steps // total_blocks

        target_step = max(0, current_step - steps_to_run)
        current_block_idx = (total_diffusion_steps - current_step) // steps_per_block
        target_block_idx = (total_diffusion_steps - target_step) // steps_per_block
        blocks_to_process = list(range(current_block_idx, min(target_block_idx + 1, total_blocks)))
        if not blocks_to_process:
            return x

        remaining_steps = steps_to_run

        for block_idx in blocks_to_process:
            if remaining_steps <= 0:
                break

            start_idx = prompt_length + block_idx * block_length
            end_idx = prompt_length + (block_idx + 1) * block_length

            block_step_start = total_diffusion_steps - block_idx * steps_per_block
            block_step_end = total_diffusion_steps - (block_idx + 1) * steps_per_block

            actual_start = min(current_step, block_step_start)
            actual_end = max(target_step, block_step_end)
            steps_needed_in_this_block = max(0, actual_start - actual_end)
            if steps_needed_in_this_block <= 0:
                continue

            done_in_block = block_step_start - current_step
            done_in_block = max(0, min(done_in_block, steps_per_block))
            remain_in_block = steps_per_block - done_in_block
            if remain_in_block <= 0:
                continue

            steps_in_this_block = min(steps_needed_in_this_block, remaining_steps, remain_in_block)
            if steps_in_this_block <= 0:
                continue

            block_mask_index_now = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens_schedule = self.get_num_transfer_tokens(block_mask_index_now, remain_in_block)

            for step_i in range(steps_in_this_block):
                block_mask_index_step = x[:, start_idx:end_idx] == mask_id
                per_step_budget = num_transfer_tokens_schedule[:, step_i]
                mask_index_full = x == mask_id

                with torch.cuda.amp.autocast(enabled=self.args.fp16), torch.no_grad():
                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[prompt_index] = mask_id
                        x_ = torch.cat([x, un_x], dim=0)
                        logits = model(x_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(x).logits

                    logits_with_noise = self.add_gumbel_noise(logits, temperature, dtype)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    del logits_with_noise

                    if remasking == "low_confidence":
                        p = F.softmax(logits.to(dtype), dim=-1)
                        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1)
                    elif remasking == "random":
                        x0_p = torch.rand_like(x0, dtype=torch.float32)
                    else:
                        raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")

                    x0_p[:, :start_idx] = -float('inf')
                    x0_p[:, end_idx:] = -float('inf')
                    x0 = torch.where(mask_index_full, x0, x)
                    confidence = torch.where(mask_index_full, x0_p, -torch.inf)
                    del x0_p

                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)

                    for j in range(x.shape[0]):
                        k_plan = int(per_step_budget[j].item())
                        if k_plan <= 0:
                            continue
                        k_avail = int(block_mask_index_step[j].sum().item())
                        if k_avail <= 0:
                            continue
                        k = min(k_plan, k_avail)

                        k_plus = min(k, k_avail)
                        values, candidate_index = torch.topk(confidence[j], k=k_plus)
                        if k_plus > k:
                            perm = torch.randperm(k_plus, device=device)
                            pick = perm[:k]
                            select_index = candidate_index[pick]
                        else:
                            select_index = candidate_index

                        transfer_index[j, select_index] = True

                    transfer_index = transfer_index & ~prompt_index
                    x[transfer_index] = x0[transfer_index]

                    del x0, confidence, transfer_index, logits

            remaining_steps -= steps_in_this_block

        return x
    
    # -------- Input preparation and buffering --------
    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        mode = "eval" if self.control.should_evaluate else "train"
        inputs_list = self._explode_batch_if_needed(inputs)
        this_itr_idx = self._step % self.num_iterations

        if mode == "eval":
            segments = self._perform_tree_search_and_compute_advantages(inputs_list)
            return segments[0] if segments else None

        # Training mode: at the start of each round (this_itr_idx == 0), fill or top up the buffer
        if this_itr_idx == 0:
            if not self._buffered_segments:
                if self.accelerator.is_local_main_process:
                    self.accelerator.print(
                        f"[DTreeRPO] Global step {self.state.global_step}: running tree-search to (re)fill buffer …"
                    )
                self._buffered_segments = self._perform_tree_search_and_compute_advantages(inputs_list)
                random.shuffle(self._buffered_segments)

            if not self._buffered_segments:
                if self.accelerator.is_local_main_process:
                    self.accelerator.print("[DTreeRPO] Warning: tree-search produced 0 segments; skipping.")
                self._current_segment_for_reuse = None
                self._step += 1
                return None

            parents_per_step = getattr(self.args, "parents_per_step", 1)
            seg_batch = []
            while self._buffered_segments and len(seg_batch) < parents_per_step:
                seg_batch.append(self._buffered_segments.pop(0))

            if len(seg_batch) < parents_per_step:
                new_segments = self._perform_tree_search_and_compute_advantages(inputs_list)
                random.shuffle(new_segments)
                self._buffered_segments.extend(new_segments)
                while self._buffered_segments and len(seg_batch) < parents_per_step:
                    seg_batch.append(self._buffered_segments.pop(0))

            self._current_segment_for_reuse = self._concat_segments(seg_batch)

        self._step += 1

        if this_itr_idx == 0 and self._current_segment_for_reuse is not None:
            seg = self._current_segment_for_reuse
            A = seg.get("local_advantages")
            if A is not None:
                A = A.detach()
                mean_abs_A = A.abs().mean().item()
                self._metrics[mode].setdefault("local_advantage_abs_mean", []).append(mean_abs_A)

        return self._current_segment_for_reuse

    @staticmethod
    def _explode_batch_if_needed(
        batch: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if isinstance(batch, dict) and isinstance(next(iter(batch.values())), (list, tuple)):
            bs = len(next(iter(batch.values())))
            return [{k: v[i] for k, v in batch.items()} for i in range(bs)]
        elif isinstance(batch, list):
            return batch
        else:
            return [batch]

    def _concat_segments(self, seg_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Concatenate multiple segments along the batch dimension. Note:
        - parent_node_id may collide when concatenating across parent segments, so this key is ignored here.
        """
        merged: Dict[str, Any] = {}

        cat_keys_dim0 = [
            "prompt_ids",
            "completion_ids",
            "parent_ids",
            "child_ids",
            "old_local_logps",
            "group_ids",
        ]
        if seg_list[0]["ref_local_logps"] is not None:
            cat_keys_dim0.append("ref_local_logps")

        for k in cat_keys_dim0:
            if seg_list[0][k] is None:
                merged[k] = None
            else:
                merged[k] = torch.cat([seg[k] for seg in seg_list], dim=0)

        skip_keys = set(cat_keys_dim0 + ["prompt_length", "parent_node_id"])
        for k in seg_list[0]:
            if k in skip_keys:
                continue
            v0 = seg_list[0][k]
            if v0 is None:
                merged[k] = None
                continue
            if isinstance(v0, torch.Tensor):
                if v0.dim() > 0:
                    merged[k] = torch.cat([seg[k] for seg in seg_list], dim=0)
                else:
                    merged[k] = torch.stack([seg[k] for seg in seg_list], dim=0)
            elif isinstance(v0, (float, int)):
                merged[k] = torch.tensor([seg[k] for seg in seg_list], device=self.accelerator.device)
            else:
                pass

        merged["prompt_length"] = seg_list[0]["prompt_length"]
        return merged

    # -------- Loss --------
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch=None
    ) -> torch.Tensor:

        device = self.accelerator.device

        epsilon = getattr(self.args, "epsilon", 0.2)
        beta = getattr(self.args, "beta", 0.0)

        parent_ids, child_ids = inputs["parent_ids"], inputs["child_ids"]
        prompt_length = inputs["prompt_length"]
        assert inputs.get("old_local_logps") is not None, "Need old_local_logps for GRPO"

        enable_self_distillation_loss = getattr(self.args, "enable_self_distillation_loss", True)
        
        # Calculate logps under current policy
        new_local_logps = self._get_local_transition_logps(
            model, parent_ids, child_ids, prompt_length
        )  # [B, Lc], B = batch_size * branch_factor

        # mask
        child_completion = child_ids[:, prompt_length:]  # [B, Lc]
        local_completion_mask = self._build_completion_mask(child_completion)  # [B, Lc]
        changed_mask = (new_local_logps != 0).float()
        active_mask = local_completion_mask * changed_mask  # [B, Lc]
        L_eff_local = active_mask.sum(dim=1).clamp_min(1)

        old_local_logps = inputs["old_local_logps"].to(device)  # [B, Lc]

        # policy gradient
        ratio_local_t = torch.exp(new_local_logps - old_local_logps)  # [B, Lc]
        clipped_ratio_local_t = torch.clamp(ratio_local_t, 1.0 - epsilon, 1.0 + epsilon)

        local_adv = inputs["local_advantages"].detach().to(device)  # [B]
        A_local_b = local_adv.unsqueeze(1)  # [B,1]

        ppo_obj_local_t = torch.min(ratio_local_t * A_local_b, clipped_ratio_local_t * A_local_b)  # [B, Lc]
        ppo_obj_local_t_masked = ppo_obj_local_t * active_mask
        policy_loss_local_per_sample = -(ppo_obj_local_t_masked.sum(dim=1) / L_eff_local)  # [B]
        local_policy_loss = policy_loss_local_per_sample.mean()

        total_loss = local_policy_loss

        # KL divergence
        kl_penalty_local = torch.tensor(0.0, device=device)
        if beta != 0.0 and inputs.get("ref_local_logps") is not None and inputs["ref_local_logps"] is not None:
            ref_local_logps = inputs["ref_local_logps"].to(device)  # [B, Lc]
            delta_local = ref_local_logps - new_local_logps  # [B, Lc]
            per_token_kl_local = torch.exp(delta_local) - delta_local - 1.0  # [B, Lc]
            kl_local_per_sample = (per_token_kl_local * active_mask).sum(dim=1) / L_eff_local  # [B]
            kl_penalty_local = beta * kl_local_per_sample.mean()
            total_loss = total_loss + kl_penalty_local

        # self-distillation loss
        self_distillation_loss = torch.tensor(0.0, device=device)
        if enable_self_distillation_loss and "group_ids" in inputs and inputs["group_ids"] is not None:
            t = int(self.state.global_step)
            T = max(1, int(getattr(self.args, "max_steps", 30000)))
            lambda_max = float(getattr(self.args, "self_distillation_lambda_max", 3e-3))
            gamma = float(getattr(self.args, "self_distillation_gamma", 2.0))
            tau_max = float(getattr(self.args, "self_distillation_tau_max", 2.0))
            beta_tau = float(getattr(self.args, "self_distillation_beta", 0.7))

            progress = min(1.0, max(0.0, t / T))
            
            # λ(t) = λ_max · (e^{γ t/T}-1)/(e^γ-1)
            lambda_t = lambda_max * (math.exp(gamma * progress) - 1.0) / (math.exp(gamma) - 1.0)

            # τ(t) = τ_max · (1-t/T)^β
            tau_t = tau_max * max(0.0, (1.0 - progress)) ** beta_tau
            tau_t = max(tau_t, 1e-6)

            group_ids = inputs["group_ids"].to(device).long()  # [B]
            uniq_groups = torch.unique(group_ids)              # [G]

            ref_indices = []
            group_to_pos = {}
            for pos, g in enumerate(uniq_groups.tolist()):
                idxs = (group_ids == g).nonzero(as_tuple=False).squeeze(-1)
                if idxs.numel() == 0:
                    continue
                ref_indices.append(idxs[0])
                group_to_pos[g] = pos
            if len(ref_indices) == 0:
                self_distillation_loss = torch.tensor(0.0, device=device)
            else:
                ref_indices = torch.stack(ref_indices, dim=0)        # [G]
                B_all, Lc = child_completion.size()
                logits_parent_unique = self.get_logits(
                    model, parent_ids[ref_indices], None, self.args.cfg_scale, self.args.mask_id
                )                                                    # [G, L, V]
                logits_parent_slice = logits_parent_unique[:, prompt_length:prompt_length + Lc, :]  # [G, Lc, V]
                logp_parent_slice = F.log_softmax(logits_parent_slice, dim=-1)                      # [G, Lc, V]

                eps = 1e-8
                consistency_sum = torch.tensor(0.0, device=device)
                consistency_count = 0

                for g in uniq_groups.tolist():
                    idxs = (group_ids == g).nonzero(as_tuple=False).squeeze(-1)  # [K]
                    if idxs.numel() <= 1:
                        continue

                    A_full = inputs["local_advantages"][idxs].detach().to(device)  # [K]
                    A_full = torch.nan_to_num(A_full, nan=0.0, posinf=0.0, neginf=0.0)

                    # Only keep positive advantages
                    pos_mask = (A_full > 0)
                    if pos_mask.sum().item() == 0:
                        continue

                    A = A_full[pos_mask]                 # [K_pos]
                    idxs_pos = idxs[pos_mask]            
                    w = F.softmax(A / tau_t, dim=0)      # [K_pos]

                    group_active = active_mask[idxs_pos] > 0  # [K, Lc]
                    toks_group = child_completion[idxs_pos]    # [K, Lc]
                    ref_pos = group_to_pos[g]              

                    for j in range(Lc):
                        active_k = group_active[:, j]
                        if not active_k.any():
                            continue

                        toks_j = toks_group[active_k, j]          # [r]
                        w_j = w[active_k]                         # [r]

                        unique_tokens, inv = torch.unique(toks_j, return_inverse=True)
                        q_weights = torch.zeros_like(unique_tokens, dtype=torch.float32)
                        q_weights.index_add_(0, inv, w_j.to(q_weights.dtype))
                        ws_sum = q_weights.sum()
                        if ws_sum <= 0:
                            continue
                        q_probs = q_weights / (ws_sum + eps)

                        logp_support = logp_parent_slice[ref_pos, j].gather(0, unique_tokens)  # [u]
                        kl_j = torch.sum(q_probs * (torch.log(q_probs + eps) - logp_support))
                        consistency_sum = consistency_sum + kl_j
                        consistency_count += 1

                self_distillation_loss = consistency_sum / float(consistency_count) if consistency_count > 0 else torch.tensor(0.0, device=device)
                total_loss = total_loss + (lambda_t * self_distillation_loss)


        # Wandb log
        with torch.no_grad():
            def masked_frac(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                num = (t.float() * mask).sum()
                den = mask.sum().clamp_min(1)
                return num / den

            local_clip_ratio_hi = masked_frac((ratio_local_t > (1.0 + epsilon)), active_mask)
            local_clip_ratio_lo = masked_frac((ratio_local_t < (1.0 - epsilon)), active_mask)

            log_dict = {}
            log_dict["local_clip_ratio_hi"] = float(local_clip_ratio_hi.item())
            log_dict["local_clip_ratio_lo"] = float(local_clip_ratio_lo.item())
            log_dict["local_loss"] = float(local_policy_loss.item())
            log_dict["kl_penalty_local"] = float(kl_penalty_local.item())
            if enable_self_distillation_loss:
                log_dict["lambda_t"] = float(lambda_t)
                log_dict["self_distillation_loss*lambda_t"] = float(lambda_t * self_distillation_loss.item())
                log_dict["self_distillation_loss"] = float(self_distillation_loss.item())
                log_dict["tau_t"] = float(tau_t)
            self.log(log_dict)


        return total_loss

