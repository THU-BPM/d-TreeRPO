from dataclasses import dataclass, field
from diffu_grpo_config import DiffuGRPOConfig

@dataclass
class DTreeRPOConfig(DiffuGRPOConfig):
    """
    Configuration class for DTreeRPOTrainer.
    """
    _name: str = "DTreeRPOConfig"

    tree_branch_factor: int = field(
        default=2,
        metadata={"help": "T: Number of independent branches to explore at each fork point in the tree search."}
    )
    tree_contraction_factor: int = field(
        default=4,
        metadata={"help": "s: Contraction factor. Branch points are at N/s, 2N/s, ..., where N is total diffusion steps."}
    )
    num_tree_samples: int = field(
        default=1,
        metadata={"help": "Number of full tree rollouts to perform per prompt in a batch."}
    )
    print_tree_prompt_idx: int = field(
        default=0,
        metadata={"help": "Which prompt index in the batch to visualize (default: first prompt)."}
    )
    parents_per_step: int = field(
        default=1,
        metadata={"help": "Number of parents to used for forward/backward per step."}
    )
    enable_consistency_loss: bool = field(
        default=False,
        metadata={"help": "Whether to enable consistency loss."}
    )
    consistency_lambda_max: float = field(
        default=0.5,
        metadata={"help": "Maximum consistency loss lambda."}
    )
    consistency_gamma: float = field(
        default=5.0,
        metadata={"help": "Consistency loss gamma."}
    )
    consistency_tau_max: float = field(
        default=3.0,
        metadata={"help": "Maximum consistency loss tau."}
    )
    consistency_beta: float = field(
        default=1.0,
        metadata={"help": "Minimum consistency loss tau."}
    )
