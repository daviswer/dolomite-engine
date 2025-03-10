from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ...config import CommonConfig
from ...enums import InitMethod
from .mlp import MLP, interleave_up_gate_tensor_for_mlp, split_up_gate_tensor_for_mlp
from .moe import AuxFreeMoE, MoE, ParameterizedExperts, ParameterizedScatteredExperts, ScatterMoE


def get_mlp_block(config: CommonConfig, use_padding_free_transformer: bool, layer_idx: int) -> MLP | MoE:
    block = config.mlp_blocks[layer_idx]
    mlp_type = block.mlp_type

    kwargs = dict(
        hidden_size=config.hidden_size,
        intermediate_size=block.intermediate_size,
        activation_function=block.activation_function,
        add_bias=block.add_bias,
        dropout=block.dropout,
        init_method=InitMethod(config.init_method),
        initializer_range=config.initializer_range,
        m_width=config.m_width,
        num_layers=config.num_layers,
    )

    if mlp_type == "MLP":
        return MLP(**kwargs)
    elif mlp_type == "MoE":
        mlp_block_class = ScatterMoE if is_kernel_allowed(Kernel.scattermoe) else MoE
        return mlp_block_class(
            **kwargs,
            shared_intermediate_size=block.shared_intermediate_size,
            num_experts=block.num_experts,
            num_experts_per_tok=block.num_experts_per_tok,
            use_padding_free_transformer=use_padding_free_transformer,
        )
    elif mlp_type == "AuxFreeMoE":
        assert is_kernel_allowed(Kernel.scattermoe)
        return AuxFreeMoE(config, use_padding_free_transformer)
    else:
        raise ValueError(f"invalid mlp_type ({mlp_type}) for layer ({layer_idx})")
