import logging

import torch.nn as nn

from ..containers import ModelContainer
from ..enums import ParamsGroupMethod
from ..hf_models import (
    GPTDolomiteForCausalLM,
    GPTDolomiteForCausalLM_TP,
    LadderResidualForCausalLM,
    LadderResidualForCausalLM_TP,
)
from ..hf_models.modeling_utils import MLP, Attention, Mamba2Base, MoE
from ..model_wrapper import ModelWrapper
from ..utils import log_rank_0


def get_normal_group_with_names(model: ModelWrapper, optimizer_class_args: dict) -> dict:
    if optimizer_class_args.get("weight_decay") == 0:
        trainable_parameters_or_param_groups = model.parameters()
        names = {"normal": [name for name, _ in model.named_parameters()]}
    else:
        if model.has_teacher_model():
            log_rank_0(logging.WARN, "found a teacher model in the ModelWrapper")
            # this is the student model
            model = model.model

        normal_params = {}
        no_weight_decay_params = {}

        # remove layernorm and rmsnorm parameters from weight decay
        for module_name, module in model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or module.__class__.__name__.lower().endswith("norm"):
                for param_name, param in module.named_parameters():
                    no_weight_decay_params[f"{module_name}.{param_name}"] = param
            elif isinstance(module, Mamba2Base):
                for param_name, param in module.named_parameters():
                    # we don't add bias or norms to mup group
                    if param_name.endswith("A_log") or param_name.endswith("D"):
                        no_weight_decay_params[f"{module_name}.{param_name}"] = param

        # remove biases from weight decay
        for param_name, param in model.named_parameters():
            if param_name not in no_weight_decay_params and param_name.endswith("bias"):
                no_weight_decay_params[param_name] = param

        # these parameters have weight decay
        for param_name, param in model.named_parameters():
            if param_name not in no_weight_decay_params:
                normal_params[param_name] = param

        assert len(normal_params) + len(no_weight_decay_params) == len(
            list(model.parameters())
        ), "params in groups don't sum up to total parameters"

        trainable_parameters_or_param_groups = []
        names = {}

        if len(normal_params) > 0:
            trainable_parameters_or_param_groups.append({"params": list(normal_params.values())})
            names["normal"] = list(normal_params.keys())
        if len(no_weight_decay_params) > 0:
            trainable_parameters_or_param_groups.append(
                {"params": list(no_weight_decay_params.values()), "weight_decay": 0}
            )
            names["no_weight_decay"] = list(no_weight_decay_params.keys())

    return trainable_parameters_or_param_groups, names


def get_mup_group_with_names(model: ModelWrapper, optimizer_class_args: dict) -> list[dict]:
    assert isinstance(
        model.model,
        (
            GPTDolomiteForCausalLM,
            GPTDolomiteForCausalLM_TP,
            LadderResidualForCausalLM,
            LadderResidualForCausalLM_TP,
        ),
    ), "mup is not supported with this model architecture"

    assert (
        model.config.init_method == "mup"
    ), "both init method for model and params group method for optimizer should be set to mup"

    if model.has_teacher_model():
        log_rank_0(logging.WARN, "found a teacher model in the ModelWrapper")
        # this is the student model
        model = model.model

    normal_params = {}
    no_weight_decay_params = {}
    mup_params = {}

    # collect parameters with mup learning rate
    for module_name, module in model.named_modules():
        if isinstance(module, (Attention, MLP, MoE)):
            for param_name, param in module.named_parameters():
                # we don't add bias or norms to mup group
                if not (param_name.endswith("bias") or "norm" in param_name):
                    # add name of module to name of subparam
                    mup_params[f"{module_name}.{param_name}"] = param
        elif isinstance(module, Mamba2Base):
            for param_name, param in module.named_parameters():
                if param_name in ["A_log", "D"] or not (param_name.endswith("bias") or "norm" in param_name):
                    mup_params[f"{module_name}.{param_name}"] = param
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or module.__class__.__name__.lower().endswith("norm"):
            for param_name, param in module.named_parameters():
                no_weight_decay_params[f"{module_name}.{param_name}"] = param

    # remove biases from weight decay
    for param_name, param in model.named_parameters():
        if param_name not in no_weight_decay_params and param_name.endswith("bias"):
            no_weight_decay_params[param_name] = param

    # collect parameters without mup learning rate
    for param_name, param in model.named_parameters():
        if param_name not in mup_params and param_name not in no_weight_decay_params:
            normal_params[param_name] = param

    assert len(normal_params) + len(no_weight_decay_params) + len(mup_params) == len(
        list(model.parameters())
    ), "params in groups don't sum up to total parameters"

    trainable_parameters_or_param_groups = []
    names = {}

    if len(normal_params) > 0:
        trainable_parameters_or_param_groups.append({"params": list(normal_params.values())})
        names["normal"] = list(normal_params.keys())
    if len(no_weight_decay_params) > 0:
        trainable_parameters_or_param_groups.append(
            {"params": list(no_weight_decay_params.values()), "weight_decay": 0}
        )
        names["no_weight_decay"] = list(no_weight_decay_params.keys())
    if len(mup_params) > 0:
        trainable_parameters_or_param_groups.append(
            {"params": list(mup_params.values()), "lr": optimizer_class_args["lr"] / model.config.m_width}
        )
        names["mup"] = list(mup_params.keys())

    return trainable_parameters_or_param_groups, names


_PARAM_GROUPS = {
    None: get_normal_group_with_names,
    ParamsGroupMethod.mup: get_mup_group_with_names,
}


def get_param_groups_list(
    model_container: ModelContainer, optimizer_class_args: dict, params_group_method: ParamsGroupMethod | None
) -> list[list[dict]]:
    if params_group_method not in _PARAM_GROUPS:
        raise ValueError(f"unexpected `params_group_method` {params_group_method}")

    return [_PARAM_GROUPS[params_group_method](model, optimizer_class_args)[0] for model in model_container]
