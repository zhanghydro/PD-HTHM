# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable, Any, Set

import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# 1) Config
# =========================================================
@dataclass
class DecoupledOptimizerConfig:
    # base
    weight_decay: float = 1e-5

    # group learning rates (module-based)
    lr_param_net: float = 1e-3
    lr_cell: float = 3e-4
    lr_others: float = 1e-3

    # group learning rates (name-based)
    lr_static_named: float = 1e-2
    lr_dynamic_named: float = 1e-3

    # scheduler (ReduceLROnPlateau)
    use_scheduler: bool = True
    sched_mode: str = "max"
    sched_factor: float = 0.8
    sched_patience: int = 5
    sched_min_lr: float = 1e-5
    sched_verbose: bool = True

    # training stability
    grad_clip_norm: float = 1.0  # set None/0 to disable

    # behavior
    prefer_name_grouping: bool = False  # if True -> use name-based grouping; else module-based
    static_param_leaf_names: Tuple[str, ...] = ("SUB", "CRAK", "SQ", "COEFF", "INSC")

    # (optional) exclude some params from weight decay
    no_weight_decay_keywords: Tuple[str, ...] = ("bias", "LayerNorm.weight", "layernorm.weight", "ln.weight")


# =========================================================
# 2) Utilities
# =========================================================
def _iter_trainable_named_params(model: nn.Module):
    for n, p in model.named_parameters():
        if p is None or (not p.requires_grad):
            continue
        yield n, p


def _should_apply_weight_decay(name: str, cfg: DecoupledOptimizerConfig) -> bool:
    for kw in cfg.no_weight_decay_keywords:
        if kw in name:
            return False
    return True


def _collect_params(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def _unique_params(params: Iterable[nn.Parameter]) -> List[nn.Parameter]:
    """Deduplicate parameters by id while preserving order."""
    out: List[nn.Parameter] = []
    seen: Set[int] = set()
    for p in params:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def _filter_nonempty_groups(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for g in groups:
        ps = g.get("params", [])
        if ps is None:
            continue
        if len(ps) == 0:
            continue
        out.append(g)
    return out


def _build_id2name(model: nn.Module) -> Dict[int, str]:
    return {id(p): n for n, p in model.named_parameters()}


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def freeze_by_name_prefix(model: nn.Module, prefixes: Iterable[str]):
    prefixes = tuple(prefixes)
    for n, p in model.named_parameters():
        if n.startswith(prefixes):
            p.requires_grad = False


def unfreeze_by_name_prefix(model: nn.Module, prefixes: Iterable[str]):
    prefixes = tuple(prefixes)
    for n, p in model.named_parameters():
        if n.startswith(prefixes):
            p.requires_grad = True


def set_trainable_by_leaf_names(model: nn.Module, leaf_names: Iterable[str], trainable: bool):
    """
    Set requires_grad for parameters whose leaf name matches any in leaf_names.
    leaf name = last token after '.'
    """
    leaf_set = set(leaf_names)
    for n, p in model.named_parameters():
        leaf = n.split(".")[-1]
        if leaf in leaf_set:
            p.requires_grad = bool(trainable)


# =========================================================
# 3) Grouping strategies
# =========================================================
def build_param_groups_by_modules(
    model: nn.Module,
    cfg: DecoupledOptimizerConfig,
    module_attr_param_net: str = "param_net",
    module_attr_cell: str = "cell",
) -> List[Dict[str, Any]]:
    """
    Group parameters by major submodules:
      - param_net (dynamic parameter module)
      - cell (process model / physical core)
      - others (everything else)

    Works as long as the model exposes attributes with those names.
    """
    groups: List[Dict[str, Any]] = []
    assigned: Set[int] = set()

    if hasattr(model, module_attr_param_net):
        m = getattr(model, module_attr_param_net)
        ps = _unique_params(_collect_params(m))
        groups.append({"name": "param_net", "params": ps, "lr": cfg.lr_param_net})
        for p in ps:
            assigned.add(id(p))

    if hasattr(model, module_attr_cell):
        m = getattr(model, module_attr_cell)
        ps = _unique_params(_collect_params(m))
        groups.append({"name": "cell", "params": ps, "lr": cfg.lr_cell})
        for p in ps:
            assigned.add(id(p))

    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in assigned]
    rest = _unique_params(rest)
    if rest:
        groups.append({"name": "others", "params": rest, "lr": cfg.lr_others})

    return _filter_nonempty_groups(groups)


def build_param_groups_by_leaf_names(
    model: nn.Module,
    cfg: DecoupledOptimizerConfig,
) -> List[Dict[str, Any]]:
    """
    Group parameters by leaf name matches:
      - static_named: leaf in cfg.static_param_leaf_names
      - dynamic_named: all other trainable parameters
    """
    static_params: List[nn.Parameter] = []
    dynamic_params: List[nn.Parameter] = []

    static_set = set(cfg.static_param_leaf_names)
    for n, p in _iter_trainable_named_params(model):
        leaf = n.split(".")[-1]
        if leaf in static_set:
            static_params.append(p)
        else:
            dynamic_params.append(p)

    groups: List[Dict[str, Any]] = []
    static_params = _unique_params(static_params)
    dynamic_params = _unique_params(dynamic_params)

    if static_params:
        groups.append({"name": "static_named", "params": static_params, "lr": cfg.lr_static_named})
    if dynamic_params:
        groups.append({"name": "dynamic_named", "params": dynamic_params, "lr": cfg.lr_dynamic_named})

    return _filter_nonempty_groups(groups)


def attach_weight_decay_per_param(
    groups: List[Dict[str, Any]],
    model: nn.Module,
    cfg: DecoupledOptimizerConfig,
) -> List[Dict[str, Any]]:
    """
    Split each group into two sub-groups:
      - wd group
      - no-wd group
    """
    id2name = _build_id2name(model)

    new_groups: List[Dict[str, Any]] = []
    for g in groups:
        ps_wd, ps_nowd = [], []
        for p in g["params"]:
            name = id2name.get(id(p), "")
            if _should_apply_weight_decay(name, cfg):
                ps_wd.append(p)
            else:
                ps_nowd.append(p)

        if ps_wd:
            ng = dict(g)
            ng["params"] = _unique_params(ps_wd)
            ng["weight_decay"] = cfg.weight_decay
            ng["name"] = f"{g.get('name','group')}_wd"
            new_groups.append(ng)

        if ps_nowd:
            ng = dict(g)
            ng["params"] = _unique_params(ps_nowd)
            ng["weight_decay"] = 0.0
            ng["name"] = f"{g.get('name','group')}_no_wd"
            new_groups.append(ng)

    return _filter_nonempty_groups(new_groups)


def validate_param_groups(groups: List[Dict[str, Any]]) -> None:
    """
    Validate optimizer groups:
      - no empty groups
      - no duplicated parameter across groups
      - at least one parameter exists
    """
    if not groups:
        raise RuntimeError("No parameter groups were constructed (no trainable parameters?).")

    seen: Set[int] = set()
    total = 0
    for g in groups:
        ps = g.get("params", [])
        if ps is None or len(ps) == 0:
            raise RuntimeError(f"Empty optimizer group: {g.get('name', 'unknown')}")
        for p in ps:
            pid = id(p)
            if pid in seen:
                raise RuntimeError(f"Parameter appears in multiple optimizer groups: {g.get('name','unknown')}")
            seen.add(pid)
        total += len(ps)

    if total <= 0:
        raise RuntimeError("No parameters found in optimizer groups.")


# =========================================================
# 4) Optimizer / Scheduler builders
# =========================================================
def build_optimizer(model: nn.Module, cfg: DecoupledOptimizerConfig) -> optim.Optimizer:
    if cfg.prefer_name_grouping:
        base_groups = build_param_groups_by_leaf_names(model, cfg)
    else:
        base_groups = build_param_groups_by_modules(model, cfg)

    groups = attach_weight_decay_per_param(base_groups, model, cfg)
    validate_param_groups(groups)

    optimizer = optim.Adam(groups)
    return optimizer


def build_scheduler(optimizer: optim.Optimizer, cfg: DecoupledOptimizerConfig):
    if not cfg.use_scheduler:
        return None
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.sched_mode,
        factor=cfg.sched_factor,
        patience=cfg.sched_patience,
        min_lr=cfg.sched_min_lr,
        verbose=cfg.sched_verbose,
    )


def clip_gradients(model: nn.Module, cfg: DecoupledOptimizerConfig):
    if cfg.grad_clip_norm is None:
        return
    if isinstance(cfg.grad_clip_norm, (int, float)) and cfg.grad_clip_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))


# =========================================================
# 5) Logging / Summary
# =========================================================
def optimizer_group_summary(optimizer: optim.Optimizer) -> List[Dict[str, Any]]:
    out = []
    for i, g in enumerate(optimizer.param_groups):
        out.append({
            "idx": i,
            "name": g.get("name", f"group_{i}"),
            "lr": float(g.get("lr", 0.0)),
            "weight_decay": float(g.get("weight_decay", 0.0)),
            "n_params": int(len(g.get("params", []))),
        })
    return out


def print_optimizer_summary(optimizer: optim.Optimizer):
    info = optimizer_group_summary(optimizer)
    txt = " | ".join([f"{d['name']}: n={d['n_params']}, lr={d['lr']}, wd={d['weight_decay']}" for d in info])
    print("[OPT]", txt)


def export_optim_config(cfg: DecoupledOptimizerConfig) -> Dict[str, Any]:
    return asdict(cfg)


# =========================================================
# 6) One-call setup
# =========================================================
def setup_optimization(
    model: nn.Module,
    cfg: DecoupledOptimizerConfig,
) -> Tuple[optim.Optimizer, Optional[Any], Dict[str, Any]]:
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    summary = {
        "optimizer_groups": optimizer_group_summary(optimizer),
        "optim_config": export_optim_config(cfg),
        "grouping_mode": "name" if cfg.prefer_name_grouping else "modules",
    }
    return optimizer, scheduler, summary