# FX → GE node converters for torchcomms collective ops on Ascend NPU.
#
# Each registration teaches torchair how to lower a torch.ops.torchcomms.*
# FX node into the matching HcomXxx GE op, so torchair-compiled graphs that
# contain torchcomms collectives can be executed natively on the NPU.
#
# Skeleton stage: registrations are guarded behind a check that the
# torchcomms functional/ path is active and the corresponding torch op
# exists. When the gate is closed (torchcomms functional/ not enabled,
# e.g. torch < 2.12 without the override env var), this module imports
# cleanly but does nothing — the caller falls back to eager mode.

from __future__ import annotations

import logging
from typing import Any

import torch

from torch_npu.dynamo.torchair._ge_concrete_graph import ge_apis as ge
from torch_npu.dynamo.torchair._ge_concrete_graph.fx2ge_converter import (
    register_fx_node_ge_converter,
)

logger = logging.getLogger(__name__)


# ---- ReduceOp string mapping ----
# ge.HcomAllReduce expects reduction in {"sum", "prod", "max", "min"}.

def _reduce_op_to_str(op: Any) -> str:
    name = getattr(op, "name", None)
    if name is None:
        name = str(op).split(".")[-1]
    return {
        "SUM": "sum",
        "PRODUCT": "prod",
        "PROD": "prod",
        "MAX": "max",
        "MIN": "min",
    }.get(name.upper(), "sum")


def _resolve_group_name(_options=None) -> str:
    # Defaults to hccl_world_group; future revisions can resolve from
    # CommOptions.hints["group_name"] or comm.get_comm_name().
    return "hccl_world_group"


# ---- Registration helpers ----
#
# torchcomms exposes collectives via torch.library only when functional/ is
# active. If the namespace doesn't exist, skip registration cleanly and let
# eager-mode users continue to work.

_REGISTERED: list[str] = []


def _has_torchcomms_op(op_name: str) -> bool:
    if not hasattr(torch.ops, "torchcomms"):
        return False
    return hasattr(torch.ops.torchcomms, op_name)


def _register_all_reduce() -> None:
    if not _has_torchcomms_op("all_reduce"):
        return

    op = torch.ops.torchcomms.all_reduce.default

    @register_fx_node_ge_converter(op)
    def _conv_all_reduce(tensor, reduce_op, async_op=False, *_, meta_outputs=None):
        return ge.HcomAllReduce(
            tensor,
            reduction=_reduce_op_to_str(reduce_op),
            group=_resolve_group_name(),
            fusion=0,
            fusion_id=-1,
        )

    _REGISTERED.append("all_reduce")


def _register_broadcast() -> None:
    if not _has_torchcomms_op("broadcast"):
        return

    op = torch.ops.torchcomms.broadcast.default

    @register_fx_node_ge_converter(op)
    def _conv_broadcast(tensor, root, async_op=False, *_, meta_outputs=None):
        return ge.HcomBroadcast(
            [tensor], root_rank=int(root), group=_resolve_group_name(),
            fusion=0, fusion_id=-1,
        )

    _REGISTERED.append("broadcast")


def _register_all_gather() -> None:
    if not _has_torchcomms_op("all_gather_single"):
        return

    op = torch.ops.torchcomms.all_gather_single.default

    @register_fx_node_ge_converter(op)
    def _conv_all_gather_single(output, input, async_op=False, *_,
                                  meta_outputs=None):
        return ge.HcomAllGather(
            input, rank_size=0,  # rank_size=0 → torchair derives from comm
            group=_resolve_group_name(), fusion=0, fusion_id=-1,
        )

    _REGISTERED.append("all_gather_single")


def _try_register_all() -> None:
    try:
        _register_all_reduce()
        _register_broadcast()
        _register_all_gather()
    except Exception as e:  # noqa: BLE001
        logger.warning("torchcomms hccl torchair: registration failed: %s", e)
        return
    if _REGISTERED:
        logger.info(
            "torchcomms hccl torchair: registered FX→GE converters for %s",
            _REGISTERED,
        )
    else:
        logger.info(
            "torchcomms hccl torchair: torch.ops.torchcomms not present, "
            "skipping registration (eager mode is unaffected)."
        )


_try_register_all()
