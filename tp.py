# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import List, Optional

import torch
import torch.distributed as dist
from functools import wraps
from torch import nn, Tensor
if os.uname().sysname != "Darwin":
    from torch.distributed import _functional_collectives as funcol
else:
    # Distributed is not supported on MacOS
    funcol = None

from model import Attention, FeedForward, Transformer


def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def rank_zero():
    return _get_rank() == 0

def local_break():
    if rank_zero():
        breakpoint()
    dist.barrier()

def get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def maybe_init_dist() -> Optional[int]:
    try:
        # provided by torchrun
        rank = _get_rank()
        world_size = get_world_size()
        
        if world_size < 2:
            # too few gpus to parallelize, tp is no-op
            return None
    except KeyError:
        # not run via torchrun, no-op
        return None
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank


def _shard_linear(linear: nn.Linear, style: str, weight_splits: List[int] = []) -> None:
    rank = _get_rank()
    world_size = get_world_size()
    
    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {
        "colwise": (0, "out_features"),
        "rowwise": (1, "in_features")
    }
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]
    
    # ensure we can shard evenly
    assert getattr(linear, size_attr) % world_size == 0
    def shard(x, dim):
        assert x.size(dim=dim) % world_size == 0
        return torch.tensor_split(x, world_size, dim=dim)[rank]
    
    def shard_qkv(qkv, dim, weight_splits):
        q, k, v = qkv.split(weight_splits, dim=dim)
        q = shard(q, dim)
        k = shard(k, dim)
        v = shard(v, dim)
        return torch.cat((q,k,v), dim=dim)
    
    # shard
    if weight_splits:
        # attention
        assert len(weight_splits) == 3
        
        sharded_weight = shard_qkv(linear.weight, shard_dim, weight_splits)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard_qkv(linear.scales, 0, weight_splits)
    else:
        sharded_weight = shard(linear.weight, shard_dim)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard(linear.scales, 0)
    
    # local_break()
    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)
    
    # shape info should still be synced
    # assert linear.weight.shape == (linear.out_features, linear.in_features)

def _apply_tp_ffn(mlp: FeedForward) -> None:
    assert hasattr(mlp, "w1")
    assert hasattr(mlp, "w3")
    assert hasattr(mlp, "w2")
    
    _shard_linear(mlp.w1, "colwise")
    _shard_linear(mlp.w3, "colwise")
    _shard_linear(mlp.w2, "rowwise")
    
    world_size = get_world_size()
    mlp.register_forward_hook(lambda _module, _input, output: funcol.all_reduce(
        output, "sum", list(range(world_size))))


def _apply_tp_lm_head(head: nn.Linear, style: Optional[str]) -> None:
    world_size = get_world_size()
    rank = _get_rank()
    
    assert world_size > 1
    
    chunk_size = head.in_features // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    
    def rowwise_pre_forward_hook(module, input: tuple[Tensor]):
        input_tensor = input[0]
        input_tensor = input_tensor[..., start_idx:end_idx]
        return (input_tensor,)
    
    def rowwise_forward_hook(module, input, output: Tensor):
        return funcol.all_reduce(output, reduceOp="sum", group=list(range(world_size)))
    
    def colwise_forward_hook(module, input, output: Tensor):
        return funcol.all_gather_tensor(output, gather_dim=-1, group=list(range(world_size)))
    
    _shard_linear(head, style)
    match style:
        case "rowwise":
            head.register_forward_pre_hook(rowwise_pre_forward_hook)
            head.register_forward_hook(rowwise_forward_hook)
        case "colwise":
            head.register_forward_hook(colwise_forward_hook)


def _apply_tp_attn(attn: Attention) -> None:
    assert hasattr(attn, "wqkv")
    assert hasattr(attn, "wo")
    
    kv_size = attn.n_local_heads * attn.head_dim
    _shard_linear(attn.wqkv, "colwise", [attn.dim, kv_size, kv_size])
    _shard_linear(attn.wo, "rowwise")
    
    # overwrite
    world_size = get_world_size()
    attn.n_head = attn.n_head // world_size
    attn.dim = attn.dim // world_size
    attn.head_dim = attn.dim // attn.n_head
    attn.n_local_heads = attn.n_local_heads // world_size
    
    attn.register_forward_hook(lambda _module, _input, output: funcol.all_reduce(
        output[0], "sum", list(range(world_size))))


def _apply_tp_Transformer(model: Transformer) -> None:
    # overwrite config before Transformer.setup_cache is called
    world_size = get_world_size()
    model.config.n_head = model.config.n_head // world_size
    model.config.dim = model.config.dim // world_size
    model.config.n_local_heads = model.config.n_local_heads // world_size



def apply_tp(model: Transformer, **tp_kwargs) -> Transformer:
    _apply_tp_Transformer(model)
    for i in range(len(model.layers)):
        _apply_tp_ffn(model.layers[i].feed_forward)
        _apply_tp_attn(model.layers[i].attention)
    
    _apply_tp_lm_head(model.output, tp_kwargs["lm_head_shard_style"])
    
    return model