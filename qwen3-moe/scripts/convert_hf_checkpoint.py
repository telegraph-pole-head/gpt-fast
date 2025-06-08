# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import glob
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs  # noqa: E402


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/Qwen/Qwen3-30B-A3B"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # First, load all safetensors files
    st_files = glob.glob(str(checkpoint_dir / "*.safetensors"))

    merged_result = {}
    for st_file in sorted(st_files):
        print(f"Loading {st_file}")
        state_dict = load_file(st_file)
        merged_result.update(state_dict)

    print(f"Total keys loaded: {len(merged_result)}")

    final_result = {}

    # Process each key in the merged state dict
    for hf_key, value in merged_result.items():
        new_key = None

        if hf_key == "model.embed_tokens.weight":
            new_key = "tok_embeddings.weight"
        elif hf_key == "model.norm.weight":
            new_key = "norm.weight"
        elif hf_key == "lm_head.weight":
            new_key = "output.weight"
        elif "model.layers." in hf_key:
            # Extract layer number
            layer_match = re.search(r'model\.layers\.(\d+)\.', hf_key)
            if layer_match:
                layer_num = layer_match.group(1)
                remaining_key = hf_key.replace(f"model.layers.{layer_num}.", "")

                if remaining_key == "input_layernorm.weight":
                    new_key = f"layers.{layer_num}.attention_norm.weight"
                elif remaining_key == "post_attention_layernorm.weight":
                    new_key = f"layers.{layer_num}.ffn_norm.weight"
                elif remaining_key == "self_attn.q_proj.weight":
                    new_key = f"layers.{layer_num}.attention.wq.weight"
                elif remaining_key == "self_attn.k_proj.weight":
                    new_key = f"layers.{layer_num}.attention.wk.weight"
                elif remaining_key == "self_attn.v_proj.weight":
                    new_key = f"layers.{layer_num}.attention.wv.weight"
                elif remaining_key == "self_attn.o_proj.weight":
                    new_key = f"layers.{layer_num}.attention.wo.weight"
                elif remaining_key == "self_attn.q_norm.weight":
                    new_key = f"layers.{layer_num}.attention.q_norm.weight"
                elif remaining_key == "self_attn.k_norm.weight":
                    new_key = f"layers.{layer_num}.attention.k_norm.weight"
                elif remaining_key == "mlp.gate.weight":
                    new_key = f"layers.{layer_num}.block_sparse_moe.gate.weight"
                elif "mlp.experts." in remaining_key:
                    # Handle expert weights
                    expert_match = re.search(r'mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight', remaining_key)
                    if expert_match:
                        expert_num = int(expert_match.group(1))
                        proj_type = expert_match.group(2)

                        if proj_type == "gate_proj":
                            new_key = f"layers.{layer_num}.block_sparse_moe.cond_ffn.w1"
                        elif proj_type == "up_proj":
                            new_key = f"layers.{layer_num}.block_sparse_moe.cond_ffn.w3"
                        elif proj_type == "down_proj":
                            new_key = f"layers.{layer_num}.block_sparse_moe.cond_ffn.w2"

                        # We'll accumulate expert weights and process them later
                        if new_key not in final_result:
                            if proj_type == "down_proj":
                                final_result[new_key] = torch.zeros(config.num_experts, config.dim, config.moe_intermediate_size, dtype=value.dtype)
                            else:
                                final_result[new_key] = torch.zeros(config.num_experts, config.moe_intermediate_size, config.dim, dtype=value.dtype)

                        if proj_type == "down_proj":
                            final_result[new_key][expert_num] = value.T  # Transpose for down_proj
                        else:
                            final_result[new_key][expert_num] = value
                        continue

        if new_key is not None and new_key not in final_result:
            final_result[new_key] = value

    # Process Q, K, V projections - combine them into wqkv for each layer
    q_keys = [k for k in final_result.keys() if ".wq.weight" in k]
    for q_key in q_keys:
        layer_prefix = q_key.replace(".wq.weight", "")
        k_key = f"{layer_prefix}.wk.weight"
        v_key = f"{layer_prefix}.wv.weight"

        if k_key in final_result and v_key in final_result:
            q = final_result[q_key]
            k = final_result[k_key]
            v = final_result[v_key]

            # Combine Q, K, V into single tensor
            wqkv_key = f"{layer_prefix}.wqkv.weight"
            final_result[wqkv_key] = torch.cat([q, k, v], dim=0)

            # Remove individual Q, K, V weights
            del final_result[q_key]
            del final_result[k_key]
            del final_result[v_key]

    # Ensure expert weights are contiguous
    for key in final_result.keys():
        if "block_sparse_moe.cond_ffn" in key:
            final_result[key] = final_result[key].contiguous()

    print(f"Final model keys: {len(final_result)}")
    for key in sorted(final_result.keys())[:20]:
        print(f"  {key}: {final_result[key].shape}")

    output_path = checkpoint_dir / "model.pth"
    print(f"Saving checkpoint to {output_path}")
    torch.save(final_result, output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert Qwen3-MoE HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/Qwen/Qwen3-30B-A3B"))
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
