import json
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import os
import argparse

def convert_mlx_to_hf(mlx_model_dir, hf_output_dir, mapping_file):
    """
    MLXモデルの重みをマッピングに従ってHF形式に変換して保存する
    """
    if not os.path.exists(mlx_model_dir):
        print(f"Error: MLX model directory '{mlx_model_dir}' does not exist.")
        return
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file '{mapping_file}' does not exist.")
        return

    # 1. マッピングリストの読み込み { mlx_key: hf_key }
    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    except Exception as e:
        print(f"Error reading mapping file '{mapping_file}': {e}")
        return

    try:
        if not os.path.exists(hf_output_dir):
            os.makedirs(hf_output_dir)
    except Exception as e:
        print(f"Error creating output directory '{hf_output_dir}': {e}")
        return

    # 保存用のstate_dict
    hf_state_dict = {}

    # 2. MLXモデルの全safetensorsファイルを走査
    try:
        mlx_files = [f for f in os.listdir(mlx_model_dir) if f.endswith(".safetensors")]
    except Exception as e:
        print(f"Error reading MLX directory '{mlx_model_dir}': {e}")
        return

    if not mlx_files:
        print(f"No .safetensors files found in '{mlx_model_dir}'.")
        return

    mlx_files.sort()

    print(f"Starting conversion of {len(mlx_files)} files...")

    for file_name in mlx_files:
        file_path = os.path.join(mlx_model_dir, file_name)
        print(f"Processing {file_name}...")

        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for mlx_key in f.keys():
                    if mlx_key in mapping:
                        hf_key = mapping[mlx_key]
                        # 重みの値をロードしてHFのキー名で保存
                        hf_state_dict[hf_key] = f.get_tensor(mlx_key)
                    else:
                        print(f"Warning: Key '{mlx_key}' not found in mapping. Skipping.")
        except Exception as e:
            print(f"Error processing file '{file_name}': {e}")

    if not hf_state_dict:
        print("No weights were successfully mapped. Aborting save.")
        return

    # 3. HF形式で保存 (index.json が存在すれば分割保存し、なければ単一ファイルで保存)
    index_file_path = os.path.join(hf_output_dir, "model.safetensors.index.json")
    
    if os.path.exists(index_file_path):
        print(f"\nFound index file: {index_file_path}. Saving weights in sharded format...")
        try:
            with open(index_file_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            
            # 各シャードファイル用のテンソル辞書を初期化
            shards = {}
            for hf_key, shard_name in weight_map.items():
                if hf_key in hf_state_dict:
                    if shard_name not in shards:
                        shards[shard_name] = {}
                    shards[shard_name][hf_key] = hf_state_dict[hf_key]
                else:
                    # 完全にマッピングされていないキーは警告のみ出す
                    pass

            # 各シャードを保存
            for shard_name, shard_tensors in shards.items():
                shard_path = os.path.join(hf_output_dir, shard_name)
                print(f"Saving shard to {shard_path} ({len(shard_tensors)} keys)...")
                save_file(shard_tensors, shard_path, metadata={"format": "pt"})
            
            # 不要になった単一 model.safetensors がもし残っていれば削除
            single_file_path = os.path.join(hf_output_dir, "model.safetensors")
            if os.path.exists(single_file_path):
                os.remove(single_file_path)
                print("Removed single model.safetensors to avoid conflicts.")

            print("\nSharded Conversion Complete! Weights have been saved in HF format.")
            print(f"Next step: Verify load using 060_check1.py")
            return
        except Exception as e:
            print(f"Error during sharded saving: {e}. Falling back to single-file save...")

    # フォールバック: 単一ファイルでの保存
    output_path = os.path.join(hf_output_dir, "model.safetensors")
    print(f"\nSaving converted weights to a single file: {output_path}...")
    try:
        save_file(hf_state_dict, output_path, metadata={"format": "pt"})
        print("\nConversion Complete! Weights have been saved in HF format (single file).")
        print(f"Next step: Copy config.json and tokenizer files to {hf_output_dir}")
    except Exception as e:
        print(f"Error saving weights to '{output_path}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MLX (VLM) model weights to Hugging Face format based on a key mapping."
    )
    parser.add_argument(
        "--mlx-dir", "-mlx",
        required=True,
        help="Path to the input MLX model directory containing .safetensors files."
    )
    parser.add_argument(
        "--hf-dir", "-hf",
        required=True,
        help="Path to the output directory where HF format model weights will be saved."
    )
    parser.add_argument(
        "--mapping", "-m",
        default="mapping_list.json",
        help="Path to the JSON mapping file (default: mapping_list.json)."
    )

    args = parser.parse_args()
    convert_mlx_to_hf(args.mlx_dir, args.hf_dir, args.mapping)
