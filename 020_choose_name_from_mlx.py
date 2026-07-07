import json
from safetensors import safe_open
import os
import argparse

def extract_mlx_keys(model_path, output_file="mlx_keys.json"):
    """
    MLXモデルのディレクトリ（またはファイル）から全てのキーを抽出して保存する
    """
    all_keys = []

    # 1. 指定されたパスがファイルかディレクトリかを確認
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        return

    if os.path.isfile(model_path):
        # 単一ファイルの場合
        try:
            with safe_open(model_path, framework="pt", device="cpu") as f:
                all_keys = list(f.keys())
        except Exception as e:
            print(f"Error reading file '{model_path}': {e}")
            return
    elif os.path.isdir(model_path):
        # ディレクトリの場合、中の.safetensorsファイルをすべて探す
        try:
            files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        except Exception as e:
            print(f"Error reading directory '{model_path}': {e}")
            return

        if not files:
            print(f"No .safetensors files found in '{model_path}'.")
            return

        files.sort()
        
        print(f"Found {len(files)} safetensors files in MLX directory. Extracting...")
        for file_name in files:
            file_path = os.path.join(model_path, file_name)
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    all_keys.extend(f.keys())
            except Exception as e:
                print(f"Error reading '{file_name}': {e}")
    else:
        print("Error: Specified path is neither a file nor a directory.")
        return

    if not all_keys:
        print("No keys were extracted.")
        return

    # キーの一覧をJSON形式で保存（リストとして保存）
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_keys, f, indent=4)
        print(f"\nSuccess! Total MLX keys: {len(all_keys)}")
        print(f"Keys saved to: {output_file}")
    except Exception as e:
        print(f"Error saving to '{output_file}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract keys from safetensors files in an MLX model directory or file."
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the MLX model directory or safetensors file."
    )
    parser.add_argument(
        "--output", "-o",
        default="mlx_keys.json",
        help="Path to the output JSON file (default: mlx_keys.json)."
    )
    
    args = parser.parse_args()
    extract_mlx_keys(args.model, args.output)
