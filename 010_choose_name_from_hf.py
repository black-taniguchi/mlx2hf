import json
from safetensors import safe_open
import os
import argparse

def extract_hf_keys(model_path, output_file="hf_keys.json"):
    """
    HFモデルのディレクトリから全てのsafetensorsファイルのキーを抽出して保存する
    """
    all_keys = []
    
    # モデルディレクトリの存在確認
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        return

    # モデルディレクトリ内の.safetensorsファイルをすべて探す
    try:
        files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    except Exception as e:
        print(f"Error reading directory '{model_path}': {e}")
        return

    if not files:
        print(f"No .safetensors files found in '{model_path}'.")
        return

    files.sort() # 順序を安定させる

    print(f"Found {len(files)} safetensors files. Extracting keys...")

    for file_name in files:
        file_path = os.path.join(model_path, file_name)
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                # ファイルに含まれるキーの一覧を取得
                keys = f.keys()
                all_keys.extend(keys)
                print(f"Extracted {len(keys)} keys from {file_name}")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    if not all_keys:
        print("No keys were extracted.")
        return

    # キーの一覧をJSON形式で保存（リストとして保存）
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_keys, f, indent=4)
        print(f"\nSuccess! Total keys: {len(all_keys)}")
        print(f"Keys saved to: {output_file}")
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract keys from safetensors files in a Hugging Face model directory."
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the Hugging Face model directory containing .safetensors files."
    )
    parser.add_argument(
        "--output", "-o",
        default="hf_keys.json",
        help="Path to the output JSON file (default: hf_keys.json)."
    )
    
    args = parser.parse_args()
    extract_hf_keys(args.model, args.output)
