import json
import argparse
import os

def create_string_mapping(hf_keys_file, mlx_keys_file, output_mapped_file="mapping_list.json", output_unmapped_file="unmapped_mlx_keys.json"):
    """
    HFとMLXのキー一覧を読み込み、文字列パターンに基づいてマッピングを作成する
    """
    if not os.path.exists(hf_keys_file):
        print(f"Error: HF keys file '{hf_keys_file}' does not exist.")
        return
    if not os.path.exists(mlx_keys_file):
        print(f"Error: MLX keys file '{mlx_keys_file}' does not exist.")
        return

    try:
        with open(hf_keys_file, "r", encoding="utf-8") as f:
            hf_keys = json.load(f)
    except Exception as e:
        print(f"Error reading HF keys file: {e}")
        return

    try:
        with open(mlx_keys_file, "r", encoding="utf-8") as f:
            mlx_keys = json.load(f)
    except Exception as e:
        print(f"Error reading MLX keys file: {e}")
        return

    # マッピング結果を格納する辞書 { mlx_key: hf_key }
    mapping = {}
    unmapped_mlx_keys = set(mlx_keys)

    # --- 置換ルールの定義 ---
    # (MLX側のパターン, HF側で置換する文字列)
    rules = [
        ("language_model.model.", "model.language_model."), 
        ("embed_vision.", "model.embed_vision."),
        ("vision_tower.", "model.vision_tower."),
    ]

    print("Applying string mapping rules...")

    for mlx_key in mlx_keys:
        for mlx_pattern, hf_replacement in rules:
            if mlx_key.startswith(mlx_pattern):
                # パターンに一致する場合、接頭辞を置換してHF側にあるか確認
                potential_hf_key = mlx_key.replace(mlx_pattern, hf_replacement, 1)
                if potential_hf_key in hf_keys:
                    mapping[mlx_key] = potential_hf_key
                    if mlx_key in unmapped_mlx_keys:
                        unmapped_mlx_keys.remove(mlx_key)
                    break # 一致したら次のキーへ

    # 結果の保存 (Mapped)
    try:
        with open(output_mapped_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=4)
        print(f"Mapped keys saved to: {output_mapped_file}")
    except Exception as e:
        print(f"Error saving mapped keys to '{output_mapped_file}': {e}")

    # 結果の保存 (Unmapped)
    try:
        sorted_unmapped = sorted(list(unmapped_mlx_keys))
        with open(output_unmapped_file, "w", encoding="utf-8") as f:
            json.dump(sorted_unmapped, f, indent=4)
        print(f"Unmapped keys saved to: {output_unmapped_file}")
    except Exception as e:
        print(f"Error saving unmapped keys to '{output_unmapped_file}': {e}")

    print(f"\nMapping Complete!")
    print(f"Total MLX keys: {len(mlx_keys)}")
    print(f"Successfully mapped: {len(mapping)}")
    print(f"Remaining unmapped keys: {len(unmapped_mlx_keys)}")

    # 未解決のキーを少し表示して傾向を確認
    if unmapped_mlx_keys:
        print("\nSample of unmapped keys:")
        for k in sorted_unmapped[:5]:
            print(f" - {k}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map MLX and HF keys based on string substitution rules."
    )
    parser.add_argument(
        "--hf-keys", "-hf",
        required=True,
        help="Path to the JSON file containing Hugging Face model keys."
    )
    parser.add_argument(
        "--mlx-keys", "-mlx",
        required=True,
        help="Path to the JSON file containing MLX model keys."
    )
    parser.add_argument(
        "--output-mapped", "-om",
        default="mapping_list.json",
        help="Path to the output JSON file for mapped keys (default: mapping_list.json)."
    )
    parser.add_argument(
        "--output-unmapped", "-ou",
        default="unmapped_mlx_keys.json",
        help="Path to the output JSON file for unmapped MLX keys (default: unmapped_mlx_keys.json)."
    )

    args = parser.parse_args()
    create_string_mapping(
        args.hf_keys,
        args.mlx_keys,
        args.output_mapped,
        args.output_unmapped
    )
