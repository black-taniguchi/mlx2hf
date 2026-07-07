import json
import argparse
import os

def analyze_config_diff(hf_path, mlx_path, output_file="config_diff.json"):
    """
    HFとMLXのコンフィグファイルを比較し、差分を抽出する
    """
    # 比較対象のファイルのペアを決定する (ファイル名, HFのパス, MLXのパス)
    pairs_to_compare = []

    if os.path.isdir(hf_path) and os.path.isdir(mlx_path):
        # ディレクトリ同士の場合は、主要な設定ファイルを自動で探索する
        candidates = ["config.json", "processor_config.json", "preprocessor_config.json"]
        for filename in candidates:
            hf_file = os.path.join(hf_path, filename)
            mlx_file = os.path.join(mlx_path, filename)
            if os.path.exists(hf_file) or os.path.exists(mlx_file):
                pairs_to_compare.append((filename, hf_file, mlx_file))
    else:
        # 直接ファイルが指定された場合
        pairs_to_compare.append((os.path.basename(hf_path), hf_path, mlx_path))

    if not pairs_to_compare:
        print("Error: No configuration files found to compare.")
        return

    diff_report = {}

    for name, hf_file, mlx_file in pairs_to_compare:
        if not os.path.exists(hf_file):
            print(f"Warning: HF file '{hf_file}' does not exist. Skipping.")
            diff_report[name] = {"error": "Missing in HF"}
            continue
        if not os.path.exists(mlx_file):
            print(f"Warning: MLX file '{mlx_file}' does not exist. Skipping.")
            diff_report[name] = {"error": "Missing in MLX"}
            continue

        try:
            with open(hf_file, "r", encoding="utf-8") as f:
                hf_cfg = json.load(f)
        except Exception as e:
            print(f"Error reading HF file '{hf_file}': {e}")
            diff_report[name] = {"error": f"Failed to read HF file: {e}"}
            continue

        try:
            with open(mlx_file, "r", encoding="utf-8") as f:
                mlx_cfg = json.load(f)
        except Exception as e:
            print(f"Error reading MLX file '{mlx_file}': {e}")
            diff_report[name] = {"error": f"Failed to read MLX file: {e}"}
            continue

        hf_keys = set(hf_cfg.keys())
        mlx_keys = set(mlx_cfg.keys())

        # 1. HFにしか存在しないキー
        only_in_hf = list(hf_keys - mlx_keys)

        # 2. MLXにしか存在しないキー
        only_in_mlx = list(mlx_keys - hf_keys)

        # 3. 両方にあるが、値が異なるキー
        common_keys = hf_keys & mlx_keys
        value_diffs = {}
        for k in common_keys:
            if hf_cfg[k] != mlx_cfg[k]:
                value_diffs[k] = {
                    "hf_value": hf_cfg[k],
                    "mlx_value": mlx_cfg[k]
                }

        diff_report[name] = {
            "only_in_hf": only_in_hf,
            "only_in_mlx": only_in_mlx,
            "value_diffs": value_diffs
        }

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(diff_report, f, indent=4)
        print(f"Detailed diff saved to: {output_file}")
    except Exception as e:
        print(f"Error saving diff report to '{output_file}': {e}")

    print("\n--- Config Analysis Report ---")
    for name, report in diff_report.items():
        print(f"\nFile: {name}")
        if "error" in report:
            print(f"  Error: {report['error']}")
        else:
            print(f"  Keys only in HF: {len(report['only_in_hf'])}")
            print(f"  Keys only in MLX: {len(report['only_in_mlx'])}")
            print(f"  Common keys with different values: {len(report['value_diffs'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare HF and MLX configuration files and generate a diff report."
    )
    parser.add_argument(
        "--hf-path", "-hf",
        required=True,
        help="Path to the HF model directory (containing config.json) or direct path to config.json."
    )
    parser.add_argument(
        "--mlx-path", "-mlx",
        required=True,
        help="Path to the MLX model directory (containing config.json) or direct path to config.json."
    )
    parser.add_argument(
        "--output", "-o",
        default="config_diff.json",
        help="Path to the output JSON file for config diff (default: config_diff.json)."
    )

    args = parser.parse_args()
    analyze_config_diff(args.hf_path, args.mlx_path, args.output)

