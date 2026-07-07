# MLX to HF Converter 🔄

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20MLX-orange.svg)](https://github.com/ml-explore/mlx)

MLX形式（Apple Silicon最適化形式）のモデル重みを、Hugging Face (HF) 形式に効率的かつ正確に変換して復元するためのユーティリティツールキットです。

---

## 🚀 なぜこれが必要なのか？

**MLX** は Mac (Apple Silicon) 上でのローカル推論やLoRA微調整において極めて強力ですが、その形式で保存されたモデルをそのままクラウド環境や他の高性能推論エンジン（例: **vLLM**, **TGI**, **LM Studio** など）で利用することは困難です。

本ツールキットを使用することで、MLX形式の重みをHugging Face形式（PyTorch互換の `safetensors`）にリネームして書き出し、他の爆速推論環境へデプロイすることが可能になります。

---

## ✨ 主な特徴

- 🔄 **キー名のマッピング変換**: 変数名（接頭辞など）の付け替えを行い、HF互換の形式に復元します。
- 📦 **シャード保存対応**: `model.safetensors.index.json` が出力先にある場合、自動的に元のモデルと全く同じ分割構造（Sharded format）でシャード保存し、巨大なモデルでもメモリ効率よく保存します。
- 🔍 **設定ファイル差分検知**: `config.json` に加え、`processor_config.json` などの画像処理設定ファイルの差分も自動で比較・検証します。
- 🚀 **OpenAI互換テストサーバー**: FastAPIを使用した、`stream` 対応の `/v1/chat/completions` エンドポイントを持つテスト用APIサーバーを同梱しています。

---

## 📁 リポジトリ構成

```text
mlx2hf/
├── 010_choose_name_from_hf.py   # HFモデルの重みキー一覧を抽出
├── 020_choose_name_from_mlx.py  # MLXモデルの重みキー一覧を抽出
├── 030_check_name_by_string.py  # 2つのキーリストをマッチングし対応表を作成
├── 040_diff_config.py           # HFとMLXの設定ファイル(config)の差分を比較
├── 050_vlm_to_hf.py             # 重みの変換と書き出し (シャード分割対応)
├── 060_check1.py                # 変換後モデルのインポート・ロードチェック
├── 070_api_server.py            # OpenAI互換の簡易APIサーバー
├── data/                        # サンプルデータ (gemma-4-31b-it での実行結果を同梱)
└── requirements.txt             # 依存ライブラリ一覧
```

---

## 🛠️ インストール

```bash
# リポジトリのクローン
git clone https://github.com/black-taniguchi/mlx2hf.git
cd mlx2hf

# 仮想環境の作成と有効化 (推奨)
python3 -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt
```

---

## 📖 使い方

### Step 1: マッピングリスト（対応表）の作成

モデルのアーキテクチャによって重み変数名の命名ルールが異なるため、まずマッピング用の対応表（JSON）を作成します。

1. **HFモデルのキー一覧を抽出 (010)**
   ```bash
   python 010_choose_name_from_hf.py --model /path/to/hf_model_dir --output data/hf_keys.json
   ```
2. **MLXモデルのキー一覧を抽出 (020)**
   ```bash
   python 020_choose_name_from_mlx.py --model /path/to/mlx_model_dir_or_file --output data/mlx_keys.json
   ```
3. **マッチングと対応表の作成 (030)**
   両者のキー一覧を比較し、命名の置換ルールに基づいてマッピング表 `mapping_list.json` と、マッチングしなかった未対応キー一覧のファイルを書き出します。
   ```bash
   python 030_check_name_by_string.py \
       --hf-keys data/hf_keys.json \
       --mlx-keys data/mlx_keys.json \
       --output-mapped data/mapping_list.json \
       --output-unmapped data/unmapped_list.json
   ```

> 💡 **参考サンプルデータについて**
> `data` ディレクトリ内には、本ツールの検証時に `gemma-4-31b-it` を用いて実際に抽出・マッチングしたデータ（`data/gemma4-31b-it-*.txt`）が格納されています。実際にどのような対応表が作成されるかの参考にしてください。

---

### Step 2: 設定ファイル（Config）の差分確認

変換前に、HFとMLXの設定ファイルの差分を確認・記録します。
```bash
python 040_diff_config.py \
    --hf-path /path/to/hf_model_dir \
    --mlx-path /path/to/mlx_model_dir \
    --output data/config_diff.json
```
*※ `config.json` に加え、`processor_config.json` や `preprocessor_config.json` がある場合は自動で検出して比較結果に含めます。*

---

### Step 3: 重みの変換・書き出し

MLXの重みをHF互換の形式にリネームして保存します。

```bash
python 050_vlm_to_hf.py \
    --mlx-dir /path/to/mlx_model_dir \
    --hf-dir /path/to/output_hf_model_dir \
    --mapping data/mapping_list.json
```
*※ 出力先ディレクトリに元の `model.safetensors.index.json` をあらかじめコピーしておくと、自動的にシャード分割（複数ファイル）で保存されます。また、LM Studio等で必要となるメタデータ `{ "format": "pt" }` も自動で付与されます。*

---

### Step 4: ロード確認 & テストサーバーの起動

1. **ロードチェックの実行 (060)**
   Hugging Face の `transformers` ライブラリを使用して、変換後のモデルが正常にメモリへロードできるか検証します。
   ```bash
   python 060_check1.py --model /path/to/output_hf_model_dir
   ```

2. **OpenAI互換APIサーバーの起動 (070)**
   外部サービスやUIクライアント（LM Studio, Dify, LobeChatなど）から接続するための、ストリーミングに対応したOpenAI互換のテストサーバーを起動できます。
   ```bash
   python 070_api_server.py --model /path/to/output_hf_model_dir --port 8000
   ```

---

## ⚠️ 注意点

- **量子化について**: 量子化されたMLXモデルを変換した場合、重みの数値は保持されますが、形式的にHFフォーマットに戻しただけであるため、元のフル精度モデルと同等の性能が回復するわけではありません。推論エンジン側で対応した量子化形式として扱う必要があります。
- **設定ファイル**: 本ツールは主に重みの変換を行います。モデルの構造自体を定義する `config.json` などの設定ファイルは、あらかじめ元のHFモデルのものを出力先ディレクトリにコピーして用意してください。