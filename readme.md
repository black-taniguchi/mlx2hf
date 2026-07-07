mlx-to-hf-converter

MLX形式（Apple Silicon最適化形式）のモデル重みを、Hugging Face (HF) 形式に変換して復元するためのシンプルなコンバーターです。


🚀 なぜこれが必要なのか？

MLXフレームワークはMac上での推論・微調整において非常に強力ですが、その形式で保存されたモデルをそのままクラウド環境や他の高性能推論エンジン（例: vLLM, TGI など）で利用することは困難です。


本ツールを使用することで、MLX形式の重みをHugging Face形式にリネームして書き出し、vLLMなどの爆速推論環境へデプロイすることが可能になります。


🛠️ 仕組み

MLXモデルとHFモデルの間で、多くの場合「重みの数値自体は変わらず、変数名（接頭辞など）だけが変更されている」という点に着目しています。
本ツールは、ユーザーが指定したマッピングファイル（JSON）に基づき、重みの名前を付け替えて保存するだけの非常に軽量な処理を行います。


📖 使い方

### 1. マッピングリスト（対応表）の作成

モデルのアーキテクチャによって重み変数名のルールが異なるため、まず以下の手順でマッピング用の JSON ファイルを作成します。

* **HFモデルのキー一覧を抽出 (010_choose_name_from_hf.py)**
  元となる Hugging Face モデルのディレクトリから、すべての重み名（キー）をリストアップします。
  ```bash
  python 010_choose_name_from_hf.py --model /path/to/hf_model_dir --output data/hf_keys.json
  ```

* **MLXモデルのキー一覧を抽出 (020_choose_name_from_mlx.py)**
  変換元となる MLX モデルのディレクトリから、すべての重み名をリストアップします。
  ```bash
  python 020_choose_name_from_mlx.py --model /path/to/mlx_model_dir_or_file --output data/mlx_keys.json
  ```

* **マッチングと対応表の作成 (030_check_name_by_string.py)**
  抽出した双方のキー一覧を比較し、命名の置換ルールに基づいてマッピング表 `{"mlx_key": "hf_key"}` と、マッチングしなかった未対応キー一覧のファイルを書き出します。
  ```bash
  python 030_check_name_by_string.py \
      --hf-keys data/hf_keys.json \
      --mlx-keys data/mlx_keys.json \
      --output-mapped data/mapping_list.json \
      --output-unmapped data/unmapped_list.json
  ```

> 💡 **参考サンプルデータについて**
> `data` ディレクトリ内には、本ツールの検証時に `gemma-4-31b-it` を用いて実際に抽出・マッチングしたデータ（`data/gemma4-31b-it-*.txt`）が格納されています。マッピングの動作確認や検証の際の参考にしてください。




2. 変換の実行

以下のコマンドを実行します。


python 050_vlm_to_hf.py \
    --mlx-dir /path/to/mlx_model \
    --hf-dir /path/to/output_hf_model \
    --mapping mapping_list.json

3. 後処理

変換後のディレクトリに、元のモデルの config.json や tokenizer.json などの設定ファイルをコピーしてください。これでHFモデルとしてロード可能になります。


✨ 特徴


シャード保存対応: model.safetensors.index.json が存在する場合、自動的に分割保存（Sharded format）を行い、巨大なモデルでもメモリ効率よく保存します。

低オーバーヘッド: safetensors ライブラリを使用し、CPU上で高速にテンソルを転送します。

汎用性: マッピングファイルさえあれば、VLMに限らず様々なアーキテクチャのモデルに適用可能です。


⚠️ 注意点


量子化について: 量子化されたMLXモデルを変換した場合、重みの数値は保持されますが、形式的にHFフォーマットに戻しただけであるため、元のフル精度モデルと同等の性能が回復するわけではありません。推論エンジン側で対応した量子化形式（AWQ/GPTQ等）として扱う必要があります。

コンフィグファイル: 本ツールは重みの変換のみを行います。モデル構造を定義する config.json は別途用意してください。