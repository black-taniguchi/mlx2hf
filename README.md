# MLX to HF Converter 🔄

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20MLX-orange.svg)](https://github.com/ml-explore/mlx)

A utility toolkit for efficiently and accurately converting model weights from Apple Silicon-optimized **MLX** format back to **Hugging Face (HF)** format.

👉 [日本語のREADMEはこちら](./README.ja.md)

---

## 🚀 Why do we need this?

**MLX** is extremely powerful for local inference and LoRA fine-tuning on Apple Silicon Mac computers. However, it is challenging to deploy models saved in the native MLX format to cloud environments or high-performance inference engines like **vLLM**, **TGI**, or **LM Studio**.

By using this toolkit, you can rename and export MLX weights back to the Hugging Face format (PyTorch-compatible `safetensors`), allowing you to deploy the model to high-performance inference servers.

---

## ✨ Features

- 🔄 **Weight Key Mapping**: Renames tensors (prefixes, suffixes, etc.) based on a mapping table to restore the HF-compatible format.
- 📦 **Sharded Saving Support**: Automatically detects `model.safetensors.index.json` in your output directory and saves weights using the exact sharded multi-file structure as the original model to optimize memory usage.
- 🔍 **Config File Diff Detection**: Automatically compares settings not only in `config.json` but also in image processing configurations like `processor_config.json` between MLX and HF.
- 🚀 **OpenAI-Compatible Test Server**: Includes a FastAPI-based server with `/v1/chat/completions` endpoint support, fully compatible with streaming (SSE) for easy local testing.

---

## 📁 Repository Structure

```text
mlx2hf/
├── 010_choose_name_from_hf.py   # Extracts weight keys from the HF model
├── 020_choose_name_from_mlx.py  # Extracts weight keys from the MLX model
├── 030_check_name_by_string.py  # Compares key lists and generates the mapping JSON
├── 040_diff_config.py           # Compares configurations between HF and MLX
├── 050_vlm_to_hf.py             # Converts and saves weight files (supports sharding)
├── 060_check1.py                # Verifies if the converted model imports and loads correctly
├── 070_api_server.py            # Simple OpenAI-compatible API server
├── data/                        # Contains sample run logs for gemma-4-31b-it
└── requirements.txt             # Dependency packages list
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/black-taniguchi/mlx2hf.git
cd mlx2hf

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Usage

### Step 1: Create a Weight Key Mapping List

Since key naming conventions differ across model architectures, you must first create a mapping JSON table.

1. **Extract Keys from the HF Model (010)**
   ```bash
   python 010_choose_name_from_hf.py --model /path/to/hf_model_dir --output data/hf_keys.json
   ```
2. **Extract Keys from the MLX Model (020)**
   ```bash
   python 020_choose_name_from_mlx.py --model /path/to/mlx_model_dir_or_file --output data/mlx_keys.json
   ```
3. **Compare and Generate Mapping (030)**
   Compares both key lists, applies string replacement rules, and generates a `mapping_list.json` along with an unmapped keys list.
   ```bash
   python 030_check_name_by_string.py \
       --hf-keys data/hf_keys.json \
       --mlx-keys data/mlx_keys.json \
       --output-mapped data/mapping_list.json \
       --output-unmapped data/unmapped_list.json
   ```

> 💡 **Sample Reference Data**
> The `data` directory contains actual output files (`data/gemma4-31b-it-*.txt`) generated when running these tools on `gemma-4-31b-it`. You can refer to them to see how the mapping lists look.

---

### Step 2: Compare Configuration Files (Optional)

Compare differences in the configuration settings between the HF and MLX models.
```bash
python 040_diff_config.py \
    --hf-path /path/to/hf_model_dir \
    --mlx-path /path/to/mlx_model_dir \
    --output data/config_diff.json
```
*Note: This script automatically detects and compares additional config files like `processor_config.json` or `preprocessor_config.json` if they are present.*

---

### Step 3: Convert Weight Files

Rename and export the MLX weight tensors into HF-compatible format.

```bash
python 050_vlm_to_hf.py \
    --mlx-dir /path/to/mlx_model_dir \
    --hf-dir /path/to/output_hf_model_dir \
    --mapping data/mapping_list.json
```
*Note: If you copy the original `model.safetensors.index.json` to the output directory before running, this script will automatically save the model in sharded (multi-file) format. It also automatically attaches the format metadata `{ "format": "pt" }` required by loaders like LM Studio.*

---

### Step 4: Verify Load & Start Local Test Server

1. **Verify Model Load (060)**
   Verify that the converted model can load successfully into CPU/GPU memory using the Hugging Face `transformers` library.
   ```bash
   python 060_check1.py --model /path/to/output_hf_model_dir
   ```

2. **Run OpenAI-Compatible Test Server (070)**
   Start a FastAPI local server featuring a streaming-compatible OpenAI API endpoint for testing integration with UI frontends like LM Studio, LobeChat, Dify, etc.
   ```bash
   python 070_api_server.py --model /path/to/output_hf_model_dir --port 8000
   ```

---

## ⚠️ Notes

- **Quantized Models**: While converting a quantized MLX model preserves the raw tensor values in HF format, this process only restores the file structure. It will not recover the model back to its original full-precision quality. You must treat it as a quantized model (e.g., AWQ/GPTQ) on the target server.
- **Config Files**: This tool primarily handles weight tensor conversion. You must manually copy the model structure configurations (like `config.json`, `processor_config.json`, tokenizer configs, etc.) from the original HF model directory to the output directory.