from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import argparse
import os

def check_model_load(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        return

    try:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path)
        
        print("Loading model (this may take a while)...")
        # メモリ節約のため device_map="auto" や torch_dtype=torch.bfloat16 を推奨
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, 
            device_map="auto", 
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("\n✅ SUCCESS: Model and Processor loaded successfully!")
    except Exception as e:
        print(f"\n❌ ERROR: Load failed.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify if the converted HF model and processor load successfully using transformers."
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the converted HF model directory."
    )
    
    args = parser.parse_args()
    check_model_load(args.model)
