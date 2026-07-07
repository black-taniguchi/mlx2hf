import argparse
import os
import io
import base64
import time
import uuid
import requests
from PIL import Image
from typing import List, Dict, Any, Union, Optional
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
import torch
import json
from threading import Thread

app = FastAPI(title="Gemma 4 VLM OpenAI-Compatible Server")

model = None
processor = None
model_name = "gemma-4-vlm"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- Pydantic Models for OpenAI Compatibility ---
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "gemma-4"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.4
    stream: Optional[bool] = False

# --- Helper Functions ---
def parse_image(url_or_data: str) -> Image.Image:
    if url_or_data.startswith("data:image"):
        # Base64 Data URL (e.g., data:image/jpeg;base64,xxxx)
        try:
            header, encoded = url_or_data.split(",", 1)
            image_data = base64.b64decode(encoded)
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse base64 image: {e}")
    else:
        # Standard URL
        try:
            response = requests.get(url_or_data, timeout=10)
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {e}")

# --- API Endpoints ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model and processor are not loaded.")

    formatted_messages = []
    images = []

    try:
        # OpenAIのマルチモーダルメッセージ形式から Transformers のチャットテンプレート形式へ変換
        for msg in request.messages:
            formatted_content = []
            if isinstance(msg.content, str):
                formatted_content.append({"type": "text", "text": msg.content})
            else:
                for item in msg.content:
                    if item.get("type") == "text":
                        formatted_content.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        img = parse_image(url)
                        images.append(img)
                        # Transformers の標準テンプレートにおける画像指示プレースホルダ
                        formatted_content.append({"type": "image"})
            
            formatted_messages.append({
                "role": msg.role,
                "content": formatted_content
            })

        # チャットテンプレートの適用 (Gemma 4 VLM に対応)
        prompt = processor.apply_chat_template(
            formatted_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # プロセッサでトークナイズおよび画像の前処理
        if images:
            inputs = processor(text=prompt, images=images, return_tensors="pt")
        else:
            inputs = processor(text=prompt, return_tensors="pt")

        # 重みがロードされているデバイスへ入力を移動
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ストリーミングと非ストリーミングの分岐
        if request.stream:
            streamer = TextIteratorStreamer(
                processor.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True if request.temperature > 0 else False,
                temperature=request.temperature if request.temperature > 0 else None,
                streamer=streamer,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            async def sse_generator():
                response_id = f"chatcmpl-{uuid.uuid4()}"
                created_time = int(time.time())
                
                # 初回チャンク (role: assistant の通知)
                first_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model or model_name,
                    "system_fingerprint": request.model or model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "logprobs": None,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

                # 逐次テキストトークンの送信
                for text in streamer:
                    if text:
                        chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": request.model or model_name,
                            "system_fingerprint": request.model or model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text},
                                    "logprobs": None,
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                # 終了チャンク
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model or model_name,
                    "system_fingerprint": request.model or model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_generator(), media_type="text/event-stream")

        else:
            # 非ストリーミング生成
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    do_sample=True if request.temperature > 0 else False,
                    temperature=request.temperature if request.temperature > 0 else None
                )

            # 生成結果のデコード (プロンプト部分をスライスして新規生成テキストのみにする)
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = output[0][input_len:]
            generated_text = processor.decode(generated_tokens, skip_special_tokens=True)

            resp_data = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model or model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                            "reasoning_content": "",
                            "tool_calls": []
                        },
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": input_len,
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": input_len + len(generated_tokens),
                    "completion_tokens_details": {
                        "reasoning_tokens": 0
                    }
                },
                "stats": {},
                "system_fingerprint": request.model or model_name
            }

            # インデントされ、末尾に改行コードが含まれたJSONとしてレスポンスを返す
            json_str = json.dumps(resp_data, indent=2, ensure_ascii=False) + "\n"
            return Response(content=json_str, media_type="application/json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user"
            }
        ]
    }

@app.get("/health")
async def health():
    if model is not None and processor is not None:
        return {"status": "healthy", "device": device}
    return {"status": "loading"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start an OpenAI-compatible API server for Gemma 4 VLM.")
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the converted HF model directory."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number (default: 8000)"
    )
    args = parser.parse_args()

    model_name = os.path.basename(os.path.normpath(args.model))
    print(f"Loading model '{model_name}' from '{args.model}' onto device: {device}...")
    
    try:
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            device_map="auto",
            dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            trust_remote_code=True
        )
        print("Model and processor loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
