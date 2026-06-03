from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_PATH = os.getenv("QWEN35_MODEL_PATH", "/home/btechuser/models/Qwen3.5-2B")
SERVED_MODEL_NAME = os.getenv("QWEN35_SERVED_MODEL_NAME", "Qwen/Qwen3.5-2B")
DEVICE = os.getenv("QWEN35_STUDENT_DEVICE", "cpu").strip().lower()

_THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_THINKING_PROCESS_RE = re.compile(
    r"^\s*(?:Thinking Process|Reasoning|Analysis)\s*:\s*.*?(?:Final Answer\s*:|Answer\s*:)",
    flags=re.IGNORECASE | re.DOTALL,
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = SERVED_MODEL_NAME
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0)
    max_tokens: int = Field(default=300, ge=1, le=2048)


app = FastAPI(title="Qwen 3.5 2B Student Server")

processor: Any | None = None
model: Any | None = None


@app.on_event("startup")
def load_model() -> None:
    global processor, model
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if DEVICE == "cuda":
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model.to("cpu")
    model.eval()


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-transformers",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty.")

    messages = [{"role": item.role, "content": item.content} for item in request.messages]
    try:
        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = processor(text=[text], return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        do_sample = request.temperature > 0
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=do_sample,
                temperature=max(request.temperature, 1e-5) if do_sample else None,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        prompt_tokens = inputs["input_ids"].shape[-1]
        output_ids = generated[0][prompt_tokens:]
        content = _clean_visible_text(processor.tokenizer.decode(output_ids, skip_special_tokens=True))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": SERVED_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(output_ids),
            "total_tokens": prompt_tokens + len(output_ids),
        },
    }


def _clean_visible_text(content: str) -> str:
    clean = _THINK_TAG_RE.sub("", content).strip()
    clean = _THINKING_PROCESS_RE.sub("", clean).strip()
    if clean.lower().startswith(("thinking process:", "reasoning:", "analysis:")):
        lines = [line for line in clean.splitlines() if line.strip()]
        answer_lines = [
            line
            for line in lines
            if not line.lstrip().startswith(("*", "1.", "2.", "3.", "-"))
            and not line.strip().lower().startswith(("thinking process:", "reasoning:", "analysis:"))
        ]
        clean = "\n".join(answer_lines[-2:]).strip()
    return clean
