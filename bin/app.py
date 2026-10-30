#!/usr/bin/env python3
"""Steve's AI for macOS.

Features:
- Local Qwen/MLX-LM chat and final synthesis
- Optional Exa web retrieval with clean Qwen-written output
- Local file attachments
- YouTube transcript/audio summarization
- Live microphone prompts with adaptive voice activity detection
- Manual microphone mute and automatic mute before Read Aloud
- Per-response language selection and local translation
- Piper speech for English and all supported non-English languages
- Detailed STDOUT dialog and diagnostics
"""
from __future__ import annotations

import atexit
import html as html_lib
import ipaddress
import socket
from datetime import datetime, timedelta, timezone
import functools
import gc
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse, urljoin
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from flask import Flask, jsonify, render_template_string, request, send_from_directory
from werkzeug.utils import secure_filename

# -----------------------------------------------------------------------------
# Paths and configuration
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
TEMP = ROOT / "temp"
UPLOADS = TEMP / "uploads"
AUDIO = TEMP / "audio"
YOUTUBE = TEMP / "youtube"
MEDIA = TEMP / "media"
PIPER_VOICES_DIR = ROOT / "voices" / "piper"
for directory in (TEMP, UPLOADS, AUDIO, YOUTUBE, MEDIA, PIPER_VOICES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MODEL_OPTIONS = {
    "default": {
        "label": "Default — Qwen3 14B (32K context)",
        "model": "mlx-community/Qwen3-14B-4bit",
        "backend": "mlx_lm",
        "max_tokens": 3000,
    },
    "reasoning": {
        "label": "DeepSeek Logical Reasoning (128K context)",
        "model": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
        "backend": "mlx_lm",
        "max_tokens": 5000,
    },
    "summary": {
        "label": "Summarization — Qwen3.5 9B (262K context)",
        "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "backend": "mlx_vlm",
        "max_tokens": 2200,
    },
    "quick": {
        "label": "Quick Summarization — Qwen3.5 4B (262K context)",
        "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "backend": "mlx_vlm",
        "max_tokens": 4000,
    },
    "gemma": {
        "label": "Gemma 4 12B — Multimodal analysis & reasoning (256K context)",
        "model": "mlx-community/gemma-4-12B-it-4bit",
        "backend": "mlx_vlm",
        "max_tokens": 4000,
    },
    "qwen3_vl": {
        "label": "Qwen3-VL 8B — Screenshots, OCR, charts & document analysis",
        "model": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
        "backend": "mlx_vlm",
        "max_tokens": 4000,
    },
}

COMMUNICATION_STYLE = """
Communication style:
- Be direct and objective. Deliver precise, actionable responses without fluff, conversational filler, or unnecessary preamble.
- Use a professional, grounded, and engaging tone without being dramatic, promotional, or performative.
- Maintain an objective viewpoint. Challenge assumptions when warranted rather than defaulting to agreement.
- Match the depth of the response to the complexity of the request. Keep simple answers concise and use structure only when it adds value.
- Use evidence-based reasoning. Clearly distinguish facts, inferences, assumptions, and speculation.
- Focus on what the information means, its practical impact, and recommended actions.
- Answer the question asked without unnecessary background explanations or unrelated tutorials.
- Complete the requested task in a single response whenever possible.
- Prioritize conclusions, recommendations, and next steps over abstract discussion.
- Be concise by default and comprehensive when needed, but never verbose for its own sake.
""".strip()

DEFAULT_MODEL_KEY = os.getenv("DEFAULT_MODEL_KEY", "default")
if DEFAULT_MODEL_KEY not in MODEL_OPTIONS:
    DEFAULT_MODEL_KEY = "default"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8080"))
MAX_FILE_CHARS = int(os.getenv("MAX_FILE_CHARS", "120000"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "12"))
MAX_WEB_RESULTS = int(os.getenv("MAX_WEB_RESULTS", "8"))
MAX_UPLOAD_FILES = 15
MAX_MEDIA_BYTES = int(os.getenv("MAX_MEDIA_BYTES", str(2 * 1024 * 1024 * 1024)))
EXA_QUERY_HISTORY_TURNS = int(os.getenv("EXA_QUERY_HISTORY_TURNS", "6"))
EXA_QUERY_MAX_TOKENS = int(os.getenv("EXA_QUERY_MAX_TOKENS", "180"))
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "500000"))
MAX_ARTICLE_CHARS = int(os.getenv("MAX_ARTICLE_CHARS", "350000"))
MAX_ARTICLE_URLS = int(os.getenv("MAX_ARTICLE_URLS", "5"))
ARTICLE_FETCH_TIMEOUT = int(os.getenv("ARTICLE_FETCH_TIMEOUT", "30"))
LIVE_MIN_THRESHOLD = float(os.getenv("LIVE_MIN_THRESHOLD", "0.006"))
LIVE_SILENCE_SECONDS = float(os.getenv("LIVE_SILENCE_SECONDS", "1.75"))
LIVE_MAX_UTTERANCE_SECONDS = float(os.getenv("LIVE_MAX_UTTERANCE_SECONDS", "30"))

ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".py", ".log",
    ".html", ".xml", ".pdf", ".docx", ".xlsx",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(40 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "80000000"))
MEDIA_EXTENSIONS = {
    ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".aiff", ".aif",
    ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".wmv", ".mpeg", ".mpg",
}
LANGUAGES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
}
# Languages available to constrain Exa retrieval. This is deliberately separate
# from LANGUAGES, which controls translation and speech playback.
SEARCH_LANGUAGES = {
    "ar": "Arabic", "bg": "Bulgarian", "ca": "Catalan", "zh": "Chinese",
    "hr": "Croatian", "cs": "Czech", "da": "Danish", "nl": "Dutch",
    "en": "English", "et": "Estonian", "fi": "Finnish", "fr": "French",
    "de": "German", "el": "Greek", "he": "Hebrew", "hi": "Hindi",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "lv": "Latvian", "lt": "Lithuanian", "ms": "Malay",
    "no": "Norwegian", "fa": "Persian", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sr": "Serbian", "sk": "Slovak",
    "sl": "Slovenian", "es": "Spanish", "sv": "Swedish", "th": "Thai",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
}
DEFAULT_SEARCH_LANGUAGES = ["en"]
# Every supported Read Aloud language uses Piper, including English.
# This gives English the same stable synthesis path that already works for French.
PIPER_VOICE_NAMES = {
    "en": os.getenv("PIPER_VOICE_EN", "en_US-libritts_r-medium"),
    "fr": os.getenv("PIPER_VOICE_FR", "fr_FR-siwis-medium"),
    "de": os.getenv("PIPER_VOICE_DE", "de_DE-thorsten-medium"),
    "es": os.getenv("PIPER_VOICE_ES", "es_ES-sharvard-medium"),
    "it": os.getenv("PIPER_VOICE_IT", "it_IT-paola-medium"),
    "nl": os.getenv("PIPER_VOICE_NL", "nl_NL-mls-medium"),
    "pt": os.getenv("PIPER_VOICE_PT", "pt_PT-tugao-medium"),
}
PIPER_AUTO_DOWNLOAD = os.getenv("PIPER_AUTO_DOWNLOAD", "1").lower() not in {"0", "false", "no"}
PIPER_USE_CUDA = os.getenv("PIPER_USE_CUDA", "0").lower() in {"1", "true", "yes"}
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))
PIPER_SENTENCE_SILENCE = float(os.getenv("PIPER_SENTENCE_SILENCE", "0.25"))
ARGOS_AUTO_DOWNLOAD = os.getenv("ARGOS_AUTO_DOWNLOAD", "1").lower() not in {"0", "false", "no"}
_argos_lock = threading.Lock()
_piper_voices: dict[str, Any] = {}
_piper_lock = threading.Lock()
_nemo_normalizer = None
_nemo_lock = threading.Lock()
YOUTUBE_URL_RE = re.compile(
    r"^https?://(?:www\.|m\.)?"
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|live/|embed/)|youtu\.be/)"
    r"[A-Za-z0-9_-]{11}(?:[?&/#].*)?$",
    re.I,
)
ARTICLE_URL_EXTRACT_RE = re.compile(r"https?://[^\s<>\'\"]+", re.I)

YOUTUBE_URL_EXTRACT_RE = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?(?:[^\s#]*&)?v=|shorts/|live/|embed/)|youtu\.be/)"
    r"[A-Za-z0-9_-]{11}(?:[^\s<>'\"]*)?",
    re.I,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MEDIA_BYTES

class PollingAccessFilter(logging.Filter):
    """Suppress routine browser polling requests from the Werkzeug access log."""
    QUIET_PATHS = ("/api/live_poll", "/api/system_health")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(path in message for path in self.QUIET_PATHS)

logging.getLogger("werkzeug").addFilter(PollingAccessFilter())

# -----------------------------------------------------------------------------
# Optional StyleTTS2 setup
# -----------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
styletts2_tts = None
AudioSegment = None
nltk = None
try:
    import nltk as _nltk
    import torch
    from pydub import AudioSegment as _AudioSegment

    nltk = _nltk
    AudioSegment = _AudioSegment
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    _original_torch_load = torch.load

    @functools.wraps(_original_torch_load)
    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    from styletts2 import tts as styletts2_tts
except Exception as style_error:
    print(f"[STARTUP] StyleTTS2 unavailable: {style_error!r}", flush=True)

# -----------------------------------------------------------------------------
# Shared state
# -----------------------------------------------------------------------------
sessions: dict[str, dict[str, Any]] = {}
sessions_lock = threading.Lock()
model_lock = threading.Lock()
_model_cache: dict[str, dict[str, Any]] = {}
_active_model_key: str | None = None

live_lock = threading.Lock()
live_stream = None
live_thread = None
live_stop = threading.Event()
live_muted = threading.Event()
live_audio: queue.Queue = queue.Queue()
live_results: queue.Queue = queue.Queue()
live_session_id: str | None = None
live_use_web = False
live_max_web_results = MAX_WEB_RESULTS
live_model_key = DEFAULT_MODEL_KEY


def log(event: str, **fields: Any) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    detail = " ".join(f"{key}={value!r}" for key, value in fields.items())
    print(f"[{stamp}] {event}" + (f" | {detail}" if detail else ""), flush=True)


def get_session(session_id: str) -> dict[str, Any]:
    with sessions_lock:
        return sessions.setdefault(session_id, {"history": [], "files": [], "images": [], "search": None, "media": None})


def normalize_model_key(model_key: str | None) -> str:
    return model_key if model_key in MODEL_OPTIONS else DEFAULT_MODEL_KEY


def release_loaded_models(next_model_key: str | None = None) -> None:
    """Release cached MLX models and reclaim Metal memory before model changes."""
    global _active_model_key
    if not _model_cache:
        _active_model_key = None
        return

    released = list(_model_cache)
    log("Releasing MLX model memory", loaded_model_keys=released, next_model_key=next_model_key)
    _model_cache.clear()
    _active_model_key = None
    gc.collect()
    try:
        import mlx.core as mx
        mx.synchronize()
        mx.clear_cache()
    except Exception as exc:
        log("MLX memory cleanup warning", error=repr(exc))
    gc.collect()
    log("MLX model memory released", released_model_keys=released)


def local_model(model_key: str | None = None):
    global _active_model_key
    selected_key = normalize_model_key(model_key)
    selected = MODEL_OPTIONS[selected_key]
    with model_lock:
        # Keep only one large MLX model resident. When Run Prompt requests a
        # different model, unload the old weights before loading the new ones.
        if _active_model_key is not None and _active_model_key != selected_key:
            release_loaded_models(next_model_key=selected_key)

        if selected_key not in _model_cache:
            backend = selected["backend"]
            log("Loading MLX model", model_key=selected_key, model=selected["model"], backend=backend)
            if backend == "mlx_vlm":
                try:
                    from mlx_vlm import load as vlm_load
                except ImportError as exc:
                    raise RuntimeError(
                        f"{selected['label']} requires mlx-vlm. Install it with: "
                        "python -m pip install -U mlx-vlm"
                    ) from exc
                model, processor = vlm_load(selected["model"])
                try:
                    from mlx_vlm.utils import load_config as vlm_load_config
                    vlm_config = vlm_load_config(selected["model"])
                except Exception as exc:
                    vlm_config = getattr(model, "config", None)
                    log("MLX-VLM config fallback", model=selected["model"], error=repr(exc))
                _model_cache[selected_key] = {
                    "backend": backend, "model": model, "processor": processor, "config": vlm_config,
                }
            else:
                from mlx_lm import load as lm_load
                model, tokenizer = lm_load(selected["model"])
                _model_cache[selected_key] = {
                    "backend": backend, "model": model, "tokenizer": tokenizer,
                }
            _active_model_key = selected_key
            log("MLX model loaded", model_key=selected_key, model=selected["model"], backend=backend)
        else:
            _active_model_key = selected_key
    return _model_cache[selected_key]

def apply_model_chat_template(
    loaded: dict[str, Any],
    messages: list[dict[str, str]],
    enable_thinking: bool | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    def render(target: Any) -> str:
        try:
            return target.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Compatibility fallback for older MLX-VLM/tokenizer versions.
            return target.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    if loaded["backend"] == "mlx_vlm":
        processor = loaded["processor"]
        if hasattr(processor, "apply_chat_template"):
            return render(processor)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return render(tokenizer)
        return "\n".join(
            f"{item['role'].upper()}: {item['content']}" for item in messages
        ) + "\nASSISTANT:"
    return render(loaded["tokenizer"])


def generate_with_model(
    model_key: str,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    enable_thinking: bool | None = None,
    image_paths: list[str] | None = None,
) -> str:
    selected_key = normalize_model_key(model_key)
    selected = MODEL_OPTIONS[selected_key]
    loaded = local_model(selected_key)
    token_limit = max_tokens or selected["max_tokens"]
    valid_images = [str(Path(item)) for item in (image_paths or []) if Path(item).is_file()]

    if loaded["backend"] == "mlx_vlm":
        from mlx_vlm import generate as vlm_generate
        if valid_images:
            from mlx_vlm.prompt_utils import apply_chat_template as apply_vlm_chat_template
            conversation = "\n\n".join(
                f"{str(item.get('role') or 'user').upper()}: {str(item.get('content') or '')}"
                for item in messages
            )
            vlm_config = loaded.get("config")
            if vlm_config is None:
                raise RuntimeError("Could not load the MLX-VLM configuration required for image encoding. Upgrade mlx-vlm and restart.")
            prompt = apply_vlm_chat_template(
                loaded["processor"], vlm_config, conversation, num_images=len(valid_images)
            )
            log("MLX-VLM visual prompt prepared", model_key=selected_key,
                image_count=len(valid_images), prompt_characters=len(prompt))
        else:
            prompt = apply_model_chat_template(loaded, messages, enable_thinking=enable_thinking)
        image_input: Any = valid_images[0] if len(valid_images) == 1 else (valid_images or None)
        started = time.perf_counter()
        with model_lock:
            try:
                result = vlm_generate(loaded["model"], loaded["processor"],
                    prompt=prompt, image=image_input, max_tokens=token_limit, verbose=False)
            except TypeError as exc:
                if image_input is not None:
                    raise RuntimeError("The installed mlx-vlm version cannot pass images. Run: python -m pip install -U mlx-vlm") from exc
                result = vlm_generate(loaded["model"], loaded["processor"],
                    prompt=prompt, max_tokens=token_limit, verbose=False)
        output = result.text if hasattr(result, "text") else str(result)
        log("MLX generation complete", model_key=selected_key, backend="mlx_vlm",
            image_count=len(valid_images), elapsed_seconds=round(time.perf_counter()-started, 2))
        return output.strip()

    prompt = apply_model_chat_template(loaded, messages, enable_thinking=enable_thinking)
    from mlx_lm import generate as lm_generate
    started = time.perf_counter()
    with model_lock:
        output = lm_generate(loaded["model"], loaded["tokenizer"], prompt=prompt,
            max_tokens=token_limit, verbose=False).strip()
    log("MLX generation complete", model_key=selected_key, backend="mlx_lm",
        elapsed_seconds=round(time.perf_counter()-started, 2))
    return output


def strip_think_block(output: str) -> tuple[str, str]:
    """Separate hidden reasoning from the user-facing answer.

    Handles complete and truncated <think> blocks plus common textual
    reasoning/final-answer headings emitted by Qwen-family models.
    """
    text = str(output or "").strip()
    if not text:
        return "", ""

    # Some non-thinking Qwen templates emit a standalone closing marker before
    # the real answer. It is a control token, not reasoning. Remove only orphan
    # markers at the start, including their HTML-escaped representation.
    text = re.sub(
        r"^(?:\s|&nbsp;)*(?:(?:</think>|&lt;/think&gt;)(?:\s|&nbsp;)*)+",
        "",
        text,
        flags=re.I,
    ).strip()
    if not text:
        return "", ""

    reasoning_parts: list[str] = []

    # DeepSeek-R1 MLX generations can omit the opening <think> token while
    # retaining the closing token. In that format, everything before the first
    # closing marker is reasoning and everything after it is the visible answer.
    # If the marker is trailing, final remains empty and the existing final-only
    # retry below generates the user-facing response.
    has_opening_marker = bool(re.search(
        r"(?:<think\b[^>]*>|&lt;think\b.*?&gt;)",
        text,
        flags=re.I | re.S,
    ))
    orphan_closing = None if has_opening_marker else re.search(
        r"(?:</think>|&lt;/think&gt;)", text, flags=re.I
    )
    if orphan_closing:
        thought = text[:orphan_closing.start()].strip()
        if thought:
            reasoning_parts.append(thought)
        text = text[orphan_closing.end():].strip()

    # Normalise escaped control markers before collecting complete blocks. The
    # response itself is still escaped by the browser before display.
    text = re.sub(r"&lt;think\b(.*?)&gt;", r"<think\1>", text, flags=re.I | re.S)
    text = re.sub(r"&lt;/think&gt;", "</think>", text, flags=re.I)

    # Collect every complete <think> block, then remove it from the answer.
    complete_blocks = re.findall(r"<think\b[^>]*>(.*?)</think>", text, flags=re.I | re.S)
    reasoning_parts.extend(part.strip() for part in complete_blocks if part.strip())
    final = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.I | re.S).strip()

    # A generation may hit its token limit before </think>. Treat everything
    # after the opening tag as reasoning and trigger a final-answer retry.
    unclosed = re.search(r"<think\b[^>]*>(.*)$", final, flags=re.I | re.S)
    if unclosed:
        prefix = final[:unclosed.start()].strip()
        thought = unclosed.group(1).strip()
        if thought:
            reasoning_parts.append(thought)
        final = prefix

    # Support models that use headings instead of XML-style think tags.
    final_heading = re.search(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:final answer|answer|response)\s*:\s*",
        final,
        flags=re.I,
    )
    thinking_heading = re.search(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:thinking process|reasoning|analysis)\s*:\s*",
        final,
        flags=re.I,
    )
    if thinking_heading:
        if final_heading and final_heading.start() > thinking_heading.end():
            thought = final[thinking_heading.end():final_heading.start()].strip()
            if thought:
                reasoning_parts.append(thought)
            final = final[final_heading.end():].strip()
        elif thinking_heading.start() == 0:
            thought = final[thinking_heading.end():].strip()
            if thought:
                reasoning_parts.append(thought)
            final = ""
    elif final_heading and final_heading.start() == 0:
        final = final[final_heading.end():].strip()

    reasoning = "\n\n".join(dict.fromkeys(part for part in reasoning_parts if part)).strip()
    return reasoning, final.strip()


def final_answer_missing(final_output: str) -> bool:
    final = str(final_output or "").strip()
    if not final:
        return True
    # Do not mistake a leaked chain-of-thought heading for the final response.
    return bool(re.match(
        r"^(?:<[^>]+>\s*)*(?:#{1,6}\s*)?(?:thinking process|reasoning|analysis)\s*:",
        final,
        flags=re.I,
    ))


def generate_final_only(
    model_key: str,
    original_messages: list[dict[str, str]],
    reasoning: str = "",
    image_paths: list[str] | None = None,
) -> str:
    """Generate the visible answer using the selected model with thinking disabled."""
    final_messages = list(original_messages)
    final_messages.append({
        "role": "user",
        "content": (
            "/no_think\n"
            "Give only the final user-facing answer. Do not output reasoning, analysis, "
            "planning, self-review, a thinking process, or <think> tags. Start directly "
            "with the substantive answer and make it complete."
        ),
    })
    return generate_with_model(
        model_key,
        final_messages,
        max_tokens=MODEL_OPTIONS[model_key]["max_tokens"],
        enable_thinking=False,
        image_paths=image_paths,
    )


# -----------------------------------------------------------------------------
# Attachments
# -----------------------------------------------------------------------------
def validate_image(path: Path) -> dict[str, Any]:
    """Verify an uploaded image and return safe metadata for the UI/model."""
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {path.suffix.lower() or 'unknown'}")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"Images are limited to {MAX_IMAGE_BYTES // (1024 * 1024)} MB each.")
    from PIL import Image
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        frames = int(getattr(image, "n_frames", 1) or 1)
        image_format = str(image.format or path.suffix.lstrip(".")).upper()
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError("The image dimensions are invalid or exceed the configured pixel limit.")
    return {"width": width, "height": height, "frames": frames, "format": image_format}


def extract_text(path: Path) -> str:
    extension = path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension or 'unknown'}")

    if extension == ".pdf":
        from pypdf import PdfReader

        text = "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    elif extension == ".docx":
        from docx import Document

        document = Document(str(path))
        parts = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            parts.extend("\t".join(cell.text for cell in row.cells) for row in table.rows)
        text = "\n".join(parts)
    elif extension == ".xlsx":
        from openpyxl import load_workbook

        workbook = load_workbook(str(path), read_only=True, data_only=True)
        parts: list[str] = []
        for worksheet in workbook.worksheets:
            parts.append(f"[Sheet: {worksheet.title}]")
            for row in worksheet.iter_rows(values_only=True):
                parts.append("\t".join("" if value is None else str(value) for value in row))
        text = "\n".join(parts)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    return text[:MAX_FILE_CHARS]


# -----------------------------------------------------------------------------
# Local audio and video transcription
# -----------------------------------------------------------------------------
def probe_media(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1200:] or "ffprobe could not inspect the media file.")
    try:
        duration = max(1, int(round(float(result.stdout.strip()))))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("The media file does not report a usable duration.") from exc
    return {"duration_seconds": duration}

def transcribe_local_media(
    media_path: Path, job_dir: Path, start_seconds: float = 0, end_seconds: float | None = None
) -> str:
    import mlx_whisper
    clipped_path = job_dir / "selected_segment.wav"
    command = ["ffmpeg", "-y", "-ss", str(max(0, start_seconds)), "-i", str(media_path)]
    if end_seconds is not None:
        command.extend(["-t", str(max(0.1, end_seconds - start_seconds))])
    command.extend(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(clipped_path)])
    log("Local media conversion started", file=media_path.name, start_seconds=start_seconds, end_seconds=end_seconds)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not clipped_path.exists():
        raise RuntimeError(result.stderr[-1600:] or "ffmpeg could not create the selected media segment.")
    transcription = mlx_whisper.transcribe(str(clipped_path), path_or_hf_repo=WHISPER_MODEL, task="transcribe")
    return str(transcription.get("text", "")).strip()

def analyze_local_media(
    prompt: str, media: dict[str, Any], segment: dict[str, Any], model_key: str = DEFAULT_MODEL_KEY
) -> tuple[str, str, list[dict[str, str]]]:
    media_path = Path(str(media.get("path") or ""))
    if not media_path.exists() or media_path.parent != MEDIA / secure_filename(str(media.get("session_id") or "")):
        raise ValueError("The uploaded media file is no longer available. Upload it again.")
    duration = max(1, int(media.get("duration_seconds") or 1))
    start_seconds = max(0, min(duration - 1, int(segment.get("start_seconds") or 0)))
    end_seconds = max(start_seconds + 1, min(duration, int(segment.get("end_seconds") or duration)))
    job_dir = media_path.parent / f"transcription-{uuid.uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        transcript = transcribe_local_media(media_path, job_dir, start_seconds, end_seconds)
        if not transcript:
            raise RuntimeError("No usable speech transcript could be generated from the selected segment.")
        transcript_path = media_path.parent / f"{media_path.stem}_transcript_{start_seconds}-{end_seconds}.txt"
        transcript_path.write_text(transcript, encoding="utf-8")
        system = (
            "You are an expert executive analyst. Answer the user's request using the supplied transcript from an uploaded "
            "audio or video file. Do not invent facts or claim to have seen visuals that are not represented in the transcript. "
            "If the user's prompt contains no specific analytical instruction, use exactly these sections: Executive Summary, "
            "Key Points, Conclusions.\n\n" + COMMUNICATION_STYLE
        )
        user = (
            f"ORIGINAL USER PROMPT:\n{prompt}\n\n"
            f"UPLOADED MEDIA FILE: {media.get('name') or media_path.name}\n"
            f"SELECTED SEGMENT: {start_seconds}s to {end_seconds}s\n"
            f"TRANSCRIPTION METHOD: local MLX Whisper\n\n"
            f"TRANSCRIPT:\n{transcript[:MAX_TRANSCRIPT_CHARS]}"
        )
        selected_key = normalize_model_key(model_key)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        output = generate_with_model(
            selected_key, messages, max_tokens=MODEL_OPTIONS[selected_key]["max_tokens"],
            enable_thinking=False if selected_key in {"default", "gemma", "qwen3_vl"} else None,
        )
        reasoning, final_output = strip_think_block(output)
        if selected_key in {"summary", "quick"} or final_answer_missing(final_output):
            if not reasoning and final_output and selected_key in {"summary", "quick"}:
                reasoning = final_output
            retry = generate_final_only(selected_key, messages, reasoning)
            leaked, final_output = strip_think_block(retry)
            if leaked:
                reasoning = "\n\n".join(part for part in (reasoning, leaked) if part)
        if final_answer_missing(final_output):
            raise RuntimeError("The selected model did not produce an analysis of the uploaded media.")
        source = {"title": str(media.get("name") or media_path.name), "url": ""}
        return clean_inline_citations(final_output), reasoning, [source]
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

# -----------------------------------------------------------------------------
# Exa retrieval and Qwen final synthesis
# -----------------------------------------------------------------------------
def value(obj: Any, name: str, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def normalize_max_web_results(value: Any) -> int:
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = MAX_WEB_RESULTS
    requested = max(0, min(80, requested))
    return (requested // 8) * 8

def parse_domains(raw: Any) -> list[str]:
    if isinstance(raw, list):
        items = raw
    else:
        items = re.split(r"[\s,;]+", str(raw or ""))
    cleaned: list[str] = []
    for item in items:
        domain = str(item).strip()
        if not domain:
            continue
        domain = re.sub(r"^https?://", "", domain, flags=re.I).rstrip("/")
        if domain and domain not in cleaned:
            cleaned.append(domain)
    return cleaned[:100]


def normalise_iso_date(value_: Any, end_of_day: bool = False) -> str | None:
    raw = str(value_ or "").strip()
    if not raw:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        raise ValueError(f"Invalid date: {raw}")
    return raw + ("T23:59:59.999Z" if end_of_day else "T00:00:00.000Z")


def normalize_search_languages(raw: Any) -> list[str]:
    """Validate and deduplicate requested search languages; default to English."""
    if isinstance(raw, list):
        candidates = raw
    elif raw is None:
        candidates = []
    else:
        candidates = re.split(r"[\s,;]+", str(raw))
    selected: list[str] = []
    for item in candidates:
        code = str(item or "").strip().lower()
        if code in SEARCH_LANGUAGES and code not in selected:
            selected.append(code)
    return selected or list(DEFAULT_SEARCH_LANGUAGES)


def language_search_query(query: str, language_code: str) -> str:
    """Express the language constraint in Exa's natural-language query."""
    language_name = SEARCH_LANGUAGES[language_code]
    return (
        f"{query}\n\n"
        f"Return sources whose main page content is written in {language_name}. "
        f"Exclude translated snippets and pages primarily written in other languages."
    )


def web_search(
    query: str,
    max_results: int = MAX_WEB_RESULTS,
    objective: str = "",
    include_domains: Any = None,
    exclude_domains: Any = None,
    freshness_mode: str = "cached",
    max_age_hours: Any = None,
    start_published_date: Any = None,
    end_published_date: Any = None,
    highlights: bool = True,
    search_type: str = "auto",
    user_location: str = "",
    moderation: bool = False,
    categories: Any = None,
    languages: Any = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    api_key = os.getenv("EXA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set.")
    max_results = normalize_max_web_results(max_results)
    if max_results == 0:
        return "", [], {}
    query = str(query or "").strip()
    if not query:
        raise ValueError("Enter an Exa search query.")

    allowed_types = {"instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning"}
    search_type = search_type if search_type in allowed_types else "auto"
    user_location = str(user_location or "").strip().upper()
    if user_location and not re.fullmatch(r"[A-Z]{2}", user_location):
        raise ValueError("User location must be a two-letter ISO country code.")
    allowed_categories = {"company", "publication", "news", "personal site", "financial report", "people"}
    if isinstance(categories, list):
        selected_categories = [str(item) for item in categories if str(item) in allowed_categories]
    else:
        selected_categories = []
    selected_languages = normalize_search_languages(languages)

    include = parse_domains(include_domains)
    exclude = parse_domains(exclude_domains)
    freshness_mode = freshness_mode if freshness_mode in {"cached", "max_age", "trawl", "date_range"} else "cached"
    contents: dict[str, Any] = {"highlights": bool(highlights)}
    publication_cutoff: datetime | None = None
    if freshness_mode == "cached":
        contents["maxAgeHours"] = -1
    elif freshness_mode == "trawl":
        contents["maxAgeHours"] = 0
    elif freshness_mode == "max_age":
        try:
            age = int(str(max_age_hours).strip())
        except (TypeError, ValueError):
            raise ValueError("Maximum publication age must be a whole number of hours.")
        if age < 1:
            raise ValueError("Maximum publication age must be at least 1 hour.")
        # maxAgeHours controls the age of Exa's cached copy, not the age of the
        # article. Use startPublishedDate to constrain article publication time.
        publication_cutoff = datetime.now(timezone.utc) - timedelta(hours=age)
        contents["maxAgeHours"] = age

    base_payload: dict[str, Any] = {
        "query": query,
        "type": search_type,
        "numResults": max_results,
        "moderation": bool(moderation),
        "contents": contents,
    }
    if objective.strip():
        base_payload["objective"] = objective.strip()
    if include:
        base_payload["includeDomains"] = include
    if exclude:
        base_payload["excludeDomains"] = exclude
    if user_location:
        base_payload["userLocation"] = user_location
    if publication_cutoff is not None:
        base_payload["startPublishedDate"] = publication_cutoff.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if freshness_mode == "date_range":
        start_date = normalise_iso_date(start_published_date)
        end_date = normalise_iso_date(end_published_date, end_of_day=True)
        if not start_date and not end_date:
            raise ValueError("Choose at least one publication date.")
        if start_date:
            base_payload["startPublishedDate"] = start_date
        if end_date:
            base_payload["endPublishedDate"] = end_date

    # Exa's category parameter is singular. A multi-selection is implemented as
    # one search per category, followed by URL deduplication and result capping.
    if publication_cutoff is not None and any(category in {"company", "people"} for category in selected_categories):
        raise ValueError("Company and People categories cannot be combined with a publication-age limit because Exa does not support publication-date filters for those categories.")
    category_runs: list[str | None] = selected_categories or [None]
    search_runs = [
        (language_code, category)
        for language_code in selected_languages
        for category in category_runs
    ]
    all_results: list[dict[str, Any]] = []
    request_ids: list[str] = []
    total_cost = 0.0
    search_times: list[Any] = []
    warnings: list[Any] = []
    notices: list[Any] = []
    per_run_limit = max(1, min(100, max_results))
    for language_code, category in search_runs:
        payload = dict(base_payload)
        payload["query"] = language_search_query(query, language_code)
        payload["contents"] = dict(contents)
        payload["numResults"] = per_run_limit
        if category:
            payload["category"] = category
            if category in {"company", "people"}:
                payload.pop("startPublishedDate", None)
                payload.pop("endPublishedDate", None)
            if category == "people":
                payload.pop("excludeDomains", None)
        log("EXA retrieval started", query=query, language=SEARCH_LANGUAGES[language_code], type=search_type, category=category, freshness_mode=freshness_mode, max_age_hours=contents.get("maxAgeHours"))
        req = Request(
            "https://api.exa.ai/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=180) as response_handle:
                response_data = json.loads(response_handle.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Exa search failed ({exc.code}): {detail[:1200]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach Exa: {exc.reason}") from exc
        request_id = str(response_data.get("requestId") or "")
        if request_id:
            request_ids.append(request_id)
        search_times.append(response_data.get("searchTime"))
        cost = response_data.get("costDollars")
        if isinstance(cost, dict):
            total_cost += float(cost.get("total") or 0)
        elif isinstance(cost, (int, float)):
            total_cost += float(cost)
        warnings.extend(response_data.get("warnings") or [])
        notices.extend(response_data.get("notices") or [])
        for result in response_data.get("results") or []:
            item = dict(result)
            item["matchedCategory"] = category or ""
            item["requestedLanguage"] = language_code
            all_results.append(item)

    # Interleave language-specific result sets so the final cap cannot be filled
    # entirely by the first selected language before later languages are seen.
    if len(selected_languages) > 1:
        language_buckets = {
            code: [item for item in all_results if item.get("requestedLanguage") == code]
            for code in selected_languages
        }
        interleaved_results: list[dict[str, Any]] = []
        offset = 0
        while any(offset < len(bucket) for bucket in language_buckets.values()):
            for code in selected_languages:
                bucket = language_buckets[code]
                if offset < len(bucket):
                    interleaved_results.append(bucket[offset])
            offset += 1
        all_results = interleaved_results

    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    filtered_out_by_age = 0
    filtered_out_undated = 0
    for result in all_results:
        if publication_cutoff is not None:
            raw_published = str(result.get("publishedDate") or "").strip()
            if not raw_published:
                filtered_out_undated += 1
                continue
            try:
                published_at = datetime.fromisoformat(raw_published.replace("Z", "+00:00"))
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                if published_at.astimezone(timezone.utc) < publication_cutoff:
                    filtered_out_by_age += 1
                    continue
            except ValueError:
                filtered_out_undated += 1
                continue
        url = str(result.get("url") or "")
        key = url.rstrip("/").lower()
        if key and key in seen_urls:
            continue
        if key:
            seen_urls.add(key)
        deduped.append(result)
        if len(deduped) >= max_results:
            break

    evidence_blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    for index, result in enumerate(deduped, 1):
        title = str(result.get("title") or "Untitled")
        url = str(result.get("url") or "")
        published_date = str(result.get("publishedDate") or "")
        highlights_list = result.get("highlights") or []
        if isinstance(highlights_list, str):
            highlights_list = [highlights_list]
        highlights_list = [str(item) for item in highlights_list if str(item).strip()]
        excerpt = "\n".join(highlights_list)[:9000]
        matched_category = str(result.get("matchedCategory") or "")
        requested_language = str(result.get("requestedLanguage") or "en")
        author = str(result.get("author") or "").strip()
        domain = (urlparse(url).hostname or "").lower()
        if domain.startswith("www."):
            domain = domain[4:]
        evidence_blocks.append(
            f"WEB EVIDENCE ITEM {index}\nTitle: {title}\nURL: {url}\nPublished: {published_date or 'Unknown'}\nCategory: {matched_category or 'General'}\nRequested language: {SEARCH_LANGUAGES.get(requested_language, requested_language)}\nRelevant passages:\n{excerpt or '[Highlights not requested or unavailable]'}"
        )
        sources.append({"title": title, "url": url, "domain": domain, "author": author, "published_date": published_date, "highlights": highlights_list, "category": matched_category, "language": requested_language, "language_name": SEARCH_LANGUAGES.get(requested_language, requested_language)})
    metadata = {
        "query": query,
        "objective": objective.strip(),
        "type": search_type,
        "user_location": user_location,
        "moderation": bool(moderation),
        "categories": selected_categories,
        "languages": selected_languages,
        "language_names": [SEARCH_LANGUAGES[code] for code in selected_languages],
        "freshness_mode": freshness_mode,
        "max_age_hours": contents.get("maxAgeHours"),
        "publication_cutoff": publication_cutoff.isoformat().replace("+00:00", "Z") if publication_cutoff else "",
        "filtered_out_by_age": filtered_out_by_age,
        "filtered_out_undated": filtered_out_undated,
        "include_domains": include,
        "exclude_domains": exclude,
        "start_published_date": str(start_published_date or ""),
        "end_published_date": str(end_published_date or ""),
        "highlights": bool(highlights),
        "request_ids": request_ids,
        "search_times": search_times,
        "cost_dollars": total_cost,
        "warnings": warnings,
        "notices": notices,
    }
    log("EXA retrieval complete", result_count=len(sources), request_ids=request_ids, cost_dollars=total_cost)
    return "\n\n".join(evidence_blocks), sources, metadata

def clean_inline_citations(output: str) -> str:
    output = re.sub(
        r"(?<!\w)\[(?:\d+(?:\s*[,;–-]\s*\d+)*|Source\s*\d+|WEB\s+SOURCE\s*\d+)\](?!\w)",
        "",
        output,
        flags=re.I,
    )
    output = re.sub(r"[ \t]+([,.;:!?])", r"\1", output)
    output = re.sub(r" {2,}", " ", output)
    return output.strip()


def compact_history_for_query(history: list[dict[str, str]]) -> str:
    recent = history[-EXA_QUERY_HISTORY_TURNS * 2 :]
    lines: list[str] = []
    for item in recent:
        role = str(item.get("role", "unknown")).upper()
        content = re.sub(r"\s+", " ", str(item.get("content", ""))).strip()
        if not content:
            continue
        # Keep the query-planning context focused while retaining the subject,
        # positions, names, countries, organisations, dates and claims.
        if len(content) > 5000:
            content = content[:5000] + " ..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def clean_exa_query(raw_query: str, fallback: str) -> str:
    query = raw_query.strip()
    query = re.sub(r"^```(?:text|json)?\s*", "", query, flags=re.I)
    query = re.sub(r"\s*```$", "", query)
    query = re.sub(r"^(?:search query|query)\s*:\s*", "", query, flags=re.I)
    query = query.strip().strip('"').strip("'")
    query = re.sub(r"\s+", " ", query).strip()
    if not query or len(query) < 4:
        return fallback
    return query[:700]


def devise_exa_query(prompt: str, current_session: dict[str, Any], model_key: str = DEFAULT_MODEL_KEY) -> str:
    history_text = compact_history_for_query(current_session.get("history", []))
    if not history_text:
        log("EXA query planner", reason="no_history", original_prompt=prompt, planned_query=prompt)
        return prompt

    system = (
        "You are a search-query planner for Exa. Read the current user question together with the recent conversation. "
        "Resolve pronouns and vague references such as 'its position', 'that proposal', 'they', 'the video', or 'this issue' using the history. "
        "Produce one self-contained web-search query that includes the relevant named entities, country, policy, event, claim, timeframe and analytical angle. "
        "The query should retrieve external evidence needed to answer the current question, not merely restate the question. "
        "Do not answer the question. Do not add commentary, labels, quotation marks, citations, JSON or multiple alternatives. Output only the query."
    )
    user = (
        "RECENT CONVERSATION:\n"
        + history_text
        + "\n\nCURRENT USER QUESTION:\n"
        + prompt
        + "\n\nWrite the single best contextual Exa search query."
    )

    planner_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    started = time.perf_counter()
    planned = generate_with_model(
        model_key,
        planner_messages,
        max_tokens=EXA_QUERY_MAX_TOKENS,
    )
    _, planned_final = strip_think_block(planned)
    query = clean_exa_query(planned_final, prompt)
    log(
        "EXA query planned by Qwen",
        original_prompt=prompt,
        planned_query=query,
        history_messages=len(current_session.get("history", [])),
        elapsed_seconds=round(time.perf_counter() - started, 2),
    )
    return query


def selected_search_evidence(saved_search: dict[str, Any] | None) -> tuple[str, list[dict[str, Any]]]:
    if not saved_search:
        return "", []
    all_sources = list(saved_search.get("sources") or [])
    selected_indices = saved_search.get("selected_indices")
    if selected_indices is None:
        selected_indices = list(range(len(all_sources)))
    selected_set = {index for index in selected_indices if isinstance(index, int) and 0 <= index < len(all_sources)}
    selected_sources = [source for index, source in enumerate(all_sources) if index in selected_set]
    evidence_blocks: list[str] = []
    for evidence_index, source in enumerate(selected_sources, 1):
        highlights = source.get("highlights") or []
        if isinstance(highlights, str):
            highlights = [highlights]
        excerpt = "\n".join(str(item) for item in highlights if str(item).strip())[:9000]
        evidence_blocks.append(
            f"WEB EVIDENCE ITEM {evidence_index}\n"
            f"Search batch: {source.get('search_batch') or 'Unknown'}\n"
            f"Search query: {source.get('search_query') or 'Unknown'}\n"
            f"Title: {source.get('title') or 'Untitled'}\n"
            f"URL: {source.get('url') or ''}\n"
            f"Website: {source.get('domain') or 'Unknown'}\n"
            f"Author or publisher: {source.get('author') or 'Unknown'}\n"
            f"Published: {source.get('published_date') or 'Unknown'}\n"
            f"Category: {source.get('category') or 'General'}\n"
            f"Relevant passages:\n{excerpt or '[Highlights not requested or unavailable]'}"
        )
    return "\n\n".join(evidence_blocks), selected_sources


def answer(prompt: str, current_session: dict[str, Any], use_web: bool, model_key: str = DEFAULT_MODEL_KEY, max_web_results: int = MAX_WEB_RESULTS) -> tuple[str, list[dict[str, str]], str]:
    model_key = normalize_model_key(model_key)
    max_web_results = normalize_max_web_results(max_web_results)
    use_web = bool(use_web and max_web_results > 0)
    saved_search = current_session.get("search") if use_web else None
    search_queries = list((saved_search or {}).get("queries") or [])
    exa_query = "\n".join(
        f"Search {index}: {item.get('query') or ''}"
        for index, item in enumerate(search_queries, 1)
    ).strip() or str((saved_search or {}).get("query") or "")
    web_context, sources = selected_search_evidence(saved_search)
    file_context = "\n\n".join(
        f"LOCAL FILE: {item['name']}\n{item['text']}" for item in current_session["files"]
    )
    attached_images = [item for item in current_session.get("images", []) if Path(str(item.get("path") or "")).is_file()]
    image_paths = [str(item["path"]) for item in attached_images]
    if image_paths:
        log("Images selected for analysis", model_key=model_key, image_names=[item["name"] for item in attached_images])
    if image_paths and MODEL_OPTIONS[model_key]["backend"] != "mlx_vlm":
        raise ValueError(
            f"{MODEL_OPTIONS[model_key]['label']} is text-only and cannot analyse images. "
            "Select Qwen3-VL 8B, Gemma 4 12B, or another multimodal model."
        )
    image_context = ""
    if attached_images:
        image_context = "ATTACHED IMAGES AVAILABLE TO THE MODEL:\n" + "\n".join(
            f"- {item['name']} ({item['width']}x{item['height']}, {item['format']})"
            for item in attached_images
        )
    evidence = "\n\n".join(part for part in (file_context, image_context, web_context) if part)

    system = (
        "You are a rigorous local research assistant. You, not the retrieval service, write the final response. "
        "When web evidence is supplied, compare and synthesize it into a clean, natural answer. "
        "Do not reproduce retrieval output verbatim. Distinguish established evidence from inference and mention genuine uncertainty. "
        "Do not include bracketed reference markers, numbered citations, source IDs, or forms such as [1], [2], [Source 1], or [WEB SOURCE]. "
        "Do not append a references or sources section because the interface displays source links separately. "
        "When images are attached, base the answer on the pixels in those images. First identify the visible document or scene, then extract only legible facts relevant to the request. "
        "Do not substitute a remembered or unrelated image. If the image is unavailable, unreadable, or inconsistent with the question, say so explicitly rather than guessing. "
        "State uncertainty where image detail is unreadable or ambiguous. "
        "Use ordinary prose, headings, and bullets where useful.\n\n"
        + COMMUNICATION_STYLE
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(current_session["history"][-MAX_HISTORY_TURNS * 2 :])
    current_prompt = prompt
    if evidence:
        current_prompt += (
            "\n\n--- CONTEXTUAL WEB QUERY USED ---\n"
            + exa_query
            + "\n--- RETRIEVED EVIDENCE FOR QWEN TO ANALYZE ---\n"
            + evidence
            + "\n--- END RETRIEVED EVIDENCE ---\n"
            + "Write the final answer yourself. Do not emit inline citation markers or square-bracket references."
        )
    messages.append({"role": "user", "content": current_prompt})

    started = time.perf_counter()
    output = generate_with_model(
        model_key,
        messages,
        max_tokens=MODEL_OPTIONS[model_key]["max_tokens"],
        enable_thinking=False if model_key in {"default", "gemma", "qwen3_vl"} else None,
        image_paths=image_paths,
    )
    reasoning, first_pass_final = strip_think_block(output)
    if model_key in {"summary", "quick"}:
        # Qwen3.5 may use the whole first pass for reasoning. Keep that reasoning
        # collapsible, but always create the visible answer in a second pass with
        # native thinking disabled.
        if not reasoning and first_pass_final:
            reasoning = first_pass_final
        log(
            "Qwen3.5 non-thinking final-answer pass",
            model_key=model_key,
            reasoning_characters=len(reasoning),
        )
        final_raw = generate_final_only(model_key, messages, reasoning, image_paths=image_paths)
        leaked_reasoning, final_output = strip_think_block(final_raw)
        if leaked_reasoning:
            reasoning = "\n\n".join(part for part in (reasoning, leaked_reasoning) if part)
    else:
        final_output = first_pass_final
        if final_answer_missing(final_output):
            retry_output = generate_final_only(model_key, messages, reasoning, image_paths=image_paths)
            retry_reasoning, final_output = strip_think_block(retry_output)
            if retry_reasoning:
                reasoning = "\n\n".join(part for part in (reasoning, retry_reasoning) if part)
    if final_answer_missing(final_output):
        raise RuntimeError(
            "The selected model completed its reasoning but did not produce a final answer. "
            "Please retry or select another model."
        )
    output = clean_inline_citations(final_output)

    current_session["history"].extend(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}]
    )
    current_session["history"] = current_session["history"][-MAX_HISTORY_TURNS * 2 :]
    log(
        "QWEN final synthesis complete",
        used_exa=use_web,
        source_count=len(sources),
        elapsed_seconds=round(time.perf_counter() - started, 2),
        model_key=model_key,
        model=MODEL_OPTIONS[model_key]["model"],
        reasoning_characters=len(reasoning),
        output_characters=len(output),
    )
    return output, sources, reasoning


def clean_search_suggestions(raw_output: str) -> list[str]:
    """Parse exactly three useful search queries from a local-model response."""
    _, visible = strip_think_block(raw_output)
    text = visible or raw_output
    text = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    candidates: list[str] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = parsed.get("suggestions") or parsed.get("queries") or []
        if isinstance(parsed, list):
            candidates = [str(item) for item in parsed]
    except (json.JSONDecodeError, TypeError):
        candidates = re.split(r"\n+", text)
    cleaned: list[str] = []
    for candidate in candidates:
        value_ = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", str(candidate)).strip()
        value_ = value_.strip('"').strip("'")
        value_ = re.sub(r"\s+", " ", value_).strip()
        if 4 <= len(value_) <= 700 and value_.lower() not in {item.lower() for item in cleaned}:
            cleaned.append(value_)
        if len(cleaned) == 3:
            break
    if len(cleaned) != 3:
        raise RuntimeError("The model did not return three usable search suggestions. Please try again.")
    return cleaned


def generate_search_suggestions(
    current_session: dict[str, Any], model_key: str = DEFAULT_MODEL_KEY
) -> list[str]:
    """Suggest three complementary follow-up searches from the latest search and its results."""
    saved_search = current_session.get("search") or {}
    queries = list(saved_search.get("queries") or [])
    sources = list(saved_search.get("sources") or [])
    if not queries or not sources:
        raise ValueError("Run a search with results before generating follow-up suggestions.")
    latest_query = queries[-1]
    result_blocks: list[str] = []
    for index, source in enumerate(sources, 1):
        highlights = source.get("highlights") or []
        if isinstance(highlights, str):
            highlights = [highlights]
        excerpt = " ".join(str(item) for item in highlights if str(item).strip())[:1800]
        result_blocks.append(
            f"RESULT {index}\nTitle: {source.get('title') or 'Untitled'}\n"
            f"URL: {source.get('url') or ''}\nPublished: {source.get('published_date') or 'Unknown'}\n"
            f"Relevant passages: {excerpt or '[No highlights available]'}"
        )
    system = (
        "You improve research coverage. Produce exactly three complementary Exa search queries that are likely "
        "to find material evidence missing, weak, contradictory, or insufficiently authoritative in the current results. "
        "Use the original query, research objective, and retrieved results. Avoid merely rephrasing the original query, "
        "avoid overlap between suggestions, and preserve relevant named entities, geography and timeframe. "
        "Return only a JSON array of three strings, with no explanation or markdown."
    )
    user = (
        f"ORIGINAL SEARCH QUERY:\n{latest_query.get('query') or saved_search.get('query') or ''}\n\n"
        f"SEARCH OBJECTIVE:\n{latest_query.get('objective') or ''}\n\n"
        "CURRENT SEARCH RESULTS:\n" + "\n\n".join(result_blocks)[:42000]
    )
    selected_key = normalize_model_key(model_key)
    output = generate_with_model(
        selected_key,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=500,
        enable_thinking=False,
    )
    suggestions = clean_search_suggestions(output)
    log("Search suggestions generated", model_key=selected_key, suggestions=suggestions)
    return suggestions


# -----------------------------------------------------------------------------
# Direct article retrieval and summarization
# -----------------------------------------------------------------------------
def extract_article_urls(text: str) -> list[str]:
    """Return unique non-YouTube HTTP(S) URLs in prompt order."""
    urls: list[str] = []
    seen: set[str] = set()
    for match in ARTICLE_URL_EXTRACT_RE.finditer(str(text or "")):
        url = match.group(0).rstrip(".,);]}>'\\\"")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or youtube_video_id(url):
            continue
        key = url.rstrip("/").lower()
        if key not in seen:
            seen.add(key)
            urls.append(url)
        if len(urls) >= MAX_ARTICLE_URLS:
            break
    return urls


def validate_public_article_url(url: str) -> str:
    """Reject unsupported schemes, credentials, localhost and private-network targets."""
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Article URLs must start with http:// or https://.")
    if parsed.username or parsed.password:
        raise ValueError("Article URLs containing credentials are not allowed.")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("Localhost URLs are not allowed.")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve article website: {host}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("Private, local, reserved and link-local article addresses are not allowed.")
    return parsed.geturl()


def fetch_article_html(url: str) -> tuple[str, str, str]:
    """Fetch a public webpage directly, following only validated public redirects."""
    current = validate_public_article_url(url)
    opener = __import__("urllib.request", fromlist=["build_opener"]).build_opener(
        __import__("urllib.request", fromlist=["HTTPHandler"]).HTTPHandler(),
        __import__("urllib.request", fromlist=["HTTPSHandler"]).HTTPSHandler(),
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8,*;q=0.5",
    }
    for _ in range(6):
        req = Request(current, headers=headers, method="GET")
        try:
            with opener.open(req, timeout=ARTICLE_FETCH_TIMEOUT) as response:
                final_url = validate_public_article_url(response.geturl())
                content_type = str(response.headers.get("Content-Type") or "").lower()
                if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    raise ValueError(f"The URL returned {content_type or 'a non-HTML document'}, not a webpage.")
                limit = 8 * 1024 * 1024
                raw = response.read(limit + 1)
                if len(raw) > limit:
                    raise ValueError("The webpage is larger than the 8 MB direct-reading limit.")
                charset = response.headers.get_content_charset() or "utf-8"
                return raw.decode(charset, errors="replace"), final_url, content_type
        except HTTPError as exc:
            raise RuntimeError(f"Website returned HTTP {exc.code} for {current}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not retrieve the article: {exc.reason}") from exc
    raise RuntimeError("Too many redirects while retrieving the article.")


def article_metadata_from_html(html: str, url: str) -> dict[str, str]:
    title = ""
    author = ""
    published = ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        def meta_value(*keys: tuple[str, str]) -> str:
            for attribute, value_ in keys:
                tag = soup.find("meta", attrs={attribute: value_})
                if tag and tag.get("content"):
                    return str(tag.get("content")).strip()
            return ""
        title = meta_value(("property", "og:title"), ("name", "twitter:title"))
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)
        author = meta_value(("name", "author"), ("property", "article:author"))
        published = meta_value(("property", "article:published_time"), ("name", "date"), ("name", "pubdate"))
    except Exception:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        title = html_lib.unescape(re.sub(r"<[^>]+>", " ", match.group(1))).strip() if match else ""
    domain = (urlparse(url).hostname or "").removeprefix("www.")
    return {"title": title or domain or "Web article", "author": author, "published_date": published, "domain": domain}


def fallback_article_extraction(html: str) -> str:
    """DOM-density fallback that prefers article/main content and removes page chrome."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "form", "button", "input", "select", "textarea", "nav", "header", "footer", "aside"]):
        tag.decompose()
    noise = re.compile(
        r"advert|advertisement|banner|breadcrumb|cookie|consent|footer|header|menu|nav|newsletter|"
        r"paywall|popup|promo|recommend|related|share|sidebar|social|sponsor|subscribe|toolbar",
        re.I,
    )
    for node in list(soup.find_all(True)):
        marker = " ".join([str(node.get("id") or ""), " ".join(node.get("class") or [])])
        if marker and noise.search(marker):
            node.decompose()
    candidates = list(soup.find_all("article")) + list(soup.find_all("main"))
    candidates += list(soup.find_all(["section", "div"], limit=1200))
    best_text = ""
    best_score = -1.0
    for node in candidates:
        paragraphs = [p.get_text(" ", strip=True) for p in node.find_all(["p", "h1", "h2", "h3", "li"], recursive=True)]
        paragraphs = [re.sub(r"\s+", " ", item).strip() for item in paragraphs if len(item.split()) >= 4]
        text = "\n\n".join(dict.fromkeys(paragraphs))
        if len(text) < 200:
            continue
        links = " ".join(a.get_text(" ", strip=True) for a in node.find_all("a"))
        punctuation = text.count(".") + text.count("!") + text.count("?")
        score = len(text) + punctuation * 35 - len(links) * 1.5
        if node.name == "article":
            score *= 1.35
        elif node.name == "main":
            score *= 1.15
        if score > best_score:
            best_score, best_text = score, text
    if best_text:
        return best_text
    return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))


def extract_article_text(html: str, url: str) -> tuple[str, str]:
    """Use article-aware extraction first, then a local DOM-density fallback."""
    text = ""
    method = "BeautifulSoup article-density fallback"
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_links=False,
            include_images=False,
            favor_precision=True,
            deduplicate=True,
            output_format="txt",
        ) or ""
        if len(text.split()) >= 80:
            method = "Trafilatura main-content extraction"
    except ImportError:
        pass
    except Exception as exc:
        log("Trafilatura extraction warning", url=url, error=repr(exc))
    if len(text.split()) < 80:
        text = fallback_article_extraction(html)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text.split()) < 50:
        raise RuntimeError("The page did not expose enough article text. It may require login, block automated access, or render its content only after browser interaction.")
    return text[:MAX_ARTICLE_CHARS], method


def summarize_article_batch(prompt: str, urls: list[str], model_key: str = DEFAULT_MODEL_KEY) -> tuple[str, str, list[dict[str, str]]]:
    """Retrieve article pages directly and ask the selected local model for an English briefing."""
    blocks: list[str] = []
    sources: list[dict[str, str]] = []
    for index, requested_url in enumerate(urls[:MAX_ARTICLE_URLS], 1):
        log("Direct article retrieval started", url=requested_url)
        html, final_url, _ = fetch_article_html(requested_url)
        metadata = article_metadata_from_html(html, final_url)
        text, method = extract_article_text(html, final_url)
        blocks.append(
            f"ARTICLE {index}\nTitle: {metadata['title']}\nURL: {final_url}\n"
            f"Website: {metadata['domain']}\nAuthor: {metadata['author'] or 'Unknown'}\n"
            f"Published: {metadata['published_date'] or 'Unknown'}\nExtraction method: {method}\n\n"
            f"ARTICLE BODY:\n{text}"
        )
        sources.append({"title": metadata["title"], "url": final_url})
        log("Direct article retrieval complete", url=final_url, words=len(text.split()), method=method)
    exact_urls_only = bool(re.fullmatch(r"(?:\s*https?://\S+\s*)+", prompt or "", flags=re.I))
    system = (
        "You are an expert executive analyst. Use only the supplied article body text. The page chrome, advertising, "
        "navigation and unrelated recommendations have been removed as far as possible. Always write the final response "
        "in English, regardless of the source language. Do not claim the article is English and do not mention translation "
        "unless language itself is relevant. Preserve important names, figures, dates, qualifications and disagreements. "
        "Do not invent missing facts. If several articles are supplied, compare and synthesize them.\n\n" + COMMUNICATION_STYLE
    )
    default_instruction = (
        "Use exactly these sections: Executive Summary, Key Points, Conclusions. The executive summary should be "
        "decision-useful and concise. The key points should be detailed, non-repetitive bullets. The conclusion should "
        "state the principal implications and any important limitations."
    )
    user_instruction = default_instruction if exact_urls_only else prompt
    user = f"ORIGINAL USER REQUEST:\n{user_instruction}\n\n" + "\n\n---\n\n".join(blocks)
    selected_key = normalize_model_key(model_key)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    output = generate_with_model(
        selected_key, messages,
        max_tokens=MODEL_OPTIONS[selected_key]["max_tokens"],
        enable_thinking=False if selected_key in {"default", "gemma", "qwen3_vl"} else None,
    )
    reasoning, final_output = strip_think_block(output)
    if selected_key in {"summary", "quick"} or final_answer_missing(final_output):
        if not reasoning and final_output and selected_key in {"summary", "quick"}:
            reasoning = final_output
        retry = generate_final_only(selected_key, messages, reasoning)
        leaked, final_output = strip_think_block(retry)
        if leaked:
            reasoning = "\n\n".join(part for part in (reasoning, leaked) if part)
    if final_answer_missing(final_output):
        raise RuntimeError("The selected model did not produce an article summary.")
    return clean_inline_citations(final_output), reasoning, sources

# -----------------------------------------------------------------------------
# YouTube
# -----------------------------------------------------------------------------
def youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().split(":")[0]
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif parts and parts[0] in {"shorts", "live", "embed"} and len(parts) > 1:
            candidate = parts[1]
        else:
            candidate = ""
    else:
        candidate = ""
    return candidate if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate or "") else None


def extract_youtube_urls(text: str) -> list[str]:
    """Return unique YouTube video URLs in prompt order."""
    urls: list[str] = []
    seen: set[str] = set()
    for match in YOUTUBE_URL_EXTRACT_RE.finditer(str(text or "")):
        url = match.group(0).rstrip(".,);]}>")
        video_id = youtube_video_id(url)
        if video_id and video_id not in seen:
            seen.add(video_id)
            urls.append(url)
    return urls

def youtube_metadata(url: str) -> dict[str, Any]:
    import yt_dlp
    video_id = youtube_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL.")
    options = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
    duration = max(0, int(round(float(info.get("duration") or 0))))
    if duration <= 0:
        raise RuntimeError("YouTube did not provide a usable video duration.")
    return {
        "url": url,
        "video_id": video_id,
        "title": str(info.get("title") or f"YouTube video {video_id}"),
        "duration_seconds": duration,
    }

def transcript_text(items: Any, start_seconds: float = 0, end_seconds: float | None = None) -> str:
    lines: list[str] = []
    for item in items:
        text = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
        item_start = float(item.get("start", 0) if isinstance(item, dict) else getattr(item, "start", 0) or 0)
        item_duration = float(item.get("duration", 0) if isinstance(item, dict) else getattr(item, "duration", 0) or 0)
        item_end = item_start + item_duration
        if item_end < start_seconds:
            continue
        if end_seconds is not None and item_start > end_seconds:
            break
        if text:
            lines.append(str(text))
    return "\n".join(lines)

def fetch_youtube_transcript(video_id: str, start_seconds: float = 0, end_seconds: float | None = None) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi
    errors: list[str] = []
    try:
        listing = YouTubeTranscriptApi().list(video_id)
        try:
            transcript = listing.find_transcript(["en", "en-GB", "en-US"])
        except Exception:
            available = list(listing)
            generated = [item for item in available if getattr(item, "is_generated", False)]
            transcript = generated[0] if generated else available[0]
        text = transcript_text(transcript.fetch(), start_seconds, end_seconds)
        if text.strip():
            return text
    except Exception as exc:
        errors.append(str(exc))
    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = listing.find_transcript(["en", "en-GB", "en-US"])
        except Exception:
            available = list(listing)
            generated = [item for item in available if getattr(item, "is_generated", False)]
            transcript = generated[0] if generated else available[0]
        text = transcript_text(transcript.fetch(), start_seconds, end_seconds)
        if text.strip():
            return text
    except Exception as exc:
        errors.append(str(exc))
    raise RuntimeError("; ".join(errors) or "No YouTube transcript is available.")

def transcribe_youtube_audio(
    url: str, video_id: str, job_dir: Path, start_seconds: float = 0, end_seconds: float | None = None
) -> str:
    import mlx_whisper
    import yt_dlp
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(job_dir / f"{video_id}.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
        ],
    }
    log("YouTube audio fallback started", url=url, start_seconds=start_seconds, end_seconds=end_seconds)
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([url])
    audio_path = job_dir / f"{video_id}.wav"
    if not audio_path.exists():
        raise RuntimeError("Audio conversion failed. Check that ffmpeg is installed.")
    input_path = audio_path
    if start_seconds > 0 or end_seconds is not None:
        clipped_path = job_dir / f"{video_id}_segment.wav"
        command = ["ffmpeg", "-y", "-ss", str(max(0, start_seconds)), "-i", str(audio_path)]
        if end_seconds is not None:
            command.extend(["-t", str(max(0.1, end_seconds - start_seconds))])
        command.extend(["-ac", "1", "-ar", "16000", str(clipped_path)])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not clipped_path.exists():
            raise RuntimeError(result.stderr[-1200:] or "ffmpeg could not create the selected audio segment.")
        input_path = clipped_path
    result = mlx_whisper.transcribe(str(input_path), path_or_hf_repo=WHISPER_MODEL)
    return str(result.get("text", "")).strip()

def collect_youtube_transcript(segment: dict[str, Any]) -> tuple[str, str, Path]:
    url = str(segment.get("url") or "").strip()
    video_id = youtube_video_id(url)
    if not video_id:
        raise ValueError(f"Invalid YouTube URL: {url}")
    duration = max(1, int(segment.get("duration_seconds") or 1))
    start_seconds = max(0, min(duration - 1, int(segment.get("start_seconds") or 0)))
    end_seconds = max(start_seconds + 1, min(duration, int(segment.get("end_seconds") or duration)))
    job_dir = YOUTUBE / f"{video_id}-{uuid.uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    method = "YouTube captions"
    try:
        transcript = fetch_youtube_transcript(video_id, start_seconds, end_seconds)
    except Exception as caption_error:
        log("YouTube captions unavailable", error=repr(caption_error))
        method = "local MLX Whisper transcription"
        transcript = transcribe_youtube_audio(url, video_id, job_dir, start_seconds, end_seconds)
    if not transcript.strip():
        raise RuntimeError(f"No usable transcript could be generated for {url}.")
    (job_dir / f"{video_id}_transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript, method, job_dir

def summarize_youtube_batch(
    prompt: str, segments: list[dict[str, Any]], model_key: str = DEFAULT_MODEL_KEY
) -> tuple[str, str, list[dict[str, str]]]:
    collected: list[tuple[dict[str, Any], str, str]] = []
    job_dirs: list[Path] = []
    try:
        for segment in segments:
            transcript, method, job_dir = collect_youtube_transcript(segment)
            job_dirs.append(job_dir)
            collected.append((segment, transcript, method))
        transcript_blocks = []
        sources = []
        for index, (segment, transcript, method) in enumerate(collected, 1):
            start_seconds = int(segment.get("start_seconds") or 0)
            end_seconds = int(segment.get("end_seconds") or segment.get("duration_seconds") or 0)
            title = str(segment.get("title") or f"YouTube video {index}")
            url = str(segment.get("url") or "")
            transcript_blocks.append(
                f"VIDEO {index}\nTitle: {title}\nURL: {url}\nSelected segment: {start_seconds}s to {end_seconds}s\n"
                f"Transcript acquisition method: {method}\nTRANSCRIPT:\n{transcript[:MAX_TRANSCRIPT_CHARS]}"
            )
            sources.append({"title": title, "url": url})
        system = (
            "You are an expert executive analyst. Answer the user's request using the supplied YouTube transcripts. "
            "When several videos are supplied, compare or synthesize them as the request requires. Do not invent facts or "
            "claim to have seen visuals not represented in the transcripts. If the prompt is only one or more URLs, use exactly "
            "these sections: Executive Summary, Key Points, Conclusions.\n\n" + COMMUNICATION_STYLE
        )
        user = f"ORIGINAL USER PROMPT:\n{prompt}\n\n" + "\n\n---\n\n".join(transcript_blocks)
        model_key = normalize_model_key(model_key)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        output = generate_with_model(model_key, messages, max_tokens=MODEL_OPTIONS[model_key]["max_tokens"], enable_thinking=False if model_key in {"default", "gemma", "qwen3_vl"} else None)
        reasoning, final_output = strip_think_block(output)
        if model_key in {"summary", "quick"} or final_answer_missing(final_output):
            if not reasoning and final_output and model_key in {"summary", "quick"}: reasoning = final_output
            retry = generate_final_only(model_key, messages, reasoning)
            leaked, final_output = strip_think_block(retry)
            if leaked: reasoning = "\n\n".join(part for part in (reasoning, leaked) if part)
        if final_answer_missing(final_output):
            raise RuntimeError("The selected model did not produce the YouTube analysis.")
        return clean_inline_citations(final_output), reasoning, sources
    finally:
        for job_dir in job_dirs:
            for media in job_dir.iterdir():
                if media.suffix.lower() != ".txt": media.unlink(missing_ok=True)

def summarize_youtube(url: str, model_key: str = DEFAULT_MODEL_KEY) -> tuple[str, str]:
    video_id = youtube_video_id(url)
    if not video_id:
        raise ValueError("That is not a valid single-video YouTube URL.")
    job_dir = YOUTUBE / f"{video_id}-{uuid.uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    method = "YouTube captions"
    try:
        try:
            transcript = fetch_youtube_transcript(video_id)
        except Exception as caption_error:
            log("YouTube captions unavailable", error=repr(caption_error))
            method = "local MLX Whisper transcription"
            transcript = transcribe_youtube_audio(url, video_id, job_dir)
        if not transcript.strip():
            raise RuntimeError("No usable transcript could be generated.")

        (job_dir / f"{video_id}_transcript.txt").write_text(transcript, encoding="utf-8")
        system = (
            "You are an expert executive analyst. Produce a faithful, decision-useful briefing from a video transcript. "
            "Do not invent facts or imply that you watched visuals not represented in the transcript.\n\n"
            + COMMUNICATION_STYLE
        )
        user = (
            "Analyze this YouTube video transcript and respond using exactly these sections:\n\n"
            "Executive Summary:\n"
            "Write a thorough 200-400 word executive summary covering the central thesis, major arguments, material evidence, and practical implications.\n\n"
            "Key Points:\n"
            "Provide up to 30 detailed, non-repetitive bullet points. Consolidate overlapping ideas while retaining important facts, qualifications, examples, and context.\n\n"
            "Conclusions:\n"
            "Write a concise concluding paragraph stating the principal takeaways and important limitations or unresolved questions.\n\n"
            f"Transcript acquisition method: {method}\nYouTube URL: {url}\n\n"
            f"VIDEO TRANSCRIPT:\n{transcript[:MAX_TRANSCRIPT_CHARS]}"
        )
        model_key = normalize_model_key(model_key)
        youtube_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        output = generate_with_model(
            model_key,
            youtube_messages,
            max_tokens=MODEL_OPTIONS[model_key]["max_tokens"],
            enable_thinking=False if model_key in {"default", "gemma", "qwen3_vl"} else None,
        )
        reasoning, first_pass_final = strip_think_block(output)
        if model_key in {"summary", "quick"}:
            if not reasoning and first_pass_final:
                reasoning = first_pass_final
            final_raw = generate_final_only(model_key, youtube_messages, reasoning)
            leaked_reasoning, final_output = strip_think_block(final_raw)
            if leaked_reasoning:
                reasoning = "\n\n".join(part for part in (reasoning, leaked_reasoning) if part)
        else:
            final_output = first_pass_final
            if final_answer_missing(final_output):
                retry_output = generate_final_only(model_key, youtube_messages, reasoning)
                retry_reasoning, final_output = strip_think_block(retry_output)
                if retry_reasoning:
                    reasoning = "\n\n".join(part for part in (reasoning, retry_reasoning) if part)
        if final_answer_missing(final_output):
            raise RuntimeError(
                "The selected model completed its reasoning but did not produce the YouTube summary. "
                "Please retry or select another model."
            )
        final_output = clean_inline_citations(final_output)
        (job_dir / f"{video_id}_executive_summary.txt").write_text(final_output, encoding="utf-8")
        log("YouTube summary complete", video_id=video_id, method=method, model_key=model_key, reasoning_characters=len(reasoning))
        return final_output, reasoning
    finally:
        for media in job_dir.iterdir():
            if media.suffix.lower() != ".txt":
                media.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Professional speech: Piper for English and all supported languages
# -----------------------------------------------------------------------------
def get_argos_translation(target_code: str):
    """Load English-to-target offline translation without invoking a chat model."""
    try:
        import argostranslate.package as package
        import argostranslate.translate as translate
    except ImportError as exc:
        raise RuntimeError(
            "Non-English Read Aloud requires Argos Translate: "
            "python -m pip install -U argostranslate"
        ) from exc

    def installed():
        languages = translate.get_installed_languages()
        source = next((item for item in languages if item.code == "en"), None)
        target = next((item for item in languages if item.code == target_code), None)
        if source is None or target is None:
            return None
        try:
            return source.get_translation(target)
        except Exception:
            return None

    translator = installed()
    if translator is not None:
        return translator
    if not ARGOS_AUTO_DOWNLOAD:
        raise RuntimeError(f"Argos package en->{target_code} is not installed.")
    package.update_package_index()
    candidate = next(
        (item for item in package.get_available_packages()
         if item.from_code == "en" and item.to_code == target_code),
        None,
    )
    if candidate is None:
        raise RuntimeError(f"No Argos package is available for en->{target_code}.")
    package.install_from_path(candidate.download())
    translator = installed()
    if translator is None:
        raise RuntimeError(f"Argos package en->{target_code} could not be loaded.")
    return translator


def split_translation_text(text: str, maximum: int = 1800) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs or [text.strip()]:
        while len(paragraph) > maximum:
            cut = paragraph.rfind(" ", 0, maximum)
            cut = cut if cut > 0 else maximum
            chunks.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if paragraph:
            chunks.append(paragraph)
    return chunks


def translate_for_speech(text: str, language_code: str) -> str:
    """Translate visible response text for TTS without using any LLM."""
    language_code = language_code if language_code in LANGUAGES else "en"
    visible_text = str(text or "").strip()
    if language_code == "en":
        return visible_text
    with _argos_lock:
        translator = get_argos_translation(language_code)
        translated = "\n\n".join(
            translator.translate(chunk).strip()
            for chunk in split_translation_text(visible_text)
        ).strip()
    if not translated:
        raise RuntimeError(f"Translation to {LANGUAGES[language_code]} returned no text.")
    log(
        "Direct TTS translation complete",
        engine="Argos Translate",
        target_language=LANGUAGES[language_code],
        input_characters=len(visible_text),
        output_characters=len(translated),
    )
    return translated


def synthesize_styletts2(text: str, output_path: Path) -> None:
    if styletts2_tts is None or AudioSegment is None or nltk is None:
        raise RuntimeError(
            "StyleTTS2 dependencies are unavailable. Install: styletts2 torch nltk pydub"
        )

    segments: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[A-Za-z0-9]+\.|[-*•]+)\s*", "", line)
        for sentence in nltk.sent_tokenize(line):
            sentence = sentence.strip()
            if not sentence:
                continue
            if not sentence.endswith((".", "!", "?")):
                sentence += "."
            words = sentence.split()
            if len(words) > 60:
                for offset in range(0, len(words), 60):
                    chunk = " ".join(words[offset : offset + 60])
                    if not chunk.endswith((".", "!", "?")):
                        chunk += "."
                    segments.append(chunk)
            else:
                segments.append(sentence)
    if not segments:
        raise RuntimeError("The response contains no text to speak.")

    log("StyleTTS2 loading", segments=len(segments))
    engine = styletts2_tts.StyleTTS2()
    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=400)
    chunk_files: list[Path] = []
    try:
        for index, segment in enumerate(segments, 1):
            chunk_path = AUDIO / f"styletts2-{uuid.uuid4().hex}-{index}.wav"
            chunk_files.append(chunk_path)
            log("StyleTTS2 segment", number=index, total=len(segments), characters=len(segment))
            engine.inference(segment, output_wav_file=str(chunk_path))
            combined += AudioSegment.from_wav(chunk_path) + pause
        combined.export(output_path, format="wav")
    finally:
        for chunk_path in chunk_files:
            chunk_path.unlink(missing_ok=True)


def piper_voice_paths(language_code: str) -> tuple[Path, Path]:
    voice_name = PIPER_VOICE_NAMES.get(language_code)
    if not voice_name:
        raise ValueError(f"No Piper voice is configured for language: {language_code}")
    model_path = PIPER_VOICES_DIR / f"{voice_name}.onnx"
    config_path = PIPER_VOICES_DIR / f"{voice_name}.onnx.json"
    return model_path, config_path


def ensure_piper_voice(language_code: str) -> tuple[Path, Path]:
    model_path, config_path = piper_voice_paths(language_code)
    if model_path.exists() and config_path.exists():
        return model_path, config_path
    voice_name = PIPER_VOICE_NAMES[language_code]
    if not PIPER_AUTO_DOWNLOAD:
        raise RuntimeError(
            f"Piper voice {voice_name} is missing. Download it with: "
            f"python -m piper.download_voices --download-dir {PIPER_VOICES_DIR} {voice_name}"
        )
    log("Piper voice download started", language=LANGUAGES[language_code], voice=voice_name, directory=str(PIPER_VOICES_DIR))
    command = [
        os.sys.executable,
        "-m",
        "piper.download_voices",
        "--download-dir",
        str(PIPER_VOICES_DIR),
        voice_name,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout.strip():
        log("Piper downloader stdout", text=result.stdout.strip())
    if result.returncode != 0:
        log("Piper downloader stderr", text=result.stderr.strip(), returncode=result.returncode)
        raise RuntimeError(result.stderr.strip() or f"Could not download Piper voice {voice_name}")
    if not model_path.exists() or not config_path.exists():
        raise RuntimeError(f"Piper downloaded {voice_name}, but its ONNX model or JSON config is missing.")
    log("Piper voice download complete", language=LANGUAGES[language_code], voice=voice_name)
    return model_path, config_path


def get_piper_voice(language_code: str):
    if language_code not in PIPER_VOICE_NAMES:
        raise ValueError(f"Piper is not configured for language: {language_code}")
    with _piper_lock:
        cached = _piper_voices.get(language_code)
        if cached is not None:
            return cached
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise RuntimeError("Read Aloud requires piper-tts. Install it with: python -m pip install piper-tts") from exc
        model_path, config_path = ensure_piper_voice(language_code)
        voice_name = PIPER_VOICE_NAMES[language_code]
        log("Piper voice loading", language=LANGUAGES[language_code], voice=voice_name, model=str(model_path))
        voice = PiperVoice.load(str(model_path), config_path=str(config_path), use_cuda=PIPER_USE_CUDA)
        _piper_voices[language_code] = voice
        log("Piper voice loaded", language=LANGUAGES[language_code], voice=voice_name)
        return voice


def normalize_english_for_piper(text: str) -> str:
    """Normalize cased English text for speech before it reaches Piper."""
    global _nemo_normalizer
    with _nemo_lock:
        if _nemo_normalizer is None:
            try:
                from nemo_text_processing.text_normalization.normalize import Normalizer
            except ImportError as exc:
                raise RuntimeError(
                    "English Read Aloud requires nemo_text_processing. Install it with: "
                    "python -m pip install nemo_text_processing"
                ) from exc
            log("NeMo English text normalizer loading", input_case="cased")
            _nemo_normalizer = Normalizer(input_case="cased", lang="en")
            log("NeMo English text normalizer loaded", input_case="cased")
        normalized = _nemo_normalizer.normalize(text, verbose=False).strip()
    if not normalized:
        raise RuntimeError("NeMo text normalization returned no English text.")
    log("NeMo English text normalization complete", input_case="cased", input_characters=len(text), output_characters=len(normalized))
    return normalized

def synthesize_piper(text: str, language_code: str, output_path: Path) -> None:
    voice = get_piper_voice(language_code)
    voice_name = PIPER_VOICE_NAMES[language_code]
    if language_code == "en":
        text = normalize_english_for_piper(text)
    log("Piper synthesis started", language=LANGUAGES[language_code], voice=voice_name, characters=len(text))
    with _piper_lock:
        with wave.open(str(output_path), "wb") as wav_file:
            # Current piper-tts API.
            if hasattr(voice, "synthesize_wav"):
                try:
                    from piper.config import SynthesisConfig
                    config = SynthesisConfig(
                        length_scale=PIPER_LENGTH_SCALE,
                        sentence_silence=PIPER_SENTENCE_SILENCE,
                    )
                    voice.synthesize_wav(text, wav_file, syn_config=config)
                except (ImportError, TypeError):
                    voice.synthesize_wav(text, wav_file)
            else:
                # Compatibility with older piper-tts releases.
                voice.synthesize(
                    text,
                    wav_file,
                    length_scale=PIPER_LENGTH_SCALE,
                    sentence_silence=PIPER_SENTENCE_SILENCE,
                )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Piper voice {voice_name} produced no audio file")
    log("Piper synthesis complete", language=LANGUAGES[language_code], voice=voice_name, bytes=output_path.stat().st_size)


def clean_audio(max_age_seconds: int = 3600) -> None:
    cutoff = time.time() - max_age_seconds
    for audio_file in AUDIO.glob("*"):
        try:
            if audio_file.is_file() and audio_file.stat().st_mtime < cutoff:
                audio_file.unlink()
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Live microphone mode
# -----------------------------------------------------------------------------
def live_callback(indata, frames, time_info, status) -> None:
    if status:
        log("LIVE audio callback status", status=str(status))
    if live_stop.is_set() or live_muted.is_set():
        return
    live_audio.put(indata.copy())


def emit_live_event(event_type: str, **payload: Any) -> None:
    event = {"type": event_type, "timestamp": time.time(), **payload}
    live_results.put(event)
    log("LIVE UI event", event_type=event_type, **{k: v for k, v in payload.items() if k != "answer"})


def live_worker() -> None:
    import mlx_whisper
    import numpy as np

    rate = 16000
    half_second = rate // 2
    buffer = np.zeros((0, 1), dtype=np.float32)
    speech = np.zeros((0, 1), dtype=np.float32)
    preroll: list[Any] = []
    speaking = False
    silence = 0.0
    noise_floor = 0.004
    log("LIVE worker started", sample_rate=rate, whisper_model=WHISPER_MODEL)

    while not live_stop.is_set():
        try:
            buffer = np.vstack((buffer, live_audio.get(timeout=0.1)))
        except queue.Empty:
            continue
        while len(buffer) >= half_second:
            chunk, buffer = buffer[:half_second], buffer[half_second:]
            rms = float(np.sqrt(np.mean(chunk**2)))
            if not speaking:
                noise_floor = (noise_floor * 0.92) + (min(rms, 0.03) * 0.08)
            threshold = max(LIVE_MIN_THRESHOLD, min(0.025, noise_floor * 2.4))
            preroll.append(chunk)
            if len(preroll) > 2:
                preroll.pop(0)

            if rms >= threshold:
                if not speaking:
                    speaking = True
                    speech = np.vstack(preroll)
                    preroll = []
                    log(
                        "LIVE speech detected",
                        rms=round(rms, 5),
                        threshold=round(threshold, 5),
                        noise_floor=round(noise_floor, 5),
                    )
                else:
                    speech = np.vstack((speech, chunk))
                silence = 0.0
            elif speaking:
                silence += 0.5
                speech = np.vstack((speech, chunk))

            if speaking and (
                silence >= LIVE_SILENCE_SECONDS or len(speech) >= rate * LIVE_MAX_UTTERANCE_SECONDS
            ):
                duration = len(speech) / rate
                utterance = speech.flatten()
                speech = np.zeros((0, 1), dtype=np.float32)
                speaking = False
                silence = 0.0
                preroll = []
                log("LIVE utterance captured", duration_seconds=round(duration, 2), samples=len(utterance))
                try:
                    started = time.perf_counter()
                    result = mlx_whisper.transcribe(
                        utterance, path_or_hf_repo=WHISPER_MODEL, task="transcribe"
                    )
                    prompt = str(result.get("text", "")).strip()
                    log(
                        "USER (voice, raw transcript)",
                        text=prompt,
                        transcription_seconds=round(time.perf_counter() - started, 2),
                    )
                    if prompt and live_session_id:
                        emit_live_event(
                            "transcript",
                            prompt=prompt,
                            message="Message received. Preparing your response...",
                        )
                        stage_message = (
                            "Searching the web with Exa, then asking Qwen to synthesize the final answer..."
                            if live_use_web
                            else "Asking Qwen to prepare the response..."
                        )
                        emit_live_event("status", message=stage_message)
                        started = time.perf_counter()
                        output, sources, reasoning = answer(prompt, get_session(live_session_id), live_use_web, live_model_key, live_max_web_results)
                        elapsed = round(time.perf_counter() - started, 2)
                        log(
                            "ASSISTANT (raw response)",
                            text=output,
                            generation_seconds=elapsed,
                            source_count=len(sources),
                        )
                        emit_live_event(
                            "result",
                            prompt=prompt,
                            answer=output,
                            reasoning=reasoning,
                            model_key=live_model_key,
                            sources=sources,
                            elapsed_seconds=elapsed,
                            message="Response ready.",
                        )
                    else:
                        log("LIVE empty transcript ignored")
                        emit_live_event("status", message="I could not detect a spoken message. Please try again.")
                except Exception as exc:
                    log("LIVE processing error", error=repr(exc))
                    emit_live_event("error", error=str(exc), message="Live processing failed.")
    log("LIVE worker stopped")


def set_live_muted(muted: bool) -> bool:
    if muted:
        live_muted.set()
    else:
        live_muted.clear()
    while not live_audio.empty():
        try:
            live_audio.get_nowait()
        except queue.Empty:
            break
    log("LIVE microphone muted" if muted else "LIVE microphone unmuted")
    return live_muted.is_set()


def start_live(session_id: str, use_web: bool = False, model_key: str = DEFAULT_MODEL_KEY, max_web_results: int = MAX_WEB_RESULTS) -> None:
    global live_stream, live_thread, live_session_id, live_use_web, live_max_web_results, live_model_key
    import sounddevice as sd

    with live_lock:
        if live_stream is not None:
            return
        live_session_id = session_id
        live_max_web_results = normalize_max_web_results(max_web_results)
        live_use_web = bool(use_web and live_max_web_results > 0)
        live_model_key = normalize_model_key(model_key)
        live_stop.clear()
        live_muted.clear()
        while not live_audio.empty():
            live_audio.get_nowait()
        live_stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="float32",
            blocksize=8000,
            callback=live_callback,
        )
        live_stream.start()
        log("LIVE microphone opened", device=getattr(live_stream, "device", None), sample_rate=16000)
        live_thread = threading.Thread(target=live_worker, daemon=True)
        live_thread.start()


def stop_live() -> None:
    global live_stream, live_thread
    with live_lock:
        live_stop.set()
        if live_stream is not None:
            live_stream.stop()
            live_stream.close()
            live_stream = None
            log("LIVE microphone closed")
        live_thread = None
        live_muted.clear()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
def system_memory_snapshot() -> dict[str, float]:
    """Return system-wide physical memory figures in bytes."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        total = int(memory.total)
        available = int(memory.available)
    except ImportError:
        # Native macOS fallback, so psutil is optional rather than required.
        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        output = subprocess.check_output(["vm_stat"], text=True)
        page_match = re.search(r"page size of (\d+) bytes", output)
        page_size = int(page_match.group(1)) if page_match else 4096
        pages = {}
        for name, value in re.findall(r"^([^:]+):\s+([0-9.]+)\.?$", output, flags=re.M):
            pages[name.strip()] = int(float(value))
        available_pages = sum(pages.get(name, 0) for name in (
            "Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"
        ))
        available = min(total, available_pages * page_size)

    used = max(0, total - available)
    percent = round((used / total * 100.0) if total else 0.0, 1)
    return {"used": used, "available": available, "total": total, "percent": percent}


@app.get("/api/system_health")
def system_health():
    try:
        response = jsonify(system_memory_snapshot())
        response.headers["Cache-Control"] = "no-store"
        return response
    except Exception as exc:
        log("System health check failed", error=repr(exc))
        return jsonify(error="Memory status unavailable"), 500


@app.get("/")
def index():
    return render_template_string(HTML)


@app.post("/api/search")
def search_route():
    data = request.get_json(force=True)
    session_id = str(data.get("session_id") or uuid.uuid4())
    started = time.perf_counter()
    try:
        evidence, sources, metadata = web_search(
            query=str(data.get("query") or ""),
            max_results=data.get("max_results", MAX_WEB_RESULTS),
            objective=str(data.get("objective") or ""),
            include_domains=data.get("include_domains"),
            exclude_domains=data.get("exclude_domains"),
            freshness_mode=str(data.get("freshness_mode") or "cached"),
            max_age_hours=data.get("max_age_hours"),
            start_published_date=data.get("start_published_date"),
            end_published_date=data.get("end_published_date"),
            highlights=bool(data.get("highlights", True)),
            search_type=str(data.get("search_type") or "auto"),
            user_location=str(data.get("user_location") or ""),
            moderation=bool(data.get("moderation", False)),
            categories=data.get("categories") or [],
            languages=data.get("languages") or DEFAULT_SEARCH_LANGUAGES,
        )
        current_session = get_session(session_id)
        existing = current_session.get("search") or {}
        existing_sources = list(existing.get("sources") or [])
        existing_queries = list(existing.get("queries") or [])

        # Capture the current browser selection before adding the next batch.
        # Persist cumulative selections using stable URLs. DOM indices can be
        # stale after multiple search batches are merged and re-rendered.
        raw_existing_urls = data.get("selected_existing_urls")
        raw_existing_selection = data.get("selected_existing_indices")
        if raw_existing_urls is not None:
            selected_urls = {
                str(url or "").rstrip("/").lower()
                for url in raw_existing_urls or []
                if str(url or "").strip()
            }
            existing_selection = [
                index for index, source in enumerate(existing_sources)
                if str(source.get("url") or "").rstrip("/").lower() in selected_urls
            ]
        elif raw_existing_selection is None:
            stored_selection = existing.get("selected_indices")
            existing_selection = list(stored_selection) if stored_selection is not None else list(range(len(existing_sources)))
        else:
            existing_selection = []
            for raw_index in raw_existing_selection or []:
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(existing_sources) and index not in existing_selection:
                    existing_selection.append(index)

        batch_number = len(existing_queries) + 1
        query_record = {
            "batch": batch_number,
            "query": metadata["query"],
            "objective": metadata["objective"],
            "metadata": metadata,
            "retrieved_at": time.time(),
        }
        existing_queries.append(query_record)

        # Append genuinely new URLs. Existing articles retain their original
        # index, selection state, and search provenance.
        combined_sources = list(existing_sources)
        seen_urls = {
            str(item.get("url") or "").rstrip("/").lower()
            for item in combined_sources
            if str(item.get("url") or "").strip()
        }
        added_indices: list[int] = []
        duplicate_count = 0
        for source in sources:
            item = dict(source)
            key = str(item.get("url") or "").rstrip("/").lower()
            if key and key in seen_urls:
                duplicate_count += 1
                continue
            if key:
                seen_urls.add(key)
            item["search_batch"] = batch_number
            item["search_query"] = metadata["query"]
            item["search_objective"] = metadata["objective"]
            combined_sources.append(item)
            added_indices.append(len(combined_sources) - 1)

        selected_indices = sorted(set(existing_selection + added_indices))
        current_session["search"] = {
            "query": metadata["query"],
            "queries": existing_queries,
            "sources": combined_sources,
            "selected_indices": selected_indices,
            "metadata": metadata,
            "retrieved_at": time.time(),
        }
        return jsonify(
            session_id=session_id,
            sources=combined_sources,
            metadata=metadata,
            selected_indices=selected_indices,
            added_count=len(added_indices),
            duplicate_count=duplicate_count,
            total_count=len(combined_sources),
            search_count=len(existing_queries),
            elapsed_seconds=round(time.perf_counter()-started, 2),
        )
    except Exception as exc:
        log("Search failed", error=repr(exc))
        app.logger.exception("Search failed")
        return jsonify(error=str(exc)), 500


@app.post("/api/search_suggestions")
def search_suggestions_route():
    data = request.get_json(force=True)
    session_id = str(data.get("session_id") or "")
    if not session_id:
        return jsonify(error="No active research session."), 400
    try:
        suggestions = generate_search_suggestions(
            get_session(session_id),
            str(data.get("model_key") or DEFAULT_MODEL_KEY),
        )
        return jsonify(session_id=session_id, suggestions=suggestions)
    except Exception as exc:
        log("Search suggestion generation failed", error=repr(exc))
        app.logger.exception("Search suggestion generation failed")
        return jsonify(error=str(exc)), 500


@app.post("/api/youtube_metadata")
def youtube_metadata_route():
    data = request.get_json(force=True)
    urls = data.get("urls") or extract_youtube_urls(str(data.get("prompt") or ""))
    if not urls:
        return jsonify(videos=[]), 200
    videos = []
    try:
        for url in urls:
            videos.append(youtube_metadata(str(url)))
        return jsonify(videos=videos)
    except Exception as exc:
        log("YouTube metadata failed", error=repr(exc))
        return jsonify(error=str(exc)), 500

@app.post("/api/media_upload")
def media_upload():
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    incoming = request.files.get("file")
    if incoming is None or not incoming.filename:
        return jsonify(error="Choose an audio or video file."), 400
    name = secure_filename(incoming.filename)
    extension = Path(name).suffix.lower()
    if extension not in MEDIA_EXTENSIONS:
        return jsonify(error=f"Unsupported audio or video type: {extension or 'unknown'}"), 400
    folder = MEDIA / secure_filename(session_id)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():
        if old.is_file():
            old.unlink(missing_ok=True)
        elif old.is_dir():
            shutil.rmtree(old, ignore_errors=True)
    destination = folder / name
    try:
        incoming.save(destination)
        if destination.stat().st_size > MAX_MEDIA_BYTES:
            raise ValueError(f"Media files are limited to {MAX_MEDIA_BYTES // (1024 * 1024)} MB.")
        metadata = probe_media(destination)
        media = {
            "session_id": session_id, "name": name, "path": str(destination),
            "duration_seconds": metadata["duration_seconds"], "bytes": destination.stat().st_size,
        }
        get_session(session_id)["media"] = media
        log("Local media uploaded", name=name, duration_seconds=media["duration_seconds"], bytes=media["bytes"])
        return jsonify(session_id=session_id, media={k: v for k, v in media.items() if k != "path"})
    except Exception as exc:
        destination.unlink(missing_ok=True)
        log("Local media upload failed", name=name, error=repr(exc))
        return jsonify(error=str(exc)), 400

@app.post("/api/media_remove")
def media_remove():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id") or "")
    current_session = get_session(session_id) if session_id else None
    if current_session is not None:
        current_session["media"] = None
    if session_id:
        shutil.rmtree(MEDIA / secure_filename(session_id), ignore_errors=True)
    return jsonify(ok=True)

@app.post("/api/chat")
def chat():
    data = request.get_json(force=True)
    session_id = str(data.get("session_id") or uuid.uuid4())
    prompt = str(data.get("prompt") or "").strip()
    max_web_results = normalize_max_web_results(data.get("max_web_results", MAX_WEB_RESULTS))
    current_session = get_session(session_id)
    saved_search = current_session.get("search")
    if saved_search is not None:
        saved_sources = list(saved_search.get("sources") or [])
        if "selected_source_urls" in data:
            selected_urls = {
                str(url or "").rstrip("/").lower()
                for url in data.get("selected_source_urls") or []
                if str(url or "").strip()
            }
            saved_search["selected_indices"] = [
                index for index, source in enumerate(saved_sources)
                if str(source.get("url") or "").rstrip("/").lower() in selected_urls
            ]
        elif "selected_source_indices" in data:
            raw_indices = data.get("selected_source_indices") or []
            source_count = len(saved_sources)
            selected_indices: list[int] = []
            for raw_index in raw_indices:
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < source_count and index not in selected_indices:
                    selected_indices.append(index)
            saved_search["selected_indices"] = selected_indices
    use_web = bool(data.get("use_search_results", True) and saved_search)
    model_key = normalize_model_key(str(data.get("model_key") or DEFAULT_MODEL_KEY))
    log("USER (typed)", text=prompt, use_web=use_web, max_web_results=max_web_results, model_key=model_key, model=MODEL_OPTIONS[model_key]["model"])
    if not prompt:
        return jsonify(error="Enter a prompt."), 400

    started = time.perf_counter()
    try:
        response_type = "standard"
        reasoning = ""
        youtube_urls = extract_youtube_urls(prompt)
        article_urls = extract_article_urls(prompt)
        youtube_segments = data.get("youtube_segments") or []
        media_segment = data.get("media_segment")
        uploaded_media = current_session.get("media")
        if media_segment and uploaded_media:
            response_type = "media_analysis"
            output, reasoning, sources = analyze_local_media(prompt, uploaded_media, media_segment, model_key)
            current_session["history"].extend(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}]
            )
            current_session["history"] = current_session["history"][-MAX_HISTORY_TURNS * 2 :]
        elif article_urls:
            response_type = "article_summary"
            output, reasoning, sources = summarize_article_batch(prompt, article_urls, model_key)
            current_session["history"].extend(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}]
            )
            current_session["history"] = current_session["history"][-MAX_HISTORY_TURNS * 2 :]
        elif youtube_urls:
            if not youtube_segments:
                youtube_segments = []
                for url in youtube_urls:
                    metadata = youtube_metadata(url)
                    youtube_segments.append({**metadata, "start_seconds": 0, "end_seconds": metadata["duration_seconds"]})
            requested_ids = [youtube_video_id(url) for url in youtube_urls]
            supplied = {youtube_video_id(str(item.get("url") or "")): item for item in youtube_segments}
            missing = [video_id for video_id in requested_ids if video_id not in supplied]
            if missing:
                raise ValueError("Segment selections are missing for one or more YouTube URLs.")
            ordered_segments = [supplied[video_id] for video_id in requested_ids]
            response_type = "youtube_summary" if len(youtube_urls) == 1 and YOUTUBE_URL_RE.fullmatch(prompt) else "standard"
            output, reasoning, sources = summarize_youtube_batch(prompt, ordered_segments, model_key)
            current_session["history"].extend(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}]
            )
            current_session["history"] = current_session["history"][-MAX_HISTORY_TURNS * 2 :]
        else:
            output, sources, reasoning = answer(prompt, current_session, use_web, model_key, max_web_results)
        elapsed = time.perf_counter() - started
        log(
            "ASSISTANT (raw response)",
            text=output,
            elapsed_seconds=round(elapsed, 2),
            source_count=len(sources),
        )
        return jsonify(
            session_id=session_id,
            answer=output,
            sources=sources,
            elapsed_seconds=round(elapsed, 2),
            response_type=response_type,
            reasoning=reasoning,
            model_key=model_key,
            model_label=MODEL_OPTIONS[model_key]["label"],
        )
    except Exception as exc:
        log("Chat processing failed", error=repr(exc))
        app.logger.exception("Chat processing failed")
        return jsonify(error=str(exc)), 500


@app.post("/api/upload")
def upload():
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    current_session = get_session(session_id)
    folder = UPLOADS / secure_filename(session_id)
    folder.mkdir(parents=True, exist_ok=True)
    added = []
    errors = []
    incoming_files = request.files.getlist("files")
    if len(incoming_files) > MAX_UPLOAD_FILES:
        return jsonify(error=f"You can upload a maximum of {MAX_UPLOAD_FILES} files at once."), 400
    for incoming in incoming_files:
        name = secure_filename(incoming.filename or "")
        if not name:
            continue
        destination = folder / name
        incoming.save(destination)
        try:
            extension = destination.suffix.lower()
            if extension in IMAGE_EXTENSIONS:
                metadata = validate_image(destination)
                current_session["images"] = [item for item in current_session.get("images", []) if item["name"] != name]
                current_session["images"].append({"name": name, "path": str(destination), **metadata})
                added.append({"name": name, "kind": "image", **metadata})
                log("Image attachment loaded", name=name, **metadata)
            else:
                text = extract_text(destination)
                current_session["files"] = [item for item in current_session["files"] if item["name"] != name]
                current_session["files"].append({"name": name, "text": text})
                added.append({"name": name, "kind": "document", "characters": len(text)})
                log("Attachment loaded", name=name, characters=len(text))
        except Exception as exc:
            errors.append({"name": name, "error": str(exc)})
            destination.unlink(missing_ok=True)
            log("Attachment failed", name=name, error=repr(exc))
    return jsonify(session_id=session_id, added=added, errors=errors)


@app.post("/api/tts")
def tts():
    data = request.get_json(force=True)
    text = re.sub(r"\s+", " ", str(data.get("text") or "")).strip()
    language_code = str(data.get("language") or "en").lower()
    if not text:
        return jsonify(error="No result to speak."), 400
    if language_code not in LANGUAGES:
        return jsonify(error="Unsupported speech language."), 400
    try:
        engine_name = "Piper"
        log(
            "TTS requested",
            engine=engine_name,
            language=LANGUAGES[language_code],
            input_characters=len(text),
        )
        clean_audio()
        spoken = translate_for_speech(text, language_code)
        filename = f"speech-{uuid.uuid4().hex}.wav"
        output_path = AUDIO / filename
        synthesize_piper(spoken, language_code, output_path)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"{engine_name} produced no audio file")
        log(
            "TTS ready",
            engine=engine_name,
            language=LANGUAGES[language_code],
            voice=PIPER_VOICE_NAMES.get(language_code),
            bytes=output_path.stat().st_size,
        )
        return jsonify(
            audio_url=f"/audio/{filename}",
            engine=engine_name,
            language=language_code,
            voice=PIPER_VOICE_NAMES.get(language_code),
        )
    except Exception as exc:
        log("TTS failed", language=language_code, error=repr(exc))
        return jsonify(error=str(exc)), 500


@app.get("/audio/<path:filename>")
def audio(filename: str):
    audio_path = AUDIO / filename
    log("Audio requested", filename=filename, exists=audio_path.exists(), bytes=(audio_path.stat().st_size if audio_path.exists() else 0))
    response = send_from_directory(
        AUDIO,
        filename,
        mimetype="audio/wav",
        conditional=True,
        as_attachment=False,
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.post("/api/live_mute")
def live_mute_route():
    data = request.get_json(force=True)
    requested = bool(data.get("muted", True))
    if live_stream is None:
        return jsonify(error="Live mode is not active.", active=False, muted=False), 409
    return jsonify(active=True, muted=set_live_muted(requested))


@app.post("/api/live_toggle")
def live_toggle():
    data = request.get_json(force=True)
    session_id = str(data.get("session_id") or uuid.uuid4())
    try:
        if live_stream is None:
            start_live(session_id, bool(data.get("use_web")), str(data.get("model_key") or DEFAULT_MODEL_KEY), data.get("max_web_results", MAX_WEB_RESULTS))
            active = True
        else:
            stop_live()
            active = False
        return jsonify(
            active=active,
            muted=(live_muted.is_set() if active else False),
            session_id=session_id,
        )
    except Exception as exc:
        log("LIVE toggle failed", error=repr(exc))
        return jsonify(error=str(exc)), 500


@app.get("/api/live_poll")
def live_poll():
    items = []
    while True:
        try:
            items.append(live_results.get_nowait())
        except queue.Empty:
            break
    return jsonify(
        active=live_stream is not None,
        muted=(live_muted.is_set() if live_stream is not None else False),
        items=items,
    )


@app.post("/api/reset")
def reset():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id") or "")
    if live_stream is not None:
        stop_live()
    with sessions_lock:
        sessions.pop(session_id, None)
    if session_id:
        shutil.rmtree(UPLOADS / secure_filename(session_id), ignore_errors=True)
        shutil.rmtree(MEDIA / secure_filename(session_id), ignore_errors=True)
    clean_audio(0)
    log("Session reset", session_id=session_id)
    return jsonify(ok=True)


# -----------------------------------------------------------------------------
# Browser UI
# -----------------------------------------------------------------------------
HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Steve's AI Tool</title>
<style>
:root{color-scheme:light;--blue:#0a66c2;--line:#d7d7dc;--muted:#6b6b73}
:root[data-theme="dark"]{color-scheme:dark;--line:#45464d;--muted:#aaaab3}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f4f5f7;color:#17171a}
.shell{max-width:1050px;margin:28px auto;padding:0 18px}.card{position:relative;background:#fff;border:1px solid #e5e5e8;border-radius:14px;box-shadow:0 8px 28px #0001;padding:22px}.theme-toggle{position:absolute;top:16px;right:16px;padding:5px 9px;font-size:12px;line-height:1.2}
h1{margin:0 0 4px;font-size:25px}.sub,.status,.files,.meta{color:var(--muted)}
.bar,.audio{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}.model-choice{display:flex;gap:10px;align-items:center;margin-top:12px}.model-choice label{font-size:13px;font-weight:650;color:var(--muted)}.model-choice select{min-width:310px}
.live-activity{display:none;margin:12px 0;padding:12px 14px;border:1px solid var(--line);border-radius:10px;background:#f8fafc}.live-activity.active{display:block}.live-label{font-size:12px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.live-transcript{margin-top:5px;white-space:pre-wrap;line-height:1.45}.live-stage{margin-top:8px;font-size:13px;color:var(--blue)}.live-dot{display:inline-block;width:7px;height:7px;margin-right:7px;border-radius:50%;background:var(--blue);animation:pulse 1.1s infinite}@keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
.search-panel{margin:18px 0;padding:16px;border:1px solid var(--line);border-radius:12px;background:#f8fafc}.search-panel h2{font-size:17px;margin:0 0 4px}.search-suggestions{margin:9px 0 4px;padding:11px;border:1px solid var(--line);border-radius:10px;background:#fff}.search-suggestions-title{font-size:12px;font-weight:700;color:var(--muted);margin-bottom:7px}.search-suggestion-list{display:grid;gap:7px}.search-suggestion{width:100%;text-align:left;padding:8px 10px;color:var(--blue)}.search-batch-tools{display:flex;justify-content:flex-end;margin:2px 0 8px}.search-batch-tools button{font-size:13px;padding:7px 11px}.search-suggestion-group{margin-bottom:10px}.search-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px 14px;margin-top:12px}[hidden]{display:none!important}.search-field{display:flex;flex-direction:column;gap:5px}.search-field.full{grid-column:1/-1}.search-field label{font-size:12px;font-weight:700;color:var(--muted)}.checkbox-line{display:flex;flex-direction:row;align-items:center;gap:7px;font-weight:500!important}.checkbox-line input{width:auto!important;margin:0;flex:0 0 auto}.search-field input,.search-field textarea{width:100%;border:1px solid var(--line);border-radius:8px;padding:9px;font:inherit}.search-field textarea{min-height:70px}.search-actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}.search-results{margin-top:13px;display:grid;gap:9px}.search-batch-heading{margin-top:8px;padding:8px 10px;border-radius:8px;background:#eef3f8;color:var(--muted);font-size:12px;font-weight:700}.search-result{padding:10px;border:1px solid var(--line);border-radius:9px}.search-result-head{display:flex;align-items:flex-start;gap:9px}.search-result-toggle{width:17px;height:17px;margin:2px 0 0;flex:0 0 auto}.search-result.excluded{opacity:.52;background:#f0f1f3}.search-result.excluded .search-highlight{text-decoration:none}.search-result a{font-weight:650;color:var(--blue)}.search-result-meta{display:flex;align-items:center;flex-wrap:wrap;gap:5px;margin-top:3px;color:var(--muted);font-size:12px}.search-result-meta .separator{opacity:.65}.search-highlights{margin:8px 0 0 26px}.search-highlights summary{cursor:pointer;color:var(--muted);font-size:12px;font-weight:650;user-select:none}.search-highlight{margin-top:5px;color:var(--muted);font-size:13px;line-height:1.4}.category-select{min-height:122px}.language-picker{position:relative}.language-picker summary{list-style:none;cursor:pointer;border:1px solid var(--line);background:#fff;border-radius:8px;padding:9px 34px 9px 10px;position:relative;user-select:none}.language-picker summary::-webkit-details-marker{display:none}.language-picker summary:after{content:"▾";position:absolute;right:11px;color:var(--muted)}.language-picker[open] summary:after{content:"▴"}.language-popout{position:absolute;z-index:20;top:calc(100% + 5px);left:0;width:min(430px,90vw);max-height:320px;overflow:auto;padding:12px;border:1px solid var(--line);border-radius:10px;background:#fff;box-shadow:0 12px 30px #0002}.language-toolbar{display:flex;gap:8px;margin-bottom:9px}.language-toolbar button{padding:5px 8px;font-size:12px}.language-options{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 10px}.language-option{display:flex;align-items:center;gap:7px;font-size:13px}.language-option input{width:auto!important;margin:0}.date-fields{display:flex;gap:8px}.date-fields>*{min-width:0}.search-ready{color:#18794e;font-weight:650}textarea{width:100%;min-height:120px;border:1px solid var(--line);border-radius:10px;padding:13px;font:inherit;resize:vertical}
button,.upload,select{border:1px solid var(--line);background:#fff;border-radius:9px;padding:9px 14px;font:inherit;cursor:pointer}
.primary{background:var(--blue);border-color:var(--blue);color:#fff}.danger{color:#b42318}button:disabled{opacity:.55;cursor:not-allowed}
.turn{border-top:1px solid #ececf0;padding-top:15px;margin-top:15px}.response-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
.response-tools{display:flex;gap:6px;align-items:center;flex-shrink:0}.response-tools select,.response-tools button{padding:5px 8px;font-size:12px}
.prompt{font-weight:650;margin-bottom:8px}.answer{line-height:1.58}.answer h1,.answer h2,.answer h3,.answer h4,.youtube-section-body h1,.youtube-section-body h2,.youtube-section-body h3,.youtube-section-body h4{margin:1.05em 0 .42em;line-height:1.25}.answer h1,.youtube-section-body h1{font-size:1.35rem}.answer h2,.youtube-section-body h2{font-size:1.2rem}.answer h3,.youtube-section-body h3{font-size:1.08rem}.answer p,.youtube-section-body p{margin:.55em 0}.answer ul,.answer ol,.youtube-section-body ul,.youtube-section-body ol{margin:.55em 0 .75em;padding-left:1.45em}.answer li,.youtube-section-body li{margin:.28em 0}.answer strong,.youtube-section-body strong{font-weight:700}.answer code,.youtube-section-body code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em;background:#0000000d;border-radius:4px;padding:.08em .3em}.answer pre,.youtube-section-body pre{overflow:auto;background:#111827;color:#f9fafb;border-radius:8px;padding:12px}.answer pre code,.youtube-section-body pre code{background:transparent;padding:0}.answer blockquote,.youtube-section-body blockquote{margin:.7em 0;padding:.15em 0 .15em 12px;border-left:3px solid var(--line);color:var(--muted)}.sources{font-size:13px;margin-top:10px}.sources a{color:var(--blue);word-break:break-word}
.reasoning-block{margin:10px 0;border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:#00000005}.reasoning-block summary{cursor:pointer;font-size:13px;font-weight:650;color:var(--muted)}.reasoning-content{margin-top:8px;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.45}.meta{display:flex;gap:14px;align-items:center;margin-top:9px;font-size:12px}.copy-link{border:0;background:transparent!important;padding:0;color:var(--blue);font-size:12px}.copy-link:hover{text-decoration:underline}
.youtube-sections{display:grid;gap:14px;margin-top:8px}.youtube-section{border:1px solid var(--line);border-radius:11px;padding:14px;background:#fbfcfe}.youtube-section-head{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:9px}.youtube-section-title{font-size:17px;font-weight:700}.youtube-section-tools{display:flex;gap:6px;align-items:center;flex-shrink:0}.youtube-section-tools select,.youtube-section-tools button{padding:5px 8px;font-size:12px}.youtube-section-body{line-height:1.58}
input[type=file]{display:none}audio{width:min(520px,100%)}
.youtube-modal-backdrop{position:fixed;inset:0;z-index:100;background:#0009;display:flex;align-items:center;justify-content:center;padding:18px}.youtube-modal{width:min(760px,100%);max-height:88vh;overflow:auto;background:#fff;border:1px solid var(--line);border-radius:14px;box-shadow:0 20px 60px #0005;padding:20px}.youtube-modal h2{margin:0 0 5px}.youtube-modal-list{display:grid;gap:14px;margin:16px 0}.youtube-segment-card{border:1px solid var(--line);border-radius:11px;padding:13px}.youtube-video-title{font-weight:700;margin-bottom:4px}.youtube-video-url-row{display:flex;gap:8px;align-items:center}.youtube-video-url{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-size:11px}.youtube-copy-url{padding:4px 7px;font-size:11px;flex:0 0 auto}.youtube-time-row{display:grid;grid-template-columns:1fr auto 1fr;gap:10px;align-items:center;margin-top:12px}.youtube-time-value{font-variant-numeric:tabular-nums;font-weight:700;text-align:center}.youtube-range{width:100%;padding:0;border:0}.youtube-modal-actions{display:flex;justify-content:flex-end;gap:9px}.youtube-modal-error{color:#b42318;font-size:13px;margin-top:8px}:root[data-theme="dark"] .youtube-modal{background:#1c1d21}
@media(max-width:700px){.search-grid{grid-template-columns:1fr}.search-field.full{grid-column:auto}.response-head{display:block}.response-tools{margin:5px 0 10px}.bar{align-items:flex-start}}
:root[data-theme="dark"] body{background:#111215;color:#f3f3f5}:root[data-theme="dark"] .card{background:#1c1d21;border-color:#34353b}:root[data-theme="dark"] textarea,:root[data-theme="dark"] button,:root[data-theme="dark"] .upload,:root[data-theme="dark"] select{background:#25262b;color:#f3f3f5;border-color:#45464d}:root[data-theme="dark"] .turn{border-color:#34353b}:root[data-theme="dark"] .live-activity,:root[data-theme="dark"] .search-panel{background:#23252a}:root[data-theme="dark"] .search-suggestions{background:#202227}:root[data-theme="dark"] .search-result.excluded{background:#191a1e}:root[data-theme="dark"] .search-batch-heading{background:#292c32}:root[data-theme="dark"] .youtube-section{background:#202227}:root[data-theme="dark"] .language-picker summary,:root[data-theme="dark"] .language-popout{background:#25262b;color:#f3f3f5;border-color:#45464d}
.system-health{position:sticky;top:0;z-index:30;display:flex;align-items:center;justify-content:center;gap:5px;min-height:24px;margin:-12px 54px 8px;padding:4px 8px;background:#fff;border-bottom:1px solid transparent;color:var(--muted);font-size:10.5px;line-height:1.25;letter-spacing:.01em;white-space:nowrap}.system-health .health-label{font-weight:600;color:inherit}.system-health .health-value{font-variant-numeric:tabular-nums}.system-health .health-separator{opacity:.55}.system-health[data-level="high"]{color:#a15c00}.system-health[data-level="critical"]{color:#b42318}:root[data-theme="dark"] .system-health[data-level="high"]{color:#f0b35b}:root[data-theme="dark"] .system-health{background:#1c1d21}:root[data-theme="dark"] .system-health[data-level="critical"]{color:#ff8178}@media(max-width:680px){.system-health{margin:-8px 46px 9px;white-space:normal;text-align:center;flex-wrap:wrap}}
</style>
</head>
<body><main class="shell"><section class="card">
<button id="themeToggle" class="theme-toggle" type="button" aria-label="Switch to dark mode" title="Switch to dark mode">Dark</button>
<div id="systemHealth" class="system-health" role="status" aria-live="polite" title="System-wide physical memory; updates every 10 seconds"><span class="health-label">Memory</span><span id="memorySummary" class="health-value">checking...</span></div>
<h1>Steve's AI Tool</h1>
<div class="sub">Local reasoning, direct article reading, web retrieval, YouTube and uploaded-media transcription, local files and images, live mode, and multilingual voice interaction.</div>
<section class="search-panel">
<h2>Exa Search</h2>
<div class="sub">Run as many searches as needed. Results accumulate below, and every selected article is available to the model for analysis.</div>
<div class="search-grid">
  <div class="search-field full"><label for="searchQuery">Search query</label><textarea id="searchQuery" placeholder="Describe the pages and evidence you want Exa to retrieve..."></textarea></div>
  <div class="search-field full"><label for="searchObjective">Search objective</label><textarea id="searchObjective" placeholder="Goal for this search: what should rank first, what should be excluded, and which facts should be extracted?"></textarea></div>
  <div class="search-field"><label for="searchType">Search type</label><select id="searchType"><option value="auto" selected>Auto (balanced)</option><option value="instant">Instant</option><option value="fast">Fast</option><option value="deep-lite">Deep Lite</option><option value="deep">Deep</option><option value="deep-reasoning">Deep Reasoning</option></select></div>
  <div class="search-field"><label for="userLocation">User location</label><select id="userLocation"><option value="" selected>Global / unspecified</option><optgroup label="Major English-speaking countries"><option value="AU">Australia</option><option value="CA">Canada</option><option value="IE">Ireland</option><option value="NZ">New Zealand</option><option value="SG">Singapore</option><option value="ZA">South Africa</option><option value="GB">Britain (UK)</option><option value="US">United States</option></optgroup><optgroup label="All countries and territories"><option value="AF">Afghanistan</option><option value="AL">Albania</option><option value="DZ">Algeria</option><option value="AD">Andorra</option><option value="AO">Angola</option><option value="AI">Anguilla</option><option value="AQ">Antarctica</option><option value="AG">Antigua & Barbuda</option><option value="AR">Argentina</option><option value="AM">Armenia</option><option value="AW">Aruba</option><option value="AT">Austria</option><option value="AZ">Azerbaijan</option><option value="BS">Bahamas</option><option value="BH">Bahrain</option><option value="BD">Bangladesh</option><option value="BB">Barbados</option><option value="BY">Belarus</option><option value="BE">Belgium</option><option value="BZ">Belize</option><option value="BJ">Benin</option><option value="BM">Bermuda</option><option value="BT">Bhutan</option><option value="BO">Bolivia</option><option value="BA">Bosnia & Herzegovina</option><option value="BW">Botswana</option><option value="BV">Bouvet Island</option><option value="BR">Brazil</option><option value="IO">British Indian Ocean Territory</option><option value="BN">Brunei</option><option value="BG">Bulgaria</option><option value="BF">Burkina Faso</option><option value="BI">Burundi</option><option value="KH">Cambodia</option><option value="CM">Cameroon</option><option value="CV">Cape Verde</option><option value="BQ">Caribbean NL</option><option value="KY">Cayman Islands</option><option value="CF">Central African Rep.</option><option value="TD">Chad</option><option value="CL">Chile</option><option value="CN">China</option><option value="CX">Christmas Island</option><option value="CC">Cocos (Keeling) Islands</option><option value="CO">Colombia</option><option value="KM">Comoros</option><option value="CD">Congo (Dem. Rep.)</option><option value="CG">Congo (Rep.)</option><option value="CK">Cook Islands</option><option value="CR">Costa Rica</option><option value="HR">Croatia</option><option value="CU">Cuba</option><option value="CW">Curaçao</option><option value="CY">Cyprus</option><option value="CZ">Czech Republic</option><option value="CI">Côte d’Ivoire</option><option value="DK">Denmark</option><option value="DJ">Djibouti</option><option value="DM">Dominica</option><option value="DO">Dominican Republic</option><option value="TL">East Timor</option><option value="EC">Ecuador</option><option value="EG">Egypt</option><option value="SV">El Salvador</option><option value="GQ">Equatorial Guinea</option><option value="ER">Eritrea</option><option value="EE">Estonia</option><option value="SZ">Eswatini (Swaziland)</option><option value="ET">Ethiopia</option><option value="FK">Falkland Islands</option><option value="FO">Faroe Islands</option><option value="FJ">Fiji</option><option value="FI">Finland</option><option value="FR">France</option><option value="GF">French Guiana</option><option value="PF">French Polynesia</option><option value="TF">French S. Terr.</option><option value="GA">Gabon</option><option value="GM">Gambia</option><option value="GE">Georgia</option><option value="DE">Germany</option><option value="GH">Ghana</option><option value="GI">Gibraltar</option><option value="GR">Greece</option><option value="GL">Greenland</option><option value="GD">Grenada</option><option value="GP">Guadeloupe</option><option value="GU">Guam</option><option value="GT">Guatemala</option><option value="GG">Guernsey</option><option value="GN">Guinea</option><option value="GW">Guinea-Bissau</option><option value="GY">Guyana</option><option value="HT">Haiti</option><option value="HM">Heard Island & McDonald Islands</option><option value="HN">Honduras</option><option value="HK">Hong Kong</option><option value="HU">Hungary</option><option value="IS">Iceland</option><option value="IN">India</option><option value="ID">Indonesia</option><option value="IR">Iran</option><option value="IQ">Iraq</option><option value="IM">Isle of Man</option><option value="IL">Israel</option><option value="IT">Italy</option><option value="JM">Jamaica</option><option value="JP">Japan</option><option value="JE">Jersey</option><option value="JO">Jordan</option><option value="KZ">Kazakhstan</option><option value="KE">Kenya</option><option value="KI">Kiribati</option><option value="KP">Korea (North)</option><option value="KR">Korea (South)</option><option value="KW">Kuwait</option><option value="KG">Kyrgyzstan</option><option value="LA">Laos</option><option value="LV">Latvia</option><option value="LB">Lebanon</option><option value="LS">Lesotho</option><option value="LR">Liberia</option><option value="LY">Libya</option><option value="LI">Liechtenstein</option><option value="LT">Lithuania</option><option value="LU">Luxembourg</option><option value="MO">Macau</option><option value="MG">Madagascar</option><option value="MW">Malawi</option><option value="MY">Malaysia</option><option value="MV">Maldives</option><option value="ML">Mali</option><option value="MT">Malta</option><option value="MH">Marshall Islands</option><option value="MQ">Martinique</option><option value="MR">Mauritania</option><option value="MU">Mauritius</option><option value="YT">Mayotte</option><option value="MX">Mexico</option><option value="FM">Micronesia</option><option value="MD">Moldova</option><option value="MC">Monaco</option><option value="MN">Mongolia</option><option value="ME">Montenegro</option><option value="MS">Montserrat</option><option value="MA">Morocco</option><option value="MZ">Mozambique</option><option value="MM">Myanmar (Burma)</option><option value="NA">Namibia</option><option value="NR">Nauru</option><option value="NP">Nepal</option><option value="NL">Netherlands</option><option value="NC">New Caledonia</option><option value="NI">Nicaragua</option><option value="NE">Niger</option><option value="NG">Nigeria</option><option value="NU">Niue</option><option value="NF">Norfolk Island</option><option value="MK">North Macedonia</option><option value="MP">Northern Mariana Islands</option><option value="NO">Norway</option><option value="OM">Oman</option><option value="PK">Pakistan</option><option value="PW">Palau</option><option value="PS">Palestine</option><option value="PA">Panama</option><option value="PG">Papua New Guinea</option><option value="PY">Paraguay</option><option value="PE">Peru</option><option value="PH">Philippines</option><option value="PN">Pitcairn</option><option value="PL">Poland</option><option value="PT">Portugal</option><option value="PR">Puerto Rico</option><option value="QA">Qatar</option><option value="RO">Romania</option><option value="RU">Russia</option><option value="RW">Rwanda</option><option value="RE">Réunion</option><option value="AS">Samoa (American)</option><option value="WS">Samoa (western)</option><option value="SM">San Marino</option><option value="ST">Sao Tome & Principe</option><option value="SA">Saudi Arabia</option><option value="SN">Senegal</option><option value="RS">Serbia</option><option value="SC">Seychelles</option><option value="SL">Sierra Leone</option><option value="SK">Slovakia</option><option value="SI">Slovenia</option><option value="SB">Solomon Islands</option><option value="SO">Somalia</option><option value="GS">South Georgia & the South Sandwich Islands</option><option value="SS">South Sudan</option><option value="ES">Spain</option><option value="LK">Sri Lanka</option><option value="BL">St Barthelemy</option><option value="SH">St Helena</option><option value="KN">St Kitts & Nevis</option><option value="LC">St Lucia</option><option value="SX">St Maarten (Dutch)</option><option value="MF">St Martin (French)</option><option value="PM">St Pierre & Miquelon</option><option value="VC">St Vincent</option><option value="SD">Sudan</option><option value="SR">Suriname</option><option value="SJ">Svalbard & Jan Mayen</option><option value="SE">Sweden</option><option value="CH">Switzerland</option><option value="SY">Syria</option><option value="TW">Taiwan</option><option value="TJ">Tajikistan</option><option value="TZ">Tanzania</option><option value="TH">Thailand</option><option value="TG">Togo</option><option value="TK">Tokelau</option><option value="TO">Tonga</option><option value="TT">Trinidad & Tobago</option><option value="TN">Tunisia</option><option value="TR">Turkey</option><option value="TM">Turkmenistan</option><option value="TC">Turks & Caicos Is</option><option value="TV">Tuvalu</option><option value="UG">Uganda</option><option value="UA">Ukraine</option><option value="AE">United Arab Emirates</option><option value="UY">Uruguay</option><option value="UM">US minor outlying islands</option><option value="UZ">Uzbekistan</option><option value="VU">Vanuatu</option><option value="VA">Vatican City</option><option value="VE">Venezuela</option><option value="VN">Vietnam</option><option value="VG">Virgin Islands (UK)</option><option value="VI">Virgin Islands (US)</option><option value="WF">Wallis & Futuna</option><option value="EH">Western Sahara</option><option value="YE">Yemen</option><option value="ZM">Zambia</option><option value="ZW">Zimbabwe</option><option value="AX">Åland Islands</option></optgroup></select></div>
  <div class="search-field"><label for="searchCategories">Categories (Cmd/Ctrl-click for multiple)</label><select id="searchCategories" class="category-select" multiple><option value="company">Company</option><option value="publication">Publication</option><option value="news">News</option><option value="personal site">Personal site</option><option value="financial report">Financial report</option><option value="people">People</option></select></div>
  <div class="search-field"><label>Content</label><label class="checkbox-line" for="contentModeration"><input type="checkbox" id="contentModeration"><span>Enable content moderation</span></label><label class="checkbox-line" for="collapseHighlights"><input type="checkbox" id="collapseHighlights" checked><span>Collapse result highlights</span></label><details class="language-picker" id="languagePicker"><summary id="languageSummary">Languages: English</summary><div class="language-popout"><div class="language-toolbar"><button type="button" id="englishOnly">English only</button><button type="button" id="clearLanguages">Clear</button></div><div class="language-options"><label class="language-option"><input type="checkbox" class="search-language" value="ar"><span>Arabic</span></label><label class="language-option"><input type="checkbox" class="search-language" value="bg"><span>Bulgarian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="ca"><span>Catalan</span></label><label class="language-option"><input type="checkbox" class="search-language" value="zh"><span>Chinese</span></label><label class="language-option"><input type="checkbox" class="search-language" value="hr"><span>Croatian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="cs"><span>Czech</span></label><label class="language-option"><input type="checkbox" class="search-language" value="da"><span>Danish</span></label><label class="language-option"><input type="checkbox" class="search-language" value="nl"><span>Dutch</span></label><label class="language-option"><input type="checkbox" class="search-language" value="en" checked><span>English</span></label><label class="language-option"><input type="checkbox" class="search-language" value="et"><span>Estonian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="fi"><span>Finnish</span></label><label class="language-option"><input type="checkbox" class="search-language" value="fr"><span>French</span></label><label class="language-option"><input type="checkbox" class="search-language" value="de"><span>German</span></label><label class="language-option"><input type="checkbox" class="search-language" value="el"><span>Greek</span></label><label class="language-option"><input type="checkbox" class="search-language" value="he"><span>Hebrew</span></label><label class="language-option"><input type="checkbox" class="search-language" value="hi"><span>Hindi</span></label><label class="language-option"><input type="checkbox" class="search-language" value="hu"><span>Hungarian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="id"><span>Indonesian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="it"><span>Italian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="ja"><span>Japanese</span></label><label class="language-option"><input type="checkbox" class="search-language" value="ko"><span>Korean</span></label><label class="language-option"><input type="checkbox" class="search-language" value="lv"><span>Latvian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="lt"><span>Lithuanian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="ms"><span>Malay</span></label><label class="language-option"><input type="checkbox" class="search-language" value="no"><span>Norwegian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="fa"><span>Persian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="pl"><span>Polish</span></label><label class="language-option"><input type="checkbox" class="search-language" value="pt"><span>Portuguese</span></label><label class="language-option"><input type="checkbox" class="search-language" value="ro"><span>Romanian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="ru"><span>Russian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="sr"><span>Serbian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="sk"><span>Slovak</span></label><label class="language-option"><input type="checkbox" class="search-language" value="sl"><span>Slovenian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="es"><span>Spanish</span></label><label class="language-option"><input type="checkbox" class="search-language" value="sv"><span>Swedish</span></label><label class="language-option"><input type="checkbox" class="search-language" value="th"><span>Thai</span></label><label class="language-option"><input type="checkbox" class="search-language" value="tr"><span>Turkish</span></label><label class="language-option"><input type="checkbox" class="search-language" value="uk"><span>Ukrainian</span></label><label class="language-option"><input type="checkbox" class="search-language" value="ur"><span>Urdu</span></label><label class="language-option"><input type="checkbox" class="search-language" value="vi"><span>Vietnamese</span></label></div></div></details></div>
  <div class="search-field"><label for="includeDomains">Include websites</label><input id="includeDomains" placeholder="example.com, *.co.nz"></div>
  <div class="search-field"><label for="excludeDomains">Exclude websites</label><input id="excludeDomains" placeholder="example.net, social.example"></div>
  <div class="search-field"><label for="freshnessMode">Freshness / timeframe</label><select id="freshnessMode"><option value="cached" selected>Cached results only (fastest)</option><option value="max_age">Articles published within...</option><option value="trawl">Fresh trawl (live crawl)</option><option value="date_range">Publication date range</option></select></div>
  <div class="search-field" id="maxAgeField" hidden><label for="maxAgeHours">Maximum article age in hours</label><input id="maxAgeHours" type="number" min="0" step="1" value="24"></div>
  <div class="search-field" id="dateRangeField" hidden><label>Publication dates</label><div class="date-fields"><input id="startPublishedDate" type="date" aria-label="Start publication date"><input id="endPublishedDate" type="date" aria-label="End publication date"></div></div>
  <div class="search-field"><label for="maxArticles">Maximum results: <span id="maxArticlesValue">8</span></label><input type="range" id="maxArticles" min="8" max="80" step="8" value="8"></div>
</div>
<div class="search-actions"><button class="primary" id="runSearch">Search</button><label><input type="checkbox" id="searchHighlights" checked> Retrieve highlights</label><label class="checkbox-line" id="selectAllResultsWrap" hidden><input type="checkbox" id="selectAllResults" checked><span>Select all results</span></label><span class="status" id="searchStatus">No search run.</span></div>
<div class="search-results" id="searchResults"></div>
</section>
<textarea id="prompt" placeholder="Paste an article URL for an English summary, or ask the model to analyse search results, attached files, or another topic..."></textarea>
<div class="model-choice"><label for="modelSelect">Model for this prompt</label><select id="modelSelect"><option value="default" selected>Default — Qwen3 14B</option><option value="reasoning">DeepSeek Logical Reasoning</option><option value="summary">Summarization — Qwen3.5 9B</option><option value="quick">Quick Summarization — Qwen3.5 4B</option><option value="gemma">Multimodal analysis &amp; reasoning — Gemma 4 12B</option><option value="qwen3_vl">Screenshots, OCR, charts &amp; documents — Qwen3-VL 8B</option></select></div>
<div class="bar">
<button class="primary" id="send">Run prompt</button>
<button id="live" title="Speak a prompt through the Mac microphone">🎙 Start live mode</button>
<button id="micMute" disabled title="Mute or unmute the microphone during live mode">🔇 Mute mic</button>
<label title="Suppress the completion beep"><input type="checkbox" id="silent"> Silent mode</label>
<label class="upload" for="files">Attach up to 15 files/images</label><input id="files" type="file" multiple accept=".txt,.md,.csv,.json,.py,.log,.html,.xml,.pdf,.docx,.xlsx,.jpg,.jpeg,.png,.webp,.bmp,.gif,.tif,.tiff,image/*">
<label class="upload" for="mediaFile">Attach audio/video</label><input id="mediaFile" type="file" accept="audio/*,video/*,.mkv,.m4a,.opus,.flac">
<button class="danger" id="reset">Reset</button><span class="status" id="status">Ready</span>
</div>
<div class="live-activity" id="liveActivity" aria-live="polite">
  <div class="live-label">Live transcript</div>
  <div class="live-transcript" id="liveTranscript">Waiting for speech...</div>
  <div class="live-stage" id="liveStage"><span class="live-dot"></span><span id="liveStageText">Listening...</span></div>
</div>
<div class="files" id="fileList">No documents or images attached.</div>
<div class="files" id="mediaStatus">No audio or video attached.</div>
<div class="audio"><label>Playback speed <select id="speed"><option value="0.75">0.75x</option><option value="1" selected>1.0x</option><option value="1.25">1.25x</option><option value="1.5">1.5x</option><option value="2">2.0x</option></select></label><audio id="player" controls></audio></div>
<div id="history"></div>
<div id="youtubeModalBackdrop" class="youtube-modal-backdrop" hidden>
  <section class="youtube-modal" role="dialog" aria-modal="true" aria-labelledby="youtubeModalTitle">
    <h2 id="youtubeModalTitle">Select YouTube transcription segments</h2>
    <div class="sub">Choose the start and finish for each video. The full duration is selected by default.</div>
    <div id="youtubeModalList" class="youtube-modal-list"></div>
    <div id="youtubeModalError" class="youtube-modal-error" aria-live="polite"></div>
    <div class="youtube-modal-actions"><button type="button" id="youtubeCancel">Cancel</button><button type="button" class="primary" id="youtubeContinue">Transcribe selected segments</button></div>
  </section>
</div>
<div id="mediaModalBackdrop" class="youtube-modal-backdrop" hidden>
  <section class="youtube-modal" role="dialog" aria-modal="true" aria-labelledby="mediaModalTitle">
    <h2 id="mediaModalTitle">Select audio or video transcription segment</h2>
    <div class="sub">Choose the start and finish. The full duration is selected by default.</div>
    <div id="mediaModalList" class="youtube-modal-list"></div>
    <div id="mediaModalError" class="youtube-modal-error" aria-live="polite"></div>
    <div class="youtube-modal-actions"><button type="button" id="mediaCancel">Cancel</button><button type="button" class="primary" id="mediaContinue">Transcribe selected segment</button></div>
  </section>
</div>
</section></main>
<script>
let sid=crypto.randomUUID(),latest='',searchSuggestionGroups={},uploadedMedia=null;
const $=id=>document.getElementById(id);
function formatMemory(bytes){
  const gib=Number(bytes)/(1024**3);
  return `${gib.toFixed(gib>=100?0:1)} GB`;
}
async function updateSystemHealth(){
  const health=$('systemHealth');
  const summary=$('memorySummary');
  try{
    const response=await fetch('/api/system_health',{cache:'no-store'});
    const data=await response.json();
    if(!response.ok)throw Error(data.error||'Memory status unavailable');
    summary.textContent=`${formatMemory(data.used)} used · ${formatMemory(data.available)} remaining · ${formatMemory(data.total)} total · ${Number(data.percent).toFixed(1)}% consumed`;
    health.dataset.level=data.percent>=90?'critical':(data.percent>=80?'high':'normal');
    health.title=`System-wide physical memory. Last updated ${new Date().toLocaleTimeString()}. Updates every 10 seconds.`;
  }catch(error){
    summary.textContent='status unavailable';
    health.dataset.level='unknown';
  }
}
updateSystemHealth();
setInterval(updateSystemHealth,10000);
function applyTheme(theme){
  const selected=theme==='dark'?'dark':'light';
  document.documentElement.dataset.theme=selected;
  const button=$('themeToggle');
  if(button){
    const dark=selected==='dark';
    button.textContent=dark?'Light':'Dark';
    button.setAttribute('aria-label',dark?'Switch to light mode':'Switch to dark mode');
    button.title=dark?'Switch to light mode':'Switch to dark mode';
  }
  localStorage.setItem('theme',selected);
}
applyTheme(localStorage.getItem('theme')||'light');
$('themeToggle').onclick=()=>applyTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function inlineMarkdown(text){
  let value=esc(text);
  value=value.replace(/`([^`]+)`/g,'<code>$1</code>');
  value=value.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  value=value.replace(/__([^_]+)__/g,'<strong>$1</strong>');
  value=value.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g,'$1<em>$2</em>');
  return value;
}
function renderMarkdown(markdown){
  const lines=String(markdown||'').replace(/\r\n/g,'\n').split('\n');
  let html='',paragraph=[],listType=null,code=false,codeLines=[];
  const flushParagraph=()=>{if(paragraph.length){html+='<p>'+inlineMarkdown(paragraph.join(' '))+'</p>';paragraph=[]}};
  const closeList=()=>{if(listType){html+=`</${listType}>`;listType=null}};
  const flushCode=()=>{if(codeLines.length){html+='<pre><code>'+esc(codeLines.join('\n'))+'</code></pre>';codeLines=[]}};
  for(const raw of lines){
    if(/^\s*```/.test(raw)){
      if(code){flushCode();code=false}else{flushParagraph();closeList();code=true}
      continue;
    }
    if(code){codeLines.push(raw);continue}
    const heading=raw.match(/^\s*(#{1,4})\s+(.+?)\s*$/);
    if(heading){flushParagraph();closeList();const level=heading[1].length;html+=`<h${level}>${inlineMarkdown(heading[2].replace(/\s+#+\s*$/,''))}</h${level}>`;continue}
    const unordered=raw.match(/^\s*[-*•]\s+(.+)$/);
    const ordered=raw.match(/^\s*(\d+)[.)]\s+(.+)$/);
    if(unordered||ordered){
      flushParagraph();
      const wanted=unordered?'ul':'ol';
      if(listType!==wanted){
        closeList();
        listType=wanted;
        if(ordered){const startNumber=Number(ordered[1]);html+=startNumber===1?'<ol>':`<ol start="${startNumber}">`}
        else html+='<ul>';
      }
      html+='<li>'+inlineMarkdown(unordered?unordered[1]:ordered[2])+'</li>';
      continue;
    }
    if(/^\s*$/.test(raw)){
      // A blank line between Markdown list items is valid and must not close
      // the list. Look ahead to determine whether the next content line is
      // another item of the same type.
      flushParagraph();
      continue;
    }
    const quote=raw.match(/^\s*>\s?(.*)$/);
    if(quote){flushParagraph();closeList();html+='<blockquote>'+inlineMarkdown(quote[1])+'</blockquote>';continue}
    closeList();paragraph.push(raw.trim());
  }
  if(code){flushCode()}
  flushParagraph();closeList();
  return html;
}
const langOptions='<option value="en">English</option><option value="fr">French</option><option value="es">Spanish</option><option value="de">German</option><option value="it">Italian</option><option value="nl">Dutch</option><option value="pt">Portuguese</option>';
function busy(on,text){$('send').disabled=on;$('runSearch').disabled=on;document.querySelectorAll('.suggest-searches-button').forEach(button=>button.disabled=on);$('status').textContent=text||'Ready'}
function showLiveActivity(transcript,message,working=true){$('liveActivity').classList.add('active');if(transcript!==null&&transcript!==undefined)$('liveTranscript').textContent=transcript;$('liveStageText').textContent=message||'';const dot=$('liveActivity').querySelector('.live-dot');dot.style.display=working?'inline-block':'none';$('status').textContent=message||$('status').textContent}
function resetLiveActivity(){ $('liveActivity').classList.remove('active');$('liveTranscript').textContent='Waiting for speech...';$('liveStageText').textContent='Listening...' }
function beep(){if($('silent').checked)return;try{const C=window.AudioContext||window.webkitAudioContext,a=new C(),o=a.createOscillator(),g=a.createGain();o.type='sine';o.frequency.value=660;g.gain.setValueAtTime(.0001,a.currentTime);g.gain.exponentialRampToValueAtTime(.035,a.currentTime+.012);g.gain.exponentialRampToValueAtTime(.0001,a.currentTime+.13);o.connect(g);g.connect(a.destination);o.start();o.stop(a.currentTime+.14);o.onended=()=>a.close()}catch(e){}}
function buildCopyPayload(turn){
  const youtubeSections=turn.querySelector('.youtube-sections');
  let html='';
  if(youtubeSections){
    for(const section of youtubeSections.querySelectorAll('.youtube-section')){
      const title=section.querySelector('.youtube-section-title')?.textContent?.trim()||'';
      const body=section.querySelector('.youtube-section-body');
      if(title)html+=`<h2>${esc(title)}</h2>`;
      if(body)html+=body.innerHTML;
    }
  }else{
    const visibleAnswer=[...turn.querySelectorAll('.answer')].find(node=>!node.hidden);
    if(visibleAnswer)html=visibleAnswer.innerHTML;
  }
  const wrapper=document.createElement('div');
  wrapper.innerHTML=html;
  const plainText=wrapper.innerText||wrapper.textContent||'';
  return {html:`<div>${html}</div>`,plainText:plainText.trim()};
}
async function legacyCopyHtml(html,plainText){
  const container=document.createElement('div');
  container.contentEditable='true';
  container.style.position='fixed';
  container.style.left='-9999px';
  container.style.top='0';
  container.innerHTML=html;
  document.body.appendChild(container);
  const range=document.createRange();range.selectNodeContents(container);
  const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);
  const copied=document.execCommand('copy');
  selection.removeAllRanges();container.remove();
  if(!copied){await navigator.clipboard.writeText(plainText)}
}
async function copyResponse(button){
  const {html,plainText}=buildCopyPayload(button.closest('.turn'));
  const old=button.textContent;
  try{
    if(navigator.clipboard&&window.ClipboardItem){
      const item=new ClipboardItem({
        'text/html':new Blob([html],{type:'text/html'}),
        'text/plain':new Blob([plainText],{type:'text/plain'})
      });
      await navigator.clipboard.write([item]);
    }else{
      await legacyCopyHtml(html,plainText);
    }
    button.textContent='Copied with formatting';
  }catch(error){
    console.warn('Rich clipboard copy failed; using formatted selection fallback.',error);
    try{await legacyCopyHtml(html,plainText);button.textContent='Copied with formatting'}
    catch(fallbackError){console.error('Copy failed',fallbackError);button.textContent='Copy failed'}
  }
  setTimeout(()=>button.textContent=old,1600);
}
function parseYoutubeSections(answer){
  const normalized=String(answer).replace(/\r\n/g,'\n');
  const lines=normalized.split('\n');
  const sections={'Executive Summary':[],'Key Points':[],'Conclusion':[]};
  let current=null;
  for(const originalLine of lines){
    const heading=originalLine
      .trim()
      .replace(/^#{1,6}\s*/, '')
      .replace(/^\*\*(.*?)\*\*\s*:??\s*$/, '$1')
      .replace(/^__(.*?)__\s*:??\s*$/, '$1')
      .replace(/:\s*$/, '')
      .trim()
      .toLowerCase();
    if(heading==='executive summary'){current='Executive Summary';continue}
    if(heading==='key points'){current='Key Points';continue}
    if(heading==='conclusion'||heading==='conclusions'){current='Conclusion';continue}
    if(current!==null)sections[current].push(originalLine);
  }
  for(const key of Object.keys(sections))sections[key]=sections[key].join('\n').trim();
  return sections['Executive Summary']&&sections['Key Points']&&sections['Conclusion']?sections:null;
}
function youtubeSection(title,body){return `<section class="youtube-section"><div class="youtube-section-head"><div class="youtube-section-title">${esc(title)}</div><div class="youtube-section-tools"><select class="read-lang" aria-label="Read ${esc(title)} aloud language">${langOptions}</select><button onclick="readSection(this)">🔊 Read aloud</button></div></div><div class="youtube-section-body">${renderMarkdown(body)}</div></section>`}
function addTurn(prompt,answer,sources,elapsed,responseType='standard',reasoning='',modelLabel=''){
  latest=answer;
  const src=(sources||[]).map((s,i)=>s.url?`<div><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title||('Source '+(i+1)))}</a></div>`:`<div>${esc(s.title||('Source '+(i+1)))}</div>`).join('');
  const sections=responseType==='youtube_summary'?parseYoutubeSections(answer):null;
  const responseHtml=sections?`<div class="youtube-sections">${youtubeSection('Executive Summary',sections['Executive Summary'])}${youtubeSection('Key Points',sections['Key Points'])}${youtubeSection('Conclusion',sections['Conclusion'])}</div>`:`<div class="response-head"><div class="prompt">${esc(prompt)}</div><div class="response-tools"><select class="read-lang" aria-label="Read aloud language">${langOptions}</select><button onclick="readTurn(this)">🔊 Read aloud</button></div></div><div class="answer">${renderMarkdown(answer)}</div>`;
  const promptHtml=sections?`<div class="prompt">${esc(prompt)}</div>`:'';
  const reasoningHtml=reasoning?`<details class="reasoning-block"><summary>Show reasoning — ${esc(modelLabel||'selected model')}</summary><div class="reasoning-content">${esc(reasoning)}</div></details>`:'';
  $('history').insertAdjacentHTML('afterbegin',`<article class="turn" data-response-type="${esc(responseType)}">${promptHtml}${reasoningHtml}${responseHtml}<div class="answer" hidden>${esc(answer)}</div><div class="sources">${src}</div><div class="meta"><span>Completed in ${Number(elapsed).toFixed(2)}s</span><button class="copy-link" onclick="copyResponse(this)">Copy response</button></div></article>`)
}
async function setMicMuted(muted){if(!$('live').textContent.includes('Stop'))return false;const r=await fetch('/api/live_mute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({muted})});const d=await r.json();if(!r.ok)throw Error(d.error||'Microphone mute failed');$('micMute').disabled=!d.active;$('micMute').textContent=d.muted?'🎤 Unmute mic':'🔇 Mute mic';$('status').textContent=d.muted?'Live mode active — microphone muted':'Listening for a spoken prompt...';showLiveActivity(null,d.muted?'Microphone muted.':'Listening for a spoken prompt...',true);return d.muted}
async function loadAudioIntoPlayer(audioUrl){
  const player=$('player');
  const baseUrl=audioUrl+(audioUrl.includes('?')?'&':'?')+'v='+Date.now();
  const loadAttempt=attempt=>new Promise((resolve,reject)=>{
    let done=false;
    const cleanup=()=>{player.removeEventListener('loadedmetadata',ready);player.removeEventListener('loadeddata',ready);player.removeEventListener('canplay',ready);player.removeEventListener('error',failed)};
    const ready=()=>{if(done)return;done=true;cleanup();resolve()};
    const failed=()=>{if(done)return;done=true;cleanup();reject(Error('Safari could not load the generated WAV file.'))};
    player.addEventListener('loadedmetadata',ready);player.addEventListener('loadeddata',ready);player.addEventListener('canplay',ready);player.addEventListener('error',failed);
    player.pause();player.removeAttribute('src');player.load();
    player.src=baseUrl+'&attempt='+attempt;player.playbackRate=Number($('speed').value);player.load();
    if(player.readyState>=1)ready();
    setTimeout(()=>{if(!done){done=true;cleanup();reject(Error('Timed out loading generated audio.'))}},20000);
  });
  try{await loadAttempt(1)}catch(firstError){console.warn('Initial Safari audio load failed; retrying automatically.',firstError);await new Promise(resolve=>setTimeout(resolve,350));await loadAttempt(2)}
}
async function playCachedAudio(button){
  if(!button.dataset.audioUrl)return false;
  try{
    if(button.dataset.loaded!=='true'){await loadAudioIntoPlayer(button.dataset.audioUrl);button.dataset.loaded='true'}
    $('player').playbackRate=Number($('speed').value);await $('player').play();
    button.textContent='⏸ Playing';$('player').onended=()=>button.textContent='▶ Play audio';$('status').textContent='Playing generated audio.';return true;
  }catch(error){console.error('Audio playback failed',error);button.textContent='▶ Play audio';$('status').textContent='Playback failed: '+error.message;return false}
}
async function generateAudio(button,text,language){
  button.disabled=true;button.textContent='Generating...';
  try{
    if($('live').textContent.includes('Stop')&&!$('micMute').textContent.includes('Unmute'))await setMicMuted(true);
    const response=await fetch('/api/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,language})});
    const data=await response.json();if(!response.ok)throw Error(data.error||'Speech generation failed.');
    button.dataset.audioUrl=data.audio_url;button.dataset.loaded='false';
    await loadAudioIntoPlayer(data.audio_url);button.dataset.loaded='true';
    try{await $('player').play();button.textContent='⏸ Playing';$('player').onended=()=>button.textContent='▶ Play audio';$('status').textContent='Playing generated audio.'}
    catch(error){console.warn('Safari blocked initial autoplay',error);button.textContent='▶ Play audio';$('status').textContent='Audio generated and ready. Click Play audio.'}
  }catch(error){console.error('Read aloud failed',error);button.textContent='Error';$('status').textContent='Read aloud error: '+error.message}
  finally{button.disabled=false}
}
async function readSection(button){
  if(button.dataset.audioUrl){await playCachedAudio(button);return}
  const section=button.closest('.youtube-section');
  await generateAudio(button,section.querySelector('.youtube-section-body').textContent,section.querySelector('.read-lang').value);
}
async function readTurn(button){
  if(button.dataset.audioUrl){await playCachedAudio(button);return}
  const turn=button.closest('.turn');
  await generateAudio(button,turn.querySelector('.answer').textContent,turn.querySelector('.read-lang').value);
}
document.addEventListener('change',event=>{if(!event.target.classList.contains('read-lang'))return;const tools=event.target.closest('.youtube-section-tools,.response-tools');const button=tools&&tools.querySelector('button');if(button){delete button.dataset.audioUrl;delete button.dataset.loaded;button.textContent='🔊 Read aloud'}});
function updateFreshnessFields(){const mode=$('freshnessMode').value;const maxAge=$('maxAgeField'),dateRange=$('dateRangeField');maxAge.hidden=mode!=='max_age';dateRange.hidden=mode!=='date_range';maxAge.style.display=mode==='max_age'?'flex':'none';dateRange.style.display=mode==='date_range'?'flex':'none'}
$('freshnessMode').onchange=updateFreshnessFields;updateFreshnessFields();
function selectedSearchResultIndices(){return Array.from(document.querySelectorAll('.search-result-toggle:checked')).map(box=>Number(box.dataset.index))}
function selectedSearchResultUrls(){return Array.from(document.querySelectorAll('.search-result-toggle:checked')).map(box=>box.dataset.url).filter(Boolean)}
function refreshSearchStatus(){const summary=$('searchStatus').dataset.summary||'';const selection=$('searchStatus').dataset.selection||'';$('searchStatus').innerHTML=`${summary}${selection?` · <strong>${selection}</strong>`:''}`}
function updateSearchSelectionStatus(){
  const boxes=Array.from(document.querySelectorAll('.search-result-toggle'));
  const selected=boxes.filter(box=>box.checked).length;
  boxes.forEach(box=>box.closest('.search-result').classList.toggle('excluded',!box.checked));
  const toggle=$('selectAllResults');
  toggle.checked=boxes.length>0&&selected===boxes.length;
  toggle.indeterminate=selected>0&&selected<boxes.length;
  if(boxes.length)$('searchStatus').dataset.selection=`${selected} of ${boxes.length} results included`;
}
function setAllSearchResults(included){document.querySelectorAll('.search-result-toggle').forEach(box=>box.checked=included);updateSearchSelectionStatus();refreshSearchStatus()}
$('selectAllResults').onchange=event=>setAllSearchResults(event.target.checked);
function applyHighlightsDisplayMode(){
  const collapse=$('collapseHighlights').checked;
  document.querySelectorAll('.search-highlights').forEach(details=>details.open=!collapse);
}
$('collapseHighlights').onchange=applyHighlightsDisplayMode;
function selectedSearchLanguages(){return Array.from(document.querySelectorAll('.search-language:checked')).map(box=>box.value)}
function updateLanguageSummary(){const selected=Array.from(document.querySelectorAll('.search-language:checked'));const names=selected.map(box=>box.nextElementSibling.textContent);$('languageSummary').textContent=!names.length?'Languages: none selected':names.length<=3?'Languages: '+names.join(', '):`Languages: ${names.length} selected`;}
function setSearchLanguages(codes){const wanted=new Set(codes);document.querySelectorAll('.search-language').forEach(box=>box.checked=wanted.has(box.value));updateLanguageSummary()}
document.querySelectorAll('.search-language').forEach(box=>box.onchange=updateLanguageSummary);
$('englishOnly').onclick=()=>setSearchLanguages(['en']);
$('clearLanguages').onclick=()=>setSearchLanguages([]);
updateLanguageSummary();
function suggestionGroupHtml(batch){
  const suggestions=searchSuggestionGroups[batch]||[];
  if(!suggestions.length)return '';
  return `<section class="search-suggestions search-suggestion-group" data-suggestion-batch="${batch}"><div class="search-suggestions-title">Suggested follow-up searches for Search ${batch}</div><div class="search-suggestion-list">${suggestions.map((suggestion,index)=>`<button type="button" class="search-suggestion" data-query="${esc(suggestion)}"><strong>${index+1}.</strong> ${esc(suggestion)}</button>`).join('')}</div></section>`;
}
function renderSearchResults(sources,selectedIndices=null){
  const collapse=$('collapseHighlights').checked;
  const selectedSet=selectedIndices===null?null:new Set((selectedIndices||[]).map(Number));
  const batches=[];
  (sources||[]).forEach((source,index)=>{
    const batch=Number(source.search_batch||1);
    let group=batches.find(item=>item.batch===batch);
    if(!group){group={batch,query:source.search_query||'Search results',items:[]};batches.push(group)}
    group.items.push({source,index});
  });
  const latestBatch=batches.length?Math.max(...batches.map(item=>item.batch)):0;
  let html='';
  batches.forEach(group=>{
    html+=`<div class="search-batch-heading">Search ${group.batch}: ${esc(group.query)}</div>`;
    group.items.forEach(({source,index})=>{
      const items=source.highlights||[];
      const highlights=items.map(h=>`<div class="search-highlight">${esc(h)}</div>`).join('');
      const highlightBlock=highlights?`<details class="search-highlights" ${collapse?'':'open'}><summary>${items.length===1?'Show highlight':`Show ${items.length} highlights`}</summary>${highlights}</details>`:'';
      const domain=source.domain||(()=>{try{return new URL(source.url).hostname.replace(/^www\./,'')}catch{return''}})();
      const meta=[domain,source.author,source.published_date?`Published ${source.published_date}`:''].filter(Boolean).map((value,metaIndex)=>`${metaIndex?'<span class="separator">·</span>':''}<span>${esc(value)}</span>`).join('');
      const checked=selectedSet===null||selectedSet.has(index);
      html+=`<article class="search-result${checked?'':' excluded'}"><div class="search-result-head"><input class="search-result-toggle" type="checkbox" data-index="${index}" data-url="${esc(source.url||'')}" ${checked?'checked':''} aria-label="Include result ${index+1} in analysis"><div><div>${index+1}. <a href="${esc(source.url)}" target="_blank" rel="noopener">${esc(source.title||source.url)}</a></div><div class="search-result-meta">${meta}</div></div></div>${highlightBlock}</article>`;
    });
    html+=suggestionGroupHtml(group.batch);
    if(group.batch===latestBatch){
      html+=`<div class="search-batch-tools"><button type="button" class="suggest-searches-button" data-batch="${group.batch}" title="Use the selected local model to propose complementary searches">Suggest follow up searches</button></div>`;
    }
  });
  $('searchResults').innerHTML=html||'<div class="sub">No results returned.</div>';
  document.querySelectorAll('.search-result-toggle').forEach(box=>box.onchange=()=>{updateSearchSelectionStatus();refreshSearchStatus()});
  document.querySelectorAll('.search-suggestion').forEach(button=>button.onclick=()=>{
    $('searchQuery').value=button.dataset.query||button.textContent.replace(/^\s*\d+\.\s*/, '');
    $('searchQuery').focus();
    $('runSearch').click();
  });
  document.querySelectorAll('.suggest-searches-button').forEach(button=>button.onclick=()=>generateFollowUpSearches(Number(button.dataset.batch),button));
  $('selectAllResultsWrap').hidden=!(sources||[]).length;updateSearchSelectionStatus();
}
async function generateFollowUpSearches(batch,button){
  button.disabled=true;
  $('status').textContent='Generating follow up searches...';
  try{
    const response=await fetch('/api/search_suggestions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,model_key:$('modelSelect').value})});
    const data=await response.json();if(!response.ok)throw Error(data.error||'Suggestion generation failed');
    searchSuggestionGroups[batch]=data.suggestions||[];
    const currentSources=Array.from(document.querySelectorAll('.search-result-toggle')).map(()=>null);
    // Re-rendering is handled by replacing only this batch's suggestion area so selections stay untouched.
    const existing=$('searchResults').querySelector(`[data-suggestion-batch="${batch}"]`);
    if(existing)existing.remove();
    button.closest('.search-batch-tools').insertAdjacentHTML('beforebegin',suggestionGroupHtml(batch));
    const group=$('searchResults').querySelector(`[data-suggestion-batch="${batch}"]`);
    group.querySelectorAll('.search-suggestion').forEach(suggestion=>suggestion.onclick=()=>{
      $('searchQuery').value=suggestion.dataset.query||suggestion.textContent.replace(/^\s*\d+\.\s*/, '');
      $('searchQuery').focus();$('runSearch').click();
    });
    $('status').textContent='Three follow up searches are ready. Click one to run it.';
  }catch(error){$('status').textContent='Suggestion error: '+error.message}
  finally{button.disabled=false}
}
$('runSearch').onclick=async()=>{
  const query=$('searchQuery').value.trim();if(!query){$('searchStatus').textContent='Enter a search query.';return}
  if(!selectedSearchLanguages().length){$('searchStatus').textContent='Select at least one search language.';$('languagePicker').open=true;return}
  $('runSearch').disabled=true;$('send').disabled=true;$('searchStatus').textContent='Searching Exa... Existing results are retained.';
  try{
    const response=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,query,objective:$('searchObjective').value.trim(),search_type:$('searchType').value,user_location:$('userLocation').value,moderation:$('contentModeration').checked,categories:Array.from($('searchCategories').selectedOptions).map(option=>option.value),languages:selectedSearchLanguages(),include_domains:$('includeDomains').value,exclude_domains:$('excludeDomains').value,freshness_mode:$('freshnessMode').value,max_age_hours:Number($('maxAgeHours').value),start_published_date:$('startPublishedDate').value,end_published_date:$('endPublishedDate').value,max_results:Number($('maxArticles').value),highlights:$('searchHighlights').checked,selected_existing_indices:selectedSearchResultIndices(),selected_existing_urls:selectedSearchResultUrls()})});
    const data=await response.json();if(!response.ok)throw Error(data.error||'Search failed');sid=data.session_id;renderSearchResults(data.sources,data.selected_indices);const cost=data.metadata.cost_dollars==null?'':` · Cost $${Number(data.metadata.cost_dollars).toFixed(4)}`;const filtered=Number(data.metadata.filtered_out_by_age||0)+Number(data.metadata.filtered_out_undated||0);const filteredText=filtered?` · ${filtered} older/undated results excluded`:'';const duplicateText=Number(data.duplicate_count||0)?` · ${data.duplicate_count} duplicate${Number(data.duplicate_count)===1?'':'s'} skipped`:'';const summary=`<span class="search-ready">${data.total_count} cumulative results from ${data.search_count} search${Number(data.search_count)===1?'':'es'}</span> · ${data.added_count} added · ${Number(data.elapsed_seconds).toFixed(2)}s${filteredText}${duplicateText}${cost}`;$('searchStatus').dataset.summary=summary;updateSearchSelectionStatus();refreshSearchStatus();
  }catch(error){$('searchStatus').textContent='Search error: '+error.message}
  finally{$('runSearch').disabled=false;$('send').disabled=false}
};
function formatYoutubeTime(totalSeconds){const seconds=Math.max(0,Math.round(Number(totalSeconds)||0));const minutes=Math.floor(seconds/60);return `${String(minutes).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`}
function extractYoutubeUrlsFromPrompt(text){const regex=/https?:\/\/(?:www\.|m\.)?(?:youtube\.com\/(?:watch\?(?:[^\s#]*&)?v=|shorts\/|live\/|embed\/)|youtu\.be\/)[A-Za-z0-9_-]{11}[^\s<>'"]*/gi;const seen=new Set();return (String(text).match(regex)||[]).map(url=>url.replace(/[.,);\]}]+$/,'')).filter(url=>{const id=url.match(/(?:v=|youtu\.be\/|shorts\/|live\/|embed\/)([A-Za-z0-9_-]{11})/i)?.[1];if(!id||seen.has(id))return false;seen.add(id);return true})}
function youtubeSegmentCard(video,index){return `<article class="youtube-segment-card" data-index="${index}" data-url="${esc(video.url)}" data-title="${esc(video.title)}" data-duration="${video.duration_seconds}"><div class="youtube-video-title">${esc(video.title)}</div><div class="youtube-video-url-row"><div class="youtube-video-url" title="${esc(video.url)}">${esc(video.url)}</div><button type="button" class="youtube-copy-url">Copy URL</button></div><div class="youtube-time-row"><div><div class="youtube-time-value start-label">00:00</div><input class="youtube-range youtube-start" type="range" min="0" max="${video.duration_seconds}" step="1" value="0" aria-label="Transcription start"></div><span>to</span><div><div class="youtube-time-value end-label">${formatYoutubeTime(video.duration_seconds)}</div><input class="youtube-range youtube-end" type="range" min="0" max="${video.duration_seconds}" step="1" value="${video.duration_seconds}" aria-label="Transcription finish"></div></div></article>`}
async function chooseYoutubeSegments(prompt){const urls=extractYoutubeUrlsFromPrompt(prompt);if(!urls.length)return [];const response=await fetch('/api/youtube_metadata',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({urls})});const data=await response.json();if(!response.ok)throw Error(data.error||'Could not read YouTube video details.');$('youtubeModalList').innerHTML=data.videos.map(youtubeSegmentCard).join('');$('youtubeModalError').textContent='';$('youtubeModalBackdrop').hidden=false;const cards=Array.from(document.querySelectorAll('.youtube-segment-card'));cards.forEach(card=>{const start=card.querySelector('.youtube-start'),end=card.querySelector('.youtube-end'),startLabel=card.querySelector('.start-label'),endLabel=card.querySelector('.end-label');const update=changed=>{let a=Number(start.value),b=Number(end.value);if(a>=b){if(changed===start)a=Math.max(0,b-1);else b=Math.min(Number(end.max),a+1);start.value=a;end.value=b}startLabel.textContent=formatYoutubeTime(a);endLabel.textContent=formatYoutubeTime(b)};start.oninput=()=>update(start);end.oninput=()=>update(end);card.querySelector('.youtube-copy-url').onclick=async event=>{await navigator.clipboard.writeText(card.dataset.url);const button=event.currentTarget,old=button.textContent;button.textContent='Copied';setTimeout(()=>button.textContent=old,1200)}});return await new Promise(resolve=>{$('youtubeCancel').onclick=()=>{$('youtubeModalBackdrop').hidden=true;resolve(null)};$('youtubeContinue').onclick=()=>{const segments=cards.map(card=>({url:card.dataset.url,title:card.dataset.title,duration_seconds:Number(card.dataset.duration),start_seconds:Number(card.querySelector('.youtube-start').value),end_seconds:Number(card.querySelector('.youtube-end').value)}));$('youtubeModalBackdrop').hidden=true;resolve(segments)}})}
function localMediaSegmentCard(media){return `<article class="youtube-segment-card" data-duration="${media.duration_seconds}"><div class="youtube-video-title">${esc(media.name)}</div><div class="youtube-video-url">${(Number(media.bytes)/(1024*1024)).toFixed(1)} MB · ${formatYoutubeTime(media.duration_seconds)}</div><div class="youtube-time-row"><div><div class="youtube-time-value start-label">00:00</div><input class="youtube-range media-start" type="range" min="0" max="${media.duration_seconds}" step="1" value="0" aria-label="Transcription start"></div><span>to</span><div><div class="youtube-time-value end-label">${formatYoutubeTime(media.duration_seconds)}</div><input class="youtube-range media-end" type="range" min="0" max="${media.duration_seconds}" step="1" value="${media.duration_seconds}" aria-label="Transcription finish"></div></div></article>`}
async function chooseLocalMediaSegment(){
  if(!uploadedMedia)return null;
  $('mediaModalList').innerHTML=localMediaSegmentCard(uploadedMedia);$('mediaModalError').textContent='';$('mediaModalBackdrop').hidden=false;
  const card=$('mediaModalList').querySelector('.youtube-segment-card'),start=card.querySelector('.media-start'),end=card.querySelector('.media-end'),startLabel=card.querySelector('.start-label'),endLabel=card.querySelector('.end-label');
  const update=changed=>{let a=Number(start.value),b=Number(end.value);if(a>=b){if(changed===start)a=Math.max(0,b-1);else b=Math.min(Number(end.max),a+1);start.value=a;end.value=b}startLabel.textContent=formatYoutubeTime(a);endLabel.textContent=formatYoutubeTime(b)};
  start.oninput=()=>update(start);end.oninput=()=>update(end);
  return await new Promise(resolve=>{$('mediaCancel').onclick=()=>{$('mediaModalBackdrop').hidden=true;resolve(null)};$('mediaContinue').onclick=()=>{$('mediaModalBackdrop').hidden=true;resolve({start_seconds:Number(start.value),end_seconds:Number(end.value)})}})
}
$('mediaFile').onchange=async()=>{
  const file=$('mediaFile').files[0];if(!file)return;
  const fd=new FormData();fd.append('session_id',sid);fd.append('file',file);busy(true,'Uploading and inspecting media...');
  try{const r=await fetch('/api/media_upload',{method:'POST',body:fd}),d=await r.json();if(!r.ok)throw Error(d.error);sid=d.session_id;uploadedMedia=d.media;$('mediaStatus').textContent=`Attached media: ${d.media.name} · ${formatYoutubeTime(d.media.duration_seconds)} · ${(Number(d.media.bytes)/(1024*1024)).toFixed(1)} MB`;busy(false,'Media ready. Enter a prompt and run it.')}catch(e){uploadedMedia=null;$('mediaStatus').textContent='Media error: '+e.message;busy(false,'Error: '+e.message)}finally{$('mediaFile').value=''}
};
$('send').onclick=async()=>{const p=$('prompt').value.trim();if(!p)return;const started=performance.now();const youtubeUrls=extractYoutubeUrlsFromPrompt(p);let youtubeSegments=[],mediaSegment=null;try{if(uploadedMedia&&!youtubeUrls.length){busy(true,'Choose the audio or video segment to transcribe...');mediaSegment=await chooseLocalMediaSegment();if(mediaSegment===null){busy(false,'Media transcription cancelled.');return}}else if(youtubeUrls.length){busy(true,'Loading YouTube video details...');youtubeSegments=await chooseYoutubeSegments(p);if(youtubeSegments===null){busy(false,'YouTube transcription cancelled.');return}}const mediaWork=Boolean(mediaSegment);busy(true,mediaWork?'Transcribing the selected audio or video segment...':youtubeUrls.length?`Transcribing ${youtubeUrls.length} selected video segment${youtubeUrls.length===1?'':'s'}...`:'Reasoning over the approved evidence...');const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,prompt:p,youtube_segments:youtubeSegments,media_segment:mediaSegment,use_search_results:true,selected_source_indices:selectedSearchResultIndices(),selected_source_urls:selectedSearchResultUrls(),max_web_results:Number($('maxArticles').value),model_key:$('modelSelect').value})});const d=await r.json();if(!r.ok)throw Error(d.error);sid=d.session_id;const elapsed=Number(d.elapsed_seconds||((performance.now()-started)/1000));addTurn(p,d.answer,d.sources,elapsed,d.response_type||'standard',d.reasoning||'',d.model_label||'');$('prompt').value='';busy(false,`Complete in ${elapsed.toFixed(2)}s`);beep()}catch(e){$('youtubeModalBackdrop').hidden=true;$('mediaModalBackdrop').hidden=true;busy(false,'Error: '+e.message)}};
$('prompt').onkeydown=e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter')$('send').click()};
$('files').onchange=async()=>{const selectedFiles=Array.from($('files').files);if(selectedFiles.length>15){$('status').textContent='Select no more than 15 documents at once.';$('files').value='';return}const fd=new FormData();fd.append('session_id',sid);for(const file of selectedFiles)fd.append('files',file);busy(true,'Reading files...');try{const r=await fetch('/api/upload',{method:'POST',body:fd}),d=await r.json();if(!r.ok)throw Error(d.error);sid=d.session_id;$('fileList').textContent=(d.added.length?'Attached: '+d.added.map(x=>x.kind==='image'?`${x.name} (image ${x.width}x${x.height})`:`${x.name} (${Number(x.characters||0).toLocaleString()} chars)`).join(', '):'No documents or images attached.')+(d.errors.length?' Errors: '+d.errors.map(x=>x.name+': '+x.error).join('; '):'');busy(false,'Files ready')}catch(e){busy(false,'Error: '+e.message)}finally{$('files').value=''}};
$('speed').onchange=()=>$('player').playbackRate=Number($('speed').value);
$('maxArticles').oninput=()=>{$('maxArticlesValue').textContent=String(Number($('maxArticles').value))};
$('micMute').onclick=async()=>{try{await setMicMuted(!$('micMute').textContent.includes('Unmute'))}catch(e){$('status').textContent='Mic mute error: '+e.message}};
$('live').onclick=async()=>{$('live').disabled=true;$('status').textContent=$('live').textContent.includes('Stop')?'Stopping live mode...':'Starting microphone...';try{const r=await fetch('/api/live_toggle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,use_search_results:true,max_web_results:Number($('maxArticles').value),model_key:$('modelSelect').value})}),d=await r.json();if(!r.ok)throw Error(d.error);sid=d.session_id;$('live').textContent=d.active?'⏹ Stop live mode':'🎙 Start live mode';$('micMute').disabled=!d.active;$('micMute').textContent=d.muted?'🎤 Unmute mic':'🔇 Mute mic';$('status').textContent=d.active?(d.muted?'Live mode active — microphone muted':'Listening for a spoken prompt...'):'Live mode stopped';if(d.active)showLiveActivity(null,d.muted?'Microphone muted.':'Listening for a spoken prompt...',true);else showLiveActivity(null,'Live mode stopped.',false)}catch(e){$('status').textContent='Live error: '+e.message}finally{$('live').disabled=false}};
setInterval(async()=>{try{const r=await fetch('/api/live_poll'),d=await r.json();$('live').textContent=d.active?'⏹ Stop live mode':'🎙 Start live mode';$('micMute').disabled=!d.active;$('micMute').textContent=d.muted?'🎤 Unmute mic':'🔇 Mute mic';for(const item of d.items){const type=item.type||('answer'in item?'result':('error'in item?'error':'status'));if(type==='transcript'){showLiveActivity(item.prompt,item.message||'Message received. Preparing your response...',true);continue}if(type==='status'){showLiveActivity(null,item.message||'Working...',true);continue}if(type==='error'){showLiveActivity(null,item.message||('Live error: '+item.error),false);$('status').textContent='Live error: '+item.error;continue}if(type==='result'){showLiveActivity(item.prompt,item.message||'Response ready.',false);addTurn(item.prompt,item.answer,item.sources,item.elapsed_seconds,'standard',item.reasoning||'',item.model_key||'');busy(false,`Live response complete in ${Number(item.elapsed_seconds).toFixed(2)}s`);beep()}}}catch(e){}},500);
$('reset').onclick=async()=>{await fetch('/api/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid})});sid=crypto.randomUUID();latest='';$('prompt').value='';$('searchQuery').value='';$('searchObjective').value='';$('includeDomains').value='';$('excludeDomains').value='';$('searchType').value='auto';$('userLocation').value='';$('contentModeration').checked=false;$('collapseHighlights').checked=true;setSearchLanguages(['en']);$('languagePicker').open=false;Array.from($('searchCategories').options).forEach(option=>option.selected=false);$('freshnessMode').value='cached';updateFreshnessFields();$('searchResults').innerHTML='';searchSuggestionGroups={};$('selectAllResultsWrap').hidden=true;$('selectAllResults').checked=true;$('selectAllResults').indeterminate=false;$('searchStatus').textContent='No search run.';$('searchStatus').dataset.summary='';$('searchStatus').dataset.selection='';$('history').innerHTML='';$('fileList').textContent='No documents or images attached.';uploadedMedia=null;$('mediaStatus').textContent='No audio or video attached.';$('player').pause();$('player').removeAttribute('src');$('player').load();$('live').textContent='🎙 Start live mode';$('micMute').disabled=true;$('micMute').textContent='🔇 Mute mic';resetLiveActivity();$('status').textContent='Reset complete'};
</script>
</body></html>'''


def shutdown_cleanup() -> None:
    try:
        if live_stream is not None:
            stop_live()
    except Exception:
        pass
    clean_audio(0)


atexit.register(shutdown_cleanup)

if __name__ == "__main__":
    log("Application starting", url=f"http://localhost:{PORT}", default_model=MODEL_OPTIONS[DEFAULT_MODEL_KEY]["model"], available_models={k:v["model"] for k,v in MODEL_OPTIONS.items()}, temp=str(TEMP))
    app.run(host=HOST, port=PORT, threaded=True, debug=False)
