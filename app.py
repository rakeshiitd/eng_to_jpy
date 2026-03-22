"""
EN / HI ↔ JP Real-time Speech Translator
FastAPI backend: Claude for translation, ElevenLabs for TTS, WebSocket rooms for multi-phone.

──────────────────────────────────────────────────────────────────────────────
PERFORMANCE FEATURE FLAGS  (set env vars to "0" to revert any change)
──────────────────────────────────────────────────────────────────────────────
  FEAT_FAST_MODEL=1        Use claude-3-5-haiku-20241022 (fastest Haiku)
                           =0 → reverts to original claude-haiku-4-5-20251001
  FEAT_STREAM_TTS=1        Pipe ElevenLabs bytes straight to browser (no buffer)
                           =0 → reverts to old batch POST /api/tts
  FEAT_STREAM_TRANSLATE=1  Push translation tokens over WS as they arrive
                           =0 → reverts to old single-shot translate-then-send
──────────────────────────────────────────────────────────────────────────────
"""
import os
import asyncio
import string
import secrets
from pathlib import Path
from typing import List, Optional, AsyncIterator

import requests
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ELEVEN_API_KEY    = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVEN_EN_VOICE   = os.environ.get("ELEVEN_EN_VOICE", "21m00Tcm4TlvDq8ikWAM")  # Rachel – EN
ELEVEN_JA_VOICE   = os.environ.get("ELEVEN_JA_VOICE", "XrExE9yKIg1WjnnlVkGX")  # Matilda – JP
ELEVEN_HI_VOICE   = os.environ.get("ELEVEN_HI_VOICE", "cgSgspJ2msm6clMCkdW9")  # Jessica – multilingual
ELEVEN_MODEL_STD  = "eleven_turbo_v2_5"       # EN/JA — low latency, supports 32 langs
ELEVEN_MODEL_MULTI= "eleven_multilingual_v2"   # fallback — higher quality, much slower

# ── Performance feature flags ─────────────────────────────────────────────────
def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip() not in ("0", "false", "no")

FEAT_FAST_MODEL        = _flag("FEAT_FAST_MODEL")         # fix model alias
FEAT_STREAM_TTS        = _flag("FEAT_STREAM_TTS")         # stream TTS bytes
FEAT_STREAM_TRANSLATE  = _flag("FEAT_STREAM_TRANSLATE")   # stream translation tokens

# Resolved model name
_MODEL_FAST = "claude-haiku-4-5-20251001"   # ← proxy alias (fastest available)
_MODEL_ORIG = "claude-haiku-4-5-20251001"   # ← same; FEAT_FAST_MODEL now a no-op
TRANSLATE_MODEL = _MODEL_FAST

print(f"[config] model={TRANSLATE_MODEL} | stream_tts={FEAT_STREAM_TTS} | stream_translate={FEAT_STREAM_TRANSLATE}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="EN/HI↔JP Translator", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_claude: Optional[anthropic.Anthropic] = None

def get_claude() -> anthropic.Anthropic:
    global _claude
    if _claude is None:
        if not ANTHROPIC_API_KEY:
            raise HTTPException(500, "ANTHROPIC_API_KEY not set")
        _claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _claude

# ── Models ────────────────────────────────────────────────────────────────────
class HistoryTurn(BaseModel):
    lang: str
    text: str
    translation: str

class TranslateRequest(BaseModel):
    text: str
    from_lang: str
    to_lang: Optional[str] = None
    history: List[HistoryTurn] = []

class TTSRequest(BaseModel):
    text: str
    lang: str  # "en" | "ja" | "hi"

# ── Build system prompt ───────────────────────────────────────────────────────
def _build_system(from_lang: str, to_lang: str, context: str) -> str:
    pairs = {
        ("en","ja"): "Real-time EN→JA translator. Output ONLY natural conversational Japanese (丁寧語 by default). No romaji, no explanations.",
        ("hi","ja"): "Real-time HI→JA translator. Output ONLY natural conversational Japanese (丁寧語 by default). No romaji, no explanations.",
        ("ja","en"): "Real-time JA→EN translator. Output ONLY natural conversational English. No Japanese, no explanations.",
        ("ja","hi"): "Real-time JA→HI translator. Output ONLY natural conversational Hindi in Devanagari. No Japanese, no English.",
    }
    base = pairs.get((from_lang, to_lang), f"Translate {from_lang}→{to_lang}. Output only the translation.")
    return base + context

def _build_context(history: list) -> str:
    if not history:
        return ""
    ctx = "\n\nConversation so far (use for context only — do NOT translate it):\n"
    labels = {"en": "English speaker", "hi": "Hindi speaker", "ja": "Japanese speaker"}
    for turn in history[-3:]:
        lang_k  = turn["lang"]        if isinstance(turn, dict) else turn.lang
        text_k  = turn["text"]        if isinstance(turn, dict) else turn.text
        trans_k = turn["translation"] if isinstance(turn, dict) else turn.translation
        ctx += f"  [{labels.get(lang_k, lang_k)}] said: {text_k}\n"
        ctx += f"  [Translation shown]: {trans_k}\n"
    return ctx

def _infer_to_lang(from_lang: str, fallback: str = "en") -> str:
    return "ja" if from_lang != "ja" else fallback

# ── Translation: batch (original) ─────────────────────────────────────────────
async def translate_text(text: str, from_lang: str, history: list,
                         to_lang: str = None) -> str:
    if to_lang is None:
        to_lang = _infer_to_lang(from_lang)
    context = _build_context(history)
    system  = _build_system(from_lang, to_lang, context)
    claude  = get_claude()
    resp = await asyncio.to_thread(
        claude.messages.create,
        model=TRANSLATE_MODEL,
        max_tokens=150,
        system=system,
        messages=[{"role": "user", "content": text}],
    )
    return resp.content[0].text.strip()

# ── Translation: streaming (FEAT_STREAM_TRANSLATE) ────────────────────────────
async def translate_text_stream(text: str, from_lang: str, history: list,
                                to_lang: str = None) -> AsyncIterator[str]:
    """Yields translation tokens as they arrive from Claude."""
    if to_lang is None:
        to_lang = _infer_to_lang(from_lang)
    context = _build_context(history)
    system  = _build_system(from_lang, to_lang, context)
    claude  = get_claude()

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _worker():
        try:
            with claude.messages.stream(
                model=TRANSLATE_MODEL,
                max_tokens=150,
                system=system,
                messages=[{"role": "user", "content": text}],
            ) as s:
                for token in s.text_stream:
                    loop.call_soon_threadsafe(q.put_nowait, token)
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, ("ERROR", str(e)))
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

    # Start worker in background thread WITHOUT awaiting it
    future = loop.run_in_executor(None, _worker)

    while True:
        token = await q.get()
        if token is None:
            break
        if isinstance(token, tuple) and token[0] == "ERROR":
            await future
            raise Exception(token[1])
        yield token

    await future  # propagate any thread exceptions

# ── TTS helpers ───────────────────────────────────────────────────────────────
def _tts_params(lang: str):
    # eleven_turbo_v2_5 supports 32 languages incl. Japanese — 3x faster than multilingual
    if lang == "ja":
        return ELEVEN_JA_VOICE, ELEVEN_MODEL_STD   # turbo for speed
    if lang == "hi":
        return ELEVEN_HI_VOICE, ELEVEN_MODEL_MULTI  # multilingual needed for Hindi quality
    return ELEVEN_EN_VOICE, ELEVEN_MODEL_STD

def _tts_payload(text: str, model_id: str) -> dict:
    return {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.80,
            "style": 0.0,
            "use_speaker_boost": False,  # adds ~200ms, not worth it for real-time speech
        },
    }

# ── Room management ───────────────────────────────────────────────────────────
rooms: dict = {}

@app.get("/api/room/new")
async def create_room():
    room_id = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    rooms[room_id] = {"clients": [], "history": []}
    return {"room_id": room_id}

@app.websocket("/ws/{room_id}")
async def ws_room(websocket: WebSocket, room_id: str):
    await websocket.accept()
    room_id = room_id.upper()

    if room_id not in rooms:
        rooms[room_id] = {"clients": [], "history": []}

    room = rooms[room_id]

    if len(room["clients"]) >= 5:
        alive = []
        for c in room["clients"]:
            try:
                await c["ws"].send_json({"type": "ping"})
                alive.append(c)
            except Exception:
                pass
        room["clients"] = alive
        if len(room["clients"]) >= 5:
            await websocket.send_json({"type": "error", "msg": "Room is full (max 5 people)"})
            await websocket.close()
            return

    client = {"ws": websocket, "role": None}
    room["clients"].append(client)

    async def broadcast(msg: dict, exclude: WebSocket = None):
        for c in room["clients"]:
            if c["ws"] is not exclude:
                try:
                    await c["ws"].send_json(msg)
                except Exception:
                    pass

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "join":
                client["role"] = data["lang"]
                await websocket.send_json({"type": "joined", "room_id": room_id, "lang": data["lang"]})
                ready = [c for c in room["clients"] if c["role"] is not None]
                if len(ready) == 2:
                    await broadcast({"type": "partner_joined"})

            elif data["type"] == "speak":
                text      = data["text"]
                from_lang = data["lang"]

                to_lang = None
                for c in room["clients"]:
                    if c["ws"] is not websocket and c["role"]:
                        to_lang = c["role"]
                        break
                if to_lang is None:
                    to_lang = _infer_to_lang(from_lang)

                for c in room["clients"]:
                    try:
                        await c["ws"].send_json({"type": "translating", "from_lang": from_lang})
                    except Exception:
                        pass

                try:
                    if FEAT_STREAM_TRANSLATE:
                        # ── Streaming path: push tokens live, then send final 'turn' ──
                        tokens = []
                        async for token in translate_text_stream(text, from_lang, room["history"], to_lang):
                            tokens.append(token)
                            partial = "".join(tokens)
                            for c in room["clients"]:
                                try:
                                    await c["ws"].send_json({
                                        "type": "partial",
                                        "to_lang": to_lang,
                                        "text": partial,
                                        "mine": c["ws"] is websocket,
                                    })
                                except Exception:
                                    pass
                        translation = "".join(tokens).strip()
                    else:
                        # ── Original batch path ──
                        translation = await translate_text(text, from_lang, room["history"], to_lang)

                    room["history"].append({"lang": from_lang, "text": text, "translation": translation})
                    if len(room["history"]) > 20:
                        room["history"].pop(0)

                    for c in room["clients"]:
                        try:
                            await c["ws"].send_json({
                                "type": "turn",
                                "from_lang": from_lang,
                                "to_lang": to_lang,
                                "original": text,
                                "translation": translation,
                                "mine": c["ws"] is websocket,
                            })
                        except Exception:
                            pass
                except Exception as e:
                    try:
                        await websocket.send_json({"type": "error", "msg": f"Translation failed: {e}"})
                    except Exception:
                        pass

    except WebSocketDisconnect:
        room["clients"] = [c for c in room["clients"] if c["ws"] is not websocket]
        if not room["clients"]:
            rooms.pop(room_id, None)
        else:
            await broadcast({"type": "partner_left"})
    except Exception as e:
        import traceback
        print(f"[WS ERROR] {e}\n{traceback.format_exc()}")
        room["clients"] = [c for c in room["clients"] if c["ws"] is not websocket]
        if not room["clients"]:
            rooms.pop(room_id, None)
        else:
            await broadcast({"type": "partner_left"})

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "translator.html").read_text()

@app.get("/solo", response_class=HTMLResponse)
async def solo():
    return (Path(__file__).parent / "translator.html").read_text()

@app.get("/api/status")
async def status():
    return {
        "anthropic": bool(ANTHROPIC_API_KEY),
        "elevenlabs": bool(ELEVEN_API_KEY),
        "feat_fast_model": FEAT_FAST_MODEL,
        "feat_stream_tts": FEAT_STREAM_TTS,
        "feat_stream_translate": FEAT_STREAM_TRANSLATE,
        "model": TRANSLATE_MODEL,
    }

@app.post("/api/translate")
async def translate(req: TranslateRequest):
    translation = await translate_text(req.text, req.from_lang, req.history, req.to_lang)
    return {"translation": translation}

# ── TTS: streaming GET (FEAT_STREAM_TTS=1) ───────────────────────────────────
@app.get("/api/tts/stream")
async def tts_stream(text: str = Query(...), lang: str = Query("en")):
    """Streams audio bytes from ElevenLabs directly to the browser.
    Browser can start playing before the full file arrives (TTFB ~200ms).
    Only used when FEAT_STREAM_TTS=1."""
    if not ELEVEN_API_KEY:
        raise HTTPException(400, "ELEVENLABS_API_KEY not set")

    voice_id, model_id = _tts_params(lang)

    async def audio_generator():
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
            f"?optimize_streaming_latency=4&output_format=mp3_22050_32"
        )
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "POST", url,
                headers={
                    "xi-api-key": ELEVEN_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json=_tts_payload(text, model_id),
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise HTTPException(resp.status_code, f"ElevenLabs error: {body[:300]}")
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    yield chunk

    return StreamingResponse(audio_generator(), media_type="audio/mpeg",
                             headers={"Cache-Control": "no-store"})

# ── TTS: batch POST (original, FEAT_STREAM_TTS=0) ────────────────────────────
@app.post("/api/tts")
async def tts(req: TTSRequest):
    """Original batch TTS — waits for full MP3 before sending.
    Kept as fallback when FEAT_STREAM_TTS=0."""
    if not ELEVEN_API_KEY:
        raise HTTPException(400, "ELEVENLABS_API_KEY not set")

    voice_id, model_id = _tts_params(req.lang)
    r = await asyncio.to_thread(
        requests.post,
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": ELEVEN_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json=_tts_payload(req.text, model_id),
        timeout=30,
    )
    if not r.ok:
        raise HTTPException(r.status_code, f"ElevenLabs error: {r.text[:300]}")

    return Response(content=r.content, media_type="audio/mpeg",
                    headers={"Cache-Control": "no-store"})

@app.get("/api/tts/status")
async def tts_status():
    return {
        "elevenlabs_configured": bool(ELEVEN_API_KEY),
        "stream_mode": FEAT_STREAM_TTS,
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, root_path="/translator")
