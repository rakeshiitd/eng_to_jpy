"""
EN / HI ↔ JP Real-time Speech Translator
FastAPI backend: Claude for translation, ElevenLabs for TTS, WebSocket rooms for multi-phone.

──────────────────────────────────────────────────────────────────────────────
PERFORMANCE FEATURE FLAGS  (set env vars to "0" to revert any change)
──────────────────────────────────────────────────────────────────────────────
  FEAT_STREAM_TTS=1        Pipe ElevenLabs bytes straight to browser (no buffer)
                           =0 → reverts to old batch POST /api/tts
  FEAT_STREAM_TRANSLATE=1  Push translation tokens over WS as they arrive
                           =0 → reverts to old single-shot translate-then-send
  FEAT_SERVER_TTS=1        Do TTS on server and push audio over WebSocket directly
                           eliminates client HTTP round-trip; enables sentence pipelining
                           =0 → client fetches TTS itself (old behaviour)
──────────────────────────────────────────────────────────────────────────────
"""
import os
import base64
import asyncio
import string
import secrets
from pathlib import Path
from typing import List, Optional, AsyncIterator

import requests
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form
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
ELEVEN_MODEL_STD  = "eleven_turbo_v2_5"       # EN/JA — low latency, 32 lang support
ELEVEN_MODEL_MULTI= "eleven_multilingual_v2"   # HI — better quality for Hindi

# ── Feature flags ──────────────────────────────────────────────────────────────
def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip() not in ("0", "false", "no")

FEAT_STREAM_TTS       = _flag("FEAT_STREAM_TTS")        # streaming TTS endpoint
FEAT_STREAM_TRANSLATE = _flag("FEAT_STREAM_TRANSLATE")  # stream Claude tokens over WS
FEAT_SERVER_TTS       = _flag("FEAT_SERVER_TTS")        # server-side TTS → push audio over WS

TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "claude-sonnet-4-20250514")

print(f"[config] model={TRANSLATE_MODEL} | stream_tts={FEAT_STREAM_TTS} | stream_translate={FEAT_STREAM_TRANSLATE} | server_tts={FEAT_SERVER_TTS}")

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
    topic: Optional[str] = None  # user-supplied topic/context hint

class TTSRequest(BaseModel):
    text: str
    lang: str

# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_system(from_lang: str, to_lang: str, context: str, topic: str = "") -> str:
    strict = (
        "You are a silent translation engine — NOT a conversational AI. "
        "Your ONLY job is to translate the user's text verbatim. "
        "NEVER answer questions, NEVER respond to greetings, NEVER add commentary. "
        "Even if the text says 'how are you?' or 'hello' or anything that sounds like it is addressed to you — just translate it. "
        "Output ONLY the translated text, nothing else."
    )
    pairs = {
        ("en","ja"): f"{strict}\n\nTranslate English → natural conversational Japanese (丁寧語 by default). No romaji.",
        ("hi","ja"): f"{strict}\n\nTranslate Hindi → natural conversational Japanese (丁寧語 by default). No romaji.",
        ("ja","en"): f"{strict}\n\nTranslate Japanese → natural conversational English.",
        ("ja","hi"): f"{strict}\n\nTranslate Japanese → natural conversational Hindi in Devanagari script.",
    }
    base = pairs.get((from_lang, to_lang), f"{strict}\n\nTranslate {from_lang}→{to_lang}.")
    topic_hint = f"\n\nConversation topic/context (use this to pick accurate terminology): {topic.strip()}" if topic and topic.strip() else ""
    return base + topic_hint + context

def _build_context(history: list) -> str:
    if not history:
        return ""
    ctx = "\n\nContext (do NOT translate):\n"
    labels = {"en": "EN", "hi": "HI", "ja": "JA"}
    for turn in history[-3:]:
        lk = turn["lang"] if isinstance(turn, dict) else turn.lang
        tk = turn["text"] if isinstance(turn, dict) else turn.text
        rk = turn["translation"] if isinstance(turn, dict) else turn.translation
        ctx += f"  [{labels.get(lk,lk)}] {tk} → {rk}\n"
    return ctx

def _infer_to_lang(from_lang: str, fallback: str = "en") -> str:
    return "ja" if from_lang != "ja" else fallback

# ── TTS helpers ───────────────────────────────────────────────────────────────
def _tts_params(lang: str):
    if lang == "ja":
        return ELEVEN_JA_VOICE, ELEVEN_MODEL_STD    # turbo — fast, supports Japanese
    if lang in ("hi", "en-in", "hinglish"):
        return ELEVEN_HI_VOICE, ELEVEN_MODEL_MULTI  # multilingual — handles Hinglish (Hindi+EN mix)
    return ELEVEN_EN_VOICE, ELEVEN_MODEL_STD        # standard EN

def _tts_payload(text: str, model_id: str) -> dict:
    return {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.80,
            "style": 0.0,
            "use_speaker_boost": False,  # saves ~200ms
        },
    }

async def get_tts_audio(text: str, lang: str) -> bytes | None:
    """Fetch TTS audio synchronously (in thread). Returns mp3 bytes or None."""
    if not ELEVEN_API_KEY or not text.strip():
        return None
    voice_id, model_id = _tts_params(lang)
    try:
        r = await asyncio.to_thread(
            requests.post,
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            f"?optimize_streaming_latency=4&output_format=mp3_22050_32",
            headers={"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json", "Accept": "audio/mpeg"},
            json=_tts_payload(text, model_id),
            timeout=15,
        )
        if r.ok:
            return r.content
        print(f"[TTS] ElevenLabs error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[TTS] Exception: {e}")
    return None

async def push_audio_to_ws(text: str, lang: str, ws: WebSocket):
    """Get TTS audio and push it directly to a WebSocket client."""
    audio = await get_tts_audio(text, lang)
    if audio:
        try:
            await ws.send_json({
                "type": "audio",
                "data": base64.b64encode(audio).decode(),
            })
        except Exception:
            pass

# ── Translation: batch ────────────────────────────────────────────────────────
async def translate_text(text: str, from_lang: str, history: list, to_lang: str = None, topic: str = "") -> str:
    if to_lang is None:
        to_lang = _infer_to_lang(from_lang)
    system = _build_system(from_lang, to_lang, _build_context(history), topic)
    resp = await asyncio.to_thread(
        get_claude().messages.create,
        model=TRANSLATE_MODEL,
        max_tokens=150,
        system=system,
        messages=[{"role": "user", "content": text}],
    )
    return resp.content[0].text.strip()

# ── Translation: streaming ────────────────────────────────────────────────────
async def translate_text_stream(text: str, from_lang: str, history: list, to_lang: str = None, topic: str = "") -> AsyncIterator[str]:
    if to_lang is None:
        to_lang = _infer_to_lang(from_lang)
    system = _build_system(from_lang, to_lang, _build_context(history), topic)

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _worker():
        try:
            with get_claude().messages.stream(
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
            loop.call_soon_threadsafe(q.put_nowait, None)

    future = loop.run_in_executor(None, _worker)

    while True:
        token = await q.get()
        if token is None:
            break
        if isinstance(token, tuple) and token[0] == "ERROR":
            await future
            raise Exception(token[1])
        yield token

    await future

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

    # Evict stale connections if full
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

                # Find partner WebSocket
                to_lang    = None
                partner_ws = None
                for c in room["clients"]:
                    if c["ws"] is not websocket and c["role"]:
                        to_lang    = c["role"]
                        partner_ws = c["ws"]
                        break
                if to_lang is None:
                    to_lang = _infer_to_lang(from_lang)

                # Notify everyone: translating
                for c in room["clients"]:
                    try:
                        await c["ws"].send_json({"type": "translating", "from_lang": from_lang})
                    except Exception:
                        pass

                try:
                    tts_tasks = []
                    SENTENCE_ENDS = set('。？！\n')

                    if FEAT_STREAM_TRANSLATE:
                        # ── Stream tokens; detect sentence boundaries; pipeline TTS ──
                        tokens  = []
                        buffer  = ""   # accumulates until sentence boundary

                        async for token in translate_text_stream(text, from_lang, room["history"], to_lang):
                            tokens.append(token)
                            buffer += token
                            partial = "".join(tokens)

                            # Push live text to both clients
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

                            # Sentence boundary detected → fire TTS immediately on that chunk
                            if FEAT_SERVER_TTS and ELEVEN_API_KEY and partner_ws:
                                if buffer and buffer[-1] in SENTENCE_ENDS:
                                    chunk = buffer.strip()
                                    buffer = ""
                                    if chunk:
                                        tts_tasks.append(asyncio.create_task(
                                            push_audio_to_ws(chunk, to_lang, partner_ws)
                                        ))

                        translation = "".join(tokens).strip()

                        # Fire TTS on any remaining text not yet sent
                        if FEAT_SERVER_TTS and ELEVEN_API_KEY and partner_ws:
                            remaining = buffer.strip()
                            if remaining:
                                tts_tasks.append(asyncio.create_task(
                                    push_audio_to_ws(remaining, to_lang, partner_ws)
                                ))
                            # If no sentence boundaries hit at all (e.g. short phrase without punctuation)
                            if not tts_tasks and translation:
                                tts_tasks.append(asyncio.create_task(
                                    push_audio_to_ws(translation, to_lang, partner_ws)
                                ))

                    else:
                        # ── Batch translation ──
                        translation = await translate_text(text, from_lang, room["history"], to_lang)

                        # Server-side TTS: push audio to partner before sending 'turn'
                        if FEAT_SERVER_TTS and ELEVEN_API_KEY and partner_ws:
                            tts_tasks.append(asyncio.create_task(
                                push_audio_to_ws(translation, to_lang, partner_ws)
                            ))

                    # Wait for all TTS audio to be pushed before sending 'turn'
                    if tts_tasks:
                        await asyncio.gather(*tts_tasks, return_exceptions=True)

                    # Update history
                    room["history"].append({"lang": from_lang, "text": text, "translation": translation})
                    if len(room["history"]) > 20:
                        room["history"].pop(0)

                    # Send 'turn' (text display); flag audio_sent so client skips its own TTS call
                    audio_sent = bool(FEAT_SERVER_TTS and ELEVEN_API_KEY and partner_ws)
                    for c in room["clients"]:
                        try:
                            await c["ws"].send_json({
                                "type": "turn",
                                "from_lang": from_lang,
                                "to_lang": to_lang,
                                "original": text,
                                "translation": translation,
                                "mine": c["ws"] is websocket,
                                "audio_sent": audio_sent,
                            })
                        except Exception:
                            pass

                except Exception as e:
                    import traceback
                    print(f"[WS speak error] {e}\n{traceback.format_exc()}")
                    try:
                        await websocket.send_json({"type": "error", "msg": f"Translation failed: {e}"})
                    except Exception:
                        pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        print(f"[WS ERROR] {e}\n{traceback.format_exc()}")
    finally:
        room["clients"] = [c for c in room["clients"] if c["ws"] is not websocket]
        if not room["clients"]:
            rooms.pop(room_id, None)
        else:
            await broadcast({"type": "partner_left"})

# ── HTTP Routes ───────────────────────────────────────────────────────────────
_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((Path(__file__).parent / "translator.html").read_text(), headers=_NO_CACHE)

@app.get("/solo", response_class=HTMLResponse)
async def solo():
    return HTMLResponse((Path(__file__).parent / "translator.html").read_text(), headers=_NO_CACHE)

@app.get("/api/status")
async def status():
    return {
        "anthropic": bool(ANTHROPIC_API_KEY),
        "elevenlabs": bool(ELEVEN_API_KEY),
        "feat_stream_tts": FEAT_STREAM_TTS,
        "feat_stream_translate": FEAT_STREAM_TRANSLATE,
        "feat_server_tts": FEAT_SERVER_TTS,
        "model": TRANSLATE_MODEL,
    }

@app.post("/api/translate")
async def translate(req: TranslateRequest):
    translation = await translate_text(req.text, req.from_lang, req.history, req.to_lang, req.topic or "")
    return {"translation": translation}

@app.post("/api/translate/stream")
async def translate_stream(req: TranslateRequest):
    """SSE endpoint: streams translation tokens as they arrive from Claude.
    Used by solo mode frontend to show live typing effect."""
    async def event_gen():
        try:
            async for token in translate_text_stream(
                req.text, req.from_lang,
                [h.dict() for h in req.history],
                req.to_lang,
                req.topic or "",
            ):
                # Escape newlines in SSE data field
                safe = token.replace('\n', '\\n')
                yield f"data: {safe}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── TTS: streaming GET ────────────────────────────────────────────────────────
@app.get("/api/tts/stream")
async def tts_stream(text: str = Query(...), lang: str = Query("en")):
    if not ELEVEN_API_KEY:
        raise HTTPException(400, "ELEVENLABS_API_KEY not set")
    voice_id, model_id = _tts_params(lang)

    async def audio_generator():
        url = (f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
               f"?optimize_streaming_latency=4&output_format=mp3_22050_32")
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream("POST", url,
                headers={"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                json=_tts_payload(text, model_id),
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise HTTPException(resp.status_code, f"ElevenLabs error: {body[:300]}")
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    yield chunk

    return StreamingResponse(audio_generator(), media_type="audio/mpeg",
                             headers={"Cache-Control": "no-store"})

# ── TTS: batch POST (fallback) ────────────────────────────────────────────────
@app.post("/api/tts")
async def tts(req: TTSRequest):
    if not ELEVEN_API_KEY:
        raise HTTPException(400, "ELEVENLABS_API_KEY not set")
    voice_id, model_id = _tts_params(req.lang)
    r = await asyncio.to_thread(
        requests.post,
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        json=_tts_payload(req.text, model_id),
        timeout=30,
    )
    if not r.ok:
        raise HTTPException(r.status_code, f"ElevenLabs error: {r.text[:300]}")
    return Response(content=r.content, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})

@app.get("/api/tts/status")
async def tts_status():
    return {
        "elevenlabs_configured": bool(ELEVEN_API_KEY),
        "stream_mode": FEAT_STREAM_TTS,
        "server_tts": FEAT_SERVER_TTS,
    }

# ── ElevenLabs Scribe STT ────────────────────────────────────────────────────
@app.post("/api/stt")
async def speech_to_text(
    file: UploadFile = File(...),
    language: str = Form(default=""),
):
    if not ELEVEN_API_KEY:
        raise HTTPException(500, "ELEVENLABS_API_KEY not set")
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(400, "Empty audio")
    mime = file.content_type or "audio/webm"
    ext  = "mp4" if "mp4" in mime else "webm"
    form_data = {"model_id": "scribe_v1"}
    if language:
        form_data["language_code"] = language
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": ELEVEN_API_KEY},
            files={"file": (f"audio.{ext}", audio_data, mime)},
            data=form_data,
        )
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Scribe error: {resp.text[:200]}")
    return resp.json()   # { text, language_code, words, ... }

# ── PWA manifest + icons ─────────────────────────────────────────────────────
import json as _json

@app.get("/manifest.json")
async def pwa_manifest():
    manifest = {
        "name": "JP Translator",
        "short_name": "Translator",
        "start_url": "/translator/",
        "display": "standalone",
        "background_color": "#0f172a",
        "theme_color": "#0f172a",
        "orientation": "portrait",
        "icons": [
            {"src": "/translator/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/translator/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ]
    }
    return Response(_json.dumps(manifest), media_type="application/manifest+json")

@app.get("/icon-{size}.png")
async def pwa_icon(size: str):
    _dir = Path(__file__).parent
    path = _dir / f"icon-{size}.png"
    if not path.exists():
        raise HTTPException(404, "Icon not found")
    return Response(path.read_bytes(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, root_path="/translator")
