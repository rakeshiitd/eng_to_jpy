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
from google import genai as _genai

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
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
FEAT_GEMINI_STT       = _flag("FEAT_GEMINI_STT", "1")   # Gemini multimodal: STT+translate in one call

TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "models/gemini-3.1-flash-lite-preview")

print(f"[config] model={TRANSLATE_MODEL} | stream_tts={FEAT_STREAM_TTS} | stream_translate={FEAT_STREAM_TRANSLATE} | server_tts={FEAT_SERVER_TTS}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="EN/HI↔JP Translator", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_claude: Optional[anthropic.Anthropic] = None
_gemini: Optional[_genai.Client]        = None

def get_claude() -> anthropic.Anthropic:
    global _claude
    if _claude is None:
        if not ANTHROPIC_API_KEY:
            raise HTTPException(500, "ANTHROPIC_API_KEY not set")
        _claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _claude

def get_gemini() -> _genai.Client:
    global _gemini
    if _gemini is None:
        key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise HTTPException(500, "GEMINI_API_KEY not set")
        _gemini = _genai.Client(api_key=key)
    return _gemini

def _is_gemini(model: str) -> bool:
    return "gemini" in model.lower()

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
    if _is_gemini(TRANSLATE_MODEL):
        prompt = f"{system}\n\n{text}"
        resp = await asyncio.to_thread(
            get_gemini().models.generate_content,
            model=TRANSLATE_MODEL,
            contents=prompt,
        )
        return resp.text.strip()
    # Claude fallback
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
            if _is_gemini(TRANSLATE_MODEL):
                prompt = f"{system}\n\n{text}"
                for chunk in get_gemini().models.generate_content_stream(
                    model=TRANSLATE_MODEL, contents=prompt
                ):
                    if chunk.text:
                        loop.call_soon_threadsafe(q.put_nowait, chunk.text)
            else:
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
        "gemini": bool(GEMINI_API_KEY),
        "feat_gemini_stt": FEAT_GEMINI_STT,
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

# ── Gemini STT+Translate (single call) ──────────────────────────────────────
@app.post("/api/stt-translate")
async def gemini_stt_translate(
    file: UploadFile = File(...),
    from_lang: str = Form(default="auto"),
    to_lang:   str = Form(default="ja"),
    topic:     str = Form(default=""),
):
    import json as _j, re as _re
    from google.genai import types as _gtypes

    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY not set")
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio")
    mime = file.content_type or "audio/webm"

    lang_hint = {
        "ja":   "The speaker is using Japanese.",
        "hi":   "The speaker may use Hindi, English, or mixed Hinglish.",
        "en":   "The speaker may use English or mixed Hinglish.",
        "auto": "The speaker may use Japanese, Hindi, English, or mixed Hinglish.",
    }.get(from_lang, "")

    to_lang_name = {"ja": "Japanese", "en": "English", "hi": "Hindi"}.get(to_lang, to_lang)
    topic_line   = f'Context/topic: "{topic}"\n' if topic else ""

    prompt = (
        f"{lang_hint}\n{topic_line}"
        f"1. Transcribe the audio exactly as spoken.\n"
        f"2. Translate the transcription to {to_lang_name}. Output ONLY the translation — no explanation.\n"
        f"3. Detect the spoken language (return ISO code: ja / hi / en).\n\n"
        f'Respond ONLY with valid JSON (no markdown, no code block):\n'
        f'{{"transcription":"...","translation":"...","language":"..."}}'
    )

    try:
        resp = await asyncio.to_thread(
            get_gemini().models.generate_content,
            model=TRANSLATE_MODEL,
            contents=[
                _gtypes.Part.from_bytes(data=audio_bytes, mime_type=mime),
                prompt,
            ],
        )
        raw = resp.text.strip()
        # Strip markdown code fences if present
        raw = _re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        data = _j.loads(raw)
    except Exception as e:
        raise HTTPException(500, f"Gemini STT error: {e}")

    return {
        "text":          data.get("transcription", ""),
        "translation":   data.get("translation", ""),
        "language_code": data.get("language", ""),
    }

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

# ── Server-side VAD + streaming STT WebSocket ────────────────────────────────
# Receives raw PCM16 LE 16kHz mono from browser AudioWorklet
# Runs energy VAD, accumulates utterances, fires STT+translate
# Events sent to client: {"type":"vad","state":"speech|silence"} | {"type":"transcript","text":"..."} | {"type":"translation","text":"...","original":"...","lang":"..."}

import struct as _struct
import wave as _wave, io as _io

class _ServerVAD:
    """Energy-based VAD matching groq_asr.py logic from MyCashflo."""
    FRAME_MS      = 20
    SAMPLE_RATE   = 16000
    FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)   # 320 samples
    FRAME_BYTES   = FRAME_SAMPLES * 2                     # 640 bytes (PCM16)

    # Tuning
    SPEECH_THRESHOLD     = 250   # RMS energy — normal listening
    SPEECH_THRESHOLD_TTS = 700   # Higher during TTS to suppress echo bleed
    SPEECH_CONFIRM    = 3     # frames (~60ms) to confirm speech start
    SILENCE_CONFIRM   = 20    # frames (400ms) to confirm utterance end
    MIN_SPEECH_FRAMES = 8     # 160ms minimum
    PRE_SPEECH_FRAMES = 15    # 300ms ring buffer before onset

    def __init__(self):
        from collections import deque
        self._buf      = bytearray()
        self._speech   = bytearray()
        self._pre      = deque(maxlen=self.PRE_SPEECH_FRAMES)
        self._speaking = False
        self._sc = self._sl = self._total = 0
        self.tts_active = False  # raised threshold during TTS playback

    def push(self, data: bytes):
        """Feed raw PCM bytes. Returns list of events: dicts with 'type' key."""
        self._buf.extend(data)
        events = []
        while len(self._buf) >= self.FRAME_BYTES:
            frame = bytes(self._buf[:self.FRAME_BYTES])
            del self._buf[:self.FRAME_BYTES]
            ev = self._process(frame)
            if ev:
                events.extend(ev)
        return events

    def flush(self):
        """Return any buffered speech on session end."""
        if self._speaking and self._total >= self.MIN_SPEECH_FRAMES:
            audio = bytes(self._speech)
            self._reset()
            return audio
        return None

    def _rms(self, frame: bytes) -> float:
        n = len(frame) // 2
        if n == 0: return 0.0
        samples = _struct.unpack(f"<{n}h", frame[:n*2])
        return (sum(s*s for s in samples) / n) ** 0.5

    def _reset(self):
        self._speech.clear(); self._speaking = False
        self._sc = self._sl = self._total = 0

    def _process(self, frame: bytes):
        rms = self._rms(frame)
        threshold = self.SPEECH_THRESHOLD_TTS if self.tts_active else self.SPEECH_THRESHOLD
        is_speech = rms > threshold
        events = []

        if not self._speaking:
            self._pre.append(frame)
            if is_speech:
                self._sc += 1
                if self._sc >= self.SPEECH_CONFIRM:
                    self._speaking = True; self._sl = 0; self._total = self._sc
                    self._speech.clear()
                    for pf in self._pre: self._speech.extend(pf)
                    self._pre.clear()
                    events.append({"type": "vad", "state": "speech"})
            else:
                self._sc = 0
        else:
            self._speech.extend(frame); self._total += 1
            if is_speech:
                self._sl = 0
            else:
                self._sl += 1
                if self._sl >= self.SILENCE_CONFIRM:
                    trim = self.FRAME_BYTES * self.SILENCE_CONFIRM
                    audio = bytes(self._speech[:-trim] if len(self._speech) > trim else self._speech)
                    sf = self._total - self.SILENCE_CONFIRM
                    self._reset()
                    events.append({"type": "vad", "state": "silence"})
                    if sf >= self.MIN_SPEECH_FRAMES:
                        events.append({"type": "utterance_ready", "audio": audio, "frames": sf})
        return events

def _pcm_to_wav(pcm: bytes, sr=16000) -> bytes:
    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()

@app.websocket("/ws/audio")
async def audio_vad_ws(ws: WebSocket):
    """
    Receive: binary PCM16 LE 16kHz mono frames from browser AudioWorklet
    Also receive JSON text frames: {"type":"config","mode":"ja","gemini_stt":true,"topic":"..."}
    Send: JSON events — vad, transcript, translation
    """
    await ws.accept()
    vad        = _ServerVAD()
    from_lang  = "ja"
    gemini_stt = True
    topic      = ""

    async def _handle_utterance(audio: bytes):
        """STT + translate in background — mic keeps listening."""
        try:
            wav = _pcm_to_wav(audio)
            to_lang = "en" if from_lang == "ja" else "ja"

            if gemini_stt and GEMINI_API_KEY:
                # Single Gemini call: STT + translate
                from google.genai import types as _gt
                import json as _j, re as _re
                lang_hint = {"ja":"The speaker is using Japanese.",
                             "hi":"The speaker may use Hindi, English or Hinglish.",
                             "en":"The speaker may use English or Hinglish."}.get(from_lang,"")
                to_name = {"ja":"Japanese","en":"English","hi":"Hindi"}.get(to_lang, to_lang)
                prompt = (f"{lang_hint}\n"
                          + (f'Context: "{topic}"\n' if topic else "")
                          + f"1. Transcribe the audio exactly.\n"
                          + f"2. Translate to {to_name}. Output ONLY the translation.\n"
                          + f"3. Detected language ISO code (ja/hi/en).\n"
                          + f'Respond ONLY valid JSON: {{"transcription":"...","translation":"...","language":"..."}}')
                resp = await asyncio.to_thread(
                    get_gemini().models.generate_content,
                    model=TRANSLATE_MODEL,
                    contents=[_gt.Part.from_bytes(data=wav, mime_type="audio/wav"), prompt],
                )
                raw = _re.sub(r"^```[a-z]*\n?","",resp.text.strip()).rstrip("`").strip()
                d = _j.loads(raw)
                text = d.get("transcription","").strip()
                translation = d.get("translation","").strip()
                lang = d.get("language","")
            else:
                # Scribe STT
                import httpx as _hx
                async with _hx.AsyncClient(timeout=20) as hc:
                    resp = await hc.post(
                        "https://api.elevenlabs.io/v1/speech-to-text",
                        headers={"xi-api-key": ELEVEN_API_KEY},
                        files={"file": ("audio.wav", wav, "audio/wav")},
                        data={"model_id": "scribe_v1", "language_code": from_lang},
                    )
                if resp.status_code != 200: return
                sd = resp.json(); text = sd.get("text","").strip(); lang = sd.get("language_code","")
                if not text: return
                to_l = "en" if (lang or from_lang).startswith("ja") else "ja"
                translation = await translate_text(text, lang or from_lang, [], to_l, topic)

            if not text: return
            actual_lang = lang[:2] if lang else from_lang
            await ws.send_json({"type":"transcript","text":text,"lang":actual_lang})
            if translation:
                await ws.send_json({"type":"translation","text":translation,
                                    "original":text,"lang":actual_lang,"to_lang":to_lang})
        except Exception as e:
            import traceback; traceback.print_exc()
            try: await ws.send_json({"type":"error","message":str(e)})
            except: pass

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text"):
                try:
                    cfg = __import__("json").loads(msg["text"])
                    t = cfg.get("type")
                    if t == "config":
                        from_lang  = cfg.get("mode", from_lang)
                        gemini_stt = cfg.get("gemini_stt", gemini_stt)
                        topic      = cfg.get("topic", topic)
                    elif t == "tts_start":
                        vad.tts_active = True   # raise threshold — TTS is playing
                    elif t == "tts_end":
                        vad.tts_active = False  # back to normal threshold
                except: pass
            elif msg.get("bytes"):
                pcm = msg["bytes"]
                events = vad.push(pcm)
                for ev in events:
                    if ev["type"] in ("vad",):
                        await ws.send_json({"type":"vad","state":ev["state"]})
                    elif ev["type"] == "utterance_ready":
                        asyncio.create_task(_handle_utterance(ev["audio"]))
    except Exception:
        pass
    finally:
        leftover = vad.flush()
        if leftover:
            try: await _handle_utterance(leftover)
            except: pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, root_path="/translator")
