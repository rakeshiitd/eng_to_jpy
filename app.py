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
DEEPGRAM_API_KEY  = os.environ.get("DEEPGRAM_API_KEY", "")
CARTESIA_API_KEY  = os.environ.get("CARTESIA_API_KEY", "")
CARTESIA_MODEL    = os.environ.get("CARTESIA_MODEL", "sonic-multilingual")
CARTESIA_JA_VOICE = os.environ.get("CARTESIA_JA_VOICE", "bdab08ad-4137-4548-b9db-6142854c7525")
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
import struct as _struct, wave as _wave, io as _io, base64 as _b64, re as _re2, uuid as _uuid

# ── Deepgram streaming ASR client ────────────────────────────────────────────
class _DeepgramStream:
    """
    Streams PCM16 16kHz to Deepgram Nova-2.
    Provides real-time interim + final transcripts without any silence wait.
    endpointing=300 means transcript is final ~300ms after speech ends vs
    our old 400ms silence window + 900ms Gemini batch = 1300ms.
    """
    WS_URL = "wss://api.deepgram.com/v1/listen"

    def __init__(self, api_key: str, language: str = "ja"):
        self._key   = api_key
        self._lang  = language
        self._ws    = None
        self._task  = None
        self._on_interim = None
        self._on_final   = None
        self._on_vad     = None
        self._interim_fired = False

    def _url(self):
        from urllib.parse import urlencode
        p = dict(
            encoding="linear16", sample_rate=16000, channels=1,
            language=self._lang, model="nova-2",
            interim_results="true", endpointing=300,
            utterance_end_ms=1000, vad_events="true", smart_format="true",
        )
        return f"{self.WS_URL}?{urlencode(p)}"

    async def start(self, on_final, on_interim=None, on_vad=None):
        import websockets as _ws
        self._on_final   = on_final
        self._on_interim = on_interim
        self._on_vad     = on_vad
        self._ws = await _ws.connect(
            self._url(),
            additional_headers={"Authorization": f"Token {self._key}"},
            compression=None, max_size=None, ping_interval=10, open_timeout=6,
        )
        self._task = asyncio.create_task(self._recv_loop())

    async def send(self, pcm: bytes):
        if self._ws:
            try: await self._ws.send(pcm)
            except: pass

    async def _recv_loop(self):
        import json as _j, websockets as _ws2
        try:
            async for msg in self._ws:
                if isinstance(msg, (bytes, bytearray)): continue
                try: data = _j.loads(msg)
                except: continue
                t = data.get("type","")
                if t == "SpeechStarted":
                    if self._on_vad:
                        await self._on_vad({"type":"vad","state":"speech"})
                elif t == "UtteranceEnd":
                    self._interim_fired = False
                    if self._on_vad:
                        await self._on_vad({"type":"vad","state":"silence"})
                elif t == "Results":
                    is_final = data.get("is_final", False)
                    alts = data.get("channel",{}).get("alternatives",[])
                    text = (alts[0].get("transcript","") if alts else "").strip()
                    if text and not is_final and not self._interim_fired:
                        self._interim_fired = True
                        if self._on_interim: await self._on_interim(text)
                    if text and is_final:
                        self._interim_fired = False
                        if self._on_final: await self._on_final(text)
        except Exception: pass

    async def close(self):
        if self._task: self._task.cancel()
        if self._ws:
            try:
                import json as _j
                await self._ws.send(_j.dumps({"type":"CloseStream"}))
                await self._ws.close()
            except: pass
        self._ws = None

# ── Cartesia PCM TTS client ───────────────────────────────────────────────────
async def _cartesia_tts_pcm(text: str, language: str = "ja",
                             voice_id: str = None, sample_rate: int = 44100) -> bytes:
    """
    Cartesia Sonic over WebSocket → raw PCM16 LE.
    ~80ms TTFB, no encoding overhead, plays via AudioWorklet instantly.
    Falls back to EL WS TTS on failure.
    """
    import websockets as _ws, json as _j
    vid = voice_id or CARTESIA_JA_VOICE
    url = (f"wss://api.cartesia.ai/tts/websocket"
           f"?cartesia_version=2025-11-04&api_key={CARTESIA_API_KEY}")
    chunks = []
    try:
        async with _ws.connect(url, compression=None, open_timeout=6) as ws:
            await ws.send(_j.dumps({
                "model_id":     CARTESIA_MODEL,
                "transcript":   text,
                "voice":        {"mode": "id", "id": vid},
                "output_format":{"container":"raw","encoding":"pcm_s16le","sample_rate": sample_rate},
                "context_id":   str(_uuid.uuid4()),
                "language":     language,
            }))
            async for msg in ws:
                d = _j.loads(msg)
                if d.get("type") == "chunk" and d.get("data"):
                    chunks.append(_b64.b64decode(d["data"]))
                if d.get("done") or d.get("type") == "done":
                    break
    except Exception:
        pass
    return b"".join(chunks)

# ── 1. Hallucination filter ───────────────────────────────────────────────────
_HALLUCINATIONS = frozenset({
    # Japanese noise patterns
    "ありがとう", "ありがとうございます", "はい", "いいえ", "うん", "そう", "おう",
    "ご視聴ありがとうございました", "チャンネル登録", "よろしくお願いします",
    # English noise
    "thanks", "thank you", "thanks for watching", "thank you for watching",
    "subscribe", "like and subscribe", "bye", "hello", "okay", "hmm", "um", "uh",
    # Hindi noise
    "धन्यवाद", "शुक्रिया", "नमस्ते", "हाँ", "ओके", "हम्म", "अच्छा",
})

def _is_hallucination(text: str) -> bool:
    t = text.strip().lower().rstrip("。！？.!?,、 \n")
    if not t or len(t) <= 2:
        return True
    if t in _HALLUCINATIONS:
        return True
    # Single word ≤4 chars is almost always noise
    if len(t) <= 4 and " " not in t and "　" not in t:
        return True
    return False

# ── 2. Segment splitter for TTS pipeline ──────────────────────────────────────
_SENT_END = _re2.compile(r'(?<=[。！？\.\!\?])\s*')

def _split_segments(text: str) -> list:
    """Split translation into sentence segments for pipelined TTS."""
    parts = [s.strip() for s in _SENT_END.split(text) if s.strip()]
    return parts if parts else [text.strip()]

# ── 3. ElevenLabs WebSocket TTS ───────────────────────────────────────────────
async def _el_ws_tts_bytes(text: str, voice_id: str, model_id: str) -> bytes:
    """Synthesize via ElevenLabs WebSocket — ~80ms TTFB vs ~400ms HTTP."""
    import websockets as _ws, json as _j
    url = (f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
           f"?model_id={model_id}&optimize_streaming_latency=4&output_format=mp3_22050_32")
    chunks = []
    try:
        async with _ws.connect(url, additional_headers={"xi-api-key": ELEVEN_API_KEY},
                               open_timeout=6, close_timeout=4) as ws:
            # BOS — prime the model
            await ws.send(_j.dumps({"text": " ", "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                                    "generation_config": {"chunk_length_schedule": [50]}}))
            # Text
            await ws.send(_j.dumps({"text": text + " ", "try_trigger_generation": True}))
            # EOS
            await ws.send(_j.dumps({"text": ""}))
            async for msg in ws:
                if isinstance(msg, bytes):
                    chunks.append(msg)
                else:
                    d = _j.loads(msg)
                    if d.get("audio"):
                        chunks.append(_b64.b64decode(d["audio"]))
                    if d.get("isFinal"):
                        break
    except Exception as e:
        import traceback; traceback.print_exc()
        # Fallback: HTTP streaming
        import httpx as _hx
        async with _hx.AsyncClient(timeout=15) as hc:
            resp = await hc.get(
                f"{os.environ.get('ELEVEN_BASE','https://api.elevenlabs.io')}"
                f"/v1/text-to-speech/{voice_id}/stream",
                headers={"xi-api-key": ELEVEN_API_KEY},
                params={"model_id": model_id, "optimize_streaming_latency": "4",
                        "output_format": "mp3_22050_32"},
                content=_j.dumps({"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}}).encode(),
            )
            if resp.status_code == 200:
                chunks = [resp.content]
    return b"".join(chunks)

class _ServerVAD:
    """Energy-based VAD matching groq_asr.py logic from MyCashflo."""
    FRAME_MS      = 20
    SAMPLE_RATE   = 16000
    FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)   # 320 samples
    FRAME_BYTES   = FRAME_SAMPLES * 2                     # 640 bytes (PCM16)

    # Tuning
    SPEECH_THRESHOLD     = 250   # RMS energy — normal listening
    SPEECH_THRESHOLD_TTS = 500   # During TTS: lower than 700, catches real speech above echo residual
    SPEECH_CONFIRM       = 3     # frames (~60ms) to confirm speech start
    SILENCE_CONFIRM      = 20    # frames (400ms) to confirm utterance end
    MIN_SPEECH_FRAMES    = 8     # 160ms minimum
    PRE_SPEECH_FRAMES    = 15    # 300ms ring buffer before onset

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
                # ── Silence onset: fire speculative on COMPLETE audio immediately ──
                # The 400ms confirmation counts down in parallel with Gemini.
                # By the time utterance is confirmed, result is already 400ms in.
                if self._sl == 1 and self._total >= self.MIN_SPEECH_FRAMES:
                    events.append({"type": "silence_onset",
                                   "audio": bytes(self._speech),
                                   "frames": self._total,
                                   "during_tts": self.tts_active})
                if self._sl >= self.SILENCE_CONFIRM:
                    trim = self.FRAME_BYTES * self.SILENCE_CONFIRM
                    audio = bytes(self._speech[:-trim] if len(self._speech) > trim else self._speech)
                    sf = self._total - self.SILENCE_CONFIRM
                    during = self.tts_active
                    self._reset()
                    events.append({"type": "vad", "state": "silence"})
                    if sf >= self.MIN_SPEECH_FRAMES:
                        events.append({"type": "utterance_ready", "audio": audio,
                                        "frames": sf, "during_tts": during})
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
    v59: Full MyCashflo-equivalent pipeline.
    Client → PCM → Deepgram streaming ASR (real-time, no batch wait)
            → interim transcript → speculative Gemini translation (fires mid-speech)
            → final transcript → use speculative or fire fresh
            → Cartesia WS TTS (raw PCM, ~80ms TTFB)
            → PCM chunks → client AudioWorklet player (zero buffer latency)

    Post-silence latency: ~380ms vs old ~1300ms (3.4× improvement)
    """
    await ws.accept()
    import json as _j

    from_lang     = "ja"
    gemini_stt    = True   # kept for fallback; Deepgram is primary
    topic         = ""
    barge_in_lvl  = "medium"
    _tts_active   = False

    # ── Speculative translation state ─────────────────────────────────────────
    _spec_task:      asyncio.Task | None = None
    _spec_result:    dict | None = None
    _spec_text:      str = ""       # interim text the spec was fired on

    # ── TTS hold-queue (user spoke while bot was talking) ─────────────────────
    _tts_held: list = []

    # ── Debounce for utterance merging ────────────────────────────────────────
    _debounce_task: asyncio.Task | None = None
    DEBOUNCE_S = 0.22

    async def _translate_only(text: str, lang: str) -> str | None:
        """Fast Gemini translation-only call (no STT)."""
        to_lang = "en" if lang == "ja" else "ja"
        try:
            return await translate_text(text, lang, [], to_lang, topic)
        except Exception:
            return None

    async def _run_speculative(text: str, lang: str):
        nonlocal _spec_result, _spec_text
        import time as _t; t0 = _t.time()
        result = await _translate_only(text, lang)
        if result:
            _spec_result = result
            _spec_text   = text
            print(f"[spec] translate done in {(_t.time()-t0)*1000:.0f}ms: {result[:30]}")

    async def _dispatch_translation(text: str, lang: str, translation: str):
        """Send translation to client and fire Cartesia TTS pipeline."""
        to_lang = "en" if lang == "ja" else "ja"
        if not text or _is_hallucination(text):
            return

        await ws.send_json({"type":"transcript","text":text,"lang":lang})
        if not translation:
            return

        # Segment pipeline: split → Cartesia per segment → PCM chunks to client
        segments = _split_segments(translation)
        for i, seg in enumerate(segments):
            if not seg: continue
            is_first = (i == 0)
            is_last  = (i == len(segments) - 1)

            if lang != "ja":   # EN/HI → JA: play audio
                # Try Cartesia PCM first (fastest), fall back to EL WS
                if CARTESIA_API_KEY:
                    pcm = await _cartesia_tts_pcm(seg, language="ja")
                    if pcm:
                        await ws.send_json({
                            "type": "tts_pcm",
                            "data": _b64.b64encode(pcm).decode(),
                            "sample_rate": 44100,
                            "text": seg,
                            "original": text if is_first else "",
                            "full_translation": translation if is_first else "",
                            "lang": lang, "to_lang": to_lang,
                            "is_first": is_first, "is_last": is_last,
                        })
                        continue

                # EL WS TTS fallback → sends tts_audio (MP3 blob)
                voice_id, model_id = _tts_params("ja")
                if ELEVEN_API_KEY:
                    audio_bytes = await _el_ws_tts_bytes(seg, voice_id, model_id)
                    if audio_bytes:
                        await ws.send_json({
                            "type": "tts_audio",
                            "data": _b64.b64encode(audio_bytes).decode(),
                            "text": seg,
                            "original": text if is_first else "",
                            "full_translation": translation if is_first else "",
                            "lang": lang, "to_lang": to_lang,
                            "is_first": is_first, "is_last": is_last,
                        })
            else:   # JA → EN: text only (no TTS)
                if is_first:
                    await ws.send_json({
                        "type": "tts_segment",
                        "text": seg,
                        "original": text,
                        "full_translation": translation,
                        "lang": lang, "to_lang": to_lang,
                        "is_first": True, "is_last": is_last,
                    })

    async def _on_interim(text: str):
        """Deepgram interim: fire speculative translation immediately."""
        nonlocal _spec_task, _spec_result, _spec_text
        if _tts_active: return
        if len(text.strip()) < 3: return  # skip noise
        print(f"[deepgram] interim: {text[:40]}")
        if _spec_task and not _spec_task.done(): _spec_task.cancel()
        _spec_result = None
        _spec_task = asyncio.create_task(_run_speculative(text, from_lang))
        try: await ws.send_json({"type":"thinking"})
        except: pass

    async def _on_final(text: str):
        """Deepgram final: use speculative if it matches, else translate fresh."""
        import time as _t; _final_t = _t.time()
        print(f"[deepgram] final: {text[:50]}")
        nonlocal _spec_task, _spec_result, _spec_text, _debounce_task

        if _tts_active and barge_in_lvl != "high":
            _tts_held.append(("final", text))
            try: await ws.send_json({"type":"speech_queued","count":len(_tts_held)})
            except: pass
            return

        async def _process(t: str):
            nonlocal _spec_result, _spec_text
            translation = None

            # Wait briefly for in-flight speculative
            if _spec_task and not _spec_task.done():
                try: await asyncio.wait_for(asyncio.shield(_spec_task), timeout=2.0)
                except: pass

            # Use speculative if it was fired on ≥70% of the final text
            # Spec is good if it was fired on text that's at least 60% of final
            # Use word count for EN/HI; char count for JA (chars ≈ words)
            spec_cov = len(_spec_text) / max(1, len(t))
            if (_spec_result and _spec_text and spec_cov >= 0.6 and
                    not _is_hallucination(t)):
                translation = _spec_result
                _spec_result = None; _spec_text = ""
            else:
                translation = await _translate_only(t, from_lang)

            if translation:
                await _dispatch_translation(t, from_lang, translation)

        # Process immediately — Deepgram's endpointing=300ms already handles merging
        asyncio.create_task(_process(text))

    async def _on_vad(ev: dict):
        t = ev.get("type"); s = ev.get("state")
        if t == "vad":
            try: await ws.send_json({"type":"vad","state":s})
            except: pass

    # ── Boot Deepgram connection ───────────────────────────────────────────────
    dg: _DeepgramStream | None = None
    use_deepgram = bool(DEEPGRAM_API_KEY)

    # Fallback VAD for when Deepgram is unavailable
    vad = _ServerVAD()

    async def _start_deepgram(lang: str):
        nonlocal dg
        if dg: await dg.close()
        dg = _DeepgramStream(DEEPGRAM_API_KEY, language=lang)
        await dg.start(on_final=_on_final, on_interim=_on_interim, on_vad=_on_vad)

    # ── Non-blocking Deepgram connect: buffer PCM while connecting ────────────
    _pcm_connect_buf = bytearray()  # hold PCM until Deepgram ready
    _dg_ready = False

    async def _connect_dg_background():
        nonlocal use_deepgram, _dg_ready
        try:
            import time as _t; t0 = _t.time()
            await _start_deepgram(from_lang)
            _dg_ready = True
            elapsed = (_t.time()-t0)*1000
            print(f"[deepgram] connected in {elapsed:.0f}ms, flushing {len(_pcm_connect_buf)} buffered bytes")
            if _pcm_connect_buf and dg:
                await dg.send(bytes(_pcm_connect_buf))
                _pcm_connect_buf.clear()
        except Exception as e:
            print(f"[deepgram] connect failed: {e}")
            use_deepgram = False

    if use_deepgram:
        asyncio.create_task(_connect_dg_background())  # non-blocking!

    # ── WebSocket message loop ─────────────────────────────────────────────────
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect": break

            if msg.get("text"):
                try:
                    cfg = _j.loads(msg["text"])
                    t   = cfg.get("type")
                    if t == "config":
                        new_lang     = cfg.get("mode", from_lang)
                        gemini_stt   = cfg.get("gemini_stt", gemini_stt)
                        topic        = cfg.get("topic", topic)
                        barge_in_lvl = cfg.get("barge_in", barge_in_lvl)
                        # Restart Deepgram if language changed
                        if new_lang != from_lang:
                            from_lang = new_lang
                            if use_deepgram:
                                asyncio.create_task(_start_deepgram(from_lang))
                        from_lang = new_lang

                    elif t == "tts_start":
                        _tts_active = True
                        vad.tts_active = True
                        if _spec_task and not _spec_task.done(): _spec_task.cancel()
                        _spec_result = None; _spec_text = ""

                    elif t == "tts_end":
                        _tts_active = False
                        vad.tts_active = False
                        # Flush hold-queue
                        if _tts_held:
                            held = list(_tts_held); _tts_held.clear()
                            await ws.send_json({"type":"processing_queued","count":len(held)})
                            await asyncio.sleep(0.15)
                            for kind, text in held:
                                await _on_final(text)
                except: pass

            elif msg.get("bytes"):
                pcm = msg["bytes"]
                if use_deepgram:
                    if _dg_ready and dg:
                        await dg.send(pcm)   # real-time stream to Deepgram
                    else:
                        _pcm_connect_buf.extend(pcm)  # buffer while connecting
                else:
                    # Fallback: server VAD + batch Gemini
                    for ev in vad.push(pcm):
                        if ev["type"] == "vad":
                            await ws.send_json({"type":"vad","state":ev["state"]})
                        elif ev["type"] == "silence_onset" and not ev.get("during_tts"):
                            if _spec_task and not _spec_task.done(): _spec_task.cancel()
                            _spec_task = asyncio.create_task(
                                _run_speculative("", from_lang))  # no-op for fallback
                        elif ev["type"] == "utterance_ready":
                            audio = ev["audio"]
                            asyncio.create_task(_on_final(
                                f"__pcm__{_b64.b64encode(audio).decode()}"))

    except Exception: pass
    finally:
        if dg: await dg.close()
        if _debounce_task: _debounce_task.cancel()
        if _spec_task and not _spec_task.done(): _spec_task.cancel()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, root_path="/translator")
