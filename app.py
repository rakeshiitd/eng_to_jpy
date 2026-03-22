"""
EN / HI ↔ JP Real-time Speech Translator
FastAPI backend: Claude for translation, ElevenLabs for TTS, WebSocket rooms for multi-phone.
"""
import os
import asyncio
import string
import secrets
from pathlib import Path
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ELEVEN_API_KEY    = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVEN_EN_VOICE   = os.environ.get("ELEVEN_EN_VOICE", "21m00Tcm4TlvDq8ikWAM")  # Rachel – EN
ELEVEN_JA_VOICE   = os.environ.get("ELEVEN_JA_VOICE", "XrExE9yKIg1WjnnlVkGX")  # Matilda – JP
ELEVEN_HI_VOICE   = os.environ.get("ELEVEN_HI_VOICE", "")                        # optional dedicated HI voice
ELEVEN_MODEL      = "eleven_turbo_v2_5"

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
    to_lang: Optional[str] = None   # inferred if omitted
    history: List[HistoryTurn] = []

class TTSRequest(BaseModel):
    text: str
    lang: str  # "en" | "ja" | "hi"

# ── Core translation helper ───────────────────────────────────────────────────
def _infer_to_lang(from_lang: str, fallback: str = "en") -> str:
    return "ja" if from_lang != "ja" else fallback

async def translate_text(text: str, from_lang: str, history: list,
                         to_lang: str = None) -> str:
    if to_lang is None:
        to_lang = _infer_to_lang(from_lang)

    claude = get_claude()

    context = ""
    if history:
        context = "\n\nConversation so far (use for context only — do NOT translate it):\n"
        for turn in history[-6:]:
            lang_k  = turn["lang"]        if isinstance(turn, dict) else turn.lang
            text_k  = turn["text"]        if isinstance(turn, dict) else turn.text
            trans_k = turn["translation"] if isinstance(turn, dict) else turn.translation
            labels  = {"en": "English speaker", "hi": "Hindi speaker", "ja": "Japanese speaker"}
            context += f"  [{labels.get(lang_k, lang_k)}] said: {text_k}\n"
            context += f"  [Translation shown]: {trans_k}\n"

    # Build prompt based on from→to pair
    if from_lang == "en" and to_lang == "ja":
        system = f"""You are a real-time spoken translator helping an English speaker communicate in Okinawa, Japan.

Translate English speech → natural conversational Japanese.
Rules:
- Match register: casual speech → casual Japanese (ね、よ), polite request → polite form
- Use 丁寧語 by default for strangers; drop to casual only if clearly warranted
- Output ONLY Japanese characters — no romanization, explanations, or quotes
- Prefer spoken natural phrasing over textbook Japanese{context}"""

    elif from_lang == "hi" and to_lang == "ja":
        system = f"""You are a real-time spoken translator helping a Hindi speaker communicate in Japan.

Translate Hindi speech → natural conversational Japanese.
Rules:
- Use 丁寧語 (polite form) by default; switch to casual only if clearly needed
- Output ONLY Japanese characters — no romanization, Hindi, or explanations
- Prefer natural spoken Japanese over literal translations{context}"""

    elif from_lang == "ja" and to_lang == "en":
        system = f"""You are a real-time spoken translator helping a Japanese speaker communicate with an English speaker in Okinawa, Japan.

Translate Japanese speech → natural conversational English.
Rules:
- Match tone: polite Japanese → polite English, casual → casual
- Output ONLY English — no Japanese, explanations, or quotes
- Keep it concise and natural — spoken language is short{context}"""

    elif from_lang == "ja" and to_lang == "hi":
        system = f"""You are a real-time spoken translator helping a Japanese speaker communicate with a Hindi speaker.

Translate Japanese speech → natural conversational Hindi.
Rules:
- Use आप (formal) by default; drop to तुम only if tone is clearly casual
- Output ONLY Hindi in Devanagari script — no Japanese, English, or transliteration
- Keep translations concise and natural — spoken language is short{context}"""

    else:
        system = f"Translate the following from {from_lang} to {to_lang}. Output only the translation, nothing else."

    resp = await asyncio.to_thread(
        claude.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": text}],
    )
    return resp.content[0].text.strip()

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

    if len(room["clients"]) >= 2:
        await websocket.send_json({"type": "error", "msg": "Room is full (max 2 people)"})
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
                    await websocket.send_json({"type": "partner_joined"})

            elif data["type"] == "speak":
                text     = data["text"]
                from_lang = data["lang"]

                # Infer to_lang from partner's role
                to_lang = None
                for c in room["clients"]:
                    if c["ws"] is not websocket and c["role"]:
                        to_lang = c["role"]
                        break
                if to_lang is None:
                    to_lang = _infer_to_lang(from_lang)

                # Notify both: translating
                for c in room["clients"]:
                    try:
                        await c["ws"].send_json({"type": "translating", "from_lang": from_lang})
                    except Exception:
                        pass

                try:
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
                    await websocket.send_json({"type": "error", "msg": f"Translation failed: {e}"})

    except WebSocketDisconnect:
        room["clients"] = [c for c in room["clients"] if c["ws"] is not websocket]
        if not room["clients"]:
            rooms.pop(room_id, None)
        else:
            await broadcast({"type": "partner_left"})

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "translator.html").read_text()

@app.get("/api/status")
async def status():
    return {"anthropic": bool(ANTHROPIC_API_KEY), "elevenlabs": bool(ELEVEN_API_KEY)}

@app.post("/api/translate")
async def translate(req: TranslateRequest):
    translation = await translate_text(req.text, req.from_lang, req.history, req.to_lang)
    return {"translation": translation}

@app.post("/api/tts")
async def tts(req: TTSRequest):
    if not ELEVEN_API_KEY:
        raise HTTPException(400, "ELEVENLABS_API_KEY not set")

    if req.lang == "ja":
        voice_id = ELEVEN_JA_VOICE
    elif req.lang == "hi":
        voice_id = ELEVEN_HI_VOICE or ELEVEN_EN_VOICE  # fallback to EN voice for Hindi
    else:
        voice_id = ELEVEN_EN_VOICE

    r = await asyncio.to_thread(
        requests.post,
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": ELEVEN_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={
            "text": req.text,
            "model_id": ELEVEN_MODEL,
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.80,
                               "style": 0.0, "use_speaker_boost": True},
        },
        timeout=30,
    )

    if not r.ok:
        raise HTTPException(r.status_code, f"ElevenLabs error: {r.text[:300]}")

    return Response(content=r.content, media_type="audio/mpeg",
                    headers={"Cache-Control": "no-store"})

@app.get("/api/tts/status")
async def tts_status():
    return {"elevenlabs_configured": bool(ELEVEN_API_KEY)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
