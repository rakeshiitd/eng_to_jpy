"""
EN ↔ JP Real-time Speech Translator
FastAPI backend: Claude for translation, ElevenLabs for TTS.
"""
import os
import asyncio
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from typing import List, Optional
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ELEVEN_API_KEY    = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVEN_EN_VOICE   = os.environ.get("ELEVEN_EN_VOICE", "21m00Tcm4TlvDq8ikWAM")  # Rachel – EN
ELEVEN_JA_VOICE   = os.environ.get("ELEVEN_JA_VOICE", "XrExE9yKIg1WjnnlVkGX")  # Matilda – multilingual JP
ELEVEN_MODEL      = "eleven_turbo_v2_5"   # 32 languages, lowest latency

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="EN↔JP Translator", docs_url=None, redoc_url=None)
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
    lang: str         # "en" or "ja"
    text: str         # original speech
    translation: str  # translated text

class TranslateRequest(BaseModel):
    text: str
    from_lang: str                   # "en" or "ja"
    history: List[HistoryTurn] = []  # last N turns for context

class TTSRequest(BaseModel):
    text: str
    lang: str  # "en" or "ja"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "translator.html").read_text()

@app.get("/api/status")
async def status():
    return {
        "anthropic": bool(ANTHROPIC_API_KEY),
        "elevenlabs": bool(ELEVEN_API_KEY),
    }

@app.post("/api/translate")
async def translate(req: TranslateRequest):
    claude = get_claude()

    context = ""
    if req.history:
        context = "\n\nConversation so far (use for context only — do NOT translate it):\n"
        for turn in req.history[-6:]:
            label = "English speaker" if turn.lang == "en" else "Japanese speaker"
            context += f"  [{label}] said: {turn.text}\n"
            context += f"  [Translation shown]: {turn.translation}\n"

    if req.from_lang == "en":
        system = f"""You are a real-time spoken translator helping an English speaker communicate in Okinawa, Japan.

Translate English speech → natural conversational Japanese.
Rules:
- Match the speaker's register: casual → casual Japanese (ね、よ、etc.), polite request → polite form
- Use 丁寧語 by default for strangers; drop to casual only if clearly warranted
- Never add romanization, explanations, or quotation marks — output ONLY the Japanese characters
- Prefer spoken, natural phrasing over textbook Japanese
- Short phrases like "how much?" should become "いくらですか？" not long formal sentences{context}"""
    else:
        system = f"""You are a real-time spoken translator helping a Japanese speaker communicate with an English speaker in Okinawa, Japan.

Translate Japanese speech → natural conversational English.
Rules:
- Match the speaker's tone: polite Japanese → polite English, casual → casual
- Produce spoken natural English, not overly literal or stiff translations
- Never add Japanese, explanations, or quotation marks — output ONLY English
- Keep it short and clear — spoken language is concise{context}"""

    resp = await asyncio.to_thread(
        claude.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": req.text}],
    )
    return {"translation": resp.content[0].text.strip()}


@app.post("/api/tts")
async def tts(req: TTSRequest):
    if not ELEVEN_API_KEY:
        raise HTTPException(400, "ELEVENLABS_API_KEY not set")

    voice_id = ELEVEN_EN_VOICE if req.lang == "en" else ELEVEN_JA_VOICE

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
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.80,
                "style": 0.0,
                "use_speaker_boost": True,
            },
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
