"""Transcriber: chunked Gemini vision calls with the Last_Known_Line handshake."""
import base64, io, os, re, time
import httpx

MODELS = [m for m in os.environ.get("GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-2.5-flash").split(",") if m]
CHUNK_IMAGES = int(os.environ.get("CHUNK_IMAGES", "5"))
MAX_WIDTH = 900

PROMPT = """You are a specialized OCR engine. The images are consecutive, vertically ordered pieces of a scrolling screen recording (top to bottom). Transcribe all visible text exactly, in reading order. Maintain original formatting (lists, tables as markdown, line breaks, speaker names and timestamps in chats).
Ignore phone status bars, navigation bars, and UI chrome (battery, clock, back buttons) unless part of the content.
Adjacent images may overlap by a few lines: never repeat text that appears twice because of overlap.
{ctx}Output ONLY the transcribed text, with no commentary and no code fences."""


def last_known_lines(text, n=2):
    parts = [p for p in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if p.strip()]
    return " ".join(parts[-n:]) if parts else ""


def _jpeg(img):
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, int(img.height * MAX_WIDTH / img.width)))
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=80)
    return base64.b64encode(b.getvalue()).decode()


def call_gemini(api_key, images, ctx):
    ctx_txt = (f'The previous chunk ended with this text (Last_Known_Line). If the images begin by repeating it, do not repeat it; continue right after it:\n"""{ctx}"""\n' if ctx else "")
    parts = [{"text": PROMPT.format(ctx=ctx_txt)}] + [
        {"inline_data": {"mime_type": "image/jpeg", "data": _jpeg(i)}} for i in images]
    body = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0}}
    err = None
    for model in MODELS:
        for attempt in range(3):
            try:
                r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                               headers={"x-goog-api-key": api_key}, json=body, timeout=120)
                if r.status_code == 200:
                    j = r.json()
                    txt = "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"])
                    u = j.get("usageMetadata", {})
                    if model != MODELS[0]:
                        import logging; logging.getLogger('vocr').info("fallback model used: %s", model)
                    return txt.strip(), u.get("totalTokenCount", 0), model
                import logging; logging.getLogger('vocr').info("gemini fail %s", f"{model}: HTTP {r.status_code} {r.text[:300]}")
                err = f"{model}: HTTP {r.status_code} {r.text[:200]}"
                if r.status_code in (429, 500, 503):
                    time.sleep(2 * (attempt + 1)); continue
                break  # 400/404 etc: try next model
            except Exception as e:
                err = f"{model}: {e}"; time.sleep(2)
    raise RuntimeError(err or "transcription failed")


def transcribe(strips, api_key, progress):
    chunks = [strips[i:i + CHUNK_IMAGES] for i in range(0, len(strips), CHUNK_IMAGES)]
    out, tokens, ctx = [], 0, ""
    for n, ch in enumerate(chunks):
        progress(n, len(chunks))
        txt, tok, _ = call_gemini(api_key, [s[1] for s in ch], ctx)
        tokens += tok
        # drop a leading repeat of the handshake text if the model echoed it
        if ctx and txt.startswith(ctx):
            txt = txt[len(ctx):].lstrip()
        if txt:
            out.append(txt)
            ctx = last_known_lines("\n".join(out))
    progress(len(chunks), len(chunks))
    return "\n\n".join(out), tokens
