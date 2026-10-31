"""Privacy Redactor backend. Finds personal details in text using regex rules
(phones, Aadhaar-like numbers, PAN, cards, emails, UPI...) plus Gemma 4 running
locally in Ollama (names and addresses). Nothing leaves your machine."""
import base64, io, json, os, re, threading
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---- Settings (change here) -------------------------------------------------
MODEL = "gemma4"            # use "gemma4:e2b" if your laptop runs out of memory
OLLAMA = "http://localhost:11434"
MAX_CHARS = 30000
CHUNK = 3000                # characters sent to the model per call
NUM_CTX = 4096
OCR_LANG = "eng"            # add Hindi with "eng+hin" (needs Tesseract's Hindi language data)
TESSERACT_CMD = None        # full path to tesseract if it is not found automatically
MAX_IMAGE_BYTES = 15_000_000
BOX_PAD = 4                 # extra pixels around each covered word
# -----------------------------------------------------------------------------

PATTERNS = [  # (type, regex); earlier types win when the same text matches twice
    ("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("upi", r"\b[\w.\-]{2,}@[A-Za-z]{2,}\b(?!\.)"),
    ("aadhaar", r"(?<!\d)(?<!\d[\s-])[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?![\s-]?\d)"),
    ("phone", r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
    ("pan", r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    ("ifsc", r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    ("passport", r"\b[A-PR-WY][1-9]\d{6}\b"),
    ("card", r"\b(?:\d[ -]?){13,19}\b"),
]
AI_TYPES = {"name", "address", "org"}

SCHEMA = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {
        "type": "object",
        "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": sorted(AI_TYPES)}},
        "required": ["text", "type"]}}},
    "required": ["entities"],
}

SYSTEM = """You find personal information in text so it can be redacted before sharing.
List every:
- name: a person's name (full or partial), including names in greetings and signatures.
- address: a home or office address, as ONE item including city, state and pincode as written.
- org: an employer, school or company name.
Rules: copy each item EXACTLY as written in the text. Do not list phone numbers, emails or ID numbers.
Never invent anything. If there is nothing, return an empty list."""

app = FastAPI()


class In(BaseModel):
    text: str
    use_ai: bool = True


def luhn(s: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", s)]
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x *= 2
            x -= 9 if x > 9 else 0
        total += x
    return total % 10 == 0


def rule_entities(text: str) -> list:
    found = {}
    for kind, rx in PATTERNS:
        for m in re.finditer(rx, text):
            t = m.group(0).strip(" -")
            if kind == "card" and not luhn(t):
                continue
            if t and t not in found:
                found[t] = {"text": t, "type": kind, "source": "rule"}
    return list(found.values())


def chunks(text: str) -> list:
    out, cur = [], ""
    for line in text.splitlines(True):
        if cur and len(cur) + len(line) > CHUNK:
            out.append(cur)
            cur = ""
        while len(line) > CHUNK:
            out.append(line[:CHUNK])
            line = line[CHUNK:]
        cur += line
    if cur.strip():
        out.append(cur)
    return out


def llm(chunk: str) -> list:
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": MODEL, "stream": False, "think": False, "keep_alive": "30m",
        "format": SCHEMA, "options": {"temperature": 0, "num_ctx": NUM_CTX},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"TEXT:\n{chunk}"}]}, timeout=300)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"]).get("entities", [])


def squash(s: str) -> str:
    return " ".join(s.split())


def ai_entities(text: str, flex: bool = False) -> list:
    found = {}
    for chunk in chunks(text):
        for e in llm(chunk):
            t = str(e.get("text", "")).strip()
            if t and (squash(t) in squash(chunk) if flex else t in chunk) and e.get("type") in AI_TYPES and t not in found:  # must exist in the text
                found[t] = {"text": t, "type": e["type"], "source": "ai"}
    return list(found.values())


@app.on_event("startup")
def warm_up():
    def go():
        try:
            requests.post(f"{OLLAMA}/api/chat", json={
                "model": MODEL, "stream": False, "keep_alive": "30m",
                "messages": [{"role": "user", "content": "hi"}]}, timeout=300)
        except Exception:
            pass
    threading.Thread(target=go, daemon=True).start()


@app.post("/api/detect")
def detect(i: In):
    if not i.text.strip():
        raise HTTPException(400, "Paste or upload some text first.")
    if len(i.text) > MAX_CHARS:
        raise HTTPException(400, f"Text is too long. Keep it under {MAX_CHARS} characters.")
    ents, ai, message = find_entities(i.text, i.use_ai)
    return {"entities": ents, "ai": ai, "message": message}


def find_entities(text: str, use_ai: bool, flex: bool = False):
    ents = rule_entities(text)
    ai, message = "skipped", ""
    if use_ai:
        try:
            seen = {e["text"] for e in ents}
            ents += [e for e in ai_entities(text, flex) if e["text"] not in seen]
            ai = "ok"
        except Exception:
            ai = "unavailable"
            message = (f"The local model could not be reached, so only pattern-based details were found. "
                       f"Start Ollama and run: ollama pull {MODEL}")
    return ents, ai, message


# ---- Image mode: OCR finds words and their positions; the same detectors pick
# ---- the private ones; the page paints over those positions. ----------------
class ImgIn(BaseModel):
    image: str              # data URL from the browser
    use_ai: bool = True


def ocr_words(img) -> list:
    try:
        import pytesseract
        from PIL import ImageOps
    except ImportError:
        raise HTTPException(500, "Image support is missing. Run: pip install pillow pytesseract")
    win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    elif os.name == "nt" and os.path.exists(win):
        pytesseract.pytesseract.tesseract_cmd = win
    scale = 2 if img.width < 1400 else 1  # small images read better when enlarged
    work = ImageOps.autocontrast(img.convert("L"))
    if scale > 1:
        work = work.resize((img.width * scale, img.height * scale))
    try:
        d = pytesseract.image_to_data(work, lang=OCR_LANG, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        raise HTTPException(500, "Tesseract is not installed. See the README section 'Image mode'.")
    except Exception as e:
        raise HTTPException(500, f"Reading the image failed: {e}")
    words = []
    for k, t in enumerate(d["text"]):
        try:
            conf = float(d["conf"][k])
        except (TypeError, ValueError):
            conf = -1
        if t.strip() and conf >= 0:
            words.append({"text": t.strip(), "line": (d["block_num"][k], d["par_num"][k], d["line_num"][k]),
                          "box": [d["left"][k] / scale, d["top"][k] / scale, d["width"][k] / scale, d["height"][k] / scale]})
    return words


def words_to_text(words: list) -> str:
    parts, pos, cur = [], 0, None
    for w in words:
        sep = "" if not parts else ("\n" if w["line"] != cur else " ")
        pos += len(sep)
        w["start"], w["end"] = pos, pos + len(w["text"])
        pos = w["end"]
        parts += [sep, w["text"]]
        cur = w["line"]
    return "".join(parts)


def find_boxes(ent: str, words: list, text: str) -> list:
    pat = r"\s+".join(re.escape(p) for p in ent.split())
    boxes = []
    for m in re.finditer(pat, text):
        hit = [w for w in words if w["start"] < m.end() and w["end"] > m.start()]
        for line in dict.fromkeys(w["line"] for w in hit):  # one box per line the match touches
            ws = [w for w in hit if w["line"] == line]
            x0 = min(w["box"][0] for w in ws); y0 = min(w["box"][1] for w in ws)
            x1 = max(w["box"][0] + w["box"][2] for w in ws); y1 = max(w["box"][1] + w["box"][3] for w in ws)
            boxes.append([x0 - BOX_PAD, y0 - BOX_PAD, x1 - x0 + 2 * BOX_PAD, y1 - y0 + 2 * BOX_PAD])
    return boxes


@app.post("/api/detect-image")
def detect_image(i: ImgIn):
    try:
        from PIL import Image
    except ImportError:
        raise HTTPException(500, "Image support is missing. Run: pip install pillow pytesseract")
    try:
        raw = base64.b64decode(i.image.split(",", 1)[-1])
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise HTTPException(400, "Could not read that image. Try a smaller PNG or JPG.")
    words = ocr_words(img)
    if not words:
        return {"entities": [], "ai": "skipped", "message": "No text found. Drag on the image to cover things by hand."}
    text = words_to_text(words)
    ents, ai, message = find_entities(text, i.use_ai, flex=True)
    out = []
    for e in ents:
        boxes = find_boxes(e["text"], words, text)
        if boxes:
            out.append({**e, "boxes": boxes})
    return {"entities": out, "ai": ai, "message": message}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
