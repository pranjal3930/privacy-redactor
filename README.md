# Privacy Redactor

Paste or load a document, review the personal details it finds, and get a safe version to share. It runs fully on your laptop: nothing is uploaded anywhere.

## Run it

1. Install Ollama from ollama.com, then pull the model (about 9.6 GB):
   ```
   ollama pull gemma4
   ```
2. Install and start the app:
   ```
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```
3. Open http://localhost:8000 and press **Try a sample**.

The pattern-based detectors work even without the model. If Ollama is not running, the app tells you and still redacts phones, IDs, emails and cards.

## Image mode

Open the **Image** tab, choose a photo or screenshot (an ID card, a bill, a chat screenshot), and press **Find personal details**. The app reads the text in the picture, finds the private details with the same detectors, and paints over exactly those words. Review the list, then **Download safe image**.

One extra install is needed, because reading text from pictures uses the free Tesseract program:

- **Windows**: download the installer from github.com/UB-Mannheim/tesseract/wiki and keep the default folder.
- **Mac**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr`

Then `pip install -r requirements.txt` again to get the two new Python packages. For Hindi text, install Tesseract's Hindi data and set `OCR_LANG = "eng+hin"` at the top of `main.py`.

Tips and limits:
- **Drag on the left image** to cover anything the reader missed, such as a face, a signature or a photo. These hand-drawn boxes are not detected automatically.
- The download is a new PNG with solid fill, so the covered pixels are gone for good and photo metadata (such as location) is dropped.
- Blurry, tilted or low-resolution pictures read worse. Check the safe image before sharing.

## What it finds

- **Pattern rules** (instant, no AI): phone numbers (with or without +91), Aadhaar-like 12-digit numbers, PAN, passport numbers, IFSC codes, emails, UPI IDs, and card numbers (checked with the Luhn test to cut false hits).
- **Local Gemma 4**: person names, addresses and organisations. Every item the model returns must appear word for word in your text, so invented results are dropped.

## Redaction styles

Label (`[PHONE]`), black blocks, or keep the last 4 characters. Untick anything that should stay, and use "Add something we missed" for items the detectors skip. Organisations are found but unticked by default.

## Limits

- Text only (.txt, .md, .csv, .log, .json). Scanned images and PDFs would need OCR, which is not included.
- Detection is a safety aid, not a guarantee. Always read the safe version before sharing.
- Long documents are scanned in 3000-character pieces, up to 30,000 characters total.

## Troubleshooting

- **"The local model could not be reached"**: start Ollama and run `ollama pull gemma4`.
- **Slow first scan**: the model is loading; the server warms it up on start.
- **Out of memory**: set `MODEL = "gemma4:e2b"` at the top of `main.py` and run `ollama pull gemma4:e2b`.
- Settings (model, size limits) sit at the top of `main.py`. New patterns go in the `PATTERNS` list.
