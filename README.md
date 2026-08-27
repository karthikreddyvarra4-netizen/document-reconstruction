# Projected Physics PDF Cleaner

This is the physics-symbol version.

Tesseract is REQUIRED and is actually executed:
- checks `tesseract --version` at startup
- runs `pytesseract.image_to_data()` on every image
- uses OCR question numbers to order pages
- displays OCR text and equation-like regions for verification
- normalizes common OCR words such as lambda/mu/omega/theta to λ/μ/ω/θ

Optional Mathpix:
- If you provide Mathpix App ID + App Key in the sidebar, detected
  equation-like regions are sent to Mathpix's v3/text endpoint with OCR=math.
- Mathpix supports math/text and returns LaTeX/MathML-oriented output.
- Without Mathpix credentials, the original equation crop is retained rather
  than inventing an equation.

Streamlit Cloud:
1. Upload app.py, requirements.txt, packages.txt to GitHub.
2. Deploy app.py.
3. `packages.txt` installs the Tesseract executable.
