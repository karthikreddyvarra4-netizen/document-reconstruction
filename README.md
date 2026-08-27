# Projected PDF Cleaner — Final Streamlit Cloud Package

Upload all four files to the root of your GitHub repository.

- `app.py` — application
- `requirements.txt` — Python dependencies
- `packages.txt` — installs the Tesseract OCR executable on Streamlit Cloud
- `README.md` — instructions

Set the Streamlit entrypoint to `app.py`.

IMPORTANT: `tesseract-ocr` cannot safely be embedded as a portable file inside
this ZIP because it is a Linux executable/package with system libraries.
`packages.txt` is the correct way to have Streamlit Cloud install it automatically.
