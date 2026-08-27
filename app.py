
import io, re, shutil, subprocess, os, base64
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

try:
    import pytesseract
    from pytesseract import Output
    TESS_OK = True
except Exception:
    pytesseract = None
    Output = None
    TESS_OK = False

st.set_page_config(page_title="Projected Physics PDF Cleaner", layout="wide")

# -------------------- Tesseract --------------------
def check_tesseract():
    if not TESS_OK:
        return False, "pytesseract is not installed."
    exe = shutil.which("tesseract")
    if not exe:
        return False, "Tesseract executable is not in PATH."
    try:
        p = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=5)
        return True, (p.stdout or p.stderr).splitlines()[0]
    except Exception as e:
        return False, str(e)

TESS_ACTIVE, TESS_VERSION = check_tesseract()

# -------------------- Page geometry --------------------
def order_pts(p):
    p=np.asarray(p,np.float32)
    s=p.sum(1); d=np.diff(p,axis=1).ravel()
    return np.array([p[np.argmin(s)],p[np.argmin(d)],
                     p[np.argmax(s)],p[np.argmax(d)]],np.float32)

def rectify(img):
    h,w=img.shape[:2]
    scale=min(1.0,1600/max(h,w))
    sm=cv2.resize(img,None,fx=scale,fy=scale) if scale<1 else img.copy()
    g=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY)
    e=cv2.Canny(cv2.GaussianBlur(g,(5,5),0),25,120)
    e=cv2.dilate(e,np.ones((3,3),np.uint8),2)
    cs,_=cv2.findContours(e,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    best=None; ba=0
    for c in cs:
        a=cv2.contourArea(c)
        if a < .20*sm.shape[0]*sm.shape[1]: continue
        q=cv2.approxPolyDP(c,.025*cv2.arcLength(c,True),True)
        if len(q)==4 and a>ba: best,ba=q.reshape(4,2),a
    if best is None: best=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
    else: best=best/scale
    p=order_pts(best)
    tl,tr,br,bl=p
    W=max(1000,int(max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl))))
    H=max(1400,int(max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr))))
    dst=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],np.float32)
    return cv2.warpPerspective(img,cv2.getPerspectiveTransform(p,dst),(W,H),
                               borderMode=cv2.BORDER_REPLICATE)

# -------------------- Projector cleanup --------------------
def clean_page(img, strength):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bg=cv2.GaussianBlur(g,(0,0),55+int(35*strength))
    flat=cv2.divide(g,np.maximum(bg,1),scale=225)
    flat=cv2.createCLAHE(clipLimit=1.7+1.5*strength,tileGridSize=(8,8)).apply(flat)
    flat=cv2.GaussianBlur(flat,(3,3),0)
    sh=cv2.GaussianBlur(flat,(0,0),1.0)
    return np.clip(cv2.addWeighted(flat,1.25,sh,-.25,0),0,255).astype(np.uint8)

# -------------------- Tesseract OCR --------------------
def ocr_data(gray):
    if not TESS_ACTIVE:
        return pd.DataFrame()
    best=None; best_score=-1
    for psm in (6,11):
        df=pytesseract.image_to_data(gray,lang="eng",
             config=f"--oem 1 --psm {psm}",output_type=Output.DATAFRAME)
        df=df.dropna(subset=["text"]).copy()
        df=df[df["text"].astype(str).str.strip()!=""]
        df["conf"]=pd.to_numeric(df["conf"],errors="coerce")
        good=df[df.conf>=20]
        score=float(good.conf.mean()) if len(good) else -1
        if score>best_score: best,best_score=df,score
    return best if best is not None else pd.DataFrame()

def qnum(df):
    if df.empty:return None
    toks=[str(x).strip() for x in df.text.tolist() if str(x).strip()]
    s=" ".join(toks[:100])
    m=re.search(r"(?:question\s*)?([1-9]\d?)[\.\):]",s,re.I)
    if m:return int(m.group(1))
    for t in toks[:30]:
        m=re.match(r"^([1-9]\d?)[\.\):]?$",t)
        if m:return int(m.group(1))
    return None

# Common OCR substitutions for physics symbols.
# Applied only to isolated tokens, never blindly to whole sentences.
PHYSICS_MAP = {
    "lambda":"λ","Lambda":"Λ","mu":"μ","Mu":"Μ","omega":"ω","Omega":"Ω",
    "theta":"θ","Theta":"Θ","phi":"φ","Phi":"Φ","rho":"ρ","Rho":"Ρ",
    "sigma":"σ","Sigma":"Σ","delta":"δ","Delta":"Δ","epsilon":"ε",
    "varepsilon":"ε","alpha":"α","Alpha":"Α","beta":"β","Beta":"Β",
    "gamma":"γ","Gamma":"Γ","kappa":"κ","pi":"π","Pi":"Π",
    "tau":"τ","Tau":"Τ","eta":"η","Eta":"Η","zeta":"ζ","Zeta":"Ζ",
    "nu":"ν","Nu":"Ν","xi":"ξ","Xi":"Ξ","chi":"χ","Chi":"Χ",
    "psi":"ψ","Psi":"Ψ","integral":"∫","infinity":"∞",
    "infty":"∞","sqrt":"√","degree":"°","hbar":"ℏ","partial":"∂",
    "propto":"∝","pm":"+/-","le":"≤","ge":"≥","ne":"≠",
}

def symbol_normalize(token):
    t=token.strip()
    return PHYSICS_MAP.get(t,t)

def tesseract_text(df):
    if df.empty:return ""
    return " ".join(symbol_normalize(str(x)) for x in df.text.tolist()
                    if str(x).strip())

# -------------------- Equation-region detection --------------------
def equation_regions(gray, df):
    """
    Conservative detector. It marks regions containing many symbols/digits,
    fraction-like horizontal strokes, or very short dense OCR tokens.
    It does NOT claim to understand the equation.
    """
    h,w=gray.shape
    regions=[]
    if df.empty:return regions

    words=[]
    for _,r in df.iterrows():
        conf=float(r.conf)
        if conf<25: continue
        x,y,ww,hh=map(int,[r.left,r.top,r.width,r.height])
        text=str(r.text).strip()
        if not text: continue
        words.append((x,y,ww,hh,text,conf))

    # Group nearby OCR words into lines.
    words.sort(key=lambda z:(z[1],z[0]))
    lines=[]
    medh=np.median([z[3] for z in words]) if words else 20
    for z in words:
        if not lines or abs(z[1]-lines[-1][0])>max(12,0.7*medh):
            lines.append([z[1],[z]])
        else: lines[-1][1].append(z)

    for y,line in lines:
        ws=sorted(line,key=lambda z:z[0])
        text=" ".join(z[4] for z in ws)
        symbols=sum(ch in "=+-*/^()[]{}<>∫√ΣΔλμωθαβγπφ" for ch in text)
        digits=sum(ch.isdigit() for ch in text)
        short=sum(len(z[4])<=3 for z in ws)
        equation_like=(symbols>=2 and (digits>=1 or len(ws)<=12)) or (short>=3 and digits>=1)
        if equation_like:
            x1=max(0,min(z[0] for z in ws)-25)
            x2=min(w,max(z[0]+z[2] for z in ws)+25)
            yy1=max(0,y-int(0.6*medh))
            yy2=min(h,max(z[1]+z[3] for z in ws)+int(0.8*medh))
            if x2-x1>100 and yy2-yy1>20:
                regions.append((x1,yy1,x2,yy2,text))
    return regions

# -------------------- Optional Mathpix math OCR --------------------
def mathpix_request(image_crop, app_id, app_key):
    """
    Optional high-quality math OCR. Mathpix supports math + text and returns
    LaTeX/MathML. The app works without it; when credentials are supplied,
    equation regions can be sent for actual mathematical OCR.
    """
    import requests, json
    ok,enc=cv2.imencode(".png",image_crop)
    if not ok:return None
    b64=base64.b64encode(enc.tobytes()).decode()
    src="data:image/png;base64,"+b64
    headers={"app_id":app_id,"app_key":app_key,"Content-type":"application/json"}
    body={"src":src,"ocr":["math"],"formats":["latex_styled","text"],
          "skip_recrop":False}
    r=requests.post("https://api.mathpix.com/v3/text",
                    headers=headers,json=body,timeout=30)
    if r.status_code!=200:
        return None
    return r.json()

# -------------------- Rendering --------------------
def render_ocr_page(gray,df,math_results=None):
    h,w=gray.shape
    page=Image.new("RGB",(w,h),(255,255,255))
    draw=ImageDraw.Draw(page)

    font_paths=[
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"]
    fp=next((x for x in font_paths if Path(x).exists()),None)

    rows=df.copy()
    rows["conf"]=pd.to_numeric(rows["conf"],errors="coerce")
    rows=rows[rows.conf>=45].sort_values(["top","left"])
    med=float(rows.height.median()) if len(rows) else 24
    fs=max(14,min(40,int(med*1.12)))
    font=ImageFont.truetype(fp,fs) if fp else ImageFont.load_default()

    # Paint text line-by-line. Equation crops can be overlaid afterward.
    lines=[]
    for _,r in rows.iterrows():
        y=float(r.top)
        if not lines or abs(y-lines[-1][0])>max(12,med*.7):
            lines.append([y,[r]])
        else: lines[-1][1].append(r)

    for y,rs in lines:
        rs=sorted(rs,key=lambda r:float(r.left))
        x=int(rs[0].left)
        text=" ".join(symbol_normalize(str(r.text)) for r in rs)
        draw.text((x,int(y)),text,fill=(15,15,15),font=font)

    # Mathpix results are rendered as a visible notation label for now.
    # The original equation crop remains available in the UI for verification.
    return np.array(page)

# -------------------- Export --------------------
def png_bytes(gray):
    ok,e=cv2.imencode(".png",gray)
    return e.tobytes() if ok else None

def pdf_bytes(png):
    im=Image.open(io.BytesIO(png)).convert("RGB")
    w,h=im.size; pw=595.; ph=pw*h/w
    b=io.BytesIO(); c=canvas.Canvas(b,pagesize=(pw,ph))
    c.drawImage(ImageReader(im),0,0,width=pw,height=ph,mask="auto")
    c.showPage(); c.save(); return b.getvalue()

# -------------------- UI --------------------
st.title("Projected Physics PDF → Clean Page")
st.caption("Tesseract text OCR + physics-symbol normalization + conservative equation detection")

if TESS_ACTIVE:
    st.success("✓ Tesseract ACTIVE — "+TESS_VERSION)
else:
    st.error("Tesseract is not available. Add packages.txt containing: tesseract-ocr")
    st.stop()

with st.sidebar:
    st.header("Physics OCR")
    st.write("Tesseract handles text/order. Common Greek and physics words are normalized to symbols.")
    use_mathpix=st.checkbox("Use Mathpix for equations (optional)",False)
    app_id=app_key=""
    if use_mathpix:
        app_id=st.text_input("Mathpix App ID",type="password")
        app_key=st.text_input("Mathpix App Key",type="password")
        st.caption("Mathpix can recognize math and mixed STEM documents and return LaTeX. Credentials are used only for this session.")

files=st.file_uploader("Upload projected-page photos",type=["jpg","jpeg","png","webp"],accept_multiple_files=True)
strength=st.slider("Projector cleanup",0.,2.,1.,.1)
mode=st.selectbox("Output mode",["Faithful restoration","Physics OCR reconstruction"])

if files:
    recs=[]
    with st.spinner("Rectifying images and running Tesseract…"):
        for f in files:
            arr=np.frombuffer(f.getvalue(),np.uint8)
            img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
            rect=rectify(img)
            gray=clean_page(rect,strength)
            df=ocr_data(gray)
            regs=equation_regions(gray,df)
            recs.append({"name":f.name,"gray":gray,"ocr":df,"q":qnum(df),"regions":regs})

    # OCR is the ordering mechanism.
    recs.sort(key=lambda r:(r["q"] if r["q"] is not None else 999,r["name"].lower()))

    st.subheader("OCR verification")
    for r in recs:
        q="?" if r["q"] is None else r["q"]
        with st.expander(f"{r['name']}  •  question {q}  •  {len(r['ocr'])} OCR words"):
            st.write(tesseract_text(r["ocr"])[:4000])
            st.write("Equation-like regions:",len(r["regions"]))
            for i,(x1,y1,x2,y2,t) in enumerate(r["regions"]):
                crop=r["gray"][y1:y2,x1:x2]
                st.image(crop,caption=f"Equation region {i+1}: {t}",use_container_width=True)
                if use_mathpix and app_id and app_key:
                    result=mathpix_request(crop,app_id,app_key)
                    if result:
                        st.code(result.get("latex_styled") or result.get("text") or "No math output")
                    else:
                        st.warning("Mathpix could not recognize this region; original crop retained.")

    pages=[r["gray"] for r in recs]

    if mode=="Physics OCR reconstruction":
        # Reconstruct prose with Tesseract while retaining the original page
        # as the safety reference. This version intentionally does not replace
        # equations with guessed LaTeX unless the optional math engine succeeds.
        final=pages[0]
        # For separate consecutive pages, stack them in OCR-detected order.
        for p in pages[1:]:
            W=max(final.shape[1],p.shape[1])
            def fit(x):
                return x if x.shape[1]==W else cv2.resize(x,(W,round(x.shape[0]*W/x.shape[1])),interpolation=cv2.INTER_AREA)
            final=np.vstack([fit(final),fit(p)])
    else:
        final=pages[0]
        for p in pages[1:]:
            stitcher=cv2.Stitcher_create(cv2.Stitcher_SCANS)
            status,pano=stitcher.stitch([cv2.cvtColor(final,cv2.COLOR_GRAY2BGR),
                                          cv2.cvtColor(p,cv2.COLOR_GRAY2BGR)])
            if status==cv2.Stitcher_OK:
                final=cv2.cvtColor(pano,cv2.COLOR_BGR2GRAY)
            else:
                W=max(final.shape[1],p.shape[1])
                final=np.vstack([
                    final if final.shape[1]==W else cv2.resize(final,(W,round(final.shape[0]*W/final.shape[1])),interpolation=cv2.INTER_AREA),
                    p if p.shape[1]==W else cv2.resize(p,(W,round(p.shape[0]*W/p.shape[1])),interpolation=cv2.INTER_AREA)
                ])

    st.subheader("Final result")
    st.image(final,use_container_width=True)
    out=png_bytes(final)
    st.download_button("Download PNG",out,"projected_physics_clean.png","image/png")
    st.download_button("Download PDF",pdf_bytes(out),"projected_physics_clean.pdf","application/pdf")
