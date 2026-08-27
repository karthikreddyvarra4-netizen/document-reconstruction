
import io, re, cv2, numpy as np, streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

try:
    import pytesseract
    OCR_OK = True
except Exception:
    pytesseract, OCR_OK = None, False

st.set_page_config(page_title="Projected PDF Cleaner", layout="wide")

def order_pts(p):
    p=np.asarray(p,np.float32); s=p.sum(1); d=np.diff(p,axis=1).ravel()
    return np.array([p[np.argmin(s)],p[np.argmin(d)],p[np.argmax(s)],p[np.argmax(d)]],np.float32)

def detect_page(img):
    h,w=img.shape[:2]; scale=min(1.,1600/max(h,w))
    sm=cv2.resize(img,None,fx=scale,fy=scale) if scale<1 else img.copy()
    g=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY)
    e=cv2.Canny(cv2.GaussianBlur(g,(5,5),0),25,110)
    e=cv2.dilate(e,np.ones((3,3),np.uint8),iterations=2)
    cs,_=cv2.findContours(e,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    best=None; area_best=0
    for c in cs:
        area=cv2.contourArea(c)
        if area<.20*sm.shape[0]*sm.shape[1]: continue
        q=cv2.approxPolyDP(c,.025*cv2.arcLength(c,True),True)
        if len(q)==4 and area>area_best: best,area_best=q.reshape(4,2),area
    if best is None: best=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
    else: best=best/scale
    return order_pts(best)

def rectify(img):
    p=detect_page(img); tl,tr,br,bl=p
    W=max(900,int(max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl))))
    H=max(1200,int(max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr))))
    dst=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],np.float32)
    return cv2.warpPerspective(img,cv2.getPerspectiveTransform(p,dst),(W,H),borderMode=cv2.BORDER_REPLICATE)

def clean(img,strength=1.0):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bg=cv2.GaussianBlur(gray,(0,0),55+int(35*strength))
    norm=cv2.divide(gray,np.maximum(bg,1),scale=220)
    norm=cv2.createCLAHE(clipLimit=1.7+1.5*strength,tileGridSize=(8,8)).apply(norm)
    norm=cv2.GaussianBlur(norm,(3,3),0)
    blur=cv2.GaussianBlur(norm,(0,0),1.0)
    sharp=cv2.addWeighted(norm,1.22,blur,-.22,0)
    return cv2.cvtColor(np.clip(sharp,0,255).astype(np.uint8),cv2.COLOR_GRAY2RGB)

def qnum(img):
    if not OCR_OK: return None
    try:
        txt=pytesseract.image_to_string(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY),config="--psm 6",timeout=6)
        n=[int(x) for x in re.findall(r'(?m)^\s*(\d{1,2})[\.\)]\s+',txt)]
        return min(n) if n else None
    except Exception: return None

def stitch(a,b):
    stc=cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status,p=stc.stitch([cv2.cvtColor(a,cv2.COLOR_RGB2BGR),cv2.cvtColor(b,cv2.COLOR_RGB2BGR)])
    if status==cv2.Stitcher_OK: return cv2.cvtColor(p,cv2.COLOR_BGR2RGB),True
    W=max(a.shape[1],b.shape[1])
    def fit(x): return x if x.shape[1]==W else cv2.resize(x,(W,round(x.shape[0]*W/x.shape[1])),interpolation=cv2.INTER_AREA)
    return np.vstack([fit(a),fit(b)]),False

def png_bytes(rgb):
    ok,e=cv2.imencode(".png",cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    return e.tobytes() if ok else None

def pdf_bytes(png):
    im=Image.open(io.BytesIO(png)).convert("RGB"); w,h=im.size
    pw=595.; ph=pw*h/w; b=io.BytesIO(); c=canvas.Canvas(b,pagesize=(pw,ph))
    c.drawImage(ImageReader(im),0,0,width=pw,height=ph,mask="auto"); c.showPage(); c.save()
    return b.getvalue()

st.title("Projected PDF → Clean Page")
st.caption("Perspective correction • projector-light correction • OCR ordering • stitching")

files=st.file_uploader("Upload projected-page images",type=["jpg","jpeg","png","webp"],accept_multiple_files=True)
c1,c2=st.columns(2)
with c1: mode=st.selectbox("Output mode",["Faithful restoration","Clean reconstruction"])
with c2: strength=st.slider("Cleanup strength",0.,2.,1.,.1)
auto=st.checkbox("Auto-order questions",True)

if files:
    raw=[]
    for f in files:
        a=np.frombuffer(f.getvalue(),np.uint8); im=cv2.imdecode(a,cv2.IMREAD_COLOR)
        if im is not None: raw.append((f.name,im))
    prepared=[(n,rectify(im)) for n,im in raw]

    if auto and OCR_OK:
        scored=[(qnum(im) if qnum(im) is not None else 999,n.lower(),n,im) for n,im in prepared]
        if any(x[0]!=999 for x in scored):
            scored.sort(key=lambda x:(x[0],x[1])); prepared=[(x[2],x[3]) for x in scored]
    elif auto and not OCR_OK:
        st.warning("OCR is unavailable; upload order is preserved.")

    pages=[clean(im,strength) for _,im in prepared]
    final=pages[0]; used=False
    for p in pages[1:]:
        final,ok=stitch(final,p); used|=ok

    if mode=="Clean reconstruction":
        pil=Image.fromarray(final).convert("L")
        pil=ImageOps.autocontrast(pil,cutoff=1)
        pil=ImageEnhance.Contrast(pil).enhance(1.25)
        pil=pil.filter(ImageFilter.UnsharpMask(radius=1.2,percent=110,threshold=3))
        final=np.array(pil.convert("RGB"))

    st.image(final,use_container_width=True)
    st.caption("Order: "+" → ".join(n for n,_ in prepared)+" | overlap stitching: "+("yes" if used else "no; stacked"))
    p=png_bytes(final)
    if p:
        st.download_button("Download PNG",p,"cleaned_projected_page.png","image/png")
        st.download_button("Download PDF",pdf_bytes(p),"cleaned_projected_page.pdf","application/pdf")

st.divider()
st.write("The app never fabricates unreadable mathematical symbols. The reconstruction mode improves the page appearance while retaining the photographed content.")
