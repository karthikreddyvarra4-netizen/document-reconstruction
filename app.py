import io,re,cv2,numpy as np,streamlit as st
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
try:
 import pytesseract; OCR_IMPORT_OK=True
except Exception:
 pytesseract=None; OCR_IMPORT_OK=False
st.set_page_config(page_title='Projected PDF Cleaner',layout='wide')
def order_pts(p):
 p=np.asarray(p,dtype=np.float32); s=p.sum(1); d=np.diff(p,axis=1).ravel(); return np.array([p[np.argmin(s)],p[np.argmin(d)],p[np.argmax(s)],p[np.argmax(d)]],np.float32)
def page_quad(img):
 h,w=img.shape[:2]; scale=min(1.,1400/max(h,w)); sm=cv2.resize(img,None,fx=scale,fy=scale) if scale<1 else img.copy(); g=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY); e=cv2.Canny(cv2.GaussianBlur(g,(5,5),0),30,120); e=cv2.dilate(e,np.ones((5,5),np.uint8),iterations=1); cs,_=cv2.findContours(e,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); best=None; ba=0
 for c in cs:
  a=cv2.contourArea(c)
  if a<.20*sm.shape[0]*sm.shape[1]: continue
  q=cv2.approxPolyDP(c,.02*cv2.arcLength(c,True),True)
  if len(q)==4 and a>ba: best,ba=q.reshape(4,2),a
 if best is None: best=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
 else: best=best/scale
 return order_pts(best)
def rectify(img):
 tl,tr,br,bl=page_quad(img); W=max(600,int(max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl)))); H=max(800,int(max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr)))); src=np.array([tl,tr,br,bl],np.float32); dst=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],np.float32); return cv2.warpPerspective(img,cv2.getPerspectiveTransform(src,dst),(W,H),borderMode=cv2.BORDER_REPLICATE)
def clean(img,strength=1.):
 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); bg=cv2.GaussianBlur(gray,(0,0),45); norm=cv2.divide(gray,bg,scale=210); norm=cv2.createCLAHE(clipLimit=2+1.5*strength,tileGridSize=(8,8)).apply(norm); out=cv2.cvtColor(cv2.GaussianBlur(norm,(3,3),0),cv2.COLOR_GRAY2BGR); blur=cv2.GaussianBlur(out,(0,0),1.2); return cv2.addWeighted(out,1.35,blur,-.35,0)
def first_question_number(img):
 if not OCR_IMPORT_OK:return None
 try:
  txt=pytesseract.image_to_string(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),config='--psm 6',timeout=5); nums=[int(x) for x in re.findall(r'(?m)^\s*(\d{1,2})[\.\)]\s+',txt)]; return min(nums) if nums else None
 except Exception:return None
def stitch_pair(a,b):
 stc=cv2.Stitcher_create(cv2.Stitcher_SCANS); status,pano=stc.stitch([a,b])
 if status==cv2.Stitcher_OK:return pano,True
 W=max(a.shape[1],b.shape[1])
 def fit(x):return x if x.shape[1]==W else cv2.resize(x,(W,int(x.shape[0]*W/x.shape[1])))
 return np.vstack([fit(a),fit(b)]),False
def make_pdf(png):
 im=Image.open(io.BytesIO(png)).convert('RGB'); w,h=im.size; pw=595.; ph=pw*h/w; buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=(pw,ph)); c.drawImage(ImageReader(im),0,0,width=pw,height=ph,mask='auto'); c.showPage(); c.save(); return buf.getvalue()
st.title('Projected PDF → Clean Page'); st.caption('Perspective correction + projector-light correction + contrast enhancement + stitching')
uploads=st.file_uploader('Upload one or more projected-page images',type=['jpg','jpeg','png','webp'],accept_multiple_files=True)
a,b=st.columns(2)
with a:auto=st.checkbox('Auto-order by question number',True)
with b:strength=st.slider('Cleanup strength',0.,2.,1.,.1)
if uploads:
 raw=[]
 for f in uploads:
  arr=np.frombuffer(f.getvalue(),np.uint8); raw.append((f.name,cv2.imdecode(arr,cv2.IMREAD_COLOR)))
 prepared=[(n,rectify(im)) for n,im in raw]
 if auto:
  scored=[(first_question_number(im),n.lower(),n,im) for n,im in prepared]
  if any(x[0] is not None for x in scored): scored.sort(key=lambda x:(x[0] if x[0] is not None else 999,x[1])); prepared=[(x[2],x[3]) for x in scored]
  else: st.info('Question-number OCR is unavailable, so uploaded image order is preserved.')
 cleaned=[clean(im,strength) for _,im in prepared]; final=cleaned[0]; used=False
 for im in cleaned[1:]: final,u=stitch_pair(final,im); used|=u
 st.subheader('Result'); st.image(cv2.cvtColor(final,cv2.COLOR_BGR2RGB),use_container_width=True); st.caption('Overlap stitching: '+('used' if used else 'not available; images stacked in upload/detected order'))
 ok,enc=cv2.imencode('.png',final)
 if ok:
  p=enc.tobytes(); st.download_button('Download PNG',p,'cleaned_page.png','image/png'); st.download_button('Download PDF',make_pdf(p),'cleaned_page.pdf','application/pdf')
st.divider(); st.markdown('**Pipeline:** page detection → perspective correction → projector-light normalization → local contrast → sharpening → optional overlap stitching → PNG/PDF.')
