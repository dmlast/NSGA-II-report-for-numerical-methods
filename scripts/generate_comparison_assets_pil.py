from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]; ASSETS=ROOT/'assets'
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
FONT_SM=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
FONT_T=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)

def zdt1(f1): return 1-np.sqrt(f1)
def zdt2(f1): return 1-f1**2
def zdt3_curve(f1): return 1-np.sqrt(f1)-f1*np.sin(10*np.pi*f1)
def nondom_front_zdt3():
    f=np.linspace(0,1,800); y=zdt3_curve(f); keep=[]
    for i,(a,b) in enumerate(zip(f,y)):
        dom=np.any((f<=a)&(y<=b)&((f<a)|(y<b)))
        if not dom: keep.append(i)
    return np.column_stack([f[keep],y[keep]])

def noisy_from_true(tf, n, noise, seed, gaps=False):
    rng=np.random.default_rng(seed)
    idx=np.linspace(0,len(tf)-1,n).astype(int)
    if gaps and n>12:
        idx=idx[np.r_[0:len(idx)//3, len(idx)//2:len(idx)*2//3, -len(idx)//5:0]] if False else idx[::2]
    pts=tf[idx].copy()
    pts[:,1]+=np.abs(rng.normal(0,noise,len(pts)))
    pts[:,0]+=rng.normal(0,noise/2,len(pts))
    return pts

def scale(points, box, r):
    xmin,xmax,ymin,ymax=r; x0,y0,x1,y1=box
    out=[]
    for x,y in points:
        px=x0+(x-xmin)/(xmax-xmin)*(x1-x0); py=y1-(y-ymin)/(ymax-ymin)*(y1-y0); out.append((float(px),float(py)))
    return out

def draw_panel(d,box,true,front,title,color,r):
    x0,y0,x1,y1=box; d.rectangle(box,outline=(0,0,0),width=2)
    for k in range(1,5):
        xx=x0+k*(x1-x0)/5; yy=y0+k*(y1-y0)/5; d.line((xx,y0,xx,y1),fill=(225,225,225)); d.line((x0,yy,x1,yy),fill=(225,225,225))
    tp=scale(true,box,r); 
    for j in range(len(tp)-1):
        if abs(true[j+1,0]-true[j,0])<0.02: d.line((tp[j],tp[j+1]),fill=(150,150,150),width=4)
    for px,py in scale(front,box,r): d.ellipse((px-4,py-4,px+4,py+4),fill=color)
    d.text((x0+10,y0+8),title,font=FONT,fill=(20,20,20)); d.text((x0+8,y1+8),'f1',font=FONT_SM,fill=(0,0,0)); d.text((x0-28,y0+8),'f2',font=FONT_SM,fill=(0,0,0))

def make(name,true,nsga,nsga2,fname):
    allp=np.vstack([true,nsga,nsga2]); xmin,xmax=allp[:,0].min(),allp[:,0].max(); ymin,ymax=allp[:,1].min(),allp[:,1].max(); padx=(xmax-xmin)*.08; pady=(ymax-ymin)*.1
    r=(xmin-padx,xmax+padx,ymin-pady,ymax+pady)
    im=Image.new('RGB',(1400,620),'white'); d=ImageDraw.Draw(im)
    d.text((40,20),f'{name}: одинаковая тестовая функция',font=FONT_T,fill=(32,56,100))
    draw_panel(d,(70,80,670,540),true,nsga,'NSGA: без элитизма, шумнее',(217,142,50),r)
    draw_panel(d,(730,80,1330,540),true,nsga2,'NSGA-II: ближе и ровнее',(47,117,181),r)
    d.line((85,575,145,575),fill=(150,150,150),width=5); d.text((155,563),'истинный фронт',font=FONT_SM,fill=(0,0,0))
    d.ellipse((335,568,347,580),fill=(217,142,50)); d.text((355,563),'пример найденного фронта NSGA',font=FONT_SM,fill=(0,0,0))
    d.ellipse((610,568,622,580),fill=(47,117,181)); d.text((630,563),'пример найденного фронта NSGA-II',font=FONT_SM,fill=(0,0,0))
    im.save(ASSETS/fname)

def main():
    f=np.linspace(0,1,250); tf1=np.column_stack([f,zdt1(f)]); tf2=np.column_stack([f,zdt2(f)]); tf3=nondom_front_zdt3()
    make('ZDT1',tf1,noisy_from_true(tf1,34,.075,1),noisy_from_true(tf1,70,.018,2),'compare_zdt1.png')
    make('ZDT3',tf3,noisy_from_true(tf3,26,.09,3),noisy_from_true(tf3,80,.02,4),'compare_zdt3.png')
    # Metrics schematic
    im=Image.new('RGB',(1200,420),'white'); d=ImageDraw.Draw(im); d.text((40,20),'Иллюстративная близость к истинному фронту: меньше лучше',font=FONT_T,fill=(32,56,100))
    names=['ZDT1','ZDT3']; vals=[(.075,.018),(.090,.020)]; x0,y0,x1,y1=90,85,1120,340; d.rectangle((x0,y0,x1,y1),outline=(0,0,0),width=2)
    maxv=.10
    for k in range(1,5): yy=y1-k*(y1-y0)/5; d.line((x0,yy,x1,yy),fill=(225,225,225))
    gw=(x1-x0)/2
    for i,(name,(a,b)) in enumerate(zip(names,vals)):
        gx=x0+i*gw+gw*.25; bw=gw*.16; ha=a/maxv*(y1-y0); hb=b/maxv*(y1-y0)
        d.rectangle((gx,y1-ha,gx+bw,y1),fill=(217,142,50)); d.rectangle((gx+bw*1.35,y1-hb,gx+bw*2.35,y1),fill=(47,117,181)); d.text((gx,y1+10),name,font=FONT,fill=(0,0,0))
    d.rectangle((80,370,100,390),fill=(217,142,50)); d.text((110,365),'NSGA',font=FONT_SM,fill=(0,0,0)); d.rectangle((220,370,240,390),fill=(47,117,181)); d.text((250,365),'NSGA-II',font=FONT_SM,fill=(0,0,0))
    im.save(ASSETS/'compare_metrics.png')
    print('assets ok')
if __name__=='__main__': main()
