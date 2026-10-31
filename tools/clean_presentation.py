"""World-only English presentation of actual native states; no transition logic."""
import colorsys,hashlib,math
from PIL import Image,ImageDraw,ImageFont

def color(label):
    h=int(hashlib.sha256(str(label).encode()).hexdigest()[:6],16)/0xffffff
    return tuple(int(x*255) for x in colorsys.hsv_to_rgb(h,.43,.76))

def abstract(frame):
    im=Image.new('RGB',(640,400),'#10232b');d=ImageDraw.Draw(im);font=ImageFont.load_default(size=14);small=ImageFont.load_default(size=11)
    d.line((174,183,466,183),fill='#759594',width=3)
    shared=[];agent=[[],[]]
    excluded={'t','done','terminated','truncated','last_action','last','last_reward'}
    for k,v in frame.get('abstract_state',{}).items():
        if k in excluded:continue
        if isinstance(v,list) and len(v)==2:
            for i in range(2):agent[i].append((k,v[i]))
        elif not isinstance(v,(list,dict)):shared.append((k,v))
    def fmt(v):
        if isinstance(v,bool):return 'yes' if v else 'no'
        if isinstance(v,(int,float)):return f'{v:.3g}'
        if isinstance(v,list):return '['+', '.join(fmt(x) for x in v[:4])+']'
        return str(v)
    for i,cx in enumerate([150,490]):
        d.rounded_rectangle((cx-128,55,cx+128,333),radius=18,fill='#1e3c46',outline=color(i),width=3)
        d.ellipse((cx-22,70,cx+22,114),fill=color(i),outline='#e4e8d4',width=2)
        d.text((cx-29,121),f'Agent {i}',fill='#edf1de',font=font)
        for j,(name,val) in enumerate(agent[i][:8]):
            d.text((cx-110,151+j*20),name.replace('_',' ')[:22],font=small,fill='#a7c6c7')
            d.text((cx+23,151+j*20),fmt(val)[:18],font=small,fill='#edf1de')
    if shared:
        d.rectangle((231,12,409,47),fill='#2f4e51')
        d.text((240,20),' / '.join(k+': '+fmt(v) for k,v in shared[:2])[:29],font=small,fill='#e1dbad')
    d.text((18,373),f'Native abstract state     t={frame["clock"]}',font=small,fill='#b8c9be')
    return im

def draw(frame):
    if frame.get('legacy'):return abstract(frame)
    im=Image.new('RGB',(640,440),'#12252c');d=ImageDraw.Draw(im);font=ImageFont.load_default(size=10)
    x0,y0,x1,y1=frame['bounds'];scale=min(612/max(1,x1-x0+1),410/max(1,y1-y0+1));ox=(640-scale*(x1-x0+1))/2;oy=(440-scale*(y1-y0+1))/2
    def pt(xy):return ox+(xy[0]-x0+.5)*scale,oy+(xy[1]-y0+.5)*scale
    palette={t.get('value'):t.get('color') for t in frame.get('tile_legend',[]) if t.get('color')}
    grid=frame.get('map')
    if grid:
        mx,my=frame['map_origin'];sx,sy=frame['cell_size']
        for y,row in enumerate(grid):
            for x,v in enumerate(row):
                px,py=pt([mx+x*sx,my+y*sy]);r=scale*.49
                d.rectangle((px-r,py-r,px+r,py+r),fill=palette.get(v,color('tile'+str(v))) if v is not None else '#172a30',outline='#30494b')
    for e in sorted(frame['entities'],key=lambda e:e['kind']=='agent'):
        x,y=pt(e['xy']);r=max(2,min(10,scale*.32));c=e.get('color',color(e['group']));kind=e.get('glyph','').lower();group=e['group'];a=e.get('attributes',{})
        if e['kind']=='agent':
            d.ellipse((x-r*.45,y-r,x+r*.45,y-r*.1),fill='#dfb99c',outline='#243936')
            d.rectangle((x-r*.6,y-r*.1,x+r*.6,y+r*.8),fill=color(next(iter(a.values()),e['index'])),outline='#dce4cb')
            if scale>=22:d.text((x+r,y-r),str(e['index']),font=font,fill='#f8f2db')
        elif 'plant' in kind or any(s in group for s in ['crop','plot','garden','tree']):
            d.rectangle((x-r,y,x+r,y+r),fill='#806249');d.line((x,y+r*.5,x,y-r),fill='#b7d19d',width=max(1,int(r*.2)))
            d.ellipse((x-r,y-r*.7,x,y+r*.1),fill='#6daa65');d.ellipse((x,y-r,x+r,y-r*.2),fill='#9ac67e')
        elif 'water' in kind or any(s in group for s in ['water','reservoir','trough']):
            d.rectangle((x-r,y-r*.6,x+r,y+r*.6),fill='#4f9bb3',outline='#abdbe0');d.line((x-r*.7,y,x+r*.7,y),fill='#c8e8e9')
        elif 'gate' in kind or 'gate' in group or 'door' in group:
            d.rectangle((x-r,y-r,x-r*.6,y+r),fill=c);d.rectangle((x+r*.6,y-r,x+r,y+r),fill=c)
            opened=next((v for k,v in a.items() if 'open' in k),False)
            if not opened:d.rectangle((x-r*.7,y-r*.2,x+r*.7,y+r*.2),fill=c)
        elif any(s in group for s in ['herd','animal','grazer','sheep']):
            d.ellipse((x-r,y-r*.6,x+r*.6,y+r*.4),fill=c,outline='#394c40');d.ellipse((x+r*.3,y-r,x+r,y-r*.2),fill=c)
        elif any(s in group for s in ['boat','barge','vessel','ship']):
            d.polygon([(x-r,y-r*.6),(x+r,y-r*.6),(x+r*.6,y+r*.7),(x-r*.6,y+r*.7)],fill=c,outline='#dbe0c4')
        elif any(s in kind+' '+group for s in ['fire','flame']):
            d.polygon([(x-r,y+r),(x,y-r),(x+r,y+r)],fill='#d78c56');d.ellipse((x-r*.4,y,x+r*.4,y+r),fill='#eed47c')
        else:
            d.rectangle((x-r,y-r,x+r,y+r),fill=c,outline='#e1d7ad');d.line((x-r*.7,y-r*.7,x+r*.7,y+r*.7),fill='#ece1bb')
    d.text((8,425),f't={frame["clock"]}',font=font,fill='#d6dfce')
    return im

def save(frames,path):
    images=[draw(f) for f in frames]
    images[0].save(path,save_all=True,append_images=images[1:],duration=230,loop=0,optimize=False)
