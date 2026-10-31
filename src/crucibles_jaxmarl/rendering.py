"""Render native state. No physics, inferred movement or invented rewards."""
from pathlib import Path
import colorsys
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def value(state,path):
    for part in path:state=state[part]
    return np.asarray(state).tolist()

def scene_frame(env,state,obs,role=-1):
    m=env.manifest
    cache={}
    def leaf(path):
        k=tuple(path)
        if k not in cache:cache[k]=value(state,path)
        return cache[k]
    frame=dict(game=env.name,title=env.metadata['title'],role=role,clock=int(np.asarray(state['t'])),
               information='FULL_STATE_DEBUG' if role==-1 else 'ROLE_FILTERED',entities=[],map=None,
               bounds=m.get('world',{}).get('bounds',[0,0,10,10]))
    if env.metadata['family']=='legacy_array':
        # Honest abstract resource view: the original games have no spatial map.
        frame['legacy_observation']=[np.asarray(obs[a]).tolist() for a in env.agents] if role==-1 else [np.asarray(obs[env.agents[role]]).tolist()]
        frame['legacy']=True
        return frame
    world=m['world'];terrain=world.get('map')
    frame['tile_legend']=terrain.get('legend',[]) if terrain else []
    if terrain:
        grid=leaf(terrain['state_path'])
        if role==-1 or terrain.get('public') is True:frame['map']=grid
        elif 'visibility' in terrain:
            mask=leaf(terrain['visibility'])[role]
            frame['map']=[[v if mask[y][x] else None for x,v in enumerate(row)] for y,row in enumerate(grid)]
        frame['map_origin']=terrain.get('cell_origin_xy',[0,0]);frame['cell_size']=terrain.get('cell_size_xy',[1,1])
    for group in m.get('entities',[]):
        active=leaf(group['active']);xy=leaf(group['position'])
        visibility=None if role==-1 else leaf(group['visibility'])[role]
        attributes={'.'.join(a['state_path']):leaf(a['state_path']) for a in group.get('attributes',[])}
        style=next((s for s in env.scene.get('entities',[]) if s['id']==group['id']),{})
        for i in range(group['count']):
            if not active[i] or (visibility is not None and not visibility[i]):continue
            attrs={k:values[i] for k,values in attributes.items()}
            frame['entities'].append(dict(group=group['id'],kind=group['kind'],index=i,xy=xy[i],attributes=attrs,
                glyph=style.get('glyph','worker' if group['kind']=='agent' else 'crate'),
                color=style.get('color',color(group['id'])),label=style.get('label',group['id'])))
    return frame

def color(label):
    # Stable semantic palette, no dependency on randomized Python hash.
    import hashlib
    h=int(hashlib.sha256(str(label).encode()).hexdigest()[:6],16)/0xffffff
    return tuple(int(x*255) for x in colorsys.hsv_to_rgb(h,.48,.72))

def draw_frame(frame,size=(640,430)):
    im=Image.new('RGB',size,'#111b25');d=ImageDraw.Draw(im)
    font=ImageFont.load_default(size=12)
    d.text((12,8),f"{frame['game'].upper()}  t={frame['clock']}  {frame['information']}",fill='#e7f2e5',font=font)
    if frame.get('legacy'):
        d.text((12,38),'Original abstract resource game / native observations',fill='#9ec8ca',font=font)
        for a,values in enumerate(frame['legacy_observation']):
            y=80+a*130
            for j,x in enumerate(values):
                x0=16+j*37;h=int(min(1,abs(float(x)))*80)
                d.rectangle((x0,y+85-h,x0+25,y+85),fill=color(j))
                d.text((x0,y+90),str(j),font=font,fill='white')
        return im
    x0,y0,x1,y1=frame['bounds'];scale=min((size[0]-40)/max(1,x1-x0+1),(size[1]-80)/max(1,y1-y0+1))
    def point(p):return (20+(p[0]-x0+.5)*scale,48+(p[1]-y0+.5)*scale)
    if frame['map']:
        ox,oy=frame['map_origin'];sx,sy=frame['cell_size']
        for y,row in enumerate(frame['map']):
            for x,v in enumerate(row):
                px,py=point([ox+x*sx,oy+y*sy]);half=scale*.48
                palette={x.get('value'):x.get('color') for x in frame.get('tile_legend',[]) if x.get('color')}
                d.rectangle((px-half,py-half,px+half,py+half),fill='#192431' if v is None else palette.get(v,color('tile'+str(v))),outline='#263541')
    for e in frame['entities']:
        px,py=point(e['xy']);r=max(2,min(8,scale*.27));c=color(e['group'])
        if e['kind']=='agent':
            d.ellipse((px-r,py-r,px+r,py+r),fill=c,outline='white',width=1)
            if scale>=14:d.text((px+r,py-r),str(e['index']),fill='white',font=font)
        else:d.rectangle((px-r,py-r,px+r,py+r),fill=c,outline='#ffda83')
    legend=list(dict.fromkeys(e['group'] for e in frame['entities']))
    d.text((12,size[1]-22),' | '.join(legend)[:95],font=font,fill='#d3dfce')
    return im

def save_gif(frames,path):
    images=[draw_frame(f) for f in frames]
    images[0].save(path,save_all=True,append_images=images[1:],duration=220,loop=0,optimize=False)
