"""Reuse the original kitchen art, with effects tied to committed native changes."""
import copy,math,base64,io
import numpy as np
from PIL import ImageDraw,ImageFont
from . import bistro_art as art
from .rendering import scene_frame

SOURCE='98ded10c93949b1019e15a9c920cdd76450fe055c69fc397a19ed80905c03534'
PUBLIC=['tank','sink_dirt','draining','divert','basin','filter_installed','filter_charge','pan_dirty','served','wasted','contaminations']

def make_view(env,state,obs,role=0):
    assert env.name=='gi215' and env.metadata['source_sha256']==SOURCE
    frame=scene_frame(env,state,obs,role)
    view=dict(map_geometry=dict(shape=[6,7],cell_origin_xy=[0,0],cell_size_xy=[1,1]),public_map=frame['map'],entities=frame['entities'],clock=frame['clock'],view='PRIVILEGED_ACTUAL_STATE'if role==-1 else 'ROLE_'+str(role),public_scene_fields={k:np.asarray(state[k]).tolist()for k in PUBLIC})
    return view

def changes(before,after):
    events=[]
    for key,label in [('clean','Washed'),('prepared','Prepared'),('filled','Soup ready')]:
        a=np.asarray(before[key]);b=np.asarray(after[key])
        for i in np.where(~a & b)[0]:events.append(dict(kind=key,label=label,xy=np.asarray(after['item_pos'][i]).tolist()))
    old=np.asarray(before['order_status']);new=np.asarray(after['order_status'])
    for i in np.where((old==0)&(new==1))[0]:events.append(dict(kind='served',label='Served!',xy=np.asarray(after['order_pos'][i]).tolist()))
    if not bool(before['pan_dirty']) and bool(after['pan_dirty']):events.append(dict(kind='contaminated',label='Pan contaminated',xy=np.asarray(after['station_pos'][0]).tolist()))
    if int(after['basin'])>int(before['basin']):events.append(dict(kind='diverted',label='Dirty water diverted',xy=np.asarray(after['station_pos'][6]).tolist()))
    if int(after['filter_charge'])<int(before['filter_charge']):events.append(dict(kind='filtered',label='Water filtered',xy=np.asarray(after['station_pos'][4]).tolist()))
    return events

def draw(env,view,previous=None,phase=1.,events=()):
    v=copy.deepcopy(view)
    if previous is not None and phase<1:
        old={(e['group'],e['index']):e for e in previous['entities']}
        for e in v['entities']:
            p=old.get((e['group'],e['index']))
            if p is not None:e['xy']=[a+(b-a)*phase for a,b in zip(p['xy'],e['xy'])]
    image=art.render(env.manifest,env.scene,v,'',v['clock'],env.max_steps+1).crop(art.SCENE)
    d=ImageDraw.Draw(image);cell,ox,oy=art.geometry(v);ox-=art.SCENE[0];oy-=art.SCENE[1]
    f=ImageFont.load_default(size=16)
    # Water infrastructure reflects native levels, switches and moving packets.
    # No extra water, physics or inferred connectivity enters the environment.
    p=v['public_scene_fields'];by={(e['group'],e['index']):e for e in v['entities']}
    def point(e):return ox+(e['xy'][0]+.5)*cell,oy+(e['xy'][1]+.5)*cell
    for index,level,capacity,tint in [(1,p['sink_dirt'],3,'#8f6744'),(2,p['tank'],12,'#68b8cb'),(6,p['basin'],3,'#8f6744')]:
        station=by.get(('stations',index))
        if station:
            x,y=point(station);height=22*level/capacity
            d.rectangle((x-14,y+11-height,x+14,y+11),fill=tint,outline='#43555a')
    valve=by.get(('stations',4))
    if valve:
        x,y=point(valve);d.ellipse((x-13,y-13,x+13,y+13),fill='#476861',outline='#e1cf9c',width=2)
        d.line((x,y,x+(0 if p['divert']else 12),y+(12 if p['divert']else 0)),fill='#efd7a3',width=4)
        if p['filter_installed']:
            d.rectangle((x-19,y-22,x+19,y-15),fill='#415f65',outline='#9cd0c7')
            for i in range(p['filter_charge']):d.rectangle((x-15+i*10,y-20,x-9+i*10,y-17),fill='#a6dec1')
    pan=by.get(('stations',0))
    if pan and p['pan_dirty']:
        x,y=point(pan)
        d.ellipse((x-15,y-13,x+15,y+10),fill='#77623e',outline='#c28b55',width=3)
        for dx,dy in [(-7,-4),(6,3),(2,-7)]:d.ellipse((x+dx-3,y+dy-2,x+dx+3,y+dy+2),fill='#494c2f')
    for entity in v['entities']:
        if entity['group']=='drainage' and entity['attributes']['flow']:
            index=entity['index'];x,y=point(entity);start=by.get(('drainage',index-1))if index else by.get(('stations',1))
            if start:
                sx,sy=point(start);x=sx+(x-sx)*phase;y=sy+(y-sy)*phase
            for offset in [-9,0,9]:d.ellipse((x+offset-4,y-5,x+offset+4,y+5),fill='#aa7950',outline='#e1be89')
    # A held object stays at its exact native owner position. A small display
    # offset reveals it beside the hand instead of hiding it behind the actor.
    for entity in v['entities']:
        if entity['group']!='kitchen_items' or entity['attributes']['holder']<0:continue
        a=entity['attributes'];x=ox+(entity['xy'][0]+.5)*cell+22;y=oy+(entity['xy'][1]+.5)*cell+8
        if a['kind']==0:
            d.ellipse((x-14,y-9,x+14,y+9),fill='#e5daca'if a['clean']else'#978a71',outline='#716b5a',width=2)
            if a['filled']:d.ellipse((x-9,y-5,x+9,y+4),fill='#d8a55c')
        elif a['kind']==1:d.ellipse((x-8,y-12,x+8,y+12),fill='#90af65'if not a['spoiled']else'#81715b',outline='#55714f',width=2)
        else:d.rounded_rectangle((x-8,y-12,x+8,y+12),3,fill='#b4d3c5',outline='#658c88',width=2)
    # Prepared ingredients become visible slices; preparation changes more than a label.
    for entity in v['entities']:
        if entity['group']=='kitchen_items' and entity['attributes']['kind']==1 and entity['attributes']['prepared']:
            x,y=point(entity)
            if entity['attributes']['holder']>=0:x+=22;y+=8
            d.rounded_rectangle((x-13,y-14,x+13,y+14),3,fill='#dac298',outline='#816b4c')
            for offset in [-8,0,8]:d.ellipse((x-9,y+offset-3,x+9,y+offset+3),fill='#8dae60',outline='#526e43')
    for event in events:
        x=ox+(event['xy'][0]+.5)*cell;y=oy+(event['xy'][1]+.5)*cell
        if event['kind']=='clean':
            for k in range(5):
                angle=2*math.pi*k/5;r=10+22*phase;px=x+r*math.cos(angle);py=y+r*math.sin(angle)
                d.ellipse((px-3,py-5,px+3,py+5),fill='#86d5eb')
        elif event['kind'] in ['prepared','filled']:
            for k in [-1,0,1]:
                y0=y-15-phase*22;d.arc((x+k*12-4,y0-12,x+k*12+8,y0+10),90,270,fill='#f4ead4',width=2)
        elif event['kind']=='served':
            d.line([(x-13,y),(x-3,y+10),(x+18,y-15)],fill='#74b989',width=5)
        elif event['kind']=='contaminated':
            d.ellipse((x-22-8*phase,y-12-8*phase,x+22+8*phase,y+12+8*phase),outline='#d88763',width=3)
        elif event['kind'] in ['diverted','filtered']:
            d.arc((x-24,y-24,x+24,y+24),int(360*phase),int(360*phase)+230,fill='#9ddac8',width=4)
        d.text((x-27,y-45-12*phase),event['label'],font=f,fill='#fff8d8',stroke_width=2,stroke_fill='#31434a')
    return image

def animation(env,state,obs,role,previous_state=None,previous_obs=None):
    view=make_view(env,state,obs,role)
    old=make_view(env,previous_state,previous_obs,role)if previous_state is not None else None
    events=changes(previous_state,state)if previous_state is not None else []
    frames=[]
    for phase in ([.25,.5,.75,1.]if old else[1.]):
        image=draw(env,view,old,phase,events);output=io.BytesIO();image.save(output,format='PNG');frames.append('data:image/png;base64,'+base64.b64encode(output.getvalue()).decode('ascii'))
    cell,ox,oy=art.geometry(view)
    return dict(frames=frames,duration_ms=130,width=730,height=586,geometry=dict(cell=cell,ox=ox-art.SCENE[0],oy=oy-art.SCENE[1]),events=[e['label']for e in events])
