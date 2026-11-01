"""Display original GIF scene regions without changing their pixels or frames."""
from pathlib import Path
from html import escape
import json,hashlib

ROOT=Path(__file__).resolve().parents[1]

def profiles():
    value=json.loads((ROOT/'verification/original_scene_viewports.json').read_bytes())
    for key,row in value['games'].items():
        assert hashlib.sha256((ROOT/'docs/previews'/(key+'.gif')).read_bytes()).hexdigest()==row['preview_sha256']
        w,h=row['canvas'];x,y,bw,bh=row['box_xywh']
        assert x>=0 and y>=0 and bw>0 and bh>0 and x+bw<=w and y+bh<=h
    return value['games']

def image_markup(key,src,title,rows,lazy=True):
    image='<img '+('loading="lazy" decoding="async" ' if lazy else '')+'src="'+escape(src,quote=True)+'" alt="'+escape(title,quote=True)+'">'
    if key not in rows:return image
    row=rows[key];w,h=row['canvas'];x,y,bw,bh=row['box_xywh']
    style=f'aspect-ratio:{bw}/{bh};--frame-w:{w/bw*100:.8f}%;--frame-h:{h/bh*100:.8f}%;--frame-x:{-x/bw*100:.8f}%;--frame-y:{-y/bh*100:.8f}%'
    return '<div class="scene-window" data-scene="'+key+'" style="'+style+'">'+image+'</div>'
