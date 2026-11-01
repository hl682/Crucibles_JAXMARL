"""Prefer the existing scene recordings; retain generated diagnostics separately."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def original_previews():
    rows=json.loads((ROOT/'verification/original_preview_derivatives.json').read_bytes())
    result={}
    for row in rows:
        key=row['game'];path=ROOT/'docs/previews'/(key+'.gif')
        assert key not in result and path.is_file(),key
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['preview_sha256'],key
        result[key]=row
    return result

def primary(key,originals):
    return ('previews/' if key in originals else 'gifs/')+key+'.gif'

def caption(key,originals,lang):
    if key not in originals:
        return '原生状态重放 · 全状态调试 · 未训练动作轨迹' if lang=='zh' else 'Native state replay · Full-state debug · Untrained actions'
    row=originals[key];static=row['original_frames']==1
    if lang=='zh':
        return ('原版场景静态预览' if static else '原版场景录像 · 抽帧预览')+' · '+row['original_information']+' · 保留原图文字'
    return ('Original scene still' if static else 'Original scene recording · Sampled preview')+' · '+('Role view' if row['original_information'].startswith('角色') else 'See original information scope below')+' · Original image text retained'
