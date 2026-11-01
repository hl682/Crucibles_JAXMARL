"""Prefer the existing scene recordings; retain generated diagnostics separately."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REUSED_ART={'gi214':'gi214-causal.gif','gi215':'gi215-first-order.gif',
            'gi216':'gi216-causal.gif','gi217':'gi217-causal.gif'}

SCOPE_EN={
    '全状态审阅视角':'Full-state audit view',
    '原回放视角':'Original replay view',
    '角色视角':'Role view',
    '公共展示视角':'Public display view',
    '角色0实际可见视角':'Role 0 actual visible view',
    '角色0实际作者动作段':'Role 0 recorded author-action segment',
    '角色0完整实际记录，含原等待':'Role 0 full actual recording, including original waits',
    '角色0保存观测，513个离散真实端点，无插值':'Role 0 saved observations; 513 discrete actual endpoints, without interpolation',
    '全状态审阅静态预览；打开后播放真实保存端点':'Full-state audit still preview; the player contains actual saved endpoints',
    **{f'角色{i}实际端点抽帧预览；完整端点在播放器':f'Role {i} sampled actual-endpoint preview; full endpoints in the player'for i in [0,1,2,6]},
}
MODE_EN={'固定脚本演示':'Fixed-script demonstration','实际诊断回放':'Actual diagnostic replay','保存状态回放':'Saved-state replay','探索记录':'Exploration recording','保存回放（离线原记录；非人工游戏或benchmark验收）':'Saved offline recording; not human-play or benchmark acceptance'}

def recording_metadata(row,lang):
    if lang=='zh':return row['original_information'],row['original_recording_mode']
    return SCOPE_EN[row['original_information']],MODE_EN[row['original_recording_mode']]

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
    if key in REUSED_ART:return 'walkthroughs/'+REUSED_ART[key]
    return ('previews/' if key in originals else 'gifs/')+key+'.gif'

def caption(key,originals,lang):
    if key in REUSED_ART:
        role=2 if key=='gi217' else 0
        return f'复用原画 · 原生动作演示 · 角色{role}视角'if lang=='zh'else f'Original artwork reused · Native action walkthrough · Role {role} view'
    if key not in originals:
        return '原生状态重放 · 全状态调试 · 未训练动作轨迹' if lang=='zh' else 'Native state replay · Full-state debug · Untrained actions'
    row=originals[key];static=row['original_frames']==1
    if lang=='zh':
        return ('原版场景静态预览' if static else '原版场景录像 · 抽帧预览')+' · '+row['original_information']+' · 保留原图文字'
    return ('Original scene still' if static else 'Original scene recording · Sampled preview')+' · '+('Role view' if row['original_information'].startswith('角色') else 'See original information scope below')+' · Original image text retained'
