from pathlib import Path
import hashlib
import importlib.util
import json
import re

ROOT = Path(__file__).resolve().parent
_rows = json.loads((ROOT/'catalog.json').read_bytes())['games']
_presentation = json.loads((ROOT/'presentation.json').read_bytes())
_registry = {r['key']:r for r in _rows if 'source_sha256' in r}
registered_envs = tuple(_registry)

def game_info(name):
    if isinstance(name,int): name=f'gi{name:03d}'
    if isinstance(name,str):
        match=re.fullmatch(r'(?:gi|GI|game-v3-)?(\d+)',name)
        if match:name=f'gi{int(match[1]):03d}'
    if name not in _registry:raise ValueError(f'Unknown environment: {name!r}')
    return dict(_registry[name],**_presentation[name])

def make(name):
    from .environment import CrucibleEnv
    row=game_info(name);path=ROOT/'games'/row['key']
    for file,key in [('candidate.py','source_sha256'),('manifest.json','manifest_sha256')]:
        if hashlib.sha256((path/file).read_bytes()).hexdigest()!=row[key]:
            raise RuntimeError(f'Installed {row["key"]}/{file} hash mismatch')
    spec=importlib.util.spec_from_file_location(f'crucibles_kernel_{row["key"]}',path/'candidate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    scene=json.loads((path/'scene.json').read_bytes()) if (path/'scene.json').exists() else {}
    return CrucibleEnv(module.GeneratedEnv(),json.loads((path/'manifest.json').read_bytes()),row,scene)
