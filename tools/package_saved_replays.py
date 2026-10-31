"""Prepare a public, data-only replay download; exclude research/operator material."""
from pathlib import Path
import hashlib,json,re,zipfile

R=Path(__file__).resolve().parents[1];G=R.parent/'game_incubator_v3';W=R.parents[2]

def main():
    a=G/'community_games226_227_actual_visualization_root246_v1'
    b=G/'GI229_saved_replay253'
    targets=[]
    for p in (a/'games').rglob('*'):
        if p.is_file() and p.suffix in ['.js','.json']:targets.append((p,p.relative_to(a).as_posix(),False))
    for p in (a/'gallery246').iterdir():
        if p.name=='font246.js':continue
        if p.suffix in ['.html','.css','.js']:targets.append((p,p.relative_to(a).as_posix(),p.suffix=='.html'))
    for name in ['legend246_static.js','legend246.js','scene_canvas246.js']:
        p=a/'native156_candidate/web'/name;targets.append((p,p.relative_to(a).as_posix(),False))
    for name in ['index.html','audit.html','player250.js']:
        targets.append((b/name,'gi229/'+name,False))
    for p in (b/'assets').glob('*.js'):targets.append((p,'gi229/assets/'+p.name,False))
    dest=R/'dist/crucibles-original-saved-replays-v0.1.0.zip';dest.parent.mkdir(exist_ok=True)
    records=[];patterns=[rb'gh[opusr]_[A-Za-z0-9]{25,}',rb'-----BEGIN [A-Z ]*PRIVATE KEY-----',rb'#session=[0-9a-f]{32}',rb'[A-Z]:\\(?:Users|lab-hub)\\']
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for i,(src,name,adapt) in enumerate(targets):
            h=hashlib.sha256();original_hash=hashlib.sha256();length=0
            with src.open('rb') as input_file,z.open('saved-replays/'+name,'w',force_zip64=True) as out:
                tail=b''
                while data:=input_file.read(1024*1024):
                    original_hash.update(data)
                    if any(re.search(pattern,tail+data) for pattern in patterns):raise ValueError('Sensitive/nonportable content in '+name)
                    tail=data[-256:]
                    if adapt:data=data.replace(b'<script src="font246.js"></script>',b'')
                    h.update(data);length+=len(data);out.write(data)
            records.append(dict(path=name,bytes=length,sha256=h.hexdigest(),original_sha256=original_hash.hexdigest(),
                original_origin=src.relative_to(W).as_posix(),adaptation='Remove embedded local font reference; system font fallback' if adapt else None))
            if i%100==0:print('packed',i,'/',len(targets),flush=True)
        index='''<!doctype html><meta charset="utf-8"><title>Original saved Crucibles replays</title><style>body{font:18px/1.7 system-ui;max-width:900px;margin:60px auto;padding:20px;background:#14232c;color:#eee}a{color:#d4ed91}</style><h1>Original saved native-state replays</h1><p>These are saved endpoints, not live Python/JAX play. Source rewards, failed tracks, masks, information conditions and numerical uncertainty are retained. No human scores are claimed.</p><ul><li>GI226, 96 independently actionable roles: <a href="gallery246/226_role.html">role view</a> / <a href="gallery246/226_audit.html">privileged full-state audit</a></li><li>GI227, 128 independently actionable roles: <a href="gallery246/227_role.html">role view</a> / <a href="gallery246/227_audit.html">privileged full-state audit</a></li><li>GI229, 24 independently actionable roles: <a href="gi229/index.html">role view</a> / <a href="gi229/audit.html">privileged full-state audit</a></li></ul><p>Keep the complete extracted directory together. Serve it with <code>python -m http.server 8770 --directory saved-replays</code>, then use the printed local address. All data is bundled. Local embedded font bytes and research/operator infrastructure are excluded.</p>'''
        z.writestr('saved-replays/index.html',index)
        manifest=dict(schema='public-saved-replay-data-v1',files=records,gi226_227_original_tracks=11,gi229_original_tracks=6,
            original_scientific_acceptance_changed=False,human_scores=None,fonts='SYSTEM_FALLBACK_NO_EMBEDDED_FONT_BYTES')
        z.writestr('saved-replays/manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(dest) as z:
        bad=z.testzip()
        if bad:raise ValueError('Corrupt ZIP member: '+bad)
    h=hashlib.sha256()
    with dest.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    receipt=dict(archive=dest.name,bytes=dest.stat().st_size,sha256=h.hexdigest(),zip_crc='PASS',files=len(records),
        gi226_227_original_tracks=11,gi229_original_tracks=6,personal_material_scan='PASS',
        download_url='https://github.com/hl682/Crucibles_JAXMARL/releases/download/v0.1.0/'+dest.name)
    (R/'verification/saved_replay_download.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
