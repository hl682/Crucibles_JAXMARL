"""Bounded source/provenance inventory; deduplicate copies and retain old revisions."""
from pathlib import Path
import ast,hashlib,json,os,re,shutil

R=Path(__file__).resolve().parents[1];W=R.parents[2];G=R.parent/'game_incubator_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    rows=json.loads((R/'src/crucibles_jaxmarl/catalog.json').read_bytes())['games'];current={r['source_sha256']:r['key'] for r in rows}
    history={sha(p) for p in (R/'source-history').rglob('candidate.py')}
    roots=[R.parent/'astra_generation',G/'sources',R.parent/'assets/astra_candidate_imports',R.parent/'assets/incubator_v1/imports',
           R.parent/'assets/astra_validation_v1',R.parent/'assets/astra_validation_v2']
    roots.extend(p for p in G.iterdir() if p.is_dir() and (p.name.startswith('community_') or p.name.startswith('GI')))
    roots.append(W/'1. 正式实验/coordination/GI248_251_source_root254/package')
    origins={};unknown=[];revisions=[]
    for root in roots:
        for directory,dirs,files in os.walk(root):
            dirs[:]=[n for n in dirs if n not in ['evidence','runs','output','outputs','assets','games','__pycache__','gallery246','gallery','previews','dist','.venv']
                     and not n.startswith(('reference','before','native','recovery','attempt00_interrupted'))]
            if 'candidate.py' not in files:continue
            p=Path(directory)/'candidate.py';b=p.read_bytes()
            if b'class GeneratedEnv' not in b:continue
            ast.parse(b);h=hashlib.sha256(b).hexdigest();origin=p.relative_to(W).as_posix();origins.setdefault(h,[]).append(origin)
            if h in current or h in history:continue
            mpath=p.parent/'manifest.json';m=json.loads(mpath.read_bytes()) if mpath.exists() else {}
            cid=m.get('candidate_id','');match=re.fullmatch(r'game-v3-(\d+)',cid)
            if match:key='gi%03d'%int(match[1])
            else:
                proposal=re.search(r'candidate_(\d+)',origin)
                key=('v1_%03d' if '/window_v1/' in origin else 'gi%03d')%int(proposal[1]) if proposal else None
            if key not in {r['key'] for r in rows}:
                unknown.append(dict(origin=origin,source_sha256=h,inferred_key=key));continue
            dest=R/'source-history'/key/('source-'+h[:16]);dest.mkdir(parents=True,exist_ok=True)
            copied=[]
            for name in ['candidate.py','manifest.json','scene.json','mechanism.md','playthrough.json']:
                q=p.parent/name
                if q.is_file():shutil.copyfile(q,dest/name);copied.append(dict(file=name,sha256=sha(q)))
            revisions.append(dict(key=key,source_sha256=h,origin=origin,files=copied,registered=False));history.add(h)
    result=dict(status='PASS' if not unknown else 'UNRESOLVED_SOURCE_IDENTITIES',registered_unique_sources=len(current),
        discovered_distinct_source_hashes=len(origins),physical_source_copies=sum(map(len,origins.values())),
        source_origins=origins,additional_preserved_revisions=revisions,unresolved_identities=unknown,
        scope='Original author streams, source directories, bounded community source/provenance trees; replay/compiled-data copies are not new games')
    (R/'verification/full_source_inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_origins','additional_preserved_revisions']},ensure_ascii=False,indent=2));print('Additional historical revisions',len(revisions))
    return bool(unknown)

if __name__=='__main__':raise SystemExit(main())
