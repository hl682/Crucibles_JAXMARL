"""Publish small derivatives of existing genuine replays and all source versions."""
from pathlib import Path
from PIL import Image
import hashlib,json,shutil

R=Path(__file__).resolve().parents[1];G=R.parent/'game_incubator_v3';L=G/'game_library_v2';W=R.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    output=R/'docs/previews';output.mkdir(exist_ok=True)
    catalog=json.loads((L/'catalog.json').read_bytes())['games'];receipts=[]
    for row in catalog:
        p=(L/row['preview']).resolve();dest=output/f"gi{row['id']:03d}.gif"
        if dest.exists():continue
        if not p.is_file():raise FileNotFoundError(p)
        with Image.open(p) as image:
            indices=sorted(set(round(i*(image.n_frames-1)/19) for i in range(min(20,image.n_frames))))
            frames=[]
            for index in indices:
                image.seek(index);frame=image.convert('RGB');frame.thumbnail((600,420));frames.append(frame.copy())
            frames[0].save(dest,save_all=True,append_images=frames[1:],duration=240,loop=0)
            receipts.append(dict(game=f"gi{row['id']:03d}",original_origin=p.relative_to(W).as_posix(),
                original_bytes=p.stat().st_size,original_sha256=sha(p),original_frames=image.n_frames,
                sampled_frame_indices=indices,preview_sha256=sha(dest),derivative=True,
                original_recording_mode=row.get('mode'),original_information=row.get('preview_scope'),
                human_scores=None))
        print('preview',row['id'],flush=True)
    receipt=R/'verification/original_preview_derivatives.json'
    old=json.loads(receipt.read_bytes()) if receipt.exists() else []
    receipt.write_text(json.dumps(old+receipts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    current=json.loads((R/'src/crucibles_jaxmarl/catalog.json').read_bytes())['games'];variants=[]
    for row in current[72:]:
        source=G/('sources/game-v3-%05d'%row['id'])
        for p in source.glob('attempt_*/candidate.py'):
            if sha(p)==row['source_sha256']:continue
            dest=R/'source-history'/row['key']/p.parent.name;dest.mkdir(parents=True,exist_ok=True)
            files=[]
            for name in ['candidate.py','manifest.json','scene.json','mechanism.md','playthrough.json']:
                q=p.parent/name
                if q.is_file():shutil.copyfile(q,dest/name);files.append(dict(file=name,sha256=sha(q)))
            variants.append(dict(game=row['key'],attempt=p.parent.name,source_sha256=sha(p),files=files,registered=False))
    (R/'verification/alternate_source_versions.json').write_text(json.dumps(variants,indent=2)+'\n',encoding='utf-8')
    print('Original previews',len(old+receipts),'alternate versions',len(variants))

if __name__=='__main__':main()
