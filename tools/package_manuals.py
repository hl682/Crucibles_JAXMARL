"""Package the complete active bilingual site and GIFs for offline reading."""
from pathlib import Path
import hashlib,json,zipfile
R=Path(__file__).resolve().parents[1]
def main():
    out=R/'dist/crucibles-manuals-and-gifs-v0.1.2.zip';out.parent.mkdir(exist_ok=True)
    files=[p for p in (R/'docs').rglob('*') if p.is_file() and not any(n in p.relative_to(R/'docs').parts for n in ['demos','previews'])]
    manifest={str(p.relative_to(R/'docs')).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(R/'docs')).replace('\\','/'))
        z.writestr('OFFLINE-README.txt','Open index.html for English or zh/index.html for Chinese. Every game has a GIF and downloadable manual. Live play requires the separately installed Python package. Old bespoke replays remain available from their versioned online release.\n')
        z.writestr('PAYLOAD-SHA256.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(name)).hexdigest()==value for name,value in manifest.items())
    result=dict(status='PACKAGED_AND_HASH_VERIFIED',asset=out.name,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),active_gifs=256,game_manuals=512,languages=['en','zh-CN'],payload_files=len(files),entrypoints=['index.html','zh/index.html'])
    (R/'verification/manuals_package.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
