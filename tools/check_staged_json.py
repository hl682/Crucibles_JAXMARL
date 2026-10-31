"""Validate the exact staged JSON bytes, including concurrent verification snapshots."""
from pathlib import Path
import hashlib,json,subprocess

def main():
    root=Path(__file__).resolve().parents[1]
    paths=[p for p in subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0') if p.endswith('.json')]
    data=subprocess.check_output(['git','cat-file','--batch'],cwd=root,input=''.join(':'+p+'\n' for p in paths).encode())
    pos=0;values={};errors=[]
    for path in paths:
        end=data.index(b'\n',pos);header=data[pos:end].decode().split();size=int(header[-1]);raw=data[end+1:end+1+size];pos=end+size+2
        try:values[path]=json.loads(raw)
        except ValueError as e:errors.append(dict(path=path,error=str(e)))
    print(json.dumps(dict(json_files=len(paths),errors=errors),indent=2))
    return bool(errors)

if __name__=='__main__':raise SystemExit(main())
