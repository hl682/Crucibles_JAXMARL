"""Check static link closure, installed hashes and accidental private material."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
import hashlib,json,re

R=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['href','src'] and value:self.links.append(value)

def main():
    errors=[];count=0
    for p in (R/'docs').rglob('*.html'):
        parser=Links();parser.feed(p.read_text(encoding='utf-8'))
        for url in parser.links:
            count+=1;parts=urlsplit(url)
            if parts.scheme or parts.netloc or not parts.path:continue
            dest=(p.parent/unquote(parts.path)).resolve()
            if not dest.is_relative_to((R/'docs').resolve()) or not dest.is_file():errors.append(f'Broken link: {p.relative_to(R)} -> {url}')
    rows=json.loads((R/'src/crucibles_jaxmarl/catalog.json').read_bytes())['games']
    for row in rows:
        d=R/'src/crucibles_jaxmarl/games'/row['key']
        for file,key in [('candidate.py','source_sha256'),('manifest.json','manifest_sha256')]:
            if hashlib.sha256((d/file).read_bytes()).hexdigest()!=row[key]:errors.append('Hash mismatch: '+str(d/file))
    # Restrict the scan to explicitly publishable trees; never inspect local sessions.
    roots=[R/'src',R/'docs',R/'tools',R/'.github',R/'source-history',R/'verification']
    patterns=[r'gh[opusr]_[A-Za-z0-9]{25,}',r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
              r'(?i)(?:[A-Z]:[\\/](?:Users|lab-hub|OneDrive)|file:///|sshpass|tailscale.*authkey)',r'#session=[0-9a-f]{32}']
    for root in roots:
        if not root.exists():continue
        for p in root.rglob('*'):
            if p.resolve()==Path(__file__).resolve():continue
            if not p.is_file() or p.suffix not in ['.py','.md','.html','.json','.js','.yml','.txt','.css']:continue
            if '__pycache__' in p.parts:continue
            text=p.read_text(encoding='utf-8-sig')
            for pattern in patterns:
                if re.search(pattern,text):errors.append('Sensitive/nonportable pattern in '+str(p.relative_to(R)))
    result=dict(status='PASS' if not errors else 'FAIL',games=len(rows),html_links_checked=count,errors=errors)
    (R/'verification/release_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(result,errors=errors[:10],error_count=len(errors)),indent=2));return bool(errors)

if __name__=='__main__':raise SystemExit(main())
