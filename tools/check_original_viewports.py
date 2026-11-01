"""Verify that scene-only presentation reuses the identical original image bytes."""
from pathlib import Path
import hashlib,json
from html.parser import HTMLParser
from scene_viewports import profiles

R=Path(__file__).resolve().parents[1]
class Page(HTMLParser):
    def __init__(self):super().__init__();self.windows=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=='div' and d.get('class')=='scene-window':self.windows.append(d['data-scene'])
        if tag=='img':self.images.append(d['src'])

def main():
    rows=profiles();originals=json.loads((R/'verification/original_preview_derivatives.json').read_bytes())
    for row in originals:
        assert hashlib.sha256((R/'docs/previews'/(row['game']+'.gif')).read_bytes()).hexdigest()==row['preview_sha256']
    for base in [R/'docs',R/'docs/zh']:
        p=Page();p.feed((base/'index.html').read_text(encoding='utf8'));assert set(p.windows)==set(rows)
        for key in rows:
            p=Page();p.feed((base/'games'/(key+'.html')).read_text(encoding='utf8'))
            assert p.windows==[key]
            assert p.images[0].endswith('previews/'+key+'.gif')
    assert '.scene-window' in (R/'docs/style.css').read_text(encoding='utf8')
    result=dict(status='PASS',cropped_original_scenes=len(rows),original_gif_hashes_unchanged=len(originals),bilingual_scene_pages=len(rows)*2,original_pixel_content_rewritten=False,remaining_original_layouts=157-len(rows),full_visual_redesign_claim=False)
    (R/'verification/original_viewport_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
