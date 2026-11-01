"""Check every preferred original, bilingual entry, and secondary diagnostic link."""
from html.parser import HTMLParser
import json
from gallery_media import ROOT,original_previews,primary

class Page(HTMLParser):
    def __init__(self):super().__init__();self.images=[];self.cards=[];self.spatial_selected=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='img':self.images.append(a['src'])
        if tag=='a' and 'data-family' in a:self.cards.append(a)
        if tag=='option' and a.get('value')=='spatial' and 'selected' in a:self.spatial_selected=True

def main():
    originals=original_previews();rows=json.loads((ROOT/'src/crucibles_jaxmarl/catalog.json').read_bytes())['games'];assert len(originals)==157
    checked=0
    for lang in ['en','zh']:
        base=ROOT/'docs'/('zh' if lang=='zh' else '');asset='../' if lang=='zh' else '';p=Page();p.feed((base/'index.html').read_text(encoding='utf8'))
        assert len(p.cards)==len(rows)==256 and p.spatial_selected
        assert all(c['data-family']=='spatial' for c in p.cards[:180])
        assert all(c['data-family']!='spatial' for c in p.cards[180:])
        for card,img in zip(p.cards,p.images):
            key=card['href'].split('/')[-1].split('.')[0]
            assert img==asset+primary(key,originals),(key,img)
        for row in rows:
            key=row['key'];p=Page();p.feed((base/'games'/(key+'.html')).read_text(encoding='utf8'));pre='../../' if lang=='zh' else '../'
            assert p.images[0]==pre+primary(key,originals),key
            if key in originals:assert p.images[1]==pre+'gifs/'+key+'.gif',key
            checked+=1
    assert (ROOT/'docs/catalog.js').read_text(encoding='utf8').strip().endswith('filter();')
    result=dict(status='PASS',original_previews_verified=157,bilingual_game_pages_checked=checked,total_cards_per_language=256,spatial_default_count=180,abstract_environments_retained=76,original_stills=sum(x['original_frames']==1 for x in originals.values()),native_or_training_execution=False)
    (ROOT/'verification/gallery_media.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))

if __name__=='__main__':main()
