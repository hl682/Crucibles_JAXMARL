"""Exercise the actual static catalog UI and validate all published GIF bytes."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
import hashlib,json,re,threading,time

R=Path(__file__).resolve().parents[1]
def main():
    from playwright.sync_api import sync_playwright,expect
    from PIL import Image
    start=time.time();errors=[];images=0
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(R/'docs')))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True);page=browser.new_page(viewport=dict(width=1500,height=1000))
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto('http://127.0.0.1:'+str(server.server_port)+'/',wait_until='networkidle')
            expect(page.locator('.card[data-search]')).to_have_count(256)
            page.locator('#search').fill('gi225');expect(page.locator('.card[data-search]:visible')).to_have_count(1)
            page.locator('#search').fill('');page.locator('#family').select_option('spatial');expect(page.locator('.card[data-search]:visible')).to_have_count(180)
            page.locator('#family').select_option('legacy_array');expect(page.locator('.card[data-search]:visible')).to_have_count(76)
            page.locator('#family').select_option('')
            expect(page.locator('.card[data-search] img')).to_have_count(256)
            page.locator('img').evaluate_all('(xs)=>xs.forEach(i=>i.loading="eager")');page.wait_for_load_state('networkidle')
            page.wait_for_function('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)',timeout=60000)
            assert page.locator('img').evaluate_all('(xs)=>xs.every(i=>i.complete&&i.naturalWidth>0)')
            page.locator('#theme').select_option('commitment');assert page.locator('.card[data-search]:visible').count()>0
            page.locator('#theme').select_option('');page.locator('#search').fill('gi225')
            page.locator('#collection').scroll_into_view_if_needed()
            page.screenshot(path=str(R/'verification/screenshots/catalog-bilingual-en.png'))
            page.locator('.card[data-search][href="games/gi225.html"]').click();expect(page.locator('h1')).to_contain_text('Seasonal Pasture Commons')
            page.locator('.language-switch').click();expect(page.locator('html')).to_have_attribute('lang','zh-CN');expect(page.locator('h1')).to_contain_text('四季牧场公地')
            page.screenshot(path=str(R/'verification/screenshots/manual-zh.png'))
            page.locator('.language-switch').click();expect(page.locator('h1')).to_contain_text('Seasonal Pasture Commons')
            page.locator('img').evaluate_all('(xs)=>xs.forEach(i=>i.loading="eager")')
            page.wait_for_function('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)',timeout=60000)
            for img in page.locator('img').all():assert img.evaluate('(i)=>i.complete && i.naturalWidth>0')
            for prefix in ['', 'zh/']:
                for key in ['gi216','gi217']:
                    page.goto('http://127.0.0.1:'+str(server.server_port)+'/'+prefix+'games/'+key+'.html',wait_until='networkidle')
                    primary=page.locator('img').first
                    expect(primary).to_have_attribute('src',('../../'if prefix else'../')+'walkthroughs/'+key+'-causal.gif')
                    assert primary.evaluate('(i)=>i.complete && i.naturalWidth>0')
                    expect(page.locator('a[href$="previews/'+key+'.gif"]')).to_have_count(1)
                page.goto('http://127.0.0.1:'+str(server.server_port)+'/'+prefix+'games/gi214.html',wait_until='networkidle')
                magnetic=page.locator('img[src$="walkthroughs/gi214-causal.gif"]')
                expect(magnetic).to_have_count(1)
                assert magnetic.evaluate('(i)=>i.complete && i.naturalWidth>0')
                page.goto('http://127.0.0.1:'+str(server.server_port)+'/'+prefix+'games/gi215.html',wait_until='networkidle')
                for suffix in ['first-order','water-unprotected','water-diverted']:
                    demo=page.locator('img[src$="walkthroughs/gi215-'+suffix+'.gif"]')
                    expect(demo).to_have_count(1)
                    assert demo.evaluate('(i)=>i.complete && i.naturalWidth>0')
            page.goto('http://127.0.0.1:'+str(server.server_port)+'/zh/');expect(page.locator('.card[data-search]')).to_have_count(256)
            page.locator('#search').fill('四季牧场');expect(page.locator('.card[data-search]:visible')).to_have_count(1)
            page.locator('#collection').scroll_into_view_if_needed()
            page.screenshot(path=str(R/'verification/screenshots/catalog-bilingual-zh.png'))
            page.set_viewport_size(dict(width=390,height=844));page.reload();assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
            page.screenshot(path=str(R/'verification/screenshots/catalog-mobile.png'))
            assert not errors,errors
            browser.close()
        for folder in ['gifs']:
            for path in (R/'docs'/folder).glob('*.gif'):
                with Image.open(path) as image:
                    assert image.n_frames>1;image.seek(image.n_frames-1);image.load();assert image.width>0 and image.height>0
                    images+=1
                receipt=json.loads((R/'verification/presentation'/(path.stem+'.json')).read_bytes())
                assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['gif_sha256']
        assert images==256
        catalog=json.loads((R/'docs/catalog.json').read_bytes())['games'];by_key={r['key']:r for r in catalog}
        assert {v['code'] for v in by_key['gi072']['themes']}=={'transport'}
        assert 'adversarial' not in {v['code'] for v in by_key['gi073']['themes']},'Arrivals must not imply rivals'
        assert 'commitment' not in {v['code'] for v in by_key['gi074']['themes']},'Committed render effects are not contracts'
        for path in (R/'docs/manuals/en').glob('*.md'):assert not re.search('[\u3400-\u9fff]',path.read_text(encoding='utf-8')),path
        assert len(list((R/'docs/manuals/en').glob('*.md')))==len(list((R/'docs/manuals/zh').glob('*.md')))==256
        for lang in ['en','zh']:
            for path in (R/'docs/manuals'/lang).glob('*.md'):
                data=path.read_text(encoding='utf-8');assert data.count('\n## ')>=10 and 'crucibles play --game '+path.stem in data
        result=dict(status='PASS',catalog_cards_per_language=256,languages=['en','zh-CN'],game_manuals=512,spatial_filter=180,legacy_filter=76,mechanism_filter='PASS',search_both_languages='PASS',corresponding_game_language_switch='PASS',mobile_no_horizontal_overflow=True,all_catalog_images_loaded=True,active_gif_files_decoded=images,active_gif_hashes_verified=256,english_manuals_cjk_free=True,seconds=time.time()-start)
    finally:server.shutdown();server.server_close();thread.join()
    (R/'verification/docs_ui.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
