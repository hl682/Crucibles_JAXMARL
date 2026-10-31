"""Exercise the actual static catalog UI and validate all published GIF bytes."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
import json,threading,time

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
            page.locator('#family').select_option('');page.screenshot(path=str(R/'verification/screenshots/catalog.png'))
            page.locator('.card[data-search][href="games/gi225.html"]').click();expect(page.locator('h1')).to_contain_text('四季牧场')
            for img in page.locator('img').all():assert img.evaluate('(i)=>i.complete && i.naturalWidth>0')
            assert not errors,errors
            browser.close()
        for folder in ['demos','previews']:
            for path in (R/'docs'/folder).glob('*.gif'):
                with Image.open(path) as image:
                    image.seek(image.n_frames-1);image.load();assert image.width>0 and image.height>0
                    images+=1
        result=dict(status='PASS',catalog_cards=256,spatial_filter=180,legacy_filter=76,search='PASS',navigation='PASS',gif_files_decoded=images,seconds=time.time()-start)
    finally:server.shutdown();server.server_close();thread.join()
    (R/'verification/docs_ui.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
