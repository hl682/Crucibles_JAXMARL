"""Read every original track through its real saved-endpoint browser player."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
import json,threading,time

R=Path(__file__).resolve().parents[1]
def main():
    from playwright.sync_api import sync_playwright,expect
    root=R/'.replaytest/saved-replays';assert (root/'manifest.json').is_file()
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(root)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_port);result=dict(status='RUNNING',tracks=[],human_scores=None);start=time.time()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            for ordinal in [226,227]:
                page=browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(base+f'/gallery246/{ordinal}_role.html',wait_until='load');expect(page.locator('#counter')).to_contain_text('保存端点',timeout=180000)
                options=page.locator('#track option').evaluate_all('(xs)=>xs.map(x=>({value:x.value,label:x.textContent}))')
                for track in options:
                    page.locator('#track').select_option(track['value']);expect(page.locator('#status')).not_to_contain_text('载入',timeout=180000)
                    page.wait_for_function('document.querySelector("#counter").textContent.includes("保存端点")')
                    before=page.locator('#counter').inner_text();page.locator('#next').click();expect(page.locator('#counter')).not_to_have_text(before)
                    page.locator('#play').click();before=page.locator('#counter').inner_text();expect(page.locator('#counter')).not_to_have_text(before);page.locator('#play').click()
                    last=page.locator('#seek').get_attribute('max');page.locator('#seek').evaluate('(e)=>{e.value=e.max;e.dispatchEvent(new Event("input",{bubbles:true}))}')
                    expect(page.locator('#counter')).to_contain_text(f'{last} / {last}')
                    result['tracks'].append(dict(game=ordinal,track=track['value'],scope='ROLE0_SAVED_ENDPOINTS',status='PASS',play_pause=True,seek_final_endpoint=True))
                assert not errors,errors;page.close()
            page=browser.new_page();page.goto(base+'/gi229/index.html',wait_until='load');expect(page.locator('#position')).to_contain_text('saved endpoint',timeout=180000)
            options=page.locator('#track option').evaluate_all('(xs)=>xs.map(x=>x.value)')
            for track in options:
                page.locator('#track').select_option(track);before=page.locator('#position').inner_text();page.locator('#next').click();expect(page.locator('#position')).not_to_have_text(before)
                page.locator('#play').click();before=page.locator('#position').inner_text();expect(page.locator('#position')).not_to_have_text(before);page.locator('#play').click()
                last=page.locator('#seek').get_attribute('max');page.locator('#seek').evaluate('(e)=>{e.value=e.max;e.dispatchEvent(new Event("input",{bubbles:true}))}')
                expect(page.locator('#position')).to_contain_text(f'{last}/{last}')
                result['tracks'].append(dict(game=229,track=track,scope='ROLE0_SAVED_ENDPOINTS',status='PASS',play_pause=True,seek_final_endpoint=True))
            browser.close()
        result['status']='PASS';assert len(result['tracks'])==17
    finally:server.shutdown();server.server_close();thread.join()
    result['seconds']=time.time()-start;(R/'verification/original_saved_browser.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
