"""Alternate-host multiplayer transport check using the real native service."""
from pathlib import Path
import json,socket,threading,time,urllib.request

R=Path(__file__).resolve().parents[1]
def main():
    from crucibles_jaxmarl.server import serve
    from playwright.sync_api import sync_playwright,expect
    with socket.socket() as probe:probe.bind(('127.0.0.2',0));port=probe.getsockname()[1]
    # A concrete alternative interface exercises origin and invitation routing
    # without opening the test to the surrounding LAN.
    thread=threading.Thread(target=serve,args=(port,72,str(R/'local-runs/network-verification'),True),kwargs=dict(bind='127.0.0.2'),daemon=True);thread.start()
    base=f'http://127.0.0.2:{port}'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(100):
        try:
            with opener.open(base+'/health',timeout=1) as r:assert json.load(r)['ok']
            break
        except (OSError,AssertionError):time.sleep(.1)
    else:raise RuntimeError('Alternate-interface server did not become ready')
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True);context=browser.new_context();host=context.new_page();host.goto(base+'/?game=72',wait_until='networkidle')
        host.locator('#create').click();expect(host.locator('#session')).to_be_visible(timeout=120000)
        host.locator('#roles').fill('0');host.locator('#assign').click();expect(host.locator('#action-0')).to_be_visible()
        host.locator('#invite').click();expect(host.locator('#invitation')).to_be_visible();invite=host.locator('#invitation').input_value();assert invite.startswith(base+'/')
        peer=context.new_page();peer.goto(invite,wait_until='networkidle');expect(peer.locator('#session')).to_be_visible()
        host.locator('#player').select_option('Player 1');host.locator('#roles').fill('1');host.locator('#assign').click();expect(host.locator('#player option[value="Player 1"]')).to_have_text('Player 1 (1 roles)')
        peer.locator('#refresh').click();expect(peer.locator('#action-1')).to_be_visible();peer.locator('#controls button').first.click()
        host.locator('#step').click();expect(host.locator('#clock')).to_contain_text('t=1 /',timeout=120000)
        assert host.locator('#error').inner_text()==''
        host.locator('#close').click();expect(host.locator('#session')).to_be_hidden();browser.close()
    result=dict(status='PASS',bind='127.0.0.2',distinct_host_origin=True,invitation_host_preserved=True,two_clients_native_step=True,
        scope='Alternate-interface/origin/peer transport; no physical two-computer human test',human_scores=None)
    (R/'verification/network_transport.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
