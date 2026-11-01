"""Exercise the real six-role scene through the local browser and JAX server."""
from pathlib import Path
import hashlib,json,os,socket,subprocess,sys,time,urllib.request,uuid
from playwright.sync_api import sync_playwright,expect
from verify_magnetic_gameplay import ACTIONS
R=Path(__file__).resolve().parents[1]

def main():
    paths=['server.py','magnetic_art.py','magnetic_presentation.py','web/app.js','web/index.html']
    runtime={p:hashlib.sha256((R/'src/crucibles_jaxmarl'/p).read_bytes()).hexdigest()for p in paths}
    port=8797;nonce=uuid.uuid4().hex;out=R/'local-runs/gi214-ui';out.mkdir(parents=True,exist_ok=True)
    with socket.socket()as sock:sock.bind(('127.0.0.1',port))
    log=(out/'server.log').open('w',encoding='utf8')
    child=subprocess.Popen([sys.executable,'-m','crucibles_jaxmarl','play','--port',str(port),'--game','214','--engineering','--runs',str(out/'sessions')],cwd=R,env=dict(os.environ,JAX_PLATFORMS='cpu',CRUCIBLES_TEST_NONCE=nonce),stdout=log,stderr=subprocess.STDOUT)
    try:
        base=f'http://127.0.0.1:{port}'
        for _ in range(100):
            try:
                health=json.load(urllib.request.urlopen(base+'/health',timeout=1));assert health['test_nonce']==nonce;break
            except(OSError,ValueError):
                if child.poll()is not None:raise RuntimeError('Owned server exited')
                time.sleep(.1)
        else:raise RuntimeError('Owned server not ready')
        with sync_playwright()as p:
            browser=p.chromium.launch(headless=True);page=browser.new_page(viewport=dict(width=1500,height=1100));errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(base+'/?game=214');page.locator('#seed').fill('214');page.locator('#quickplay').click()
            expect(page.locator('#controls .role')).to_have_count(6,timeout=180000);page.wait_for_function('current && current.roles.length===6')
            before=page.evaluate('current.views[0].entities.filter(e=>e.kind==="agent").map(e=>e.xy)')
            page.locator('#scene-zoom').select_option('2')
            assert page.evaluate('sceneViewport(current.views[0]).scale / Math.min(world.width/730,world.height/586)')==2
            assert page.evaluate('current.views[0].entities.filter(e=>e.kind==="agent").map(e=>e.xy)')==before
            page.locator('#direct').uncheck()
            for tick,actions in enumerate(ACTIONS,1):
                for role,action in enumerate(actions):
                    if tick==1 and role==0:
                        point=page.locator('#world').evaluate('(c)=>{const a=current.views[0].original_animation,v=sceneViewport(current.views[0]),r=c.getBoundingClientRect();return {x:(v.x+(a.geometry.ox+8.5*a.geometry.cell)*v.scale)*r.width/c.width,y:(v.y+(a.geometry.oy+10.5*a.geometry.cell)*v.scale)*r.height/c.height}}')
                        page.locator('#world').click(position=point);page.locator('#object-actions').get_by_role('button',name='Gather resource',exact=True).click()
                    else:
                        page.locator(f'#action-{role}').select_option(str(action));page.locator('#controls .role').nth(role).locator('button').click()
                    expect(page.locator('#controls .role').nth(role).locator('button')).to_have_text('Queued')
                page.locator('#step').click();expect(page.locator('#clock')).to_have_text(f't={tick} / 384',timeout=180000);page.wait_for_function('!busy')
                assert page.evaluate('current.views[0].entities.filter(e=>e.kind==="agent").map(e=>e.xy)')==before
                if tick==2:
                    assert page.evaluate('current.views[0].map[11][11]')==2
                    assert page.evaluate('current.views[0].map[8][9]')==3
                    assert page.evaluate('current.views[0].entities.find(e=>e.group==="freight"&&e.index===2).xy')==[13,10]
                    page.locator('#world').screenshot(path=str(out/'world-built-powered.png'))
            assert not errors,errors
            assert page.evaluate('current.views[0].entities.find(e=>e.group==="magnets"&&e.index===0).attributes.magnet_powered')is False
            page.locator('#close').click();expect(page.locator('#session')).to_be_hidden();browser.close()
        latest=max((out/'sessions').glob('*/events.jsonl'),key=lambda p:p.stat().st_mtime)
        events=[json.loads(x)for x in latest.read_text(encoding='utf8').splitlines()]
        assert [x['value']['actions']for x in events if x['type']=='transition']==ACTIONS
        assert runtime=={p:hashlib.sha256((R/'src/crucibles_jaxmarl'/p).read_bytes()).hexdigest()for p in paths}
        result=dict(status='PASS',game='gi214',native_joint_steps=len(ACTIONS),roles_owned=6,all_actor_positions_unchanged=True,
            clicked_deposit=True,zoomed_click_preserves_coordinates=True,bridge_and_rail_visible=True,powered_freight_moved=True,power_off_visible=True,browser_errors=errors,
            runtime_sha256=runtime,human_scores=None,scope='Four native browser steps, not full-game success or learning')
        (R/'verification/gameplay/gi214_live_ui.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)
    finally:
        if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        else:child.terminate()
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:child.kill();child.wait()
        log.close()

if __name__=='__main__':main()
