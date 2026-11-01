"""Replay verified native landscape actions via actual browser controls."""
from pathlib import Path
import hashlib,json,os,socket,subprocess,sys,time,urllib.request,uuid
from playwright.sync_api import sync_playwright,expect
R=Path(__file__).resolve().parents[1]

def main():
    files=['server.py','web/app.js','web/index.html','flood_art.py','flood_presentation.py','orchard_art.py','orchard_presentation.py']
    runtime={p:hashlib.sha256((R/'src/crucibles_jaxmarl'/p).read_bytes()).hexdigest()for p in files}
    port=8798;nonce=uuid.uuid4().hex;out=R/'local-runs/landscape-ui';out.mkdir(parents=True,exist_ok=True)
    with socket.socket()as sock:sock.bind(('127.0.0.1',port))
    log=(out/'server.log').open('w',encoding='utf8')
    child=subprocess.Popen([sys.executable,'-m','crucibles_jaxmarl','play','--port',str(port),'--game','216','--engineering','--runs',str(out/'sessions')],cwd=R,env=dict(os.environ,JAX_PLATFORMS='cpu',CRUCIBLES_TEST_NONCE=nonce),stdout=log,stderr=subprocess.STDOUT)
    try:
        base=f'http://127.0.0.1:{port}'
        for _ in range(100):
            try:
                h=json.load(urllib.request.urlopen(base+'/health',timeout=1));assert h['test_nonce']==nonce;break
            except(OSError,ValueError):
                if child.poll()is not None:raise RuntimeError('Owned server exited')
                time.sleep(.1)
        else:raise RuntimeError('Owned server not ready')
        with sync_playwright()as p:
            browser=p.chromium.launch(headless=True)
            for game,roles,horizon in [('gi216',4,384),('gi217',8,512)]:
                original=json.loads((R/'verification/gameplay'/f'{game}_causal_gameplay.json').read_bytes())
                sequence=[r['actions']for r in original['sequence']]
                page=browser.new_page(viewport=dict(width=1500,height=1100));errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(base+'/?game='+game);expect(page.locator('#game')).to_have_value(game,timeout=30000)
                page.locator('#seed').fill(str(int(game[2:])));page.locator('#quickplay').click()
                page.wait_for_function('document.querySelectorAll("#controls .role").length>0 || document.getElementById("error").textContent.length>0',timeout=180000)
                issue=page.locator('#error').inner_text()
                if issue:
                    (out/f'{game}-failure.json').write_text(json.dumps(dict(error=issue,browser_errors=errors),indent=2),encoding='utf8')
                    raise RuntimeError(issue)
                expect(page.locator('#controls .role')).to_have_count(roles,timeout=180000)
                session=page.evaluate('session')
                # Switching input mode must preserve a selected, not-yet-queued action.
                # Await the actual change handler so this catches a delayed refresh too.
                page.evaluate('()=>{const handler=document.getElementById("direct").onchange;document.getElementById("direct").onchange=e=>window.modeChangeDone=Promise.resolve(handler(e))}')
                page.locator('#action-0').select_option(str(sequence[0][0]))
                page.locator('#direct').uncheck();page.evaluate('window.modeChangeDone')
                expect(page.locator('#action-0')).to_have_value(str(sequence[0][0]))
                page.locator('#scene-zoom').select_option('2')
                for tick,actions in enumerate(sequence,1):
                    for role,value in enumerate(actions):
                        if game=='gi216' and tick==1 and role==0:
                            point=page.locator('#world').evaluate('(c)=>{const a=current.views[0].original_animation,v=sceneViewport(current.views[0]),r=c.getBoundingClientRect();return {x:(v.x+(a.geometry.ox+9.5*a.geometry.cell)*v.scale)*r.width/c.width,y:(v.y+(a.geometry.oy+5.5*a.geometry.cell)*v.scale)*r.height/c.height}}')
                            page.locator('#world').click(position=point);page.locator('#object-actions').get_by_role('button',name='Build levee · 2 stone',exact=True).click()
                        elif game=='gi217' and tick==2 and role==7:
                            page.locator('#camera').select_option('7')
                            page.wait_for_function('document.getElementById("world").dataset.role==="7"')
                            point=page.locator('#world').evaluate('(c)=>{const f=current.views[7],a=f.original_animation,v=sceneViewport(f),e=f.entities.find(e=>e.group==="facilities"&&e.index===0),r=c.getBoundingClientRect();return {x:(v.x+(a.geometry.ox+(e.xy[0]+.5)*a.geometry.cell)*v.scale)*r.width/c.width,y:(v.y+(a.geometry.oy+(e.xy[1]+.5)*a.geometry.cell)*v.scale)*r.height/c.height}}')
                            page.locator('#world').click(position=point);page.locator('#object-actions').get_by_role('button',name='Draw water from well',exact=True).click()
                        else:
                            page.locator(f'#action-{role}').select_option(str(value));page.locator('#controls .role').nth(role).locator('button').click()
                        expect(page.locator('#controls .role').nth(role).locator('button')).to_have_text('Queued')
                    page.locator('#step').click();expect(page.locator('#clock')).to_have_text(f't={tick} / {horizon}',timeout=180000);page.wait_for_function('!busy')
                if game=='gi216':
                    assert page.evaluate('current.views[0].visible_grid_fields.structure[5][9]')==3
                    assert page.evaluate('current.views[0].visible_grid_fields.gate_open[5][9]')==1
                    assert page.evaluate('current.views[0].observed_fields.report_tick[0]')==9
                    assert page.evaluate('current.views[0].map.flat().some(x=>x===null)')
                    assert page.evaluate('current.views[0].visible_grid_fields.depth.flat().some(x=>x===null)')
                else:
                    assert page.evaluate('current.views[2].entities.find(e=>e.group==="plots"&&e.index===20).attributes.trait')==0
                    assert page.evaluate('JSON.stringify(current.views).includes("achieved")')is False
                selected='0'if game=='gi216'else'2';page.locator('#camera').select_option(selected)
                page.locator('#world').screenshot(path=str(out/f'{game}-normal.png'))
                page.locator('#debug').check();page.locator('#camera').select_option('-1')
                expect(page.locator('#scope')).to_contain_text('FULL_STATE_DEBUG')
                assert page.evaluate('current.views[-1].information')=='FULL_STATE_DEBUG'
                assert not errors,errors
                page.locator('#close').click();expect(page.locator('#session')).to_be_hidden();page.close()
                events=[json.loads(x)for x in (out/'sessions'/session/'events.jsonl').read_text(encoding='utf8').splitlines()]
                actual=[e['value']['actions']for e in events if e['type']=='transition']
                assert actual==sequence,dict(game=game,session=session,expected=sequence,actual=actual)
                result=dict(status='PASS',game=game,native_joint_steps=len(sequence),roles_owned=roles,
                    object_menu_executes_native_action=True,zoomed_click=True,role_privacy_checks=True,debug_logged_separately=True,
                    browser_errors=errors,runtime_sha256=runtime,human_scores=None,
                    scope='Finite native browser walkthrough; not full-game or scientific acceptance')
                (R/'verification/gameplay'/f'{game}_live_ui.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)
            browser.close()
        assert runtime=={p:hashlib.sha256((R/'src/crucibles_jaxmarl'/p).read_bytes()).hexdigest()for p in files}
    finally:
        if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        else:child.terminate()
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:child.kill();child.wait()
        log.close()

if __name__=='__main__':main()
