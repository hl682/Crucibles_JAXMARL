"""Drive the live original kitchen through browser controls to a served order."""
from pathlib import Path
import json,os,socket,subprocess,sys,time,urllib.request,uuid,hashlib
from playwright.sync_api import sync_playwright,expect
from verify_bistro_gameplay import ACTIONS

R=Path(__file__).resolve().parents[1]

def main():
    runtime_files=['server.py','bistro_art.py','bistro_presentation.py','web/app.js','web/index.html']
    runtime={p:hashlib.sha256((R/'src/crucibles_jaxmarl'/p).read_bytes()).hexdigest()for p in runtime_files}
    port=8796;nonce=uuid.uuid4().hex;out=R/'local-runs/bistro-ui';out.mkdir(parents=True,exist_ok=True)
    with socket.socket()as sock:sock.bind(('127.0.0.1',port))
    log=(out/'server.log').open('w',encoding='utf8')
    child=subprocess.Popen([sys.executable,'-m','crucibles_jaxmarl','play','--port',str(port),'--game','215','--engineering','--runs',str(out/'sessions')],cwd=R,env=dict(os.environ,JAX_PLATFORMS='cpu',CRUCIBLES_TEST_NONCE=nonce),stdout=log,stderr=subprocess.STDOUT)
    try:
        base=f'http://127.0.0.1:{port}'
        for _ in range(100):
            try:
                health=json.load(urllib.request.urlopen(base+'/health',timeout=1));assert health['test_nonce']==nonce;break
            except (OSError,ValueError):
                if child.poll()is not None:raise RuntimeError('Owned server exited')
                time.sleep(.1)
        else:raise RuntimeError('Owned server not ready')
        with sync_playwright()as p:
            browser=p.chromium.launch(headless=True);page=browser.new_page(viewport=dict(width=1500,height=1100));errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(base+'/?game=215');page.locator('#quickplay').click();expect(page.locator('#controls .role')).to_have_count(2,timeout=180000)
            expect(page.locator('#direct')).to_be_checked();expect(page.locator('#clock')).to_have_text('t=0 / 256')
            page.keyboard.press('d');expect(page.locator('#clock')).to_have_text('t=1 / 256',timeout=180000)
            page.wait_for_function('!busy')
            # Browser state retains original entity coordinates; the canvas uses original art.
            assert page.evaluate('current.views[0].entities.find(e=>e.kind==="agent"&&e.index===0).xy')==[6,3]
            assert page.evaluate('current.views[0].original_animation.frames.length')==4
            page.locator('#camera').select_option('1')
            point=page.locator('#world').evaluate('(c)=>{const a=current.views[1].original_animation,s=Math.min(c.width/a.width,c.height/a.height),r=c.getBoundingClientRect();return {x:((c.width-a.width*s)/2+(a.geometry.ox+3.5*a.geometry.cell)*s)*r.width/c.width,y:((c.height-a.height*s)/2+(a.geometry.oy+3.5*a.geometry.cell)*s)*r.height/c.height}}')
            page.locator('#world').click(position=point);expect(page.locator('#clock')).to_have_text('t=2 / 256',timeout=180000);page.wait_for_function('!busy')
            assert page.evaluate('current.views[1].entities.find(e=>e.group==="kitchen_items"&&e.index===1).attributes.holder')==1
            page.locator('#direct').uncheck();expect(page.locator('#controls .role').first.locator('button')).to_have_text('Queue')
            sequence=ACTIONS[1:]
            for t,actions in enumerate(sequence,start=3):
                for role,action in enumerate(actions):
                    page.locator(f'#action-{role}').select_option(str(action));page.locator('#controls .role').nth(role).locator('button').click()
                    expect(page.locator('#controls .role').nth(role).locator('button')).to_have_text('Queued')
                page.locator('#step').click();expect(page.locator('#clock')).to_have_text(f't={t} / 256',timeout=180000)
                page.wait_for_function('!busy')
                expect(page.locator('#controls .role')).to_have_count(2)
                if t==3:
                    expect(page.locator('#feedback')).to_contain_text('Washed');page.locator('#world').screenshot(path=str(out/'washed.png'))
            expect(page.locator('#feedback')).to_contain_text('Orders served: 1/4');expect(page.locator('#feedback')).to_contain_text('Served!')
            page.locator('#world').screenshot(path=str(out/'served.png'))
            assert page.evaluate('current.bistro.order_status')==[1,0,0,0]
            assert not errors,errors
            page.locator('#close').click();expect(page.locator('#session')).to_be_hidden();browser.close()
        files=list((out/'sessions').glob('*/events.jsonl'));latest=max(files,key=lambda x:x.stat().st_mtime);events=[json.loads(x)for x in latest.read_text(encoding='utf8').splitlines()]
        transitions=[x['value']['actions']for x in events if x['type']=='transition'];assert transitions==[[2,0],[0,5],*sequence]
        result=dict(status='PASS',game='gi215',quickplay_owns_two_roles=True,direct_keyboard_changes_native_position=True,canvas_item_click_changes_native_holder=True,real_native_joint_steps=len(transitions),washed_feedback=True,served_orders=1,original_scene_animation=True,browser_errors=errors,human_scores=None,scope='Browser engineering walkthrough, not human performance or all-game acceptance')
        assert runtime=={p:hashlib.sha256((R/'src/crucibles_jaxmarl'/p).read_bytes()).hexdigest()for p in runtime_files}
        result['runtime_sha256']=runtime
        target=R/'verification/gameplay/gi215_live_ui.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)
    finally:
        if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        else:child.terminate()
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:child.kill();child.wait()
        log.close()

if __name__=='__main__':main()
