"""Real Chromium DOM controls against the real local JAX server, per game."""
from pathlib import Path
import argparse,hashlib,json,os,re,socket,subprocess,sys,time,urllib.request,uuid

R=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--games',nargs='*');ap.add_argument('--port',type=int,default=8768);ap.add_argument('--retry',action='store_true');ap.add_argument('--shard');a=ap.parse_args()
    from crucibles_jaxmarl import registered_envs
    from playwright.sync_api import sync_playwright,expect
    expect.set_options(timeout=180000)
    out=R/'verification/browser';out.mkdir(parents=True,exist_ok=True)
    ui_hash=hashlib.sha256(b''.join((R/'src/crucibles_jaxmarl'/p).read_bytes() for p in ['server.py','rendering.py','web/app.js','web/index.html','web/style.css'])).hexdigest()
    logs=R/'verification/logs';logs.mkdir(exist_ok=True)
    base=f'http://127.0.0.1:{a.port}'
    boot_nonce=uuid.uuid4().hex;actual_server_pid=None
    # Never attach a test to an older or unrelated service occupying the port.
    with socket.socket() as probe:probe.bind(('127.0.0.1',a.port))
    env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',CRUCIBLES_TEST_NONCE=boot_nonce)
    log=(logs/('browser_server-'+str(a.port)+'.log')).open('w',encoding='utf-8')
    server=subprocess.Popen([sys.executable,'-m','crucibles_jaxmarl','play','--port',str(a.port),'--engineering','--runs',str(R/'local-runs/browser-verification')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        failures=0
        for _ in range(100):
            try:
                with urllib.request.urlopen(base+'/health',timeout=1) as resp:health=json.load(resp)
                if health.get('test_nonce')!=boot_nonce:raise RuntimeError('Health endpoint belongs to a different process')
                actual_server_pid=health['pid'];assert health['ok']
                break
            except (urllib.error.URLError,TimeoutError,ConnectionError):
                if server.poll() is not None:raise RuntimeError('Owned test server exited before becoming healthy')
                time.sleep(.1)
        else:raise RuntimeError('Test server did not become healthy')
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            games=a.games or registered_envs
            if a.shard:
                index,total=map(int,a.shard.split('/'))
                if not 0<=index<total:ap.error('shard must be index/total')
                games=games[index::total]
            for game in games:
                dest=out/(game+'.json')
                if not a.retry and dest.exists():
                    prior=json.loads(dest.read_bytes())
                    if prior['status']=='PASS' and prior.get('ui_sha256')==ui_hash:continue
                if dest.exists():
                    history=out/'history';history.mkdir(exist_ok=True)
                    import shutil
                    shutil.copyfile(dest,history/(game+'-'+str(time.time_ns())+'.json'))
                started=time.time();record=dict(game=game,status='RUNNING',human_scores=None,engineering=True,checks=[],ui_sha256=ui_hash)
                context=browser.new_context(viewport=dict(width=1500,height=1000));page=context.new_page();errors=[]
                page.on('pageerror',lambda error:errors.append(str(error)))
                session=None;token=None
                try:
                    page.goto(base+'/?game='+game,wait_until='networkidle');page.set_default_timeout(180000)
                    expect(page.locator('#health')).to_contain_text('Python ready')
                    page.locator('#create').click();expect(page.locator('#session')).to_be_visible(timeout=180000)
                    expect(page.locator('#progress')).to_contain_text('ready',timeout=180000)
                    assert not re.search('[\u3400-\u9fff]',page.locator('body').inner_text()), 'Game UI contains Chinese text'
                    record['checks'].append('english_only_game_ui')
                    # Read only our own test capability; never include it in evidence.
                    from urllib.parse import parse_qs,urlsplit
                    parsed=parse_qs(urlsplit(page.url).fragment);session=parsed['session'][0];token=parsed['token'][0]
                    record['checks'].append('catalog_selected_and_real_native_reset')
                    page.locator('#roles').fill('0');page.locator('#assign').click();expect(page.locator('#action-0')).to_be_visible()
                    phase_setup={'gi134':6,'gi196':1}.get(game,0)
                    for phase_step in range(phase_setup):
                        page.locator('#step').click();expect(page.locator('#clock')).to_contain_text(f't={phase_step+1} /')
                    if phase_setup:record['phase_setup_native_steps']=phase_setup
                    page.locator('#controls button').first.click()
                    options=page.locator('#action-0 option').evaluate_all('(xs)=>xs.map(x=>({id:x.value,label:x.textContent}))')
                    directions={'north':'ArrowUp','east':'ArrowRight','south':'ArrowDown','west':'ArrowLeft'}
                    choice=None
                    for o in options:
                        label=o['label'].replace('_',' ').lower()
                        if not re.search(r'^(north|east|south|west)\b|\b(move|walk|step|throttle|climb|swim|attempt|steer|pull|thrust|scout|request|travel|face)\b',label):continue
                        direction=next((d for d in directions if re.search(r'\b(?:'+d+r'(?:ward)?|'+d[0]+r')\b',label)),None)
                        if direction:choice=(o,directions[direction]);break
                    if choice:
                        page.locator('#world').click();page.keyboard.press(choice[1]);expect(page.locator('#action-0')).to_have_value(choice[0]['id'])
                        expect(page.locator('#progress')).to_contain_text('Queued ')
                        record['keyboard_action_id']=int(choice[0]['id']);record['checks'].append('keyboard_native_direction_command')
                    else:record['keyboard']='NO_LEGAL_CARDINAL_COMMAND_FOR_ROLE0_AT_RESET'
                    page.locator('#step').click();expect(page.locator('#clock')).to_contain_text(f't={phase_setup+1} /')
                    record['checks'].append('dom_queue_and_native_step')
                    page.locator('#invite').click();expect(page.locator('#invitation')).to_be_visible()
                    invite=page.locator('#invitation').input_value()
                    player=context.new_page();player.goto(invite,wait_until='networkidle')
                    expect(player.locator('#session')).to_be_visible();expect(player.locator('#step')).to_be_disabled()
                    page.locator('#player').select_option('Player 1');page.locator('#roles').fill('1');page.locator('#assign').click()
                    expect(page.locator('#player option[value="Player 1"]')).to_have_text('Player 1 (1 roles)')
                    page.locator('#player').select_option('Host');player.locator('#refresh').click();expect(player.locator('#action-1')).to_be_visible()
                    assert player.locator('#action-0').count()==0
                    player.locator('#controls button').first.click();page.locator('#step').click();expect(page.locator('#clock')).to_contain_text(f't={phase_setup+2} /')
                    record['checks'].append('second_participant_ownership_and_native_joint_step')
                    page.locator('#all').click()
                    expected=int(page.locator('#scope').inner_text().split(' · ')[1].split()[0])
                    expect(page.locator('#controls .role')).to_have_count(expected)
                    record['checks'].append('all_roles_individually_controllable_no_role_cap')
                    page.locator('#debug').check();expect(page.locator('#scope')).to_contain_text('FULL_STATE_DEBUG')
                    page.locator('#camera').select_option('-1')
                    record['checks'].append('explicit_debug_camera')
                    # The rendered canvas must contain real non-background pixels.
                    assert page.locator('#world').evaluate('(c)=>{const a=c.getContext("2d").getImageData(0,0,c.width,c.height).data;let n=0;for(let i=0;i<a.length;i+=4)if(a[i]>70||a[i+1]>70||a[i+2]>70)n++;return n>1000}')
                    shots=R/'verification/screenshots';shots.mkdir(exist_ok=True)
                    page.locator('#world').screenshot(path=str(shots/(game+'.png')))
                    page.locator('#player').select_option('AI');page.locator('#all').click();expect(page.locator('#controls .role')).to_have_count(0)
                    page.locator('#step').click();expect(page.locator('#clock')).to_contain_text(f't={phase_setup+3} /')
                    record['checks'].append('release_to_pure_ai_without_reset')
                    assert page.locator('#error').inner_text()=='',page.locator('#error').inner_text()
                    assert not errors,errors
                    page.locator('#close').click();expect(page.locator('#session')).to_be_hidden();expect(page.locator('#progress')).to_contain_text('Logs preserved')
                    session=None;token=None
                    record['checks'].append('close_session_preserves_log')
                    record['status']='PASS'
                except Exception as error:
                    record.update(status='FAIL',error=f'{type(error).__name__}: {error}',javascript_errors=errors)
                finally:
                    if session and token:
                        try:context.request.post(base+'/api',data=dict(op='close',session=session,token=token))
                        except Exception:pass
                    context.close()
                record['seconds']=time.time()-started
                temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');os.replace(temp,dest)
                print(game,record['status'],round(record['seconds'],1),flush=True)
                failures+=record['status']!='PASS'
            browser.close()
    finally:
        if os.name=='nt':
            # Windows venv redirectors launch the actual interpreter as a child.
            # Kill only our verified Popen process tree, including that child.
            subprocess.run(['taskkill','/PID',str(server.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        else:server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:server.kill();server.wait()
        log.close()
    return bool(failures)

if __name__=='__main__':raise SystemExit(main())
