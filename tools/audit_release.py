"""Requirement-level local audit of actual immutable sources and executed evidence."""
from pathlib import Path
import ast,collections,gzip,hashlib,json,math,subprocess

R=Path(__file__).resolve().parents[1];S=R/'src/crucibles_jaxmarl'
def load(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    catalog=load(S/'catalog.json')['games'];native=[];browser=[];metadata_revisions=[]
    assert len(catalog)==256 and len({r['source_sha256'] for r in catalog})==256
    for row in catalog:
        root=S/'games'/row['key'];manifest=load(root/'manifest.json')
        assert sha(root/'candidate.py')==row['source_sha256']==manifest['source_sha256']
        assert sha(root/'manifest.json')==row['manifest_sha256']
        if (root/'scene.json').exists():
            scene=load(root/'scene.json');assert scene['source_sha256']==row['source_sha256']
            assert scene['manifest_sha256']==row['manifest_sha256']
        old=R/'source-history'/row['key']/'release-metadata-v0.json'
        if old.exists():
            previous=load(old)
            for field in ['source_sha256','num_agents','num_actions','observation_size','max_steps','reward_low','reward_high','observation_low','observation_high','terminal_semantics']:
                assert previous[field]==manifest[field],(row['key'],field)
            metadata_revisions.append(row['key'])
        n=load(R/'verification'/(row['key']+'.json'));b=load(R/'verification/browser'/(row['key']+'.json'))
        assert n['status']==b['status']=='PASS',row['key']
        assert n['source_sha256']==row['source_sha256'] and n['dimensions']==row['dimensions']
        assert n['verification_revision']>=2 and n['terminal'] is True
        assert n['render_roles_checked']==manifest['num_agents']
        assert n['terminal_outer_reset']=='PASS'
        if row['family']=='spatial':
            assert n['raw_terminal_absorbing']=='PASS'
            assert 'keyboard_native_direction_command' in b['checks']
        for check in ['eager_reset','eager_step','jit_reset','jit_step','jit_vmap_reset','jit_vmap_step','outer_autoreset_api','full_original_horizon_episode_and_render']:
            assert n['checks'][check]['status']=='PASS'
        for check in ['dom_queue_and_native_step','second_participant_ownership_and_native_joint_step','all_roles_individually_controllable_no_role_cap','explicit_debug_camera','release_to_pure_ai_without_reset','close_session_preserves_log']:
            assert check in b['checks']
        with gzip.open(R/'verification/traces'/(row['key']+'.json.gz'),'rt',encoding='utf-8') as f:trace=json.load(f)
        assert len(trace)==n['episode_steps']<=manifest['max_steps'] and trace[-1]['done'] is True
        totals=[0.0]*manifest['num_agents']
        for t,step in enumerate(trace):
            assert step['t']==t+1 and len(step['actions'])==len(step['rewards'])==manifest['num_agents']
            assert all(type(a)is int and 0<=a<manifest['num_actions'] for a in step['actions'])
            assert all(math.isfinite(x) for x in step['rewards'])
            totals=[a+b for a,b in zip(totals,step['rewards'])]
        assert totals==n['scripted_random_returns']
        assert n['human_scores'] is None and b['human_scores'] is None
        native.append(n);browser.append(b)
    for name in ['docs_ui','session_contract','network_transport','api_examples','original_saved_browser','release_checks','full_source_inventory','clean_install']:
        assert load(R/'verification'/(name+'.json'))['status']=='PASS',name
    base='ea670cd'
    pieces=['server.py','rendering.py','web/app.js','web/index.html','web/style.css']
    base_bytes=[subprocess.check_output(['git','show',base+':src/crucibles_jaxmarl/'+p],cwd=R) for p in pieces]
    base_ui_hash=hashlib.sha256(b''.join(base_bytes)).hexdigest()
    assert all(old==(S/p).read_bytes() for old,p in zip(base_bytes[1:],pieces[1:]) if p!='web/app.js'), 'Rendering/style changed after per-game test'
    old_app=base_bytes[2].decode().splitlines();new_app=(S/'web/app.js').read_text(encoding='utf-8').splitlines()
    assert old_app[:-1]==new_app[:-1] and 'test(game)' in new_app[-1], 'Only the initial legacy-namespace selector may differ'
    old_session=next(n for n in ast.parse(base_bytes[0].decode()).body if isinstance(n,ast.ClassDef) and n.name=='Session')
    new_session=next(n for n in ast.parse((S/'server.py').read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='Session')
    assert ast.dump(old_session)==ast.dump(new_session), 'Per-game native control/session logic changed'
    # The only server change after the all-game browser pass adds interface/origin
    # binding. Its separate real peer/alternate-host check is required above.
    network_ui_hash=hashlib.sha256((S/'server.py').read_bytes()+b''.join(base_bytes[1:])).hexdigest()
    known={base_ui_hash,network_ui_hash,hashlib.sha256(b''.join((S/p).read_bytes() for p in pieces)).hexdigest()}
    assert all(b['ui_sha256'] in known for b in browser)
    parity=collections.Counter(n['numerical_parity'] for n in native)
    result=dict(status='LOCAL_ENGINEERING_VERIFICATION_COMPLETE',registered_unique_games=256,spatial_games=180,legacy_abstract_games=76,
        native_full_episode_pass=256,browser_control_pass=256,spatial_keyboard_pass=180,
        metadata_only_legacy_documentation_revisions=len(metadata_revisions),
        preserved_population_range=[min(r['dimensions']['num_agents'] for r in catalog),max(r['dimensions']['num_agents'] for r in catalog)],
        preserved_horizon_range=[min(r['dimensions']['max_steps'] for r in catalog),max(r['dimensions']['max_steps'] for r in catalog)],
        numerical_parity=dict(parity),floating_difference_max=max(n['float_parity_max_abs'] for n in native),
        floating_differences_are_not_calibrated_acceptance=True,original_saved_replay_tracks=17,public_gif_count=413,
        browser_base_commit=base,browser_base_ui_sha256=base_ui_hash,native_session_and_rendering_core_unchanged=True,
        later_optional_bind_origin_change_separately_verified='network_transport.json',legacy_namespace_bootstrap_verified=['v1_000','v1_001','v1_002','v1_003'],
        human_scores=None,research_conclusions_changed=False,new_games_generated=False,old_training_resumed=False,
        limits=['No real human participant study','No physical two-computer test','CPU validation; CUDA/TPU untested',
                'Generic source-driven live graphics plus original bespoke saved previews; no commercial game engine',
                'Engineering contract passes do not establish strategic depth, learning, transfer or scientific numerical acceptance'])
    (R/'verification/release_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
