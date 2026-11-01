"""Verify presentation-only scope against the immutable v0.1.0 release."""
from pathlib import Path
import hashlib,json,re,subprocess
R=Path(__file__).resolve().parents[1];S=R/'src/crucibles_jaxmarl'
def main():
    from crucibles_jaxmarl import registered_envs,game_info
    # This check intentionally fails if scientific sources, contracts or transition wrappers change.
    paths=['src/crucibles_jaxmarl/games','src/crucibles_jaxmarl/catalog.json','src/crucibles_jaxmarl/environment.py','src/crucibles_jaxmarl/rendering.py','src/crucibles_jaxmarl/web/style.css']
    subprocess.run(['git','diff','--exit-code','0b0ed5d737cb8a1d6fa64a7d5fc8092cbf0ed529','--',*paths],cwd=R,check=True,stdout=subprocess.DEVNULL)
    assert len(registered_envs)==256
    for key in registered_envs:
        info=game_info(key);assert not re.search('[\u3400-\u9fff]',info['title']+info['summary']),key
        m=json.loads((S/'games'/key/'manifest.json').read_bytes());sp=S/'games'/key/'scene.json';sc=json.loads(sp.read_bytes()) if sp.exists() else {}
        visible=dict(actions=m['actions'],roles=m['agents'],task=m.get('task',{}),rules=sc.get('rules',[]),entities=sc.get('entities',[]))
        assert not re.search('[\u3400-\u9fff]',json.dumps(visible,ensure_ascii=False)),key
        p=json.loads((R/'verification/presentation'/(key+'.json')).read_bytes());assert p['status']=='PASS' and p['source_sha256']==info['source_sha256']
        assert p['captured_endpoints'][0]==0 and p['captured_endpoints'][-1]==p['actual_native_steps']
    live=json.loads((R/'verification/gameplay/gi215_live_ui.json').read_bytes())
    assert live['status']=='PASS' and live['served_orders']==1 and live['canvas_item_click_changes_native_holder']
    for path,digest in live['runtime_sha256'].items():assert hashlib.sha256((S/path).read_bytes()).hexdigest()==digest,path
    water=json.loads((R/'verification/gameplay/gi215_water_pair.json').read_bytes())
    assert water['status']=='PASS' and water['only_changed_action_step']==2
    for name,suffix in [('unprotected','unprotected'),('diverted','diverted')]:
        assert water[name]['source_sha256']==game_info('gi215')['source_sha256']
        assert hashlib.sha256((R/'docs/walkthroughs'/('gi215-water-'+suffix+'.gif')).read_bytes()).hexdigest()==water[name]['gif_sha256']
    magnetic=json.loads((R/'verification/gameplay/gi214_live_ui.json').read_bytes())
    assert magnetic['status']=='PASS' and magnetic['all_actor_positions_unchanged'] and magnetic['bridge_and_rail_visible']
    for path,digest in magnetic['runtime_sha256'].items():assert hashlib.sha256((S/path).read_bytes()).hexdigest()==digest,path
    for game in ('gi216','gi217'):
        actual=json.loads((R/'verification/gameplay'/f'{game}_live_ui.json').read_bytes())
        assert actual['status']=='PASS' and actual['role_privacy_checks'] and actual['object_menu_executes_native_action']
        for path,digest in actual['runtime_sha256'].items():assert hashlib.sha256((S/path).read_bytes()).hexdigest()==digest,path
    result=dict(status='PASS',reference='v0.1.0',native_rules_manifests_scenes_catalog_transition_wrapper_base_projection_byte_unchanged=True,
        english_visible_game_metadata=256,presentation_replays=256,game_manuals=512,
        classification='CURATED_PER_GAME_RULE_READING',classification_identities=256,classification_sha256=hashlib.sha256((R/'manuals/classification.tsv').read_bytes()).hexdigest(),
        modified_runtime_scope=['registry English display title/summary overlay','catalog HTTP response uses the same display overlay','Quick play and optional direct native-action controls','Original-art live views and object actions for GI214-217','Display-only follow zoom; native coordinates preserved'],
        gameplay_verified_scope=['gi214 four-step construction and power','gi215 first order','gi215 dirty-water diversion pair','gi216 15-step local construction and transfer','gi217 22-step local harvest/dry/plant/water'],all_game_playability_accepted=False,
        original_all_game_execution_evidence='v0.1.0 verification archive; source identities unchanged',new_browser_regressions='verification/browser; use receipt ui_sha256 and history',
        no_training_or_performance_claim=True)
    (R/'verification/presentation_contract.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
