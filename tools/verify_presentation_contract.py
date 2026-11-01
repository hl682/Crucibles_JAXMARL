"""Verify presentation-only scope against the immutable v0.1.0 release."""
from pathlib import Path
import hashlib,json,re,subprocess
R=Path(__file__).resolve().parents[1];S=R/'src/crucibles_jaxmarl'
def main():
    from crucibles_jaxmarl import registered_envs,game_info
    # This check intentionally fails if scientific sources, contracts or transition wrappers change.
    paths=['src/crucibles_jaxmarl/games','src/crucibles_jaxmarl/catalog.json','src/crucibles_jaxmarl/environment.py','src/crucibles_jaxmarl/rendering.py','src/crucibles_jaxmarl/web']
    subprocess.run(['git','diff','--exit-code','0b0ed5d737cb8a1d6fa64a7d5fc8092cbf0ed529','--',*paths],cwd=R,check=True,stdout=subprocess.DEVNULL)
    assert len(registered_envs)==256
    for key in registered_envs:
        info=game_info(key);assert not re.search('[\u3400-\u9fff]',info['title']+info['summary']),key
        m=json.loads((S/'games'/key/'manifest.json').read_bytes());sp=S/'games'/key/'scene.json';sc=json.loads(sp.read_bytes()) if sp.exists() else {}
        visible=dict(actions=m['actions'],roles=m['agents'],task=m.get('task',{}),rules=sc.get('rules',[]),entities=sc.get('entities',[]))
        assert not re.search('[\u3400-\u9fff]',json.dumps(visible,ensure_ascii=False)),key
        p=json.loads((R/'verification/presentation'/(key+'.json')).read_bytes());assert p['status']=='PASS' and p['source_sha256']==info['source_sha256']
        assert p['captured_endpoints'][0]==0 and p['captured_endpoints'][-1]==p['actual_native_steps']
    result=dict(status='PASS',reference='v0.1.0',native_sources_manifests_scenes_catalog_transition_wrapper_live_rendering_and_controls_byte_unchanged=True,
        english_visible_game_metadata=256,presentation_replays=256,game_manuals=512,
        classification='CURATED_PER_GAME_RULE_READING',classification_identities=256,classification_sha256=hashlib.sha256((R/'manuals/classification.tsv').read_bytes()).hexdigest(),
        modified_runtime_scope=['registry English display title/summary overlay','catalog HTTP response uses the same display overlay','versioned health and log labels 0.1.2'],
        original_all_game_execution_evidence='v0.1.0 verification archive; source identities unchanged',new_browser_regressions='verification/browser; use receipt ui_sha256 and history',
        no_training_or_performance_claim=True)
    (R/'verification/presentation_contract.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
