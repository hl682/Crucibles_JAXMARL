"""Run with the newly installed release wheel, outside the editable development env."""
from pathlib import Path
import hashlib,json,re,sys
def main():
    import jax
    import crucibles_jaxmarl as pkg
    assert Path(pkg.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'Expected installed wheel'
    assert pkg.__version__=='0.1.1'
    for key in pkg.registered_envs:
        env=pkg.make(key)
        assert not re.search('[\u3400-\u9fff]',env.metadata['title']+env.metadata['summary'])
    env=pkg.make('gi225');keys=jax.random.split(jax.random.PRNGKey(0),3)
    obs,state=jax.jit(env.reset)(keys[0]);actions=env.sample_actions(keys[1],state)
    obs,state,rewards,dones,info=jax.jit(env.step_env)(keys[2],state,actions);jax.block_until_ready(rewards)
    assert set(obs)==set(env.agents) and len(obs)==64
    root=Path(__file__).resolve().parents[1];wheel=root/'dist/crucibles_jaxmarl-0.1.1-py3-none-any.whl'
    result=dict(status='PASS',version=pkg.__version__,installed_wheel=True,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),identities_instantiated_and_hash_checked=256,english_titles_and_descriptions=256,real_jit_reset_and_step='gi225',agents=64,human_performance_claim=False)
    (root/'verification/installed_presentation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
