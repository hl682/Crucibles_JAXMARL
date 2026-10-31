"""Run with the clean environment's Python -I to verify the shipped wheel."""
from pathlib import Path
import hashlib,importlib.metadata,json,sys

def main():
    import crucibles_jaxmarl as cj
    import jax
    root=Path(__file__).resolve().parents[1]
    assert sys.prefix!=sys.base_prefix and Path(cj.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert len(cj.registered_envs)==256
    for name in cj.registered_envs:
        env=cj.make(name);assert env.name==name and len(env.agents)==env.num_agents
    env=cj.make('gi225');key=jax.random.PRNGKey(0)
    obs,state=jax.jit(env.reset)(key);actions=env.sample_actions(jax.random.fold_in(key,1),state)
    result=jax.jit(env.step_env)(jax.random.fold_in(key,2),state,actions);jax.block_until_ready(result)
    wheel=next((root/'dist').glob('*.whl'))
    receipt=dict(status='PASS',package_version=cj.__version__,python=sys.version.split()[0],
        installed_module_origin='clean isolated venv site-packages, not editable workspace',
        registered_and_instantiated_games=256,quickstart_game='gi225',quickstart_real_jit_step=True,
        packages={k:importlib.metadata.version(k) for k in ['jax','jaxlib','numpy','pillow','scipy']},
        wheel=wheel.name,wheel_bytes=wheel.stat().st_size,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    (root/'verification/clean_install.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
