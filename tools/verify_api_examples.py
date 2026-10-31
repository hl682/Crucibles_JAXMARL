"""Execute the documented JIT, 16-environment vmap and scan examples."""
from pathlib import Path
import json,time

def main():
    import jax
    from crucibles_jaxmarl import make
    start=time.time();env=make('gi072');key=jax.random.PRNGKey(0)
    key,reset_key,action_key,step_key=jax.random.split(key,4)
    obs,state=jax.jit(env.reset)(reset_key);actions=env.sample_actions(action_key,state)
    obs,state,rewards,dones,info=jax.jit(env.step_env)(step_key,state,actions);jax.block_until_ready(rewards)
    assert set(obs)==set(env.agents) and '__all__' in dones
    batch_keys=jax.random.split(key,16)
    batch_obs,batch_state=jax.jit(jax.vmap(env.reset))(batch_keys)
    batch_actions=jax.vmap(env.sample_actions)(batch_keys,batch_state)
    batch_obs,batch_state,r,d,info=jax.jit(jax.vmap(env.step))(batch_keys,batch_state,batch_actions);jax.block_until_ready(r)
    assert all(v.shape==(16,) for v in r.values())
    def rollout(carry,_):
        key,state=carry;key,ak,sk=jax.random.split(key,3);actions=env.sample_actions(ak,state)
        obs,state,rewards,dones,info=env.step(sk,state,actions)
        return (key,state),(rewards,dones)
    (_,final_state),trajectory=jax.jit(lambda k,s:jax.lax.scan(rollout,(k,s),None,length=128))(key,state)
    jax.block_until_ready(trajectory);assert all(v.shape==(128,) for v in trajectory[0].values())
    result=dict(status='PASS',game='gi072',single_jit_step=True,vmap_batch=16,scan_steps=128,seconds=time.time()-start)
    root=Path(__file__).resolve().parents[1];(root/'verification/api_examples.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
