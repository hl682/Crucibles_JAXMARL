"""Run one game's real contract checks and a full unmodified-horizon episode."""
from pathlib import Path
import argparse
import json
import os
import time
import traceback

def main():
    ap=argparse.ArgumentParser();ap.add_argument('game');ap.add_argument('--output',default='verification');args=ap.parse_args()
    # Bound only this verification process, without changing other sessions.
    if os.name=='nt':
        import ctypes
        k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=ctypes.c_void_p
        k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
        k.SetProcessAffinityMask(k.GetCurrentProcess(),0xff)
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    import jax
    import jax.numpy as jnp
    import numpy as np
    import crucibles_jaxmarl as cj
    from crucibles_jaxmarl.rendering import scene_frame,save_gif
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    start=time.time();env=cj.make(args.game);key=jax.random.PRNGKey(20261101)
    result=dict(game=env.name,status='RUNNING',verification_revision=2,source_sha256=env.metadata['source_sha256'],
                seed=20261101,runtime=dict(jax=jax.__version__,numpy=np.__version__,devices=[str(d) for d in jax.devices()]),
                dimensions=env.metadata['dimensions'],checks={},human_scores=None,browser_controls='PENDING',failures=[])
    path=out/(env.name+'.json')
    def write():
        result['elapsed_seconds']=time.time()-start
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(temp,path)
    def check(name,fn):
        t=time.time()
        try:
            value=fn();result['checks'][name]=dict(status='PASS',seconds=time.time()-t);write();return value
        except Exception as e:
            result['checks'][name]=dict(status='FAIL',seconds=time.time()-t,error=f'{type(e).__name__}: {e}')
            result['failures'].append(traceback.format_exc());write();raise
    write()
    try:
        obs,state=check('eager_reset',lambda:jax.block_until_ready(env.reset(key)))
        def structure():
            assert set(obs)==set(env.agents)
            for a in env.agents:
                assert np.asarray(obs[a]).shape==(env.observation_size,)
                assert np.isfinite(obs[a]).all()
                assert bool(env.observation_space(a).contains(obs[a])),a
            masks=env.legal_actions(state)
            assert masks.shape==(env.num_agents,env.num_actions) and masks.dtype==jnp.bool_
            assert np.asarray(masks).any(axis=1).all()
        check('observation_spaces_legal_masks',structure)
        actions=env.sample_actions(jax.random.fold_in(key,1),state)
        eager=check('eager_step',lambda:jax.block_until_ready(env.step_env(key,state,actions)))
        compiled_reset=jax.jit(env.reset);compiled_step=jax.jit(env.step_env)
        compiled=check('jit_reset',lambda:jax.block_until_ready(compiled_reset(key)))
        compiled_transition=check('jit_step',lambda:jax.block_until_ready(compiled_step(key,state,actions)))
        def parity():
            diffs=[]
            for left,right in [(obs,compiled[0]),(state,compiled[1]),(eager,compiled_transition)]:
                for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right)):
                    a,b=np.asarray(a),np.asarray(b)
                    assert a.shape==b.shape and a.dtype==b.dtype
                    if a.dtype.kind in 'biu':assert np.array_equal(a,b)
                    else:
                        assert np.isfinite(a).all() and np.isfinite(b).all()
                        diffs.append(float(np.max(np.abs(a-b))) if a.size else 0.)
            result['float_parity_max_abs']=max(diffs,default=0.)
            result['numerical_parity']='EXACT' if result['float_parity_max_abs']==0 else 'UNDETERMINED_NONZERO_FLOAT_DIFFERENCE'
        check('eager_jit_integer_parity_float_measurement',parity)
        keys=jax.random.split(key,2)
        batch=check('jit_vmap_reset',lambda:jax.block_until_ready(jax.jit(jax.vmap(env.reset))(keys)))
        batched_actions=jax.tree.map(lambda a:jnp.stack([a,a]),actions)
        check('jit_vmap_step',lambda:jax.block_until_ready(jax.jit(jax.vmap(env.step_env))(keys,batch[1],batched_actions)))
        compiled_outer=jax.jit(env.step)
        check('outer_autoreset_api',lambda:jax.block_until_ready(compiled_outer(key,state,actions)))
        def episode():
            sample=jax.jit(env.sample_actions)
            current_obs,current=compiled_reset(key)
            score=np.zeros(env.num_agents,dtype=np.float64);frames=[];trace=[];done=False
            stride=max(1,env.max_steps//24)
            for t in range(env.max_steps):
                if t%stride==0:frames.append(scene_frame(env,current,current_obs,role=-1))
                act=sample(jax.random.fold_in(key,2*t+1),current)
                current_obs,current,rewards,dones,info=compiled_step(jax.random.fold_in(key,2*t+2),current,act)
                jax.block_until_ready(rewards)
                r=np.array([rewards[a] for a in env.agents],dtype=np.float64)
                assert np.isfinite(r).all()
                assert r.shape==(env.num_agents,)
                assert np.all(r>=env.manifest['reward_low']) and np.all(r<=env.manifest['reward_high'])
                assert set(dones)==set(env.agents)|{'__all__'}
                done=bool(dones['__all__']);score+=r
                trace.append(dict(t=t+1,actions=[int(act[a]) for a in env.agents],rewards=r.tolist(),done=done))
                if done:break
            assert done,'No joint terminal within original horizon'
            terminal_actions=sample(jax.random.fold_in(key,9998),current)
            if env.metadata['family']=='spatial':
                absorbed=compiled_step(jax.random.fold_in(key,9999),current,terminal_actions)
                for a,b in zip(jax.tree.leaves(current),jax.tree.leaves(absorbed[1])):
                    assert np.array_equal(np.asarray(a),np.asarray(b)),'Raw terminal state is not absorbing'
                assert all(float(v)==0 for v in absorbed[2].values())
                result['raw_terminal_absorbing']='PASS'
            else:result['raw_terminal_absorbing']='NOT_PART_OF_ORIGINAL_LEGACY_CONTRACT'
            terminal_key=jax.random.fold_in(key,10000)
            outer=compiled_outer(terminal_key,current,terminal_actions)
            jax.block_until_ready(outer)
            expected=compiled_reset(jax.random.split(terminal_key)[1])
            for a,b in zip(jax.tree.leaves(outer[1]),jax.tree.leaves(expected[1])):
                assert np.array_equal(np.asarray(a),np.asarray(b)),'Outer terminal reset differs from explicit reset key'
            result['terminal_outer_reset']='PASS'
            frames.append(scene_frame(env,current,current_obs,role=-1))
            # Role view exercises filtering independently from the debug demo.
            for role in range(env.num_agents):scene_frame(env,current,current_obs,role=role)
            result.update(episode_steps=len(trace),scripted_random_returns=score.tolist(),terminal=True,
                          demo_information='FULL_STATE_DEBUG',demo_frames=len(frames),render_roles_checked=env.num_agents,
                          human_scores=None)
            target=out/'traces';target.mkdir(exist_ok=True)
            import gzip
            with gzip.open(target/(env.name+'.json.gz'),'wt',encoding='utf-8') as f:json.dump(trace,f,separators=(',',':'))
            demos=out.parent/'docs/demos';demos.mkdir(parents=True,exist_ok=True)
            save_gif(frames,demos/(env.name+'.gif'))
        check('full_original_horizon_episode_and_render',episode)
        result['status']='PASS'
    except Exception:
        result['status']='FAIL'
    write();print(env.name,result['status'],round(result['elapsed_seconds'],2),flush=True)
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
