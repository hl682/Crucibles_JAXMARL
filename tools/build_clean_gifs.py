"""Re-execute frozen action traces through their original JAX kernel for presentation."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed

R=Path(__file__).resolve().parents[1]
def worker(key):
    import jax,jax.numpy as jnp,numpy as np
    from crucibles_jaxmarl import make
    from crucibles_jaxmarl.rendering import scene_frame
    from clean_presentation import save
    start=time.time();env=make(key);receipt=json.loads((R/'verification'/(key+'.json')).read_bytes())
    with gzip.open(R/'verification/traces'/(key+'.json.gz'),'rt',encoding='utf-8') as f:trace=json.load(f)
    seed=receipt['seed'];key0=jax.random.PRNGKey(seed);obs,state=jax.jit(env.kernel.reset)(key0);step=jax.jit(env.kernel.raw_step)
    selected=set(round(i*len(trace)/min(24,len(trace))) for i in range(min(24,len(trace))+1));frames=[];endpoints=[];maxdiff=0.
    def capture(t,obs,state):
        frame=scene_frame(env,state,{a:obs[i] for i,a in enumerate(env.agents)},-1)
        if env.metadata['family']=='legacy_array':frame['abstract_state']=jax.tree.map(lambda x:np.asarray(x).tolist(),state)
        frames.append(frame);endpoints.append(t)
    if 0 in selected:capture(0,obs,state)
    for t,row in enumerate(trace):
        obs,state,rewards,dones,info=step(jax.random.fold_in(key0,2*t+2),state,jnp.asarray(row['actions'],dtype=jnp.int32));jax.block_until_ready(rewards)
        maxdiff=max(maxdiff,float(np.max(np.abs(np.asarray(rewards)-np.asarray(row['rewards'],dtype=np.float32)))))
        assert bool(info['done_all'])==row['done']
        if t+1 in selected:capture(t+1,obs,state)
    cache=R/'.presentation-cache';cache.mkdir(exist_ok=True)
    with gzip.open(cache/(key+'.json.gz'),'wt',encoding='utf-8') as f:json.dump(frames,f,separators=(',',':'))
    out=R/'docs/gifs';out.mkdir(exist_ok=True);save(frames,out/(key+'.gif'))
    result=dict(game=key,status='PASS',source_sha256=env.metadata['source_sha256'],seed=seed,source_trace='verification/traces/'+key+'.json.gz',
        actual_native_steps=len(trace),captured_endpoints=endpoints,render_language='en',presentation='WORLD_ONLY_NO_INSTRUCTION_PANELS',
        information_condition='FULL_STATE_DEBUG',new_rewards_vs_previous_trace_max_abs=maxdiff,numerical_acceptance='NO_NEW_TOLERANCE_OR_SCIENTIFIC_ACCEPTANCE',
        gif_sha256=hashlib.sha256((out/(key+'.gif')).read_bytes()).hexdigest(),seconds=time.time()-start)
    target=R/'verification/presentation';target.mkdir(exist_ok=True);(target/(key+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(key,'PASS',round(result['seconds'],1),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.worker:return worker(a.worker)
    from crucibles_jaxmarl import registered_envs
    logs=R/'verification/logs';logs.mkdir(exist_ok=True)
    def run(key):
        done=R/'verification/presentation'/(key+'.json')
        if done.exists() and json.loads(done.read_bytes())['status']=='PASS':return key,'EXISTING'
        with (logs/('presentation-'+key+'.log')).open('w',encoding='utf-8') as log:
            proc=subprocess.run([sys.executable,str(Path(__file__)), '--worker',key],cwd=R,env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONIOENCODING='utf-8'),stdout=log,stderr=subprocess.STDOUT)
        return key,'PASS' if proc.returncode==0 else 'FAIL'
    failures=0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(run,key) for key in registered_envs]):
            key,status=f.result();print(key,status,flush=True);failures+=status=='FAIL'
    return bool(failures)

if __name__=='__main__':raise SystemExit(main())
