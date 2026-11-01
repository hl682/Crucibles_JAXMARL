"""Paired native dirty-water consequences; no state injection or rule changes."""
from pathlib import Path
import hashlib,json
import jax,jax.numpy as jnp
from crucibles_jaxmarl import make
from crucibles_jaxmarl.bistro_presentation import make_view,draw,changes

R=Path(__file__).resolve().parents[1]

def run(divert):
    env=make('gi215');key=jax.random.PRNGKey(215)
    obs,state=jax.jit(env.kernel.reset)(key);step=jax.jit(env.kernel.raw_step)
    actions=[[4,5],[10 if divert else 0,7],[0,9],[0,0],[0,0],[2,0],[2,0],[5,0],[1,0],[7,0]]
    views=[];frames=[];records=[];previous=make_view(env,state,obs)
    frames.append(draw(env,previous));durations=[500]
    for t,a in enumerate(actions):
        assert all(bool(env.kernel.legal_actions(state)[i,value]) for i,value in enumerate(a))
        before=state
        obs,state,reward,done,info=step(jax.random.fold_in(key,t+1),state,jnp.array(a,jnp.int32))
        jax.block_until_ready(reward);view=make_view(env,state,obs);events=changes(before,state)
        for phase in [.2,.4,.6,.8,1.]:
            frames.append(draw(env,view,previous,phase=phase,events=events));durations.append(120)
        records.append(dict(step=t+1,actions=a,flow=state['flow'].tolist(),sink_dirt=int(state['sink_dirt']),
            basin=int(state['basin']),pan_dirty=bool(state['pan_dirty']),prepared=state['prepared'].tolist(),
            rewards=reward.tolist(),events=events))
        previous=view
    durations[-1]=1800
    out=R/'docs/walkthroughs'/('gi215-water-diverted.gif'if divert else'gi215-water-unprotected.gif')
    out.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(out,save_all=True,append_images=frames[1:],duration=durations,loop=0)
    preview=R/'local-runs/bistro-water';preview.mkdir(parents=True,exist_ok=True)
    frames[-1].save(preview/('diverted.png'if divert else'unprotected.png'))
    return dict(actions=actions,records=records,source_sha256=env.metadata['source_sha256'],
        gif_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),native_steps=len(actions),
        final_pan_dirty=bool(state['pan_dirty']),final_prepared=int(jnp.sum(state['prepared'])),
        final_basin=int(state['basin']),final_contaminations=int(state['contaminations']))

def main():
    unprotected=run(False);protected=run(True)
    assert unprotected['source_sha256']==protected['source_sha256']
    differing=[i for i,(a,b)in enumerate(zip(unprotected['actions'],protected['actions']))if a!=b]
    assert differing==[1]
    assert unprotected['final_pan_dirty'] and unprotected['final_contaminations']==1 and unprotected['final_prepared']==0
    assert not protected['final_pan_dirty'] and protected['final_contaminations']==0 and protected['final_prepared']==1
    assert protected['final_basin']==1
    result=dict(status='PASS',game='gi215',seed=215,only_changed_action_step=2,
        unprotected=unprotected,diverted=protected,human_scores=None,
        scope='Two 10-step native action sequences; engineering causal walkthrough, not learning or human performance')
    out=R/'verification/gameplay/gi215_water_pair.json';out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items()if k not in ['unprotected','diverted']}),flush=True)

if __name__=='__main__':main()
