"""A source-identical first-order walkthrough and a one-action control.

This is an engineering demonstration, never a human score or learning result.
Both branches start from the original reset, without injecting any resources.
"""
from pathlib import Path
import json,hashlib
import jax
import jax.numpy as jnp
import numpy as np
from crucibles_jaxmarl import make

R=Path(__file__).resolve().parents[1]
ACTIONS=[[2,5],[1,7],[5,4],[7,1],[6,1],[1,2],[0,2],[0,2],[0,2],[0,6],[0,4],[4,0],[5,0],[12,0],[2,0],[3,0],[3,0],[4,0],[13,0]]
FIELDS=['agent_pos','carrying','holder','clean','prepared','filled','tank','sink_dirt','pan_dirty','served','order_status']

def run(env,sequence):
    key=jax.random.PRNGKey(215);obs,state=jax.jit(env.kernel.reset)(key);step=jax.jit(env.kernel.raw_step);rows=[]
    for t,actions in enumerate(sequence):
        legal=np.asarray(env.kernel.legal_actions(state));assert all(legal[i,a] for i,a in enumerate(actions)),(t,actions)
        before={k:np.asarray(state[k]).tolist()for k in FIELDS}
        obs,state,reward,done,info=step(jax.random.fold_in(key,t+1),state,jnp.array(actions,jnp.int32));jax.block_until_ready(reward)
        after={k:np.asarray(state[k]).tolist()for k in FIELDS}
        rows.append(dict(step=t+1,actions=actions,before=before,after=after,reward=np.asarray(reward).tolist()))
    return rows

def main():
    env=make('gi215');success=run(env,ACTIONS);control=[a.copy()for a in ACTIONS];control[1][1]=0;without_wash=run(env,control)
    assert success[-1]['after']['served']==1 and without_wash[-1]['after']['served']==0
    assert success[13]['after']['filled'][1] and not without_wash[13]['after']['filled'][1]
    out=R/'verification/gameplay';out.mkdir(exist_ok=True)
    result=dict(status='PASS',game='gi215',source_sha256=env.metadata['source_sha256'],seed=215,native_reset=True,state_injection=False,steps=19,all_commands_legal=True,successful_orders=1,control_successful_orders=0,single_changed_input=dict(step=2,role=1,from_action=7,to_action=0),result='Cleaner washing the persistent bowl enables the cook to assemble and serve the first order.',success=success,control=without_wash,human_scores=None,learning_claim=False,all_games_playable_claim=False)
    path=out/'gi215_first_order.json';path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:result[k]for k in ['status','game','steps','successful_orders','control_successful_orders','source_sha256']}),flush=True)

if __name__=='__main__':main()
