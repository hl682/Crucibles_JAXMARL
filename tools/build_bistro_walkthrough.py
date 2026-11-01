"""Illustrate a real cooperative order, preserving the original kitchen art."""
from pathlib import Path
import hashlib,json
import jax,jax.numpy as jnp
from crucibles_jaxmarl import make
from crucibles_jaxmarl.bistro_presentation import make_view,draw,changes
from verify_bistro_gameplay import ACTIONS

def main():
    R=Path(__file__).resolve().parents[1];env=make('gi215');key=jax.random.PRNGKey(215)
    obs,state=jax.jit(env.kernel.reset)(key);step=jax.jit(env.kernel.raw_step);images=[];events=[];view=make_view(env,state,obs)
    images.append(draw(env,view))
    for t,actions in enumerate(ACTIONS):
        old=state;previous=view;obs,state,reward,done,info=step(jax.random.fold_in(key,t+1),state,jnp.array(actions,jnp.int32));jax.block_until_ready(reward)
        view=make_view(env,state,obs);effects=changes(old,state);events.extend(dict(step=t+1,**e)for e in effects)
        for k in range(1,6):images.append(draw(env,view,previous,phase=k/5,events=effects))
    assert int(state['served'])==1
    out=R/'local-runs/bistro-playability';out.mkdir(parents=True,exist_ok=True);gif=out/'first-order-original-art.gif'
    images[0].save(gif,save_all=True,append_images=images[1:],duration=130,loop=0)
    images[-1].save(out/'first-order-final.png')
    receipt=dict(status='NATIVE_WALKTHROUGH_RENDERED',game='gi215',seed=215,native_transitions=len(ACTIONS),display_frames=len(images),source_sha256=env.metadata['source_sha256'],art='Original scene214 renderer, kitchen labels translated, instruction panel cropped',events=events,served=int(state['served']),gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest(),intermediate_frames='Display-only interpolation; never observations or native transitions',human_scores=None)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
