"""Native GI217 causal demonstration. No state/resources injection or training."""
from pathlib import Path
import copy
import hashlib
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from crucibles_jaxmarl import make
from crucibles_jaxmarl import orchard_presentation as presentation
from crucibles_jaxmarl import magnetic_art as original_art

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'local-runs/gi217'
FIELDS=('t','agent_pos','trait','growth','moisture','seedbank','water','reservoir',
        'wet','dry','dry_timer','basket_slot','carrying','seed_pos','seed_active')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def snapshot(state): return {k:np.asarray(state[k]).tolist() for k in FIELDS}
def state_hash(state):
    h=hashlib.sha256()
    for k in sorted(state):
        a=np.asarray(state[k]);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    start=time.perf_counter();env=make('gi217');OUT.mkdir(parents=True,exist_ok=True)
    originals=[ROOT/'src/crucibles_jaxmarl/games/gi217'/f for f in ('candidate.py','manifest.json','scene.json')]
    originals += [ROOT/'docs'/d/'gi217.gif' for d in ('gifs','demos','previews')]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in originals}
    key=jax.random.PRNGKey(217);obs,state=jax.jit(env.kernel.reset)(key);step=jax.jit(env.kernel.raw_step)
    states=[state];observations=[obs];rows=[];pairs=[];checks={}
    # Agent2 takes its genuine basket to the shared shed, dries it, retrieves
    # it and walks to an initially empty bed to plant. Agent7 draws well water.
    actions=[]
    for tick in range(1,23):
        a=[0]*8
        if tick==1: a[0]=5;a[1]=12;a[2]=5;a[7]=1
        if tick==2: a[0]=7;a[7]=10
        if tick==3: a[7]=3
        if tick==4: a[7]=11
        if 2<=tick<=5:a[2]=3
        if tick==6:a[2]=9
        if tick==10:a[2]=5
        if tick==11:a[2]=2
        if 12<=tick<=19:a[2]=3
        if tick==20:a[2]=4
        if tick==21:a[2]=8
        actions.append(a)
    forks={1:[(1,'prune')],2:[(0,'harvest'),(7,'well')],4:[(7,'water')],6:[(2,'drying_slot')],21:[(2,'plant')]}
    def transition(s,a,tick):
        legal=np.asarray(env.legal_actions(s));assert all(legal[i,x] for i,x in enumerate(a)),(tick,a,np.asarray(s['agent_pos']).tolist())
        out=step(jax.random.fold_in(key,tick),s,jnp.asarray(a,jnp.int32));jax.block_until_ready(out[2]);return out
    for tick,a in enumerate(actions,1):
        before=state;obs,state,reward,done,info=transition(before,a,tick)
        rows.append(dict(tick=tick,actions=a,before=snapshot(before),after=snapshot(state),reward=np.asarray(reward).tolist(),state_hash=state_hash(state)))
        for role,name in forks.get(tick,[]):
            control_a=list(a);control_a[role]=0
            _,control,_,_,_=transition(before,control_a,tick)
            pairs.append(dict(mechanism=name,tick=tick,role=role,action=a[role],control_action=0,common_before_hash=state_hash(before),treatment=snapshot(state),control=snapshot(control)))
        states.append(state);observations.append(obs)
    byname={p['mechanism']:p for p in pairs}
    t,c=(byname['harvest'][k] for k in ('treatment','control'))
    checks['harvest_resets_growth_collects_seeds']=t['growth'][0]<c['growth'][0] and sum(t['wet'][0])>sum(c['wet'][0])
    t,c=(byname['well'][k] for k in ('treatment','control'))
    checks['well_shared_reservoir_and_carried_water']=t['reservoir']==c['reservoir']-3 and t['water'][7]==c['water'][7]+3
    t,c=(byname['water'][k] for k in ('treatment','control'))
    checks['watering_changes_soil_and_inventory']=t['moisture'][15]>c['moisture'][15] and t['water'][7]==c['water'][7]-1
    t,c=(byname['drying_slot'][k] for k in ('treatment','control'))
    checks['basket_placed_on_shared_drying_slot']=t['basket_slot'][2]>=0 and c['basket_slot'][2]==-1
    checks['native_drying_converts_wet_seeds']=sum(np.asarray(states[9]['dry'])[2])==2 and sum(np.asarray(states[9]['wet'])[2])==0
    t,c=(byname['plant'][k] for k in ('treatment','control'))
    checks['plant_consumes_dry_seed_and_changes_empty_bed']=t['trait'][20]==0 and c['trait'][20]==-1 and sum(t['dry'][2])==sum(c['dry'][2])-1
    checks['native_growth_changes']=any(np.any(np.asarray(s['growth'])!=np.asarray(states[i-1]['growth'])) for i,s in enumerate(states[1:],1))
    checks['native_seed_drift']=any(np.any(np.asarray(states[i]['seed_active']) & np.asarray(states[i-1]['seed_active']) & np.any(np.asarray(states[i]['seed_pos'])!=np.asarray(states[i-1]['seed_pos']),axis=1)) for i in range(1,len(states)))
    # Independently compare every displayed record against its native row slice.
    specs={'gardeners':(608,8,10,['role','facing','wood','carrying','carry_break','water','selected']),
           'plots':(688,24,11,['bed_owner','trait','growth','canopy','moisture','fertility','seedbank','shade']),
           'baskets':(952,8,12,['basket_holder','basket_slot','wet','dry','dry_timer']),
           'windbreaks':(1048,8,6,['break_holder','break_axis','deployed']),
           'drifting_seeds':(1096,16,6,['seed_trait','seed_owner','seed_age'])}
    privacy=True
    for s,o in zip(states,observations):
        native=np.asarray(o)
        for role in range(8):
            v=presentation.make_view(env,s,o,role)
            privacy &= 'own_scene_fields' not in v and 'achieved' not in json.dumps(v)
            for group,(offset,count,width,keys) in specs.items():
                records=native[role,offset:offset+count*width].reshape(count,width)
                actual={e['index']:e for e in v['entities'] if e['group']==group}
                privacy &= set(actual)==set(np.flatnonzero(records[:,0]))
                for i,e in actual.items():
                    attrs=e['attributes'];flat=[]
                    for k in keys:flat.extend(np.asarray(attrs[k]).reshape(-1).tolist())
                    privacy &= np.array_equal(records[i,1:],np.array(e['xy']+flat))
    checks['all_roles_all_steps_match_native_observation_records']=bool(privacy)
    v=presentation.make_view(env,state,obs,2);h=state_hash(state)
    anim=presentation.animation(env,state,obs,2,states[-2],observations[-2])
    checks['animation_contract']=len(anim['frames'])==4 and anim['width']==730 and anim['height']==586
    checks['render_no_state_side_effects']=h==state_hash(state)
    checks['original_geometry_background']=original_art.geometry(v)==presentation.art.geometry(v) and original_art.background(217,json.dumps(v['public_map'])).tobytes()==presentation.art.background(217,json.dumps(v['public_map'])).tobytes()
    wrong=copy.copy(env);wrong.metadata=dict(env.metadata,source_sha256='0'*64)
    for name,fn in [('source_gate',lambda:presentation.make_view(wrong,state,obs,2)),('draw_source_gate',lambda:presentation.draw(wrong,v)),('mismatched_obs_rejected',lambda:presentation.make_view(env,state,observations[0],2))]:
        try:fn();checks[name]=False
        except ValueError:checks[name]=True
    frames=[];events=[]
    for i,(s,o) in enumerate(zip(states,observations)):
        view=presentation.make_view(env,s,o,2);old=presentation.make_view(env,states[i-1],observations[i-1],2) if i else None
        changes=presentation.changes(old,view) if old else [];events.append(dict(tick=i,events=changes))
        frames.extend(presentation.draw(env,view,old,p,changes) for p in ([.5,1.] if old else [1.]))
        if i in (0,6,9,21,22):presentation.draw(env,view,full_frame=True).save(OUT/f'review_role2_tick{i}.png')
    debug=presentation.make_view(env,state,obs,-1)
    checks['debug_explicit']=debug['view']=='PRIVILEGED_ACTUAL_STATE'
    presentation.draw(env,debug,full_frame=True).save(OUT/'review_DEBUG_tick22.png')
    frames[0].save(OUT/'native_orchard_walkthrough.gif',save_all=True,append_images=frames[1:],duration=200,loop=0,optimize=False)
    checks['original_candidate_manifest_scene_and_old_gifs_unchanged']=hashes=={str(p.relative_to(ROOT)):digest(p) for p in originals}
    checks={k:bool(v) for k,v in checks.items()}
    report=dict(status='PASS' if all(checks.values()) else 'FAIL',game='gi217',source_sha256=presentation.SOURCE,seed=217,native_reset=True,state_injection=False,all_commands_legal=True,main_steps=22,paired_single_action_forks=len(pairs),native_joint_steps=22+len(pairs),assertions=checks,sequence=rows,causal_pairs=pairs,role2_visible_events=events,original_hashes=hashes,backend=jax.default_backend(),jax_version=jax.__version__,elapsed_seconds=time.perf_counter()-start,scope='Short native causal engineering demonstration; no training or scientific efficacy claim',outputs=[str(p.relative_to(ROOT)) for p in OUT.glob('*')])
    target=ROOT/'verification/gameplay/gi217_causal_gameplay.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ('status','assertions','elapsed_seconds','backend')}),flush=True)
    assert all(checks.values()),checks

if __name__=='__main__':main()
