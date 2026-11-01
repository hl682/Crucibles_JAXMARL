"""GI216 native causal walkthrough, without injected state or resources.

All ordinary artwork is decoded from a source-pinned 3001-value local row.
Single-action forks share the exact preceding real state and transition key.
"""
from pathlib import Path
import copy
import hashlib
import json
import time
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from crucibles_jaxmarl import make
from crucibles_jaxmarl import flood_presentation as p
from crucibles_jaxmarl import magnetic_art as original_art

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'local-runs/gi216'
REPORT=ROOT/'verification/gameplay/gi216_causal_gameplay.json'
ACTIONS=[[9,10,9,10],[4,2,0,0],[3,3,0,0],[5,7,0,0],[6,0,0,0],
         [1,4,0,0],[2,4,0,0],[11,4,0,0],[0,1,0,0],[22,21,0,0],
         [0,21,0,0],[0,20,0,0],[23,0,0,0],[12,0,0,0],[12,0,0,0]]
FORKS={1:[(0,'levee'),(1,'road')],8:[(0,'culvert')],10:[(0,'inspect')],
       12:[(1,'resource_share')],13:[(0,'report_share')],14:[(0,'gate_close')],15:[(0,'gate_open')]}
FIELDS=('t','agent_pos','backpack','stores','structure','gate_open','depth',
        'water_pos','water_active','report','report_tick')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def snap(s):return {k:np.asarray(s[k]).tolist() for k in FIELDS}
def state_hash(s):
    h=hashlib.sha256()
    for k in sorted(s):
        a=np.asarray(s[k]);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);REPORT.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='FAIL',game='gi216',source_sha256=p.SOURCE,state_injection=False,
                resource_injection=False,training=False,learning_claim=False,human_scores=None)
    assertions={};rows=[];pairs=[]
    originals=[ROOT/'src/crucibles_jaxmarl/games/gi216'/f for f in ('candidate.py','manifest.json','scene.json')]
    hashes={str(f.relative_to(ROOT)).replace('\\','/'):digest(f) for f in originals}
    try:
        env=make('gi216');key=jax.random.PRNGKey(216)
        obs,state=jax.jit(env.kernel.reset)(key);step=jax.jit(env.kernel.raw_step)
        states=[state];observations=[obs]
        def transition(s,cmd,tick):
            legal=np.asarray(env.legal_actions(s))
            if not all(legal[i,a] for i,a in enumerate(cmd)):
                raise AssertionError(('illegal command',tick,cmd,legal.tolist()))
            r=step(jax.random.fold_in(key,tick),s,jnp.asarray(cmd,jnp.int32));jax.block_until_ready(r[2]);return r
        privacy=True
        def check_privacy(s,o):
            for role in range(4):
                v=p.make_view(env,s,o,role);r=np.asarray(o)[role];g=r[41:2841].reshape(20,20,7)
                for y in range(20):
                    for x in range(20):
                        if not g[y,x,0]:
                            assert v['role_map']['values'][y][x] is None
                            assert all(v['visible_grid_fields'][k][y][x] is None for k in p.GRID_FIELDS)
                        else:
                            assert all(v['visible_grid_fields'][k][y][x]==int(g[y,x,j+1]) for j,k in enumerate(p.GRID_FIELDS))
                for group,records in [('envoys',r[2841:2873].reshape(4,8)),('settlements',r[2873:2905].reshape(4,8)),('floodwater',r[2905:3001].reshape(32,3))]:
                    assert {e['index'] for e in v['entities'] if e['group']==group}=={i for i,a in enumerate(records) if a[0]}
                own=v['own_scene_fields']
                assert own['stores']==r[23:26].astype(int).tolist()
                assert own['report']==r[27:31].astype(int).tolist()
                assert own['report_tick']==r[31:35].astype(int).tolist()
        check_privacy(state,obs)
        for tick,cmd in enumerate(ACTIONS,1):
            before=state;prior_obs=obs
            obs,state,reward,done,info=transition(before,cmd,tick)
            prior=p.make_view(env,before,prior_obs,0);view=p.make_view(env,state,obs,0)
            events=p.changes(prior,view)
            rows.append(dict(tick=tick,actions=cmd,before_hash=state_hash(before),after_hash=state_hash(state),
                             before=snap(before),after=snap(state),reward=np.asarray(reward).tolist(),events=events))
            for role,name in FORKS.get(tick,[]):
                alternate=list(cmd);alternate[role]=0
                _,control,_,_,_=transition(before,alternate,tick)
                pairs.append(dict(mechanism=name,tick=tick,role=role,action=cmd[role],control_action=0,
                                  common_before_hash=state_hash(before),treatment_after=snap(state),control_after=snap(control)))
            check_privacy(state,obs);states.append(state);observations.append(obs)
        named={a['mechanism']:a for a in pairs}
        def pair(name):return named[name]['treatment_after'],named[name]['control_after']
        t,c=pair('levee');assertions['levee_consumes_real_stone']=t['structure'][5][9]==1 and c['structure'][5][9]==0 and t['backpack'][0][0]==c['backpack'][0][0]-2
        t,c=pair('road');assertions['road_consumes_real_timber']=t['structure'][5][10]==2 and c['structure'][5][10]==0 and t['backpack'][1][1]==c['backpack'][1][1]-2
        t,c=pair('culvert');assertions['culvert_converts_real_levee']=t['structure'][5][9]==3 and c['structure'][5][9]==1 and t['gate_open'][5][9] and t['backpack'][0][:2]==[v-1 for v in c['backpack'][0][:2]]
        t,c=pair('inspect');assertions['inspection_records_real_dated_memory']=t['report_tick'][0][0]==9 and c['report_tick'][0][0]==-1
        t,c=pair('resource_share');assertions['share_moves_actual_grain']=t['backpack'][0][2]==c['backpack'][0][2]+1 and t['backpack'][1][2]==c['backpack'][1][2]-1
        t,c=pair('report_share');assertions['report_share_preserves_observation_date']=t['report_tick'][1][0]==9 and c['report_tick'][1][0]==-1 and t['report'][1][0]==t['report'][0][0]
        t,c=pair('gate_close');assertions['gate_closes_under_single_action_control']=not t['gate_open'][5][9] and c['gate_open'][5][9]
        t,c=pair('gate_open');assertions['gate_opens_under_single_action_control']=t['gate_open'][5][9] and not c['gate_open'][5][9]
        assertions['local_grid_entities_stores_reports_match_exact_observation']=True
        assertions['ordinary_view_hides_remote_cells']=any(v is None for row in p.make_view(env,state,obs,0)['role_map']['values'] for v in row)
        assertions['native_flood_visible_and_moving']=any(any(e['kind']=='water' for e in row['events']) for row in rows)
        assertions['debug_explicitly_labeled']=p.make_view(env,state,obs,-1)['view']=='PRIVILEGED_ACTUAL_STATE'
        before=state_hash(state);v=p.make_view(env,state,obs,0)
        anim=p.animation(env,state,obs,0,states[-2],observations[-2]);p.draw(env,v,full_frame=True)
        assertions['render_no_state_side_effects']=before==state_hash(state)
        assertions['animation_contract']=len(anim['frames'])==4 and anim['width']==730 and anim['height']==586
        for target in ('metadata','manifest','scene'):
            wrong=copy.copy(env);setattr(wrong,target,dict(getattr(env,target),source_sha256='0'*64))
            for op,label in [(lambda:p.make_view(wrong,state,obs,0),'view'),(lambda:p.draw(wrong,v),'draw')]:
                try:op();assertions[target+'_'+label+'_source_gate']=False
                except ValueError:assertions[target+'_'+label+'_source_gate']=True
        wrong=copy.copy(env);wrong.manifest=dict(env.manifest,observability='fully_observable')
        try:p.make_view(wrong,state,obs,0);assertions['observability_mismatch_rejected']=False
        except ValueError:assertions['observability_mismatch_rejected']=True
        try:p.make_view(env,state,observations[0],0);assertions['mismatched_observation_rejected']=False
        except ValueError:assertions['mismatched_observation_rejected']=True
        assertions['original_background_geometry_preserved']=(original_art.geometry(v)==p.art.geometry(v) and original_art.background(216,json.dumps(p.art.map_values(v))).tobytes()==p.art.background(216,json.dumps(p.art.map_values(v))).tobytes())
        frames=[p.draw(env,p.make_view(env,states[0],observations[0],0))]
        for i in range(1,len(states)):
            a=p.make_view(env,states[i-1],observations[i-1],0);b=p.make_view(env,states[i],observations[i],0);events=p.changes(a,b)
            frames.extend(p.draw(env,b,a,phase,events) for phase in [.25,.5,.75,1.])
        frames[0].save(OUT/'causal_walkthrough.gif',save_all=True,append_images=frames[1:],duration=180,loop=0,optimize=False)
        for i in [0,1,8,12,15]:p.draw(env,p.make_view(env,states[i],observations[i],0),full_frame=True).save(OUT/f'role0_state_{i}.png')
        p.draw(env,p.make_view(env,state,obs,-1),full_frame=True).save(OUT/'debug_full_state_15.png')
        assertions['native_candidate_manifest_scene_unchanged']=hashes=={str(f.relative_to(ROOT)).replace('\\','/'):digest(f) for f in originals}
        result.update(status='PASS' if all(assertions.values()) else 'FAIL',seed=216,native_reset=True,
                      main_steps=len(rows),paired_single_action_forks=len(pairs),native_joint_steps=len(rows)+len(pairs),
                      scope='Fifteen real native ticks and eight paired one-command forks. Local artwork/privacy engineering checks; not full-game coverage or scientific efficacy.',
                      backend=jax.default_backend(),jax_version=jax.__version__)
    except Exception:
        result['failure']=traceback.format_exc()
    result.update(assertions=assertions,original_hashes=hashes,sequence=rows,causal_pairs=pairs,
                  elapsed_seconds=time.perf_counter()-start,
                  outputs=[dict(path=str(f.relative_to(ROOT)).replace('\\','/'),bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(OUT.glob('*')) if f.is_file()])
    REPORT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sequence','causal_pairs','outputs')}),flush=True)
    if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
