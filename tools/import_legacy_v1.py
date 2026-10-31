"""Preserve the four earlier, distinct original window-v1 games without GI renumbering."""
from pathlib import Path
import ast,hashlib,json,shutil

R=Path(__file__).resolve().parents[1];S=R/'src/crucibles_jaxmarl';W=R.parents[2]
TITLES=['Delayed pledges with endogenous credibility','Renewing commons with delayed runoff','Manufacturing and persistent custom offers','Private revelation and directional mode pressure']
LABELS=[['Wait','Open promise','Work on promise','Endorse partner','Withdraw promise'],
        ['Rest','Gentle draw','Deep draw','Restore soil','Install filter'],
        ['Wait','Manufacture','Post offer','Accept partner offer','Order custom good'],
        ['Wait','Privately sense mode','Start mode-0 project','Start mode-1 project','Push mode toward own index']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    catalog=json.loads((S/'catalog.json').read_bytes());existing={r['key'] for r in catalog['games']}
    for i in range(4):
        key=f'v1_{i:03d}'
        if key in existing:continue
        source=R.parent/f'astra_generation/window_v1/candidate_{i:05d}/attempt_00';dest=S/'games'/key;dest.mkdir(exist_ok=True)
        files=[]
        for name in ['candidate.py','mechanism.md']:
            p=source/name;shutil.copyfile(p,dest/name);files.append(dict(file=name,bytes=p.stat().st_size,sha256=sha(p),origin=p.relative_to(W).as_posix()))
        code=(dest/'candidate.py').read_text(encoding='utf-8-sig');tree=ast.parse(code);attrs={}
        for node in ast.walk(tree):
            if isinstance(node,ast.ClassDef) and node.name=='GeneratedEnv':
                for child in node.body:
                    if isinstance(child,ast.Assign):
                        try:attrs[child.targets[0].id]=ast.literal_eval(child.value)
                        except (ValueError,AttributeError):pass
        observe=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='observe')
        transition=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='raw_step')
        m=dict(schema='legacy-array-release-metadata-v1',candidate_id=key,source_sha256=sha(dest/'candidate.py'),
               **{k:attrs[k] for k in ['num_agents','observation_size','num_actions','reward_low','reward_high']},max_steps=32,
               observation_low=None,observation_high=None,observability='source_defined',
               agents=[dict(id=f'agent_{a}',role='Original window-v1 agent; abilities and private parameters follow the original mechanism and observe()') for a in range(attrs['num_agents'])],
               actions=[dict(id=a,label=label) for a,label in enumerate(LABELS[i])],
               terminal_semantics='legacy_horizon_no_absorbing_contract',
               observation_fields=[dict(slice=[0,attrs['observation_size']],visibility='exact original observe() below',source_code=ast.get_source_segment(code,observe))],
               task=dict(objective=TITLES[i]+'; accumulate original per-agent rewards under mechanism.md.',natural_termination='Original t >= 32 horizon.',exact_transition_and_reward_source=ast.get_source_segment(code,transition)),
               metadata_origin='release adapter metadata, not an original author manifest',original_namespace='astra_generation/window_v1',original_proposal_index=i)
        dump(dest/'manifest.json',m)
        row=dict(id=key,key=key,title='Window v1 · '+TITLES[i],summary=(dest/'mechanism.md').read_text(encoding='utf-8-sig').split('. ')[0]+'.',tags=['historical window-v1'],family='legacy_array',
                 dimensions={k:m[k] for k in ['num_agents','observation_size','num_actions','max_steps']},source_sha256=sha(dest/'candidate.py'),manifest_sha256=sha(dest/'manifest.json'),
                 files=files,original_status='EARLIER_WINDOW_V1_SOURCE',original_namespace='astra_generation/window_v1',original_proposal_index=i,
                 original_saved_replay=False,release_validation='PENDING',human_scores=None)
        assert row['source_sha256'] not in {r['source_sha256'] for r in catalog['games']}
        dump(dest/'provenance.json',row);catalog['games'].append(row)
    dump(S/'catalog.json',catalog);print('Registered identities',len(catalog['games']))

if __name__=='__main__':main()
