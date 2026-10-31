"""One-time local importer. Release runtime never depends on this workspace."""
from pathlib import Path
import ast
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
W = ROOT.parents[2]
G = ROOT.parent / 'game_incubator_v3'
L = G / 'game_library_v2'
N = G / 'community_games222_225_visualization_root222_v1/native154_candidate'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main():
    native = {r['id']: r for r in json.loads((N/'registry/v2/registry154_candidate.json').read_bytes())['games']}
    catalog = {r['id']: r for r in json.loads((L/'catalog.json').read_bytes())['games']}
    rows = []
    for i in range(252):
        if i < 72:
            source = W / f'work/exp_2/astra_generation/discovery_v2/initial/candidate_{i:05d}/attempt_00'
        elif i in native:
            source = (N / native[i]['source_path']).parent
        else:
            attempts = sorted((G/f'sources/game-v3-{i:05d}').glob('attempt_*'))
            source = attempts[-1] if attempts else None
        if source is None or not (source/'candidate.py').is_file():
            rows.append(dict(id=i, status='MISSING_SOURCE'))
            continue
        dest = ROOT / f'src/crucibles_jaxmarl/games/gi{i:03d}'
        dest.mkdir(parents=True, exist_ok=True)
        records = []
        for name in ['candidate.py','manifest.json','scene.json','mechanism.md','playthrough.json']:
            p = source/name
            if not p.is_file() and i in native:
                p = G/f'sources/game-v3-{i:05d}/attempt_00'/name
            if not p.is_file():
                continue
            shutil.copyfile(p, dest/name)
            records.append(dict(file=name, sha256=sha(p), bytes=p.stat().st_size, origin=p.relative_to(W).as_posix()))
        code = (dest/'candidate.py').read_text(encoding='utf-8-sig')
        tree = ast.parse(code)
        attrs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'GeneratedEnv':
                for child in node.body:
                    if isinstance(child, ast.Assign) and isinstance(child.value, (ast.Constant, ast.UnaryOp)):
                        try: attrs[child.targets[0].id] = ast.literal_eval(child.value)
                        except (ValueError,AttributeError): pass
        mechanism = (dest/'mechanism.md').read_text(encoding='utf-8-sig') if (dest/'mechanism.md').exists() else ''
        c = catalog.get(i,{})
        title = c.get('chinese_name') or c.get('title') or next((s.lstrip('# ').strip() for s in mechanism.splitlines() if s.startswith('#')), f'GI{i:03d}')
        if not (dest/'manifest.json').exists():
            # Original early family has a fixed 32-step kernel. No invented geometry.
            assert i < 72
            m = dict(schema='legacy-array-release-metadata-v1', candidate_id=f'gi{i:03d}', source_sha256=sha(dest/'candidate.py'),
                     **{k:attrs[k] for k in ['num_agents','observation_size','num_actions','reward_low','reward_high']},
                     max_steps=32, observation_low=-float('inf'), observation_high=float('inf'),
                     agents=[dict(id=f'agent_{a}',role='Original legacy agent; see mechanism and source') for a in range(attrs['num_agents'])],
                     actions=[dict(id=a,label=f'Action {a}; see mechanism and source') for a in range(attrs['num_actions'])],
                     terminal_semantics='legacy_horizon_no_absorbing_contract', observability='source_defined',
                     metadata_origin='release adapter, not original author manifest')
            # JSON has no Infinity. Box uses None for unbounded endpoints.
            m['observation_low']=None; m['observation_high']=None
            dump(dest/'manifest.json',m)
        m = json.loads((dest/'manifest.json').read_bytes())
        assert m['source_sha256'] == sha(dest/'candidate.py'), (i,'source binding')
        row = dict(id=i,key=f'gi{i:03d}',title=title,summary=c.get('summary',''),tags=c.get('tags',[]),
                   family='legacy_array' if i<72 else 'spatial',
                   dimensions={k:m[k] for k in ['num_agents','observation_size','num_actions','max_steps']},
                   source_sha256=sha(dest/'candidate.py'),manifest_sha256=sha(dest/'manifest.json'),
                   files=records, original_status='LEGACY_SOURCE' if i<72 else ('NATIVE154_REFERENCE' if i<=225 else 'SOURCE_ONLY'),
                   original_saved_replay=i in catalog, release_validation='PENDING',human_scores=None)
        dump(dest/'provenance.json',row)
        rows.append(row)
    dump(ROOT/'src/crucibles_jaxmarl/catalog.json',dict(schema='crucibles-catalog-v1',games=rows))
    dump(ROOT/'verification/import.json',dict(games=len(rows),source_count=sum('files' in r for r in rows),
         preserved_originals=True,research_results_changed=False,source_records=rows))
    print('Imported',len(rows),'identities;',sum('files' in r for r in rows),'sources')

if __name__=='__main__': main()
