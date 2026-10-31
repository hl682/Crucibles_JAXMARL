"""Real terminal benchmarks and capability boundaries, not human scores."""
from pathlib import Path
import hashlib,json,time

R=Path(__file__).resolve().parents[1]
def main():
    from crucibles_jaxmarl.server import Session
    start=time.time();records=[]
    for mode in ['AI_ONLY','HUMAN_INPUT','MIXED']:
        s=Session(0,0,R/'local-runs/session-contract',engineering=True)
        guest=s.command(s.host,'invite',dict(name='Guest'))['invite']
        try:s.command(guest,'assign',dict(player='Guest',roles=[0]))
        except PermissionError:pass
        else:raise AssertionError('Guest obtained Host assignment authority')
        try:s.command(guest,'actions',dict(actions={'0':0}))
        except PermissionError:pass
        else:raise AssertionError('Guest submitted another role action')
        try:s.snapshot(guest,True)
        except PermissionError:pass
        else:raise AssertionError('Guest read full-state debug data')
        assert s.snapshot(guest)['views']=={}
        if mode!='AI_ONLY':s.command(s.host,'assign',dict(player='Host',roles=[0,1]))
        if mode=='MIXED':s.snapshot(s.host,True)
        for t in range(32):
            if mode=='MIXED' and t==5:s.command(s.host,'assign',dict(player='AI',roles=[1]))
            if mode=='MIXED' and t==15:s.command(s.host,'assign',dict(player='Host',roles=[1]))
            roles=s.clients[s.host]['roles']
            if roles:s.command(s.host,'actions',dict(actions={str(r):0 for r in roles}))
            s.command(s.host,'step',{})
        assert s.terminal
        benchmark=json.loads((s.logdir/'benchmark.json').read_bytes())
        assert benchmark['mode']==mode and benchmark['human_scores'] is None and benchmark['engineering'] is True
        assert benchmark['information']==('DEBUG_EXPOSED' if mode=='MIXED' else 'ROLE_FILTERED')
        raw=(s.logdir/'events.jsonl').read_text(encoding='utf-8')
        assert s.host not in raw and guest not in raw
        previous='0'*64
        for line in raw.splitlines():
            row=json.loads(line);digest=row.pop('sha256');assert row['previous']==previous
            previous=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();assert digest==previous
        assert benchmark['log_chain']==previous
        records.append(dict(mode=mode,steps=32,terminal=True,hash_chain='PASS',capability_boundaries='PASS',human_scores=None,information=benchmark['information']))
    result=dict(status='PASS',engineering_only=True,cases=records,seconds=time.time()-start)
    (R/'verification/session_contract.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
