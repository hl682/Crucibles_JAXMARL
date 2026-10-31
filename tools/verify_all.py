"""Isolated bounded workers; every game produces its own real receipt."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,json,os,subprocess,sys,time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--timeout',type=int,default=900)
    ap.add_argument('--games',nargs='*');ap.add_argument('--retry',action='store_true');ap.add_argument('--shard');a=ap.parse_args()
    from crucibles_jaxmarl import registered_envs
    root=Path(__file__).resolve().parents[1];out=root/'verification';logs=out/'logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1')
    games=a.games or registered_envs
    if a.shard:
        index,total=map(int,a.shard.split('/'))
        if not 0<=index<total:ap.error('shard must be index/total, 0 <= index < total')
        games=games[index::total]
    def run(g):
        path=out/(g+'.json')
        if not a.retry and path.exists():
            prior=json.loads(path.read_bytes())
            if prior.get('status')=='PASS' and prior.get('verification_revision',0)>=2:return g,'ALREADY_PASS'
        t=time.time()
        try:
            with (logs/(g+'.log')).open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,str(root/'tools/verify_game.py'),g,'--output',str(out)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
            status='PASS' if result.returncode==0 else 'FAIL'
        except subprocess.TimeoutExpired:
            status='TIMEOUT'
            value=json.loads(path.read_bytes()) if path.exists() else dict(game=g)
            value.update(status=status,timeout_seconds=a.timeout,human_scores=None)
            path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        return g,status,round(time.time()-t,1)
    failures=0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(run,g) for g in games]):
            value=f.result();print(*value,flush=True)
            failures+=value[1] not in ['PASS','ALREADY_PASS']
    return bool(failures)

if __name__=='__main__':raise SystemExit(main())
