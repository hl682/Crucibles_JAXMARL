import argparse
import json

def main():
    p=argparse.ArgumentParser(prog='crucibles');sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    play=sub.add_parser('play');play.add_argument('--game',default='225');play.add_argument('--port',type=int,default=8765)
    play.add_argument('--bind',default='127.0.0.1',help='Bind a local/LAN interface; use 0.0.0.0 for LAN multiplayer')
    play.add_argument('--runs',default='local-runs');play.add_argument('--engineering',action='store_true')
    smoke=sub.add_parser('smoke');smoke.add_argument('--game',default='072')
    a=p.parse_args()
    if a.command=='list':
        from .registry import _rows
        for r in _rows:print(f"{r['key']}  A{r['dimensions']['num_agents']} H{r['dimensions']['max_steps']}  {r['title']}")
    elif a.command=='play':
        if not 1024<=a.port<=65535:p.error('port must be in 1024..65535')
        from .registry import game_info
        try:selected=game_info(a.game)['id']
        except ValueError as e:p.error(str(e))
        from .server import serve
        try:serve(a.port,selected,a.runs,a.engineering,bind=a.bind)
        except OSError as e:p.exit(1,f'Unable to listen on {a.bind}:{a.port}: {e}\nUse --port 8766 if another service owns 8765.\n')
    else:
        import jax
        from . import make
        e=make(a.game);key=jax.random.PRNGKey(0);obs,s=jax.jit(e.reset)(key)
        actions=e.sample_actions(jax.random.fold_in(key,1),s)
        obs,s,r,d,i=jax.jit(e.step_env)(jax.random.fold_in(key,2),s,actions)
        jax.block_until_ready(r)
        print(json.dumps(dict(game=e.name,agents=e.num_agents,step=1,rewards={k:float(v) for k,v in r.items()},done=bool(d['__all__']))))

if __name__=='__main__':main()
