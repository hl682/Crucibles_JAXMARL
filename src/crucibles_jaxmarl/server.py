"""Local/LAN multiplayer server. Every command reaches the original JAX kernel."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import hashlib,json,os,secrets,threading,time,uuid

WEB=Path(__file__).parent/'web'
from . import __version__

class Session:
    def __init__(self,game,seed,runs,engineering=False):
        import jax
        import numpy as np
        from .registry import make
        self.env=make(game);self.id=uuid.uuid4().hex;self.seed=seed;self.key=jax.random.PRNGKey(seed)
        self.lock=threading.RLock();self.clients={};self.host=secrets.token_urlsafe(24)
        self.clients[self.host]=dict(name='Host',host=True,roles=set(),pending={})
        self.owners={};self.step_count=0;self.terminal=False;self.engineering=engineering
        self.ever_human=False;self.ever_ai=False;self.ever_debug=False;self.events=[]
        self.returns=np.zeros(self.env.num_agents,dtype=np.float64)
        self.reset=jax.jit(self.env.reset);self.transition=jax.jit(self.env.step_env);self.sample=jax.jit(self.env.sample_actions)
        self.obs,self.state=self.reset(self.key);jax.block_until_ready(self.obs)
        self.logdir=Path(runs)/self.id;self.logdir.mkdir(parents=True)
        self.chain='0'*64
        self.log('create',dict(game=self.env.name,seed=seed,source_sha256=self.env.metadata['source_sha256'],
            manifest_sha256=self.env.metadata['manifest_sha256'],version=__version__,engineering=engineering,
            ai_policy='UNTRAINED_UNIFORM_LEGAL',human_scores=None))
    def log(self,kind,value):
        row=dict(type=kind,step=self.step_count,utc_unix=time.time(),value=value,previous=self.chain)
        text=json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False)
        self.chain=hashlib.sha256(text.encode()).hexdigest();row['sha256']=self.chain
        with (self.logdir/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    def auth(self,token):
        if token not in self.clients:raise PermissionError('Invalid session capability')
        return self.clients[token]
    def require_host(self,client):
        if not client['host']:raise PermissionError('Host capability required')
    def snapshot(self,token,debug=False):
        from .rendering import scene_frame
        c=self.auth(token)
        if debug:self.require_host(c)
        if debug and not self.ever_debug:
            self.ever_debug=True;self.log('information_condition',dict(full_state_debug=True))
        roles=sorted(c['roles']);views={str(r):scene_frame(self.env,self.state,self.obs,r) for r in roles}
        if debug:views['-1']=scene_frame(self.env,self.state,self.obs,-1)
        masks=self.env.get_avail_actions(self.state)
        return dict(id=self.id,game=self.env.metadata,step=self.step_count,terminal=self.terminal,
            roles=roles,views=views,legal={str(r):masks[self.env.agents[r]].tolist() for r in roles},
            own_returns={str(r):float(self.returns[r]) for r in roles},
            host=c['host'],players=[dict(name=p['name'],roles=sorted(p['roles']),host=p['host']) for p in self.clients.values()],
            actions=self.env.manifest['actions'],agent_roles=self.env.manifest['agents'],
            task=self.env.manifest.get('task',{}),observation_fields=self.env.manifest.get('observation_fields',[]),
            rules=self.env.scene.get('rules',[]),reward_bounds=[self.env.manifest['reward_low'],self.env.manifest['reward_high']],
            information='FULL_STATE_DEBUG' if debug else 'ROLE_FILTERED',ai_policy='UNTRAINED_UNIFORM_LEGAL',
            human_scores=None,engineering=self.engineering)
    def command(self,token,op,payload):
        import jax
        import numpy as np
        c=self.auth(token)
        if op=='invite':
            self.require_host(c);name=str(payload.get('name','Player'))[:50];invite=secrets.token_urlsafe(24)
            if name in ['AI','Host'] or any(p['name']==name for p in self.clients.values()):raise ValueError('Participant name must be unique')
            self.clients[invite]=dict(name=name,host=False,roles=set(),pending={})
            self.log('invite',dict(player=name))
            return dict(session=self.id,invite=invite)
        if op=='assign':
            self.require_host(c)
            target=payload.get('player','Host');target_token=next((t for t,p in self.clients.items() if p['name']==target),None)
            if target_token is None and target!='AI':raise ValueError('Unknown player')
            roles=payload.get('roles',[])
            if any(type(r)is not int or not 0<=r<self.env.num_agents for r in roles):raise ValueError('Invalid role')
            for r in roles:
                prev=self.owners.pop(r,None)
                if prev:self.clients[prev]['roles'].discard(r);self.clients[prev]['pending'].pop(r,None)
                if target_token:
                    self.owners[r]=target_token;self.clients[target_token]['roles'].add(r)
            self.log('takeover',dict(player=target,roles=roles,effective_before_step=self.step_count+1))
        elif op=='actions':
            masks=self.env.legal_actions(self.state)
            for r,a in payload.get('actions',{}).items():
                r=int(r)
                if r not in c['roles']:raise PermissionError('Role is not assigned to this participant')
                if type(a)is not int or not 0<=a<self.env.num_actions or not bool(masks[r,a]):raise ValueError(f'Illegal action for role {r}')
            for r,a in payload.get('actions',{}).items():c['pending'][int(r)]=a
        elif op=='step':
            self.require_host(c)
            if self.terminal:raise ValueError('Episode ended. Create a new session to preserve this record.')
            key=jax.random.fold_in(self.key,self.step_count*2+1)
            actions=self.sample(key,self.state);inputs={};human=[];ai=[]
            mask=np.asarray(self.env.legal_actions(self.state))
            for r,agent in enumerate(self.env.agents):
                owner=self.owners.get(r);p=self.clients.get(owner)
                if p is not None and r in p['pending']:
                    action=p['pending'][r]
                    if not mask[r,action]:raise ValueError(f'Queued action is no longer legal: {r}')
                    actions[agent]=np.int32(action);human.append(r);inputs[str(r)]=p['name']
                else:ai.append(r)
            self.obs,self.state,rewards,dones,info=self.transition(jax.random.fold_in(self.key,self.step_count*2+2),self.state,actions)
            jax.block_until_ready(rewards);self.step_count+=1;self.terminal=bool(dones['__all__'])
            r=np.array([rewards[a] for a in self.env.agents],dtype=np.float64);self.returns+=r
            self.ever_human|=bool(human);self.ever_ai|=bool(ai)
            self.log('transition',dict(actions=[int(actions[a]) for a in self.env.agents],rewards=r.tolist(),
                human_input_roles=human,ai_roles=ai,participants=inputs,terminal=self.terminal,
                key=np.asarray(jax.random.fold_in(self.key,(self.step_count-1)*2+2)).tolist()))
            for p in self.clients.values():p['pending'].clear()
            if self.terminal:
                mode='MIXED' if self.ever_human and self.ever_ai else ('HUMAN_INPUT' if self.ever_human else 'AI_ONLY')
                record=dict(game=self.env.name,seed=self.seed,version=__version__,source_sha256=self.env.metadata['source_sha256'],
                    mode=mode,information='DEBUG_EXPOSED' if self.ever_debug else 'ROLE_FILTERED',
                    engineering=self.engineering,returns=self.returns.tolist(),steps=self.step_count,
                    human_scores=None,human_verification='NOT_INDEPENDENTLY_VERIFIED',log_chain=self.chain)
                (self.logdir/'benchmark.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        else:raise ValueError('Unknown operation')
        return dict(ok=True)

def serve(port=8765,game=225,runs='local-runs',engineering=False,bind='127.0.0.1'):
    from .registry import _rows
    import socket
    local_addresses={'127.0.0.1','localhost',bind}
    if bind=='0.0.0.0':
        local_addresses.update(a[4][0] for a in socket.getaddrinfo(socket.gethostname(),None,socket.AF_INET))
    allowed_origins={f'http://{host}:{port}' for host in local_addresses}
    sessions={};creation_lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def send_json(self,value,status=200):
            raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            from urllib.parse import urlsplit
            p=urlsplit(self.path).path
            if p=='/health':return self.send_json(dict(ok=True,version=__version__,games=len(_rows),sessions=len(sessions),pid=os.getpid(),test_nonce=os.environ.get('CRUCIBLES_TEST_NONCE')))
            if p=='/api/catalog':
                from .registry import game_info
                return self.send_json(dict(games=[{k:v[k] for k in ['id','key','title','family','dimensions','summary']} for r in _rows for v in [game_info(r['key'])]]))
            name={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}.get(p)
            if not name:return self.send_json(dict(error='Not found'),404)
            data=(WEB/name).read_bytes();self.send_response(200);self.send_header('Content-Type',{'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css'}[name.split('.')[-1]]);self.end_headers();self.wfile.write(data)
        def do_POST(self):
            try:
                origin=self.headers.get('Origin')
                if origin and origin not in allowed_origins:raise PermissionError('Cross-origin request rejected')
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=65536:raise ValueError('Request size invalid')
                body=json.loads(self.rfile.read(n));op=body.get('op')
                if op=='create':
                    seed=body.get('seed',0)
                    if type(seed)is not int or not 0<=seed<2**32:raise ValueError('Seed must be uint32')
                    with creation_lock:
                        if len(sessions)>=4:raise ValueError('Four sessions already active. Restart the server to release compiled memory; logs persist.')
                        s=Session(body.get('game',game),seed,runs,engineering);sessions[s.id]=s
                    return self.send_json(dict(session=s.id,token=s.host))
                s=sessions[body['session']]
                with s.lock:
                    token=body['token']
                    if op=='close':
                        s.require_host(s.auth(token));s.log('close',dict(terminal=s.terminal,human_scores=None))
                        sessions.pop(s.id)
                        if not sessions:
                            import jax
                            jax.clear_caches()
                        return self.send_json(dict(ok=True,closed=True))
                    if op=='snapshot':return self.send_json(s.snapshot(token,body.get('debug',False)))
                    return self.send_json(s.command(token,op,body))
            except PermissionError as e:self.send_json(dict(error=str(e)),403)
            except Exception as e:self.send_json(dict(error=f'{type(e).__name__}: {e}'),400)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer((bind,port),Handler)
    display_host='127.0.0.1' if bind=='0.0.0.0' else bind
    print(f'Crucibles JAXMARL ready: http://{display_host}:{port}/?game={game}',flush=True)
    if bind=='0.0.0.0':
        print('LAN participants: open one of these host addresses before creating invitations: '+', '.join(f'http://{host}:{port}/' for host in sorted(local_addresses) if host not in ['localhost','0.0.0.0','127.0.0.1']),flush=True)
    print(f'Health: http://{display_host}:{port}/health | Logs: {Path(runs).resolve()} | Ctrl+C stops this server',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
