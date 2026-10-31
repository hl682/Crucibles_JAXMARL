"""Beacon Council. Native dated reports and competing mountain caravans."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 145
    num_actions = 12
    max_steps = 150
    public_keys = ('t','done','terminated','truncated','terrain','settlement_pos','reservation','lease','delivered')

    def reset(self,key):
        s = {
            't':jnp.array(0,jnp.int32),'done':jnp.array(False),'terminated':jnp.array(False),'truncated':jnp.array(False),
            'terrain':jnp.array([[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,2,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0]],jnp.int32),
            'agent_pos':jnp.array([[2,2],[2,3],[6,4]],jnp.int32),
            'agent_active':jnp.ones((3,),jnp.bool_),'agent_visible':jnp.ones((3,3),jnp.bool_),
            'fuel':jnp.full((3,),5,jnp.int32),'stock':jnp.full((3,),2,jnp.int32),'guard':jnp.zeros((3,),jnp.int32),
            'load':jnp.ones((3,),jnp.bool_),'report':jnp.full((3,),-1,jnp.int32),'stamp':jnp.full((3,),-1,jnp.int32),
            'settlement_pos':jnp.array([[1,1],[1,5],[7,5],[7,1],[7,5],[1,3]],jnp.int32),
            'settlement_active':jnp.ones((6,),jnp.bool_),'settlement_visible':jnp.ones((3,6),jnp.bool_),
            'beacon_pos':jnp.array([[3,2],[5,2],[5,4]],jnp.int32),
            'beacon_active':jnp.ones((3,),jnp.bool_),'beacon_visible':jnp.ones((3,3),jnp.bool_),
            'beacon_report':jnp.full((3,),-1,jnp.int32),'beacon_stamp':jnp.full((3,),-1,jnp.int32),
            'pending_report':jnp.full((3,),-1,jnp.int32),'pending_stamp':jnp.full((3,),-1,jnp.int32),'delay':jnp.zeros((3,),jnp.int32),
            'pass_pos':jnp.array([[4,3]],jnp.int32),'pass_active':jnp.ones((1,),jnp.bool_),'pass_visible':jnp.ones((3,1),jnp.bool_),
            'weather':jnp.array([0],jnp.int32),
            'cargo_pos':jnp.array([[2,2],[2,3],[6,4]],jnp.int32),
            'cargo_active':jnp.ones((3,),jnp.bool_),'cargo_visible':jnp.ones((3,3),jnp.bool_),
            'holder':jnp.array([0,1,2],jnp.int32),
            'reservation':jnp.array(-1,jnp.int32),'lease':jnp.array(0,jnp.int32),'delivered':jnp.zeros((3,),jnp.int32)
        }
        s = self.visibility(s)
        return self.observe(s),s

    def visibility(self,s):
        n = dict(s)
        p = s['agent_pos']
        n['agent_visible'] = jnp.sum(jnp.abs(p[:,None,:]-p[None,:,:]),axis=2)<=2
        n['beacon_visible'] = jnp.sum(jnp.abs(p[:,None,:]-s['beacon_pos'][None,:,:]),axis=2)<=2
        n['pass_visible'] = jnp.sum(jnp.abs(p[:,None,:]-s['pass_pos'][None,:,:]),axis=2)<=2
        n['cargo_visible'] = (jnp.sum(jnp.abs(p[:,None,:]-s['cargo_pos'][None,:,:]),axis=2)<=2)&s['cargo_active'][None,:]
        direct = n['pass_visible'][:,0]
        n['report'] = jnp.where(direct,(s['weather'][0]>0).astype(jnp.int32),s['report'])
        n['stamp'] = jnp.where(direct,s['t'],s['stamp'])
        return n

    def observe(self,s):
        public = jnp.concatenate([s[k].ravel().astype(jnp.float32) for k in self.public_keys])
        ar = jnp.concatenate((s['agent_pos'],s['fuel'][:,None],s['guard'][:,None],s['stock'][:,None],s['load'][:,None]),axis=1)
        br = jnp.concatenate((s['beacon_pos'],s['beacon_report'][:,None],s['beacon_stamp'][:,None]),axis=1)
        pr = jnp.concatenate((s['pass_pos'],s['weather'][:,None]),axis=1)
        cr = jnp.concatenate((s['cargo_pos'],s['holder'][:,None],s['cargo_active'][:,None]),axis=1)
        rows = []
        for i in range(3):
            av,bv,pv,cv = s['agent_visible'][i],s['beacon_visible'][i],s['pass_visible'][i],s['cargo_visible'][i]
            a = jnp.concatenate((av[:,None],jnp.where(av[:,None],ar,-1)),axis=1).ravel()
            b = jnp.concatenate((bv[:,None],jnp.where(bv[:,None],br,-1)),axis=1).ravel()
            p = jnp.concatenate((pv[:,None],jnp.where(pv[:,None],pr,-1)),axis=1).ravel()
            c = jnp.concatenate((cv[:,None],jnp.where(cv[:,None],cr,-1)),axis=1).ravel()
            own = jnp.stack((jnp.array(i,jnp.int32),s['fuel'][i],s['stock'][i],s['report'][i],s['stamp'][i],s['guard'][i]))
            rows = rows + [jnp.concatenate((public,own,a,b,p,c)).astype(jnp.float32)]
        return jnp.stack(rows)

    def legal_actions(self,s):
        p = s['agent_pos']
        d = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        q = p[:,None,:]+d[None,:,:]
        inside = jnp.all((q>=0)&(q<=jnp.array([8,6],jnp.int32)),axis=2)
        tile = s['terrain'][jnp.clip(q[:,:,1],0,6),jnp.clip(q[:,:,0],0,8)]
        entrance = (tile==2)&jnp.any(q!=p[:,None,:],axis=2)
        permit = (s['reservation']<0)|(s['reservation']==jnp.arange(3,dtype=jnp.int32))
        weatherok = (s['weather'][0]==0)|(s['guard']>0)
        move = inside&(tile!=1)&(~entrance|(permit&weatherok)[:,None])
        nearbeacon = jnp.any(jnp.sum(jnp.abs(p[:,None,:]-s['beacon_pos'][None,:,:]),axis=2)<=1,axis=1)
        home = jnp.sum(jnp.abs(p-s['settlement_pos'][:3]),axis=1)<=1
        market = jnp.sum(jnp.abs(p-s['settlement_pos'][3:]),axis=1)<=1
        extra = jnp.stack(((home|nearbeacon)&(s['fuel']>0),s['reservation']==jnp.arange(3,dtype=jnp.int32),s['fuel']>0,nearbeacon,nearbeacon&(s['fuel']>0)&(s['stamp']>=0),market&s['load'],home&(s['stock']>0)&(s['fuel']<=7)),axis=1)
        mask = jnp.concatenate((move,extra),axis=1)&~s['done']
        return mask.at[:,0].set(True)

    def advance(self,key,s,actions):
        n = dict(s)
        ac = jnp.clip(actions,0,11)
        ac = jnp.where((actions>=0)&(actions<12)&self.legal_actions(s)[jnp.arange(3,dtype=jnp.int32),ac],ac,0)
        d = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        target = s['agent_pos']+d[jnp.minimum(ac,4)]*((ac>0)&(ac<5))[:,None]
        others = ~jnp.eye(3,dtype=jnp.bool_)
        blocked = jnp.any(jnp.all(target[:,None,:]==s['agent_pos'][None,:,:],axis=2)&others,axis=1)|jnp.any(jnp.all(target[:,None,:]==target[None,:,:],axis=2)&others,axis=1)
        n['agent_pos'] = jnp.where(blocked[:,None],s['agent_pos'],target)
        entered = jnp.all(n['agent_pos']==s['pass_pos'][0],axis=1)&jnp.any(n['agent_pos']!=s['agent_pos'],axis=1)
        n['guard'] = jnp.where(entered,0,jnp.maximum(s['guard']-1,0))
        n['lease'] = jnp.maximum(s['lease']-1,0)
        n['reservation'] = jnp.where(n['lease']>0,s['reservation'],-1)
        arrival = s['delay']==1
        fresh = arrival&(s['pending_stamp']>=s['beacon_stamp'])
        n['beacon_report'] = jnp.where(fresh,s['pending_report'],s['beacon_report'])
        n['beacon_stamp'] = jnp.where(fresh,s['pending_stamp'],s['beacon_stamp'])
        n['delay'] = jnp.maximum(s['delay']-1,0)
        for k in range(3):
            i = (s['t']+k)%3
            p = n['agent_pos'][i]
            a = ac[i]
            bd = jnp.sum(jnp.abs(n['beacon_pos']-p),axis=1)
            b = jnp.argmin(bd)
            near = bd[b]<=1
            home = jnp.sum(jnp.abs(p-n['settlement_pos'][i]))<=1
            reserve = (a==5)&(near|home)&(n['fuel'][i]>0)&(n['reservation']<0)
            release = (a==6)&(n['reservation']==i)
            n['reservation'] = jnp.where(reserve,i,jnp.where(release,-1,n['reservation']))
            n['lease'] = jnp.where(reserve,5,jnp.where(release,0,n['lease']))
            neighbors = (jnp.sum(jnp.abs(n['agent_pos']-p),axis=1)<=1)&(jnp.arange(3,dtype=jnp.int32)!=i)&n['load']
            partner = jnp.argmax(neighbors).astype(jnp.int32)
            escort = (a==7)&(n['fuel'][i]>0)&jnp.any(neighbors)
            n['guard'] = n['guard'].at[partner].set(jnp.where(escort,3,n['guard'][partner]))
            listen = (a==8)&near&(n['beacon_stamp'][b]>=n['stamp'][i])
            n['report'] = n['report'].at[i].set(jnp.where(listen,n['beacon_report'][b],n['report'][i]))
            n['stamp'] = n['stamp'].at[i].set(jnp.where(listen,n['beacon_stamp'][b],n['stamp'][i]))
            relay = (a==9)&near&(n['fuel'][i]>0)&(s['stamp'][i]>=0)
            # Physical chain delays: source two ticks, then two ticks per beacon hop.
            accept = relay&(s['stamp'][i]>=n['pending_stamp'])
            n['pending_report'] = jnp.where(accept,s['report'][i],n['pending_report'])
            n['pending_stamp'] = jnp.where(accept,s['stamp'][i],n['pending_stamp'])
            n['delay'] = jnp.where(accept,2+2*jnp.abs(jnp.arange(3,dtype=jnp.int32)-b),n['delay'])
            spend = reserve|escort|relay
            n['fuel'] = n['fuel'].at[i].add(-spend.astype(jnp.int32))
            refill = (a==11)&home&(n['stock'][i]>0)&(n['fuel'][i]<=7)
            n['fuel'] = n['fuel'].at[i].add(2*refill.astype(jnp.int32))
            n['stock'] = n['stock'].at[i].add(-refill.astype(jnp.int32))
            unload = (a==10)&n['load'][i]&(jnp.sum(jnp.abs(p-n['settlement_pos'][i+3]))<=1)
            n['load'] = n['load'].at[i].set(n['load'][i]&~unload)
            n['cargo_active'] = n['cargo_active'].at[i].set(n['cargo_active'][i]&~unload)
            n['holder'] = n['holder'].at[i].set(jnp.where(unload,-1,n['holder'][i]))
            n['delivered'] = n['delivered'].at[i].add(unload.astype(jnp.int32))
        n['cargo_pos'] = jnp.where(s['cargo_active'][:,None],n['agent_pos'],s['cargo_pos'])
        storm = jax.random.bernoulli(key,0.18)
        n['weather'] = jnp.where(s['weather']>0,jnp.maximum(s['weather']-1,0),jnp.where(storm,3,0)).astype(jnp.int32)
        n['t'] = s['t']+1
        n['terminated'] = jnp.all(n['delivered']==1)
        n['truncated'] = (n['t']>=self.max_steps)&~n['terminated']
        n['done'] = n['terminated']|n['truncated']
        n = self.visibility(n)
        deliveries = (n['delivered']-s['delivered']).astype(jnp.float32)
        reward = 1.75*deliveries+0.25*jnp.sum(deliveries)-0.01
        return n,reward.astype(jnp.float32)

    def raw_step(self,key,state,actions):
        n,r = jax.lax.cond(state['done'],lambda _: (state,jnp.zeros((3,),jnp.float32)),lambda _: self.advance(key,state,actions),operand=None)
        return self.observe(n),n,r,jnp.full((3,),n['done'],jnp.bool_),{'done_all':n['done'],'terminated':n['terminated'],'truncated':n['truncated']}
