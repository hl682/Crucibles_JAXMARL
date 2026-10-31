"""Carousel Supper. Original kitchen rules; authored without execution."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 232
    num_actions = 15
    max_steps = 180
    obs_keys = ('t','done','terminated','truncated','terrain','agent_pos','load','station_pos','slot_pos','phase','item_pos','item_active','kind','holder','slot','prepared','filling','heat','age','spoiled','owner','served','discarded','rotations')

    def reset(self, key):
        s = {
            't':jnp.array(0,jnp.int32),'done':jnp.array(False),'terminated':jnp.array(False),'truncated':jnp.array(False),
            'terrain':jnp.array([[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,2,2,2,0,0,0],[0,0,0,2,2,2,0,0,0],[0,0,0,2,2,2,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],jnp.int32),
            'agent_pos':jnp.array([[2,4],[6,4],[4,6]],jnp.int32),
            'agent_active':jnp.ones((3,),jnp.bool_),'agent_visible':jnp.ones((3,3),jnp.bool_),
            'load':jnp.full((3,),-1,jnp.int32),
            'station_pos':jnp.array([[1,4],[7,4],[4,2],[4,7],[1,7]],jnp.int32),
            'station_active':jnp.ones((5,),jnp.bool_),'station_visible':jnp.ones((3,5),jnp.bool_),
            'slot_pos':jnp.array([[3,3],[4,3],[5,3],[5,4],[5,5],[4,5],[3,5],[3,4]],jnp.int32),
            'slot_active':jnp.ones((8,),jnp.bool_),'slot_visible':jnp.ones((3,8),jnp.bool_),
            'phase':jnp.array(0,jnp.int32),
            'item_pos':jnp.array([[1,4],[1,3],[1,5],[7,4],[7,3],[7,5],[2,5],[6,5],[4,7]],jnp.int32),
            'item_active':jnp.ones((9,),jnp.bool_),'item_visible':jnp.ones((3,9),jnp.bool_),
            'kind':jnp.array([1,1,1,2,2,2,3,3,3],jnp.int32),
            'holder':jnp.full((9,),-1,jnp.int32),'slot':jnp.full((9,),-1,jnp.int32),
            'prepared':jnp.zeros((9,),jnp.bool_),'filling':jnp.zeros((9,),jnp.int32),
            'heat':jnp.zeros((9,),jnp.int32),'age':jnp.zeros((9,),jnp.int32),'spoiled':jnp.zeros((9,),jnp.bool_),
            'owner':jnp.full((9,),-1,jnp.int32),
            'served':jnp.array(0,jnp.int32),'discarded':jnp.array(0,jnp.int32),'rotations':jnp.array(0,jnp.int32)
        }
        return self.observe(s),s

    def observe(self,s):
        flat = jnp.concatenate([s[k].ravel().astype(jnp.float32) for k in self.obs_keys])
        return jnp.broadcast_to(flat,(3,self.observation_size))

    def legal_actions(self,s):
        p = s['agent_pos']
        d = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        q = p[:,None,:]+d[None,:,:]
        inside = jnp.all((q>=0)&(q<=8),axis=2)
        move = inside&(s['terrain'][jnp.clip(q[:,:,1],0,8),jnp.clip(q[:,:,0],0,8)]==0)
        ring = jnp.any(jnp.sum(jnp.abs(p[:,None,:]-s['slot_pos'][None,:,:]),axis=2)<=1,axis=1)
        st = jnp.sum(jnp.abs(p[:,None,:]-s['station_pos'][None,:,:]),axis=2)<=1
        extra = jnp.stack((s['load']<0,s['load']>=0,ring,ring,ring&(s['load']>=0),st[:,0]|st[:,1],s['load']>=0,st[:,2],st[:,3],st[:,4]),axis=1)
        mask = jnp.concatenate((move,extra),axis=1)&~s['done']
        return mask.at[:,0].set(True)

    def advance(self,s,actions):
        n = dict(s)
        ac = jnp.clip(actions,0,14)
        ac = jnp.where((actions>=0)&(actions<15)&self.legal_actions(s)[jnp.arange(3,dtype=jnp.int32),ac],ac,0)
        d = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        target = s['agent_pos']+d[jnp.minimum(ac,4)]*((ac>0)&(ac<5))[:,None]
        others = ~jnp.eye(3,dtype=jnp.bool_)
        block = jnp.any(jnp.all(target[:,None,:]==s['agent_pos'][None,:,:],axis=2)&others,axis=1)|jnp.any(jnp.all(target[:,None,:]==target[None,:,:],axis=2)&others,axis=1)
        n['agent_pos'] = jnp.where(block[:,None],s['agent_pos'],target)
        for k in range(3):
            i = (s['t']+k)%3
            p = n['agent_pos'][i]
            a = ac[i]
            held = n['holder']==i
            h = jnp.argmax(held).astype(jnp.int32)
            has = jnp.any(held)
            near = jnp.sum(jnp.abs(n['item_pos']-p),axis=1)<=1
            loose = near&n['item_active']&(n['holder']==-1)
            j = jnp.argmax(loose).astype(jnp.int32)
            pick = (a==5)&~has&jnp.any(loose)
            n['holder'] = n['holder'].at[j].set(jnp.where(pick,i,n['holder'][j]))
            n['slot'] = n['slot'].at[j].set(jnp.where(pick,-1,n['slot'][j]))
            n['owner'] = n['owner'].at[j].set(jnp.where(pick,i,n['owner'][j]))
            drop = (a==6)&has
            n['holder'] = n['holder'].at[h].set(jnp.where(drop,-1,n['holder'][h]))
            n['item_pos'] = n['item_pos'].at[h].set(jnp.where(drop,p,n['item_pos'][h]))
            occupied = jnp.any((n['slot'][None,:]==jnp.arange(8,dtype=jnp.int32)[:,None])&n['item_active'][None,:],axis=1)
            available = (jnp.sum(jnp.abs(n['slot_pos']-p),axis=1)<=1)&~occupied
            slot = jnp.argmax(available).astype(jnp.int32)
            put = (a==9)&has&jnp.any(available)
            n['slot'] = n['slot'].at[h].set(jnp.where(put,slot,n['slot'][h]))
            n['holder'] = n['holder'].at[h].set(jnp.where(put,-1,n['holder'][h]))
            n['item_pos'] = n['item_pos'].at[h].set(jnp.where(put,n['slot_pos'][slot],n['item_pos'][h]))
            prepstation = jnp.clip(n['kind'][h]-1,0,1)
            prep = (a==10)&has&(n['kind'][h]<3)&(jnp.sum(jnp.abs(p-n['station_pos'][prepstation]))<=1)
            n['prepared'] = n['prepared'].at[h].set(n['prepared'][h]|prep)
            bit = n['kind'][h]
            trays = near&n['item_active']&(n['kind']==3)&~n['spoiled']&((n['filling']&bit)==0)&(n['holder']==-1)
            tr = jnp.argmax(trays).astype(jnp.int32)
            combine = (a==11)&has&(n['kind'][h]<3)&n['prepared'][h]&jnp.any(trays)
            n['filling'] = n['filling'].at[tr].set(jnp.where(combine,n['filling'][tr]|bit,n['filling'][tr]))
            n['holder'] = n['holder'].at[h].set(jnp.where(combine,-2,n['holder'][h]))
            n['item_active'] = n['item_active'].at[h].set(n['item_active'][h]&~combine)
            warm = (a==12)&has&(n['kind'][h]==3)&(n['filling'][h]==3)&(jnp.sum(jnp.abs(p-n['station_pos'][2]))<=1)
            n['heat'] = n['heat'].at[h].set(jnp.minimum(9,n['heat'][h]+warm.astype(jnp.int32)))
            serve = (a==13)&has&(n['kind'][h]==3)&(n['filling'][h]==3)&(n['heat'][h]>=3)&(n['heat'][h]<=6)&~n['spoiled'][h]&(jnp.sum(jnp.abs(p-n['station_pos'][3]))<=1)
            discard = (a==14)&has&(jnp.sum(jnp.abs(p-n['station_pos'][4]))<=1)
            consume = serve|discard
            n['item_active'] = n['item_active'].at[h].set(n['item_active'][h]&~consume)
            n['holder'] = n['holder'].at[h].set(jnp.where(consume,-3,n['holder'][h]))
            n['served'] = n['served']+serve.astype(jnp.int32)
            n['discarded'] = n['discarded']+discard.astype(jnp.int32)
        turn = jnp.clip(jnp.sum((ac==7).astype(jnp.int32)-(ac==8).astype(jnp.int32),dtype=jnp.int32),-1,1)
        n['phase'] = (s['phase']+turn)%8
        ring = jnp.array([[3,3],[4,3],[5,3],[5,4],[5,5],[4,5],[3,5],[3,4]],jnp.int32)
        n['slot_pos'] = ring[(jnp.arange(8,dtype=jnp.int32)+n['phase'])%8]
        n['rotations'] = n['rotations']+(turn!=0).astype(jnp.int32)
        n['item_pos'] = jnp.where((n['slot']>=0)[:,None],n['slot_pos'][jnp.clip(n['slot'],0,7)],n['item_pos'])
        n['item_pos'] = jnp.where((n['holder']>=0)[:,None],n['agent_pos'][jnp.clip(n['holder'],0,2)],n['item_pos'])
        onheat = n['item_active']&(n['kind']==3)&(n['filling']==3)&(n['slot']>=0)&(jnp.sum(jnp.abs(n['item_pos']-n['station_pos'][2]),axis=1)<=1)
        n['heat'] = jnp.minimum(9,n['heat']+onheat.astype(jnp.int32))
        aging = n['item_active']&(n['kind']==3)&(n['filling']>0)
        n['age'] = jnp.minimum(45,n['age']+aging.astype(jnp.int32))
        n['spoiled'] = n['spoiled']|((n['heat']>6)|(n['age']>=45))
        match = n['holder'][None,:]==jnp.arange(3,dtype=jnp.int32)[:,None]
        n['load'] = jnp.where(jnp.any(match,axis=1),jnp.argmax(match,axis=1),-1).astype(jnp.int32)
        n['item_visible'] = jnp.broadcast_to(n['item_active'],(3,9))
        n['t'] = s['t']+1
        usable = n['item_active']&(n['kind']==3)&~n['spoiled']
        n['terminated'] = (n['served']==3)|~jnp.any(usable)
        n['truncated'] = (n['t']>=self.max_steps)&~n['terminated']
        n['done'] = n['terminated']|n['truncated']
        gain = (n['served']-s['served']).astype(jnp.float32)
        waste = jnp.sum((n['spoiled']&~s['spoiled']).astype(jnp.float32))
        reward = jnp.full((3,),gain-0.15*waste-0.01,jnp.float32)
        return n,reward

    def raw_step(self,key,state,actions):
        n,r = jax.lax.cond(state['done'],lambda _: (state,jnp.zeros((3,),jnp.float32)),lambda _: self.advance(state,actions),operand=None)
        return self.observe(n),n,r,jnp.full((3,),n['done'],jnp.bool_),{'done_all':n['done'],'terminated':n['terminated'],'truncated':n['truncated']}
