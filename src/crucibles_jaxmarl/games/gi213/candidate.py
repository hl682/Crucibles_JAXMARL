"""Hedgehaven. Original spatial gardening with mobile shelter and ecology."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 265
    num_actions = 16
    max_steps = 180
    obs_keys = ('t','done','terminated','truncated','terrain','agent_pos','load','bed_pos','bed_owner','growth','moisture','soil','frost','alive','harvested','pollinated','covered','sheltered','shaded','item_pos','item_active','kind','holder','cover_bed','deployed','water','insect_pos','route_index','route_direction','facility_pos','daylight','wind','sun','delivered','pollinations')

    def reset(self,key):
        s = {
            't':jnp.array(0,jnp.int32),'done':jnp.array(False),'terminated':jnp.array(False),'truncated':jnp.array(False),
            'terrain':jnp.zeros((7,9),jnp.int32),
            'agent_pos':jnp.array([[2,2],[6,2],[4,5]],jnp.int32),'agent_active':jnp.ones((3,),jnp.bool_),'agent_visible':jnp.ones((3,3),jnp.bool_),
            'load':jnp.full((3,),-1,jnp.int32),
            'bed_pos':jnp.array([[3,2],[5,2],[3,4],[5,4],[2,3],[6,3]],jnp.int32),
            'bed_active':jnp.ones((6,),jnp.bool_),'bed_visible':jnp.ones((3,6),jnp.bool_),
            'bed_owner':jnp.array([0,1,0,1,2,2],jnp.int32),
            'growth':jnp.zeros((6,),jnp.int32),'moisture':jnp.full((6,),4,jnp.int32),'soil':jnp.full((6,),3,jnp.int32),'frost':jnp.zeros((6,),jnp.int32),
            'alive':jnp.ones((6,),jnp.bool_),'harvested':jnp.zeros((6,),jnp.bool_),'pollinated':jnp.zeros((6,),jnp.bool_),
            'covered':jnp.zeros((6,),jnp.bool_),'sheltered':jnp.zeros((6,),jnp.bool_),'shaded':jnp.zeros((6,),jnp.bool_),
            'item_pos':jnp.array([[2,1],[6,1],[1,5],[2,5],[4,5],[7,5],[3,2],[5,2],[3,4],[5,4],[2,3],[6,3]],jnp.int32),
            'item_active':jnp.array([True,True,True,True,True,True,False,False,False,False,False,False]),
            'item_visible':jnp.zeros((3,12),jnp.bool_),
            'kind':jnp.array([0,0,1,1,2,3,4,4,4,4,4,4],jnp.int32),
            'holder':jnp.full((12,),-1,jnp.int32),'cover_bed':jnp.full((12,),-1,jnp.int32),
            'deployed':jnp.array([True,True,False,False,False,False,False,False,False,False,False,False]),
            'water':jnp.array([0,0,0,0,0,4,0,0,0,0,0,0],jnp.int32),
            'insect_pos':jnp.array([[3,1]],jnp.int32),'insect_active':jnp.ones((1,),jnp.bool_),'insect_visible':jnp.ones((3,1),jnp.bool_),
            'route_index':jnp.array(1,jnp.int32),'route_direction':jnp.array(1,jnp.int32),
            'facility_pos':jnp.array([[4,6],[7,5]],jnp.int32),'facility_active':jnp.ones((2,),jnp.bool_),'facility_visible':jnp.ones((3,2),jnp.bool_),
            'daylight':jnp.array(True),'wind':jnp.array(1,jnp.int32),'sun':jnp.array(1,jnp.int32),
            'delivered':jnp.zeros((3,),jnp.int32),'pollinations':jnp.array(0,jnp.int32)
        }
        s = self.shelter(s)
        return self.observe(s),s

    def shelter(self,s):
        n = dict(s)
        hedges = s['item_active']&(s['kind']==0)&s['deployed']&(s['holder']==-1)
        dx = s['bed_pos'][:,None,0]-s['item_pos'][None,:,0]
        dy = jnp.abs(s['bed_pos'][:,None,1]-s['item_pos'][None,:,1])
        n['sheltered'] = jnp.any(hedges[None,:]&(dx==s['wind'])&(dy<=1),axis=1)
        n['shaded'] = jnp.any(hedges[None,:]&(dx*s['sun']>=1)&(dx*s['sun']<=2)&(dy==0),axis=1)&s['daylight']
        n['covered'] = jnp.any((s['cover_bed'][None,:]==jnp.arange(6,dtype=jnp.int32)[:,None])&s['item_active'][None,:]&(s['holder'][None,:]==-1),axis=1)
        n['item_visible'] = jnp.broadcast_to(s['item_active'],(3,12))
        return n

    def observe(self,s):
        flat = jnp.concatenate([s[k].ravel().astype(jnp.float32) for k in self.obs_keys])
        return jnp.broadcast_to(flat,(3,self.observation_size))

    def legal_actions(self,s):
        p = s['agent_pos']
        d = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        q = p[:,None,:]+d[None,:,:]
        hedges = s['item_active']&(s['kind']==0)&s['deployed']&(s['holder']==-1)
        blocked = jnp.any(jnp.all(q[:,:,None,:]==s['item_pos'][None,None,:,:],axis=3)&hedges[None,None,:],axis=2)
        move = jnp.all((q>=0)&(q<=jnp.array([8,6],jnp.int32)),axis=2)&~blocked
        nearbed = jnp.any(jnp.sum(jnp.abs(p[:,None,:]-s['bed_pos'][None,:,:]),axis=2)<=1,axis=1)
        nearfac = jnp.sum(jnp.abs(p[:,None,:]-s['facility_pos'][None,:,:]),axis=2)<=1
        has = s['load']>=0
        extra = jnp.stack((~has,has,has,has,has,has,nearbed&has,nearfac[:,1]&has,nearbed&has,nearbed&~has,nearfac[:,0]&has),axis=1)
        mask = jnp.concatenate((move,extra),axis=1)&~s['done']
        return mask.at[:,0].set(True)

    def advance(self,s,actions):
        n = dict(s)
        ac = jnp.clip(actions,0,15)
        ac = jnp.where((actions>=0)&(actions<16)&self.legal_actions(s)[jnp.arange(3,dtype=jnp.int32),ac],ac,0)
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
            loose = n['item_active']&(n['holder']==-1)&(jnp.sum(jnp.abs(n['item_pos']-p),axis=1)<=1)
            j = jnp.argmax(loose).astype(jnp.int32)
            pick = (a==5)&~has&jnp.any(loose)
            n['holder'] = n['holder'].at[j].set(jnp.where(pick,i,n['holder'][j]))
            n['cover_bed'] = n['cover_bed'].at[j].set(jnp.where(pick,-1,n['cover_bed'][j]))
            n['deployed'] = n['deployed'].at[j].set(n['deployed'][j]&~pick)
            drop = (a==6)&has
            n['holder'] = n['holder'].at[h].set(jnp.where(drop,-1,n['holder'][h]))
            n['item_pos'] = n['item_pos'].at[h].set(jnp.where(drop,p,n['item_pos'][h]))
            n['deployed'] = n['deployed'].at[h].set(jnp.where(drop,n['kind'][h]==0,n['deployed'][h]))
            placepos = p+d[jnp.clip(a-6,1,4)]
            inside = jnp.all((placepos>=0)&(placepos<=jnp.array([8,6],jnp.int32)))
            onbed = jnp.all(n['bed_pos']==placepos,axis=1)
            bed = jnp.argmax(onbed).astype(jnp.int32)
            blanketthere = jnp.any((n['cover_bed']==bed)&n['item_active']&(n['holder']==-1))
            obstacle = jnp.any(jnp.all(n['agent_pos']==placepos,axis=1))|jnp.any(jnp.all(n['item_pos']==placepos,axis=1)&n['item_active']&n['deployed'])
            hedgeplace = (n['kind'][h]==0)&~jnp.any(onbed)&~obstacle
            blanketplace = (n['kind'][h]==1)&jnp.any(onbed)&~blanketthere&n['alive'][bed]&~n['harvested'][bed]
            put = (a>=7)&(a<=10)&has&inside&(hedgeplace|blanketplace)
            n['holder'] = n['holder'].at[h].set(jnp.where(put,-1,n['holder'][h]))
            n['item_pos'] = n['item_pos'].at[h].set(jnp.where(put,placepos,n['item_pos'][h]))
            n['cover_bed'] = n['cover_bed'].at[h].set(jnp.where(put&blanketplace,bed,n['cover_bed'][h]))
            n['deployed'] = n['deployed'].at[h].set(n['deployed'][h]|put)
            covered = jnp.any((n['cover_bed'][None,:]==jnp.arange(6,dtype=jnp.int32)[:,None])&n['item_active'][None,:]&(n['holder'][None,:]==-1),axis=1)
            near = (jnp.sum(jnp.abs(n['bed_pos']-p),axis=1)<=1)&n['alive']&~n['harvested']&~covered
            needs = near&(((a==11)&(n['soil']<6))|((a==13)&(n['moisture']<6))|((a==14)&(n['growth']>=6)&n['pollinated']))
            b = jnp.argmax(needs).astype(jnp.int32)
            till = (a==11)&has&(n['kind'][h]==2)&jnp.any(needs)
            fill = (a==12)&has&(n['kind'][h]==3)&(jnp.sum(jnp.abs(p-n['facility_pos'][1]))<=1)
            water = (a==13)&has&(n['kind'][h]==3)&(n['water'][h]>0)&jnp.any(needs)
            n['soil'] = n['soil'].at[b].set(jnp.minimum(6,n['soil'][b]+3*till.astype(jnp.int32)))
            n['moisture'] = n['moisture'].at[b].set(jnp.minimum(6,n['moisture'][b]+3*water.astype(jnp.int32)))
            n['water'] = n['water'].at[h].set(jnp.where(fill,4,n['water'][h]-water.astype(jnp.int32)))
            harvest = (a==14)&~has&jnp.any(needs)
            n['harvested'] = n['harvested'].at[b].set(n['harvested'][b]|harvest)
            n['item_active'] = n['item_active'].at[b+6].set(n['item_active'][b+6]|harvest)
            n['holder'] = n['holder'].at[b+6].set(jnp.where(harvest,i,n['holder'][b+6]))
            delivery = (a==15)&has&(n['kind'][h]==4)&(jnp.sum(jnp.abs(p-n['facility_pos'][0]))<=1)
            n['item_active'] = n['item_active'].at[h].set(n['item_active'][h]&~delivery)
            n['holder'] = n['holder'].at[h].set(jnp.where(delivery,-3,n['holder'][h]))
            n['delivered'] = n['delivered'].at[i].add(delivery.astype(jnp.int32))
        n['item_pos'] = jnp.where((n['holder']>=0)[:,None],n['agent_pos'][jnp.clip(n['holder'],0,2)],n['item_pos'])
        n['t'] = s['t']+1
        n['daylight'] = (n['t']%12)<7
        n['wind'] = jnp.where((n['t']//12)%2==0,1,-1).astype(jnp.int32)
        n['sun'] = jnp.where((n['t']%12)<4,1,-1).astype(jnp.int32)
        n = self.shelter(n)
        route = jnp.array([[2,1],[3,1],[4,1],[5,1],[6,1],[6,2],[6,3],[6,4],[6,5],[5,5],[4,5],[3,5],[2,5],[2,4],[2,3],[2,2]],jnp.int32)
        ri = (s['route_index']+s['route_direction'])%16
        hedge = n['item_active']&(n['kind']==0)&n['deployed']&(n['holder']==-1)
        blocked = jnp.any(jnp.all(n['item_pos']==route[ri],axis=1)&hedge)
        n['route_direction'] = jnp.where(blocked,-s['route_direction'],s['route_direction'])
        n['route_index'] = jnp.where(blocked,s['route_index'],ri)
        n['insect_pos'] = route[n['route_index']][None,:]
        visit = (jnp.sum(jnp.abs(n['bed_pos']-n['insect_pos'][0]),axis=1)<=1)&~n['covered']&n['alive']&~n['harvested']&n['daylight']
        n['pollinations'] = s['pollinations']+jnp.sum((visit&~n['pollinated']).astype(jnp.int32),dtype=jnp.int32)
        n['pollinated'] = n['pollinated']|visit
        growing = n['alive']&~n['harvested']
        growth = growing&n['daylight']&~n['covered']&~n['shaded']&(n['moisture']>0)&(n['soil']>0)&((n['growth']<3)|n['pollinated'])
        n['growth'] = jnp.minimum(6,n['growth']+growth.astype(jnp.int32))
        n['soil'] = jnp.maximum(0,n['soil']-growth.astype(jnp.int32))
        dry = growing&n['daylight']&((n['t']%3)==0)&~n['covered']
        n['moisture'] = jnp.maximum(0,n['moisture']-dry.astype(jnp.int32))
        cold = growing&~n['daylight']&~n['sheltered']&~n['covered']
        n['frost'] = jnp.minimum(6,n['frost']+cold.astype(jnp.int32))
        n['alive'] = n['alive']&(n['frost']<6)
        match = n['holder'][None,:]==jnp.arange(3,dtype=jnp.int32)[:,None]
        n['load'] = jnp.where(jnp.any(match,axis=1),jnp.argmax(match,axis=1),-1).astype(jnp.int32)
        n['terminated'] = jnp.all(n['harvested']|~n['alive'])&~jnp.any(n['item_active']&(n['kind']==4))
        n['truncated'] = (n['t']>=self.max_steps)&~n['terminated']
        n['done'] = n['terminated']|n['truncated']
        gain = (n['delivered']-s['delivered']).astype(jnp.float32)
        lost = jnp.sum((s['alive']&~n['alive']).astype(jnp.float32))
        reward = jnp.sum(gain)+0.2*gain-0.1*lost-0.01
        return n,reward.astype(jnp.float32)

    def raw_step(self,key,state,actions):
        n,r = jax.lax.cond(state['done'],lambda _: (state,jnp.zeros((3,),jnp.float32)),lambda _: self.advance(state,actions),operand=None)
        return self.observe(n),n,r,jnp.full((3,),n['done'],jnp.bool_),{'done_all':n['done'],'terminated':n['terminated'],'truncated':n['truncated']}
