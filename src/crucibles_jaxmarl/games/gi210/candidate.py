"""Lantern Grid. Original blind-authored native rules; not executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 167
    num_actions = 14
    max_steps = 160
    public_keys = ('t', 'done', 'terminated', 'truncated', 'terrain', 'branch_pos', 'branch_on', 'wired', 'fuse', 'powered', 'plant_pos', 'fuel', 'charge', 'bypass', 'delivered', 'trips')

    def reset(self, key):
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'terrain': jnp.array([[0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,2,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0]], jnp.int32),
            'agent_pos': jnp.array([[1,2],[3,3],[6,3]], jnp.int32),
            'agent_active': jnp.ones((3,), jnp.bool_), 'agent_visible': jnp.ones((3,3), jnp.bool_),
            'load': jnp.full((3,), -1, jnp.int32),
            'branch_pos': jnp.array([[2,2],[3,3],[6,4]], jnp.int32),
            'branch_active': jnp.ones((3,), jnp.bool_), 'branch_visible': jnp.ones((3,3), jnp.bool_),
            'branch_on': jnp.array([True,False,False]), 'wired': jnp.array([True,False,True]),
            'fuse': jnp.ones((3,), jnp.bool_), 'powered': jnp.array([True,False,False]),
            'plant_pos': jnp.array([[1,3],[1,5]], jnp.int32),
            'plant_active': jnp.ones((2,), jnp.bool_), 'plant_visible': jnp.ones((3,2), jnp.bool_),
            'item_pos': jnp.array([[6,2],[7,2],[8,2],[6,5],[7,5],[8,5],[3,3],[2,5],[1,4],[7,4]], jnp.int32),
            'item_active': jnp.ones((10,), jnp.bool_), 'item_visible': jnp.ones((3,10), jnp.bool_),
            'kind': jnp.array([0,0,0,0,0,0,1,1,2,2], jnp.int32),
            'holder': jnp.full((10,), -1, jnp.int32), 'sorted': jnp.zeros((10,), jnp.bool_),
            'fuel': jnp.array(9, jnp.int32), 'charge': jnp.array(5, jnp.int32),
            'bypass': jnp.array(0, jnp.int32), 'delivered': jnp.array(0, jnp.int32),
            'trips': jnp.array(0, jnp.int32)
        }
        s = self.visibility(s)
        return self.observe(s), s

    def visibility(self, s):
        n = dict(s)
        lit = s['powered'][0] & (jnp.sum(jnp.abs(s['agent_pos'] - s['branch_pos'][0]), axis=1) <= 4)
        radius = jnp.where(lit, 4, 1)
        n['agent_visible'] = jnp.sum(jnp.abs(s['agent_pos'][:,None,:] - s['agent_pos'][None,:,:]), axis=2) <= radius[:,None]
        n['item_visible'] = (jnp.sum(jnp.abs(s['agent_pos'][:,None,:] - s['item_pos'][None,:,:]), axis=2) <= radius[:,None]) & s['item_active'][None,:]
        return n

    def observe(self, s):
        public = jnp.concatenate([s[k].ravel().astype(jnp.float32) for k in self.public_keys])
        agents = jnp.concatenate((s['agent_pos'], s['load'][:,None]), axis=1)
        items = jnp.concatenate((s['item_pos'], s['kind'][:,None], s['holder'][:,None], s['sorted'][:,None]), axis=1)
        rows = []
        for i in range(3):
            av = s['agent_visible'][i]
            iv = s['item_visible'][i]
            ar = jnp.concatenate((av[:,None], jnp.where(av[:,None], agents, -1)), axis=1).ravel()
            ir = jnp.concatenate((iv[:,None], jnp.where(iv[:,None], items, -1)), axis=1).ravel()
            rows = rows + [jnp.concatenate((public, jnp.array([i], jnp.float32), ar, ir)).astype(jnp.float32)]
        return jnp.stack(rows)

    def legal_actions(self, s):
        p = s['agent_pos']
        d = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]], jnp.int32)
        q = p[:,None,:] + d[None,:,:]
        inside = jnp.all((q >= 0) & (q <= jnp.array([8,6], jnp.int32)), axis=2)
        tile = s['terrain'][jnp.clip(q[:,:,1],0,6), jnp.clip(q[:,:,0],0,8)]
        move = inside & (tile != 1) & ((tile != 2) | s['powered'][1] | (s['bypass'] > 0))
        nearb = jnp.any(jnp.sum(jnp.abs(p[:,None,:] - s['branch_pos'][None,:,:]),axis=2) <= 1,axis=1)
        nearp = jnp.sum(jnp.abs(p[:,None,:] - s['plant_pos'][None,:,:]),axis=2) <= 1
        extra = jnp.stack((s['load'] < 0,s['load'] >= 0,nearb,nearp[:,0],nearb,nearb,jnp.sum(jnp.abs(p-jnp.array([4,3],jnp.int32)),axis=1)<=1,jnp.sum(jnp.abs(p-s['branch_pos'][2]),axis=1)<=1,nearp[:,1]),axis=1)
        mask = jnp.concatenate((move,extra),axis=1)
        mask = mask & ~s['done']
        return mask.at[:,0].set(True)

    def advance(self, s, actions):
        n = dict(s)
        ac = jnp.clip(actions,0,13)
        valid = (actions >= 0) & (actions < 14) & self.legal_actions(s)[jnp.arange(3,dtype=jnp.int32),ac]
        ac = jnp.where(valid,ac,0)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]], jnp.int32)
        target = s['agent_pos'] + delta[jnp.minimum(ac,4)] * ((ac >= 1)&(ac <= 4))[:,None]
        other = ~jnp.eye(3,dtype=jnp.bool_)
        occupied = jnp.any(jnp.all(target[:,None,:] == s['agent_pos'][None,:,:],axis=2)&other,axis=1)
        contested = jnp.any(jnp.all(target[:,None,:] == target[None,:,:],axis=2)&other,axis=1)
        n['agent_pos'] = jnp.where((~occupied & ~contested)[:,None],target,s['agent_pos'])
        n['bypass'] = jnp.maximum(s['bypass']-1,0)
        # Sequential interaction priority rotates every tick; movement is simultaneous.
        for k in range(3):
            i = (s['t']+k)%3
            p = n['agent_pos'][i]
            a = ac[i]
            held = n['holder'] == i
            h = jnp.argmax(held).astype(jnp.int32)
            has = jnp.any(held)
            loose = n['item_active'] & (n['holder'] == -1) & (jnp.sum(jnp.abs(n['item_pos']-p),axis=1)<=1)
            item = jnp.argmax(loose).astype(jnp.int32)
            pick = (a==5) & ~has & jnp.any(loose)
            n['holder'] = n['holder'].at[item].set(jnp.where(pick,i,n['holder'][item]))
            drop = (a==6)&has
            n['holder'] = n['holder'].at[h].set(jnp.where(drop,-1,n['holder'][h]))
            n['item_pos'] = n['item_pos'].at[h].set(jnp.where(drop,p,n['item_pos'][h]))
            dist = jnp.sum(jnp.abs(n['branch_pos']-p),axis=1)
            b = jnp.argmin(dist)
            near = dist[b]<=1
            flip = (a==7)&near
            n['branch_on'] = n['branch_on'].at[b].set(n['branch_on'][b]^flip)
            crank = (a==8)&(jnp.sum(jnp.abs(p-n['plant_pos'][0]))<=1)&(n['fuel']>0)&(n['charge']<=6)
            n['fuel'] = n['fuel']-crank.astype(jnp.int32)
            n['charge'] = n['charge']+6*crank.astype(jnp.int32)
            cable = (a==9)&near&has&(n['kind'][h]==1)&~n['wired'][b]
            repair = (a==10)&near&has&(n['kind'][h]==2)&~n['fuse'][b]
            n['wired'] = n['wired'].at[b].set(n['wired'][b]|cable)
            n['fuse'] = n['fuse'].at[b].set(n['fuse'][b]|repair)
            use = cable|repair
            n['holder'] = n['holder'].at[h].set(jnp.where(use,-2,n['holder'][h]))
            n['item_active'] = n['item_active'].at[h].set(n['item_active'][h]&~use)
            hand = (a==11)&(jnp.sum(jnp.abs(p-jnp.array([4,3],jnp.int32)))<=1)
            n['bypass'] = jnp.where(hand,3,n['bypass'])
            sort = (a==12)&has&(n['kind'][h]==0)&s['powered'][2]&(jnp.sum(jnp.abs(p-n['branch_pos'][2]))<=1)
            n['sorted'] = n['sorted'].at[h].set(n['sorted'][h]|sort)
            deliver = (a==13)&has&(n['kind'][h]==0)&n['sorted'][h]&(jnp.sum(jnp.abs(p-n['plant_pos'][1]))<=1)
            n['holder'] = n['holder'].at[h].set(jnp.where(deliver,-3,n['holder'][h]))
            n['item_active'] = n['item_active'].at[h].set(n['item_active'][h]&~deliver)
            n['delivered'] = n['delivered']+deliver.astype(jnp.int32)
        demand = (n['branch_on']&n['wired']&n['fuse']).astype(jnp.int32)*jnp.array([1,3,2],jnp.int32)
        overload = jnp.sum(demand)>4
        breaker = jnp.argmax(demand)
        n['fuse'] = n['fuse'].at[breaker].set(n['fuse'][breaker]&~overload)
        n['trips'] = n['trips']+overload.astype(jnp.int32)
        demand = (n['branch_on']&n['wired']&n['fuse']).astype(jnp.int32)*jnp.array([1,3,2],jnp.int32)
        supplied = jnp.sum(demand)<=n['charge']
        n['powered'] = (demand>0)&supplied
        n['charge'] = n['charge']-jnp.where(supplied,jnp.sum(demand,dtype=jnp.int32),0)
        carried = n['holder']>=0
        n['item_pos'] = jnp.where(carried[:,None],n['agent_pos'][jnp.clip(n['holder'],0,2)],n['item_pos'])
        match = n['holder'][None,:]==jnp.arange(3,dtype=jnp.int32)[:,None]
        n['load'] = jnp.where(jnp.any(match,axis=1),jnp.argmax(match,axis=1),-1).astype(jnp.int32)
        n['t'] = s['t']+1
        n['terminated'] = n['delivered']==6
        n['truncated'] = (n['t']>=self.max_steps)&~n['terminated']
        n['done'] = n['terminated']|n['truncated']
        n = self.visibility(n)
        reward = jnp.full((3,), (n['delivered']-s['delivered']).astype(jnp.float32)-0.01-0.1*overload.astype(jnp.float32),jnp.float32)
        return n,reward

    def raw_step(self, key, state, actions):
        n,r = jax.lax.cond(state['done'],lambda _: (state,jnp.zeros((3,),jnp.float32)),lambda _: self.advance(state,actions),operand=None)
        return self.observe(n),n,r,jnp.full((3,),n['done'],jnp.bool_),{'done_all':n['done'],'terminated':n['terminated'],'truncated':n['truncated']}
