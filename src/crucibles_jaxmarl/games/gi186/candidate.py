"""Floodstone Builders: original prospective source, never executed by its author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 171
    num_actions = 21
    max_steps = 96

    def reset(self, key):
        tile = jnp.array([
            [1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,0,6,1],
            [1,1,1,1,1,1,2,0,1],
            [1,0,0,0,0,1,2,0,1],
            [1,0,0,0,0,1,2,0,1],
            [1,7,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1]], dtype=jnp.int32)
        s = {
            "t": jnp.array(0, jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "tile": tile, "water": jnp.zeros((7,9),jnp.bool_).at[1,7].set(True),
            "water_pos": jnp.stack((jnp.mod(jnp.arange(63,dtype=jnp.int32),9),jnp.arange(63,dtype=jnp.int32)//9),axis=1),
            "water_active": jnp.zeros((63,),jnp.bool_).at[16].set(True),
            "water_visible": jnp.ones((3,63),jnp.bool_),
            "pos": jnp.array([[2,4],[3,4],[2,5]],jnp.int32),
            "face": jnp.array([1,0,0],jnp.int32),
            "breath": jnp.full((3,),6,jnp.int32),
            "stone": jnp.array([2,1,2],jnp.int32),
            "tool": jnp.array([1,0,2],jnp.int32),
            "tool_stock": jnp.array([1,0],jnp.int32),
            "ore": jnp.zeros((3,3),jnp.int32), "bank": jnp.zeros((3,),jnp.int32),
            "priority": jax.random.permutation(key,jnp.arange(3,dtype=jnp.int32)),
            "vein_pos": jnp.array([[6,2],[6,3],[6,4]],jnp.int32),
            "vein_active": jnp.ones((3,),jnp.bool_),
            "vein_visible": jnp.ones((3,3),jnp.bool_),
            "agent_active": jnp.ones((3,),jnp.bool_),
            "agent_visible": jnp.ones((3,3),jnp.bool_),
            "site_pos": jnp.array([[1,5],[7,1]],jnp.int32),
            "site_active": jnp.ones((2,),jnp.bool_),
            "site_visible": jnp.ones((3,2),jnp.bool_),
            "events": jnp.zeros((8,),jnp.bool_),
            "spent_stone": jnp.array(0,jnp.int32)}
        return self.observe(s), s

    def front(self, s):
        dirs = jnp.array([[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        return s["pos"] + dirs[s["face"]]

    def legal_actions(self, s):
        q = self.front(s)
        x, y = jnp.clip(q[:,0],0,8), jnp.clip(q[:,1],0,6)
        inside = (q[:,0]>0)&(q[:,0]<8)&(q[:,1]>0)&(q[:,1]<6)
        kind = s["tile"][y,x]
        occupied = jnp.any(jnp.all(q[:,None,:]==s["pos"][None,:,:],axis=2),axis=1)
        # Four-neighbour support, with bounded indices and explicit geometry.
        support = (s["tile"][jnp.clip(y-1,0,6),x]==4)|(s["tile"][jnp.clip(y+1,0,6),x]==4)|(s["tile"][y,jnp.clip(x-1,0,8)]==4)|(s["tile"][y,jnp.clip(x+1,0,8)]==4)
        camp = jnp.sum(jnp.abs(s["pos"]-s["site_pos"][0]),axis=1)<=1
        near = jnp.all(q[:,None,:]==s["pos"][None,:,:],axis=2)
        recipient = jnp.any(near&(s["stone"][None,:]<4),axis=1)
        rescue = jnp.any(near&(s["breath"][None,:]==0),axis=1)
        mask = jnp.zeros((3,21),jnp.bool_).at[:,0:5].set(True).at[:,17:21].set(True)
        mask = mask.at[:,5].set(inside&((kind==1)|(kind==2))&(s["tool"]==1)&(s["stone"]<4)&(jnp.sum(s["ore"],axis=1)<2))
        mask = mask.at[:,6].set(inside&(kind==0)&~occupied&(s["stone"]>=1))
        mask = mask.at[:,7].set(inside&(kind==0)&~occupied&support&(s["stone"]>=2)&(s["tool"]==2))
        mask = mask.at[:,8].set((kind==5)&support&(s["tool"]==2))
        mask = mask.at[:,9].set(inside&(kind==0)&~occupied&(s["stone"]>=1))
        salvage = jnp.where(kind==5,2,1)
        mask = mask.at[:,10].set(((kind==3)|(kind==4)|(kind==5))&~occupied&(s["stone"]+salvage<=4))
        mask = mask.at[:,11].set(camp&(jnp.sum(s["ore"],axis=1)>0))
        mask = mask.at[:,12].set(recipient&(s["stone"]>0))
        mask = mask.at[:,13].set(rescue&(s["stone"]>0))
        mask = mask.at[:,14].set(camp&(s["tool"]==0)&(s["tool_stock"][0]>0))
        mask = mask.at[:,15].set(camp&(s["tool"]==0)&(s["tool_stock"][1]>0))
        mask = mask.at[:,16].set(camp&(s["tool"]>0))
        return mask&((s["breath"]>0)&~s["done"])[:,None] | (jnp.arange(21)[None,:]==0)

    def observe(self, s):
        public = jnp.concatenate((
            s["t"][None]/96.0,s["tile"].reshape(-1)/7.0,s["water"].reshape(-1),
            s["pos"].reshape(-1)/8.0,s["face"]/3.0,s["breath"]/6.0,
            s["stone"]/4.0,s["tool"]/2.0,s["ore"].reshape(-1)/3.0,
            s["bank"]/3.0,s["tool_stock"]/2.0,s["events"]))
        return jnp.concatenate((jnp.broadcast_to(public,(3,167)),s["priority"][:,None]/2.0,jnp.eye(3)),axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        old_done = state["done"]
        s = dict(state)
        safe = jnp.clip(actions,0,20)
        allowed = self.legal_actions(state)[jnp.arange(3),safe]&(actions==safe)
        a = jnp.where(allowed,actions,0)
        s["events"] = jnp.zeros((8,),jnp.bool_)
        reward = jnp.zeros((3,),jnp.float32)
        moving = (a>=1)&(a<=4)
        turning = (a>=17)&(a<=20)
        s["face"] = jnp.where(moving,a-1,jnp.where(turning,a-17,s["face"]))
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)[jnp.clip(a,0,4)]
        q = s["pos"]+jnp.where(moving[:,None],delta,0)
        x,y = jnp.clip(q[:,0],0,8),jnp.clip(q[:,1],0,6)
        knd = s["tile"][y,x]
        free = (q[:,0]>0)&(q[:,0]<8)&(q[:,1]>0)&(q[:,1]<6)&(knd!=1)&(knd!=2)&(knd!=3)
        occupied = jnp.any(jnp.all(q[:,None,:]==s["pos"][None,:,:],axis=2)&~jnp.eye(3,dtype=jnp.bool_),axis=1)
        rank = jnp.mod(jnp.arange(3)-s["t"],3)
        same = jnp.all(q[:,None,:]==q[None,:,:],axis=2)
        lose = jnp.any(same&(rank[None,:]<rank[:,None])&moving[None,:]&free[None,:]&~occupied[None,:],axis=1)
        s["pos"] = jnp.where((moving&free&~occupied&~lose)[:,None],q,s["pos"])
        # Resource transactions use cyclic priority; each rechecks the current state.
        for turn in range(3):
            i = jnp.mod(s["t"]+turn,3)
            valid = self.legal_actions(s)[i,a[i]]
            act = jnp.where(valid,a[i],0)
            f = self.front(s)[i]
            fx,fy = jnp.clip(f[0],0,8),jnp.clip(f[1],0,6)
            kind = s["tile"][fy,fx]
            dig = act==5
            vein = jnp.all(s["vein_pos"]==f[None,:],axis=1)&s["vein_active"]&dig
            s["ore"] = s["ore"].at[i].add(vein.astype(jnp.int32))
            s["vein_active"] = s["vein_active"]&~vein
            s["stone"] = s["stone"].at[i].add(dig.astype(jnp.int32))
            support, pump, wall = act==6,act==7,act==9
            remove = act==10
            refund = jnp.where(kind==5,2,1)*remove.astype(jnp.int32)
            s["stone"] = s["stone"].at[i].add(refund-support.astype(jnp.int32)-2*pump.astype(jnp.int32)-wall.astype(jnp.int32))
            new_tile = jnp.where(dig|remove,0,jnp.where(support,4,jnp.where(pump,5,jnp.where(wall,3,kind))))
            s["tile"] = s["tile"].at[fy,fx].set(new_tile)
            s["water"] = s["water"].at[fy,fx].set(jnp.where(wall,False,s["water"][fy,fx]))
            deposit = act==11
            cargo = s["ore"][i]*deposit.astype(jnp.int32)
            s["bank"] = s["bank"]+cargo
            s["ore"] = s["ore"].at[i].add(-cargo)
            reward = reward+0.25*jnp.sum(cargo).astype(jnp.float32)
            reward = reward.at[i].add(0.5*cargo[s["priority"][i]].astype(jnp.float32))
            recipient = jnp.argmax(jnp.all(s["pos"]==f[None,:],axis=1))
            give, rescue = act==12,act==13
            s["stone"] = s["stone"].at[i].add(-give.astype(jnp.int32)-rescue.astype(jnp.int32))
            s["stone"] = s["stone"].at[recipient].add(give.astype(jnp.int32))
            s["breath"] = s["breath"].at[recipient].set(jnp.where(rescue,3,s["breath"][recipient]))
            s["spent_stone"] = s["spent_stone"]+rescue.astype(jnp.int32)
            take_pick,take_wrench,put = act==14,act==15,act==16
            old_tool = s["tool"][i]
            s["tool_stock"] = s["tool_stock"]+jnp.array([put&(old_tool==1),put&(old_tool==2)],jnp.int32)-jnp.array([take_pick,take_wrench],jnp.int32)
            s["tool"] = s["tool"].at[i].set(jnp.where(take_pick,1,jnp.where(take_wrench,2,jnp.where(put,0,old_tool))))
            flags = jnp.array([dig,support|pump|wall,False,False,deposit,give,rescue,take_pick|take_wrench|put])
            s["events"] = s["events"]|flags
        # Connected inundation spreads one cardinal cell every third tick.
        wet = s["water"]
        neighbour = jnp.roll(wet,1,0)|jnp.roll(wet,-1,0)|jnp.roll(wet,1,1)|jnp.roll(wet,-1,1)
        floor = (s["tile"]!=1)&(s["tile"]!=2)&(s["tile"]!=3)
        spread = (jnp.mod(s["t"]+1,3)==0)&neighbour&floor
        water = (wet|spread).at[1,7].set(True)
        # A supported installed pump works only while a wrench bearer works beside it.
        fronts = self.front(s)
        for i in range(3):
            fx,fy = jnp.clip(fronts[i,0],0,8),jnp.clip(fronts[i,1],0,6)
            works = (a[i]==8)&(s["tool"][i]==2)&(s["tile"][fy,fx]==5)&(s["breath"][i]>0)
            support = (s["tile"][jnp.clip(fy-1,0,6),fx]==4)|(s["tile"][jnp.clip(fy+1,0,6),fx]==4)|(s["tile"][fy,jnp.clip(fx-1,0,8)]==4)|(s["tile"][fy,jnp.clip(fx+1,0,8)]==4)
            s["events"] = s["events"].at[3].set(s["events"][3]|(works&support))
            radius = jnp.abs(jnp.arange(7)[:,None]-fy)+jnp.abs(jnp.arange(9)[None,:]-fx)<=2
            water = water&~(works&support&radius)
        water = water.at[1,7].set(True)&floor
        s["events"] = s["events"].at[2].set(jnp.any(water&~wet))
        s["water"] = water
        s["water_active"] = water.reshape(-1)
        exposed = water[s["pos"][:,1],s["pos"][:,0]]
        awake = s["breath"]>0
        s["breath"] = jnp.where(awake,jnp.where(exposed,jnp.maximum(s["breath"]-1,0),jnp.minimum(s["breath"]+1,6)),0)
        reward = reward-0.05*(exposed&awake).astype(jnp.float32)
        s["t"] = s["t"]+jnp.array(1,jnp.int32)
        s["terminated"] = (jnp.sum(s["bank"])==3)|jnp.all(s["breath"]==0)
        s["truncated"] = (s["t"]>=96)&~s["terminated"]
        s["done"] = s["terminated"]|s["truncated"]
        s = jax.tree.map(lambda new,old:jnp.where(old_done,old,new),s,state)
        reward = jnp.where(old_done,jnp.zeros((3,),jnp.float32),reward).astype(jnp.float32)
        return self.observe(s),s,reward,jnp.full((3,),s["done"],jnp.bool_),{"done_all":s["done"],"terminated":s["terminated"],"truncated":s["truncated"]}
