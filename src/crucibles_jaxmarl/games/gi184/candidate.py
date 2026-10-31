"""Rotating Keep. Original source-only authoring; no runtime claims."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 88
    num_actions = 10
    max_steps = 96

    def reset(self, key):
        bit = jax.random.randint(key, (), 0, 2, dtype=jnp.int32)
        s = {
            "t": jnp.array(0,jnp.int32),
            "done": jnp.array(False), "natural": jnp.array(False),
            "map": jnp.array([[1,1,1,1,1,1,1,1,1],[1,0,0,0,1,0,0,0,1],
                [1,0,0,0,2,0,0,0,1],[1,0,0,0,1,0,0,0,1],
                [1,0,0,0,2,0,0,0,1],[1,0,0,0,1,0,0,0,1],
                [1,1,1,1,1,1,1,1,1]],jnp.int32),
            "pos": jnp.array([[2,3],[3,2],[6,3]],jnp.int32),
            "active": jnp.ones(3,jnp.bool_), "visible": jnp.ones((3,3),jnp.bool_),
            "delivered": jnp.zeros(3,jnp.bool_),
            "preference": jnp.stack((bit,1-bit,bit)),
            "door_pos": jnp.array([[4,2],[4,4]],jnp.int32),
            "door_active": jnp.ones(2,jnp.bool_), "door_visible": jnp.ones((3,2),jnp.bool_),
            "door_open": jnp.array([False,True]),
            "door_target": jnp.array([False,True]),
            "door_jammed": jnp.zeros(2,jnp.bool_),
            "phase": jnp.array(1,jnp.int32),
            "pending": jnp.array(-1,jnp.int32), "timer": jnp.array(0,jnp.int32),
            "switch_pos": jnp.array([[2,3],[6,3]],jnp.int32),
            "switch_active": jnp.ones(2,jnp.bool_), "switch_visible": jnp.ones((3,2),jnp.bool_),
            "switch_charge": jnp.array([3,3],jnp.int32),
            "switch_key": jnp.array([0,1],jnp.int32),
            "key_pos": jnp.array([[2,3],[6,3]],jnp.int32),
            "key_active": jnp.ones(2,jnp.bool_), "key_visible": jnp.ones((3,2),jnp.bool_),
            "key_holder": jnp.array([0,2],jnp.int32),
            "key_color": jnp.array([0,1],jnp.int32),
            "goal_pos": jnp.array([[7,1],[1,5]],jnp.int32),
            "goal_active": jnp.ones(2,jnp.bool_), "goal_visible": jnp.ones((3,2),jnp.bool_),
            "goal_color": jnp.array([0,1],jnp.int32),
            "gate_event": jnp.zeros(2,jnp.bool_),
            "request_event": jnp.zeros(2,jnp.bool_),
            "claim_event": jnp.zeros(3,jnp.bool_),
        }
        return self.observe(s),s

    def movement(self,s):
        d = jnp.array([[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        p = s["pos"][:,None,:]+d[None,:,:]
        inside = (p[:,:,0]>=0)&(p[:,:,0]<=8)&(p[:,:,1]>=0)&(p[:,:,1]<=6)
        tile = s["map"][jnp.clip(p[:,:,1],0,6),jnp.clip(p[:,:,0],0,8)]
        doors = jnp.all(p[:,:,None,:] == s["door_pos"][None,None,:,:],axis=-1)
        threshold = jnp.any(doors & s["door_open"][None,None,:],axis=-1)
        blocked = jnp.any(jnp.all(p[:,:,None,:] == s["pos"][None,None,:,:],axis=-1),axis=-1)
        return inside & ((tile == 0) | ((tile == 2)&threshold)) & ~blocked

    def options(self,s):
        ids = jnp.arange(3,dtype=jnp.int32)
        holds = s["key_holder"][None,:] == ids[:,None]
        has = jnp.any(holds,axis=1)
        atswitch = jnp.all(s["pos"][:,None,:] == s["switch_pos"][None,:,:],axis=-1)
        request = atswitch & holds & (s["switch_charge"][None,:]>0) & (s["pending"]<0)
        take = jnp.all(s["pos"][:,None,:] == s["key_pos"][None,:,:],axis=-1)
        take = take & (s["key_holder"][None,:]<0) & ~has[:,None]
        goals = jnp.all(s["pos"][:,None,:] == s["goal_pos"][None,:,:],axis=-1)
        claim = jnp.any(goals,axis=1) & ~s["delivered"]
        adjacent = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["pos"][None,:,:]),axis=-1)==1
        passing = adjacent & has[:,None] & ~has[None,:]
        return holds,request,take,goals,claim,passing

    def legal_actions(self,s):
        holds,request,take,goals,claim,passing = self.options(s)
        m = jnp.concatenate((jnp.ones((3,1),jnp.bool_),self.movement(s),
            jnp.stack((jnp.any(request,1),jnp.any(take,1),claim,jnp.any(holds,1),
                       jnp.any(passing,1)),axis=1)),axis=1)
        return m & ((~s["done"]) | (jnp.arange(10)[None,:]==0))

    def observe(self,s):
        pub = jnp.concatenate((s["map"].reshape(-1),s["pos"].reshape(-1),
            s["delivered"].astype(jnp.int32),s["key_pos"].reshape(-1),s["key_holder"],
            s["door_open"].astype(jnp.int32),s["phase"].reshape(1),s["pending"].reshape(1),
            s["timer"].reshape(1),s["switch_charge"],s["t"].reshape(1))).astype(jnp.float32)
        return jnp.concatenate((jnp.broadcast_to(pub,(3,86)),s["preference"][:,None].astype(jnp.float32),
            jnp.arange(3,dtype=jnp.float32)[:,None]),axis=1)

    def raw_step(self,key,state,actions):
        legal = self.legal_actions(state)
        a = jnp.clip(actions,0,9)
        valid = (actions>=0)&(actions<10)&jnp.take_along_axis(legal,a[:,None],axis=1)[:,0]
        a = jnp.where(valid,a,0)
        s = dict(state)
        ids = jnp.arange(3,dtype=jnp.int32)
        priority = jnp.mod(ids-state["t"],3)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        proposed = state["pos"] + delta[jnp.where(a<=4,a,0)]
        same = jnp.all(proposed[:,None,:]==proposed[None,:,:],axis=-1)
        loses = jnp.any(same & (priority[None,:]<priority[:,None]),axis=1)
        s["pos"] = jnp.where(loses[:,None],state["pos"],proposed)
        s["key_pos"] = jnp.where((s["key_holder"]>=0)[:,None],
            s["pos"][jnp.maximum(s["key_holder"],0)],s["key_pos"])
        s["request_event"] = jnp.zeros(2,jnp.bool_)
        s["claim_event"] = jnp.zeros(3,jnp.bool_)
        r = jnp.zeros(3,jnp.float32)
        # Key transactions use cyclic priority and never create another key.
        for k in range(3):
            i = jnp.mod(state["t"]+k,3)
            holds,request,take,goals,claim,passing = self.options(s)
            h = jnp.argmax(holds[i])
            q = jnp.argmax(take[i])
            taking = (a[i]==6)&jnp.any(take[i])
            s["key_holder"] = s["key_holder"].at[q].set(jnp.where(taking,i,s["key_holder"][q]))
            dropping = (a[i]==8)&jnp.any(holds[i])
            s["key_holder"] = s["key_holder"].at[h].set(jnp.where(dropping,-1,s["key_holder"][h]))
            s["key_pos"] = s["key_pos"].at[h].set(jnp.where(dropping,s["pos"][i],s["key_pos"][h]))
            recipients = passing[i] & (a==0)
            j = jnp.argmin(jnp.where(recipients,priority,4))
            passing_now = (a[i]==9)&jnp.any(recipients)
            s["key_holder"] = s["key_holder"].at[h].set(jnp.where(passing_now,j,s["key_holder"][h]))
            claiming = (a[i]==7)&claim[i]
            g = jnp.argmax(goals[i])
            s["delivered"] = s["delivered"].at[i].set(s["delivered"][i]|claiming)
            s["claim_event"] = s["claim_event"].at[i].set(claiming)
            r = r.at[i].add(claiming.astype(jnp.float32)*
                (1.0+(s["preference"][i]==g).astype(jnp.float32)))
        # Existing timers mature after movement, so opening cannot aid that movement.
        due = (state["pending"]>=0)&(state["timer"]==1)
        s["phase"] = jnp.where(due,state["pending"],state["phase"])
        s["pending"] = jnp.where(due,-1,state["pending"])
        s["timer"] = jnp.maximum(state["timer"]-1,0)
        holds,request,take,goals,claim,passing = self.options(state)
        want = jnp.any(request,axis=1)&(a==5)
        actor = jnp.argmin(jnp.where(want,priority,4))
        console = jnp.argmax(request[actor])
        accepted = jnp.any(want)
        s["pending"] = jnp.where(accepted,1-state["phase"],s["pending"])
        s["timer"] = jnp.where(accepted,2,s["timer"])
        s["switch_charge"] = s["switch_charge"].at[console].add(-accepted.astype(jnp.int32))
        s["request_event"] = s["request_event"].at[console].set(accepted)
        target = jnp.array([0,1],jnp.int32)==s["phase"]
        occupied = jnp.any(jnp.all(s["door_pos"][:,None,:]==s["pos"][None,:,:],axis=-1),axis=1)
        s["door_target"] = target
        s["door_open"] = target | occupied
        s["door_jammed"] = ~target & occupied
        s["gate_event"] = s["door_open"] != state["door_open"]
        s["key_pos"] = jnp.where((s["key_holder"]>=0)[:,None],
            s["pos"][jnp.maximum(s["key_holder"],0)],s["key_pos"])
        r = r + 0.25*jnp.sum(s["claim_event"].astype(jnp.float32))
        s["t"] = state["t"]+1
        s["natural"] = jnp.all(s["delivered"])
        s["done"] = s["natural"]|(s["t"]>=self.max_steps)
        s = jax.tree.map(lambda new,old:jnp.where(state["done"],old,new),s,state)
        r = jnp.where(state["done"],jnp.zeros(3,jnp.float32),r)
        terminated = s["natural"]
        truncated = (s["t"]>=self.max_steps)&~terminated
        done = terminated|truncated
        return self.observe(s),s,r,jnp.full((3,),done,jnp.bool_),{
            "done_all":done,"terminated":terminated,"truncated":truncated}
