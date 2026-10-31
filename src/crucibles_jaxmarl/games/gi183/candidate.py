"""Mist Runners. Original source-only authoring; no runtime claims."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 162
    num_actions = 11
    max_steps = 112

    def reset(self,key):
        s = {
            "t": jnp.array(0,jnp.int32),
            "done": jnp.array(False), "natural": jnp.array(False),
            "map": jnp.array([[1,1,1,1,1,1,1,1,1],[1,0,0,0,1,0,0,0,1],
                [1,0,0,0,0,0,0,0,1],[1,0,0,0,1,0,0,0,1],
                [1,0,0,0,0,0,0,0,1],[1,0,0,0,1,0,0,0,1],
                [1,1,1,1,1,1,1,1,1]],jnp.int32),
            "pos": jnp.array([[2,2],[2,4],[6,2],[6,4]],jnp.int32),
            "active": jnp.ones(4,jnp.bool_),
            "agent_visible": jnp.zeros((4,4),jnp.bool_),
            "team": jnp.array([0,0,1,1],jnp.int32),
            "cargo": jnp.full(4,-1,jnp.int32),
            "charge": jnp.full(4,2,jnp.int32),
            "frozen": jnp.zeros(4,jnp.int32), "cooldown": jnp.zeros(4,jnp.int32),
            "packet_pos": jnp.array([[1,3],[7,3]],jnp.int32),
            "packet_holder": jnp.full(2,-1,jnp.int32),
            "packet_team": jnp.array([0,1],jnp.int32),
            "packet_active": jnp.ones(2,jnp.bool_),
            "packet_visible": jnp.zeros((4,2),jnp.bool_),
            "delivered": jnp.zeros(2,jnp.bool_),
            "scan_pos": jnp.array([[2,2],[2,4],[6,2],[6,4]],jnp.int32),
            "scan_life": jnp.zeros(4,jnp.int32),
            "scan_active": jnp.zeros(4,jnp.bool_),
            "scan_visible": jnp.zeros((4,4),jnp.bool_),
            "decoy_pos": jnp.array([[2,2],[2,4],[6,2],[6,4]],jnp.int32),
            "decoy_life": jnp.zeros(4,jnp.int32),
            "decoy_active": jnp.zeros(4,jnp.bool_),
            "decoy_visible": jnp.zeros((4,4),jnp.bool_),
            "base_pos": jnp.array([[1,3],[7,3]],jnp.int32),
            "base_active": jnp.ones(2,jnp.bool_),
            "base_visible": jnp.ones((4,2),jnp.bool_),
            "base_team": jnp.array([0,1],jnp.int32),
            "report_pos": jnp.array([[2,4],[2,2],[6,4],[6,2]],jnp.int32),
            "report_age": jnp.zeros(4,jnp.int32),
            "report_cargo": jnp.full(4,-1,jnp.int32),
            "report_active": jnp.ones(4,jnp.bool_),
            "report_visible": jnp.eye(4,dtype=jnp.bool_),
            "delivery_event": jnp.zeros(2,jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s),s

    def carried(self,s):
        held = s["packet_holder"][None,:] == jnp.arange(4,dtype=jnp.int32)[:,None]
        return jnp.sum(held.astype(jnp.int32)*jnp.array([1,2],jnp.int32)[None,:],axis=1,dtype=jnp.int32)-1

    def seen(self,s,points,owner):
        local = jnp.sum(jnp.abs(s["pos"][:,None,:]-points[None,:,:]),axis=-1)<=2
        distance = jnp.sum(jnp.abs(s["scan_pos"][:,None,:]-points[None,:,:]),axis=-1)
        own_scan = s["team"][:,None,None] == s["team"][None,:,None]
        warm = (s["scan_life"]>0)&(s["scan_life"]<4)
        remote = jnp.any(own_scan & warm[None,:,None] & (distance[None,:,:]<=4),axis=1)
        decoy_distance = jnp.sum(jnp.abs(s["decoy_pos"][:,None,:]-points[None,:,:]),axis=-1)
        shield = jnp.any((s["decoy_life"][:,None]>0)&(decoy_distance<=2)&
                        (s["team"][:,None]==owner[None,:]),axis=0)
        jammed = shield[None,:] & (s["team"][:,None]!=owner[None,:])
        return local | (remote & ~jammed)

    def refresh(self,state):
        s = dict(state)
        s["cargo"] = self.carried(s)
        held = s["packet_holder"]>=0
        s["packet_pos"] = jnp.where(held[:,None],s["pos"][jnp.maximum(s["packet_holder"],0)],s["packet_pos"])
        s["packet_active"] = ~s["delivered"]
        s["scan_active"] = s["scan_life"]>0
        s["decoy_active"] = s["decoy_life"]>0
        team_link = s["team"][:,None]==s["team"][None,:]
        s["agent_visible"] = self.seen(s,s["pos"],s["team"])
        s["packet_visible"] = self.seen(s,s["packet_pos"],s["packet_team"]) & s["packet_active"][None,:]
        s["scan_visible"] = (self.seen(s,s["scan_pos"],s["team"])|team_link)&s["scan_active"][None,:]
        # The decoy transmitter is identifiable to scanners; its protected agents are not.
        s["decoy_visible"] = (self.seen(s,s["decoy_pos"],jnp.full(4,-1,jnp.int32))|team_link)&s["decoy_active"][None,:]
        return s

    def movement(self,s):
        d = jnp.array([[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        p = s["pos"][:,None,:]+d[None,:,:]
        inside = (p[:,:,0]>=0)&(p[:,:,0]<=8)&(p[:,:,1]>=0)&(p[:,:,1]<=6)
        tile = s["map"][jnp.clip(p[:,:,1],0,6),jnp.clip(p[:,:,0],0,8)]
        blocked = jnp.any(jnp.all(p[:,:,None,:]==s["pos"][None,None,:,:],axis=-1),axis=-1)
        return inside & (tile==0)&~blocked

    def tag_targets(self,s):
        close = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["pos"][None,:,:]),axis=-1)==1
        enemy = s["team"][:,None]!=s["team"][None,:]
        safe = jnp.all(s["pos"]==s["base_pos"][s["team"]],axis=1)
        return close & enemy & ~safe[None,:]

    def options(self,s):
        cargo = self.carried(s)
        pickup = jnp.all(s["pos"][:,None,:]==s["packet_pos"][None,:,:],axis=-1)
        pickup = pickup & (s["packet_holder"][None,:]<0)&(~s["delivered"][None,:])
        pickup = pickup & (s["team"][:,None]==s["packet_team"][None,:])&(cargo[:,None]<0)
        close = jnp.sum(jnp.abs(s["pos"][:,None,:]-s["pos"][None,:,:]),axis=-1)==1
        passing = close & (s["team"][:,None]==s["team"][None,:]) & (cargo[:,None]>=0)&(cargo[None,:]<0)
        destination = jnp.all(s["pos"]==s["base_pos"][1-s["team"]],axis=1)
        delivery = destination & (cargo>=0)
        return pickup,passing,delivery

    def legal_actions(self,s):
        pickup,passing,delivery = self.options(s)
        scan = (s["charge"]>0)&(s["scan_life"]==0)
        decoy = (s["charge"]>0)&(s["decoy_life"]==0)
        tag = (s["cooldown"]==0)&jnp.any(self.tag_targets(s),axis=1)
        m = jnp.concatenate((jnp.ones((4,1),jnp.bool_),self.movement(s),
            jnp.stack((jnp.any(pickup,1),scan,decoy,tag,jnp.any(passing,1),delivery),axis=1)),axis=1)
        mobile = (s["frozen"]==0)&~s["done"]
        return m & (mobile[:,None] | (jnp.arange(11)[None,:]==0))

    def observe(self,s):
        own = jnp.concatenate((s["pos"],s["charge"][:,None],s["frozen"][:,None],
            s["cooldown"][:,None],s["cargo"][:,None],jnp.arange(4,dtype=jnp.int32)[:,None],
            s["team"][:,None]),axis=1)
        report = jnp.concatenate((s["report_pos"],s["report_age"][:,None],s["report_cargo"][:,None]),axis=1)
        public = jnp.concatenate((s["map"].reshape(-1),s["base_pos"].reshape(-1),
            s["delivered"].astype(jnp.int32),s["t"].reshape(1)))
        agents = jnp.concatenate((s["pos"],s["team"][:,None],s["frozen"][:,None],
            s["cooldown"][:,None],s["cargo"][:,None]),axis=1)
        av = s["agent_visible"]
        agents = jnp.concatenate((av[:,:,None].astype(jnp.int32),
            jnp.where(av[:,:,None],agents[None,:,:],0)),axis=2).reshape(4,28)
        packets = jnp.concatenate((s["packet_pos"],s["packet_holder"][:,None],s["packet_team"][:,None],
            s["packet_active"][:,None].astype(jnp.int32)),axis=1)
        pv = s["packet_visible"]
        packets = jnp.concatenate((pv[:,:,None].astype(jnp.int32),
            jnp.where(pv[:,:,None],packets[None,:,:],0)),axis=2).reshape(4,12)
        scans = jnp.concatenate((s["scan_pos"],s["scan_life"][:,None],s["team"][:,None]),axis=1)
        sv = s["scan_visible"]
        scans = jnp.concatenate((sv[:,:,None].astype(jnp.int32),
            jnp.where(sv[:,:,None],scans[None,:,:],0)),axis=2).reshape(4,20)
        decoys = jnp.concatenate((s["decoy_pos"],s["decoy_life"][:,None],s["team"][:,None]),axis=1)
        dv = s["decoy_visible"]
        decoys = jnp.concatenate((dv[:,:,None].astype(jnp.int32),
            jnp.where(dv[:,:,None],decoys[None,:,:],0)),axis=2).reshape(4,20)
        return jnp.concatenate((own,report,jnp.broadcast_to(public,(4,70)),agents,packets,scans,decoys),axis=1).astype(jnp.float32)

    def raw_step(self,key,state,actions):
        legal = self.legal_actions(state)
        a = jnp.clip(actions,0,10)
        valid = (actions>=0)&(actions<11)&jnp.take_along_axis(legal,a[:,None],axis=1)[:,0]
        a = jnp.where(valid,a,0)
        s = dict(state)
        ids = jnp.arange(4,dtype=jnp.int32)
        priority = jnp.mod(ids-state["t"],4)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        proposed = state["pos"]+delta[jnp.where(a<=4,a,0)]
        same = jnp.all(proposed[:,None,:]==proposed[None,:,:],axis=-1)
        loses = jnp.any(same & (priority[None,:]<priority[:,None]),axis=1)
        s["pos"] = jnp.where(loses[:,None],state["pos"],proposed)
        scanning = a==6
        decoying = a==7
        s["charge"] = s["charge"]-(scanning|decoying).astype(jnp.int32)
        s["scan_pos"] = jnp.where(scanning[:,None],s["pos"],s["scan_pos"])
        s["scan_life"] = jnp.where(scanning,4,jnp.maximum(s["scan_life"]-1,0))
        s["decoy_pos"] = jnp.where(decoying[:,None],s["pos"],s["decoy_pos"])
        s["decoy_life"] = jnp.where(decoying,4,jnp.maximum(s["decoy_life"]-1,0))
        for k in range(4):
            i = jnp.mod(state["t"]+k,4)
            pickup,passing,delivery = self.options(s)
            p = s["team"][i]
            taking = (a[i]==5)&pickup[i,p]
            s["packet_holder"] = s["packet_holder"].at[p].set(jnp.where(taking,i,s["packet_holder"][p]))
            recipient = passing[i]&(a==0)&(state["frozen"]==0)
            j = jnp.argmin(jnp.where(recipient,priority,5))
            handoff = (a[i]==9)&jnp.any(recipient)
            s["packet_holder"] = s["packet_holder"].at[p].set(jnp.where(handoff,j,s["packet_holder"][p]))
        # All tag requests are evaluated together; being tagged does not cancel a simultaneous tag.
        targets = self.tag_targets(s)
        target = jnp.argmin(jnp.where(targets,priority[None,:],5),axis=1)
        tags = (a==8)&jnp.any(targets,axis=1)
        hit = jnp.any(tags[:,None]&(target[:,None]==ids[None,:]),axis=0)
        s["frozen"] = jnp.where(hit,2,jnp.maximum(state["frozen"]-1,0))
        s["cooldown"] = jnp.where(tags,3,jnp.maximum(state["cooldown"]-1,0))
        holder = s["packet_holder"]
        s["packet_pos"] = jnp.where((holder>=0)[:,None],s["pos"][jnp.maximum(holder,0)],s["packet_pos"])
        dropped = (holder>=0)&hit[jnp.maximum(holder,0)]
        s["packet_holder"] = jnp.where(dropped,-1,holder)
        pickup,passing,delivery = self.options(s)
        sending = (a==10)&delivery&(s["frozen"]==0)
        sent = jnp.any(sending[:,None]&(s["team"][:,None]==jnp.array([0,1],jnp.int32)[None,:]),axis=0)
        s["delivery_event"] = sent
        s["delivered"] = s["delivered"]|sent
        s["packet_holder"] = jnp.where(sent,-1,s["packet_holder"])
        s["t"] = state["t"]+1
        peers = jnp.array([1,0,3,2],jnp.int32)
        radio_tick = jnp.mod(s["t"],2)==0
        s["report_pos"] = jnp.where(radio_tick,state["pos"][peers],state["report_pos"])
        s["report_cargo"] = jnp.where(radio_tick,state["cargo"][peers],state["report_cargo"])
        s["report_age"] = jnp.where(radio_tick,1,jnp.minimum(state["report_age"]+1,2))
        s = self.refresh(s)
        r = sent[s["team"]].astype(jnp.float32)-0.5*sent[1-s["team"]].astype(jnp.float32)
        s["natural"] = jnp.all(s["delivered"])
        s["done"] = s["natural"]|(s["t"]>=self.max_steps)
        s = jax.tree.map(lambda new,old:jnp.where(state["done"],old,new),s,state)
        r = jnp.where(state["done"],jnp.zeros(4,jnp.float32),r)
        terminated = s["natural"]
        truncated = (s["t"]>=self.max_steps)&~terminated
        done = terminated|truncated
        return self.observe(s),s,r,jnp.full((4,),done,jnp.bool_),{
            "done_all":done,"terminated":terminated,"truncated":truncated}
