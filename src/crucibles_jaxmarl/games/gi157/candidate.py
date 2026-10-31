"""Ghostlight Tag: beacon wards, movable jamming decoys and rescues; unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 171
    num_actions = 14
    max_steps = 128

    def refresh(self, s):
        s = dict(s)
        team = s["team"]
        distance = jnp.sum(jnp.abs(s["agent_pos"][:,None,:]-s["agent_pos"][None,:,:]),axis=2)
        beacon_distance = jnp.sum(jnp.abs(s["agent_pos"][:,None,:]-s["beacon_pos"][None,:,:]),axis=2)
        radar = jnp.any((s["beacon_owner"][None,None,:] == team[:,None,None]) & (s["beacon_charge"] >= 2)[None,None,:] & (beacon_distance <= 4)[None,:,:],axis=2)
        s["decoy_active"] = s["decoy_ttl"] > 0
        decoy_distance = jnp.sum(jnp.abs(s["agent_pos"][:,None,:]-s["decoy_pos"][None,:,:]),axis=2)
        jammed = jnp.any((s["decoy_team"][None,None,:] != team[:,None,None]) & s["decoy_active"][None,None,:] & (decoy_distance <= 2)[None,:,:],axis=2)
        s["agent_visible"] = (team[:,None] == team[None,:]) | (distance <= 1) | (((distance <= 2) | radar) & ~jammed)
        s["decoy_visible"] = ((team[:,None] == s["decoy_team"][None,:]) | (decoy_distance <= 2)) & s["decoy_active"][None,:]
        s["captured_count"] = jnp.array([jnp.sum(s["captured"][:2],dtype=jnp.int32),jnp.sum(s["captured"][2:],dtype=jnp.int32)],jnp.int32)
        return s

    def reset(self, key):
        board = jnp.array([[1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,1,0,0,0,0,1],[1,0,0,2,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1],[1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1],[1,0,0,0,0,0,0,2,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1],[1,1,1,1,1,1,1,1,1,1,1]],jnp.int32)
        s = {"t":jnp.array(0,jnp.int32),"episode_done":jnp.array(False),"terminated":jnp.array(False),"truncated":jnp.array(False),
             "map":board,"agent_pos":jnp.array([[2,2],[2,6],[8,2],[8,6]],jnp.int32),"agent_active":jnp.ones(4,jnp.bool_),"agent_visible":jnp.zeros((4,4),jnp.bool_),
             "team":jnp.array([0,0,1,1],jnp.int32),"stamina":jnp.full(4,6,jnp.int32),"stun":jnp.zeros(4,jnp.int32),"captured":jnp.zeros(4,jnp.bool_),"facing":jnp.array([2,2,4,4],jnp.int32),
             "beacon_pos":jnp.array([[3,2],[7,6]],jnp.int32),"beacon_active":jnp.ones(2,jnp.bool_),"beacon_visible":jnp.ones((4,2),jnp.bool_),"beacon_owner":jnp.full(2,-1,jnp.int32),"beacon_charge":jnp.zeros(2,jnp.int32),
             "decoy_pos":jnp.array([[1,1],[9,7]],jnp.int32),"decoy_active":jnp.zeros(2,jnp.bool_),"decoy_visible":jnp.zeros((4,2),jnp.bool_),"decoy_team":jnp.array([0,1],jnp.int32),"decoy_ttl":jnp.zeros(2,jnp.int32),
             "score":jnp.zeros(2,jnp.int32),"captured_count":jnp.zeros(2,jnp.int32),"winner":jnp.array(-1,jnp.int32),"capture_event":jnp.zeros(4,jnp.bool_),"rescue_event":jnp.zeros(4,jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s),s

    def legal_actions(self, s):
        d = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        p = jnp.clip(s["agent_pos"][:,None,:]+d[None,:,:],jnp.array([0,0],jnp.int32),jnp.array([10,8],jnp.int32))
        mask = jnp.ones((4,14),jnp.bool_).at[:,:5].set((s["map"][p[:,:,1],p[:,:,0]] != 1) & (s["stamina"] >= 1)[:,None]).at[:,0].set(True)
        mask = mask.at[:,5].set(s["stamina"] >= 2).at[:,6].set(s["stamina"] >= 1)
        mask = mask.at[:,7].set((s["stamina"] >= 2) & (s["decoy_ttl"][s["team"]] == 0))
        mask = mask.at[:,8:12].set(jnp.broadcast_to(((s["stamina"] >= 1) & (s["decoy_ttl"][s["team"]] > 0))[:,None],(4,4)))
        mask = mask.at[:,12].set(s["stamina"] >= 2).at[:,13].set(s["stamina"] >= 1)
        disabled = s["captured"] | (s["stun"] > 0) | s["episode_done"]
        return mask & (~disabled[:,None] | (jnp.arange(14)[None,:] == 0))

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions,0,13)
        mask = self.legal_actions(state)
        a = jnp.where((actions >= 0) & (actions < 14) & mask[jnp.arange(4),safe],safe,0)
        ids = jnp.arange(4,dtype=jnp.int32)
        rank = jnp.mod(ids-state["t"],4)
        direction = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        s = dict(state)
        base_cost = jnp.where((a >= 1) & (a <= 4),1,jnp.where((a == 5) | (a == 12),2,jnp.where((a == 6) | (a == 13),1,0))).astype(jnp.int32)
        s["stamina"] = jnp.clip(state["stamina"]-base_cost+2*(a == 0).astype(jnp.int32),0,6)
        s["stun"] = jnp.maximum(state["stun"]-1,0)
        s["facing"] = jnp.where((a >= 1) & (a <= 4),a,state["facing"]).astype(jnp.int32)
        s["decoy_ttl"] = jnp.maximum(state["decoy_ttl"]-1,0)
        # Each team owns one real movable jammer; competing commands have rotating priority.
        for offset in range(2):
            g = jnp.mod(state["t"]+offset,2)
            eligible = (s["team"] == g) & (a >= 7) & (a <= 11)
            i = jnp.argmin(jnp.where(eligible,rank,99))
            deploy = (a[i] == 7) & (s["decoy_ttl"][g] == 0)
            steer = (a[i] >= 8) & (a[i] <= 11) & (s["decoy_ttl"][g] > 0)
            d = direction[jnp.clip(jnp.where(deploy,state["facing"][i],a[i]-7),0,4)]
            origin = jnp.where(deploy,s["agent_pos"][i],s["decoy_pos"][g])
            target = origin+d
            clipped = jnp.clip(target,jnp.array([0,0],jnp.int32),jnp.array([10,8],jnp.int32))
            blocked = jnp.any(jnp.all(target == s["agent_pos"],axis=1)) | ((s["decoy_ttl"][1-g] > 0) & jnp.all(target == s["decoy_pos"][1-g]))
            operate = jnp.any(eligible) & (deploy | steer) & jnp.all(target == clipped) & (s["map"][clipped[1],clipped[0]] != 1) & ~blocked
            s["decoy_pos"] = s["decoy_pos"].at[g].set(jnp.where(operate,target,s["decoy_pos"][g]))
            s["decoy_ttl"] = s["decoy_ttl"].at[g].set(jnp.where(operate & deploy,6,s["decoy_ttl"][g]))
            s["stamina"] = s["stamina"].at[i].add(-jnp.where(operate,jnp.where(deploy,2,1),0).astype(jnp.int32))
        # An attempted enemy-decoy entry consumes the decoy and stuns without advancing.
        proposal = s["agent_pos"]+direction[jnp.clip(a,0,4)]*((a >= 1) & (a <= 4))[:,None]
        proposal = jnp.clip(proposal,jnp.array([0,0],jnp.int32),jnp.array([10,8],jnp.int32))
        occupied = jnp.any(jnp.all(proposal[:,None,:] == state["agent_pos"][None,:,:],axis=2) & (ids[:,None] != ids[None,:]),axis=1)
        can = (a >= 1) & (a <= 4) & ~occupied & (s["map"][proposal[:,1],proposal[:,0]] != 1)
        loser = jnp.any(jnp.all(proposal[:,None,:] == proposal[None,:,:],axis=2) & can[None,:] & (rank[None,:] < rank[:,None]),axis=1)
        chosen = can & ~loser
        decoy_hit = chosen[:,None] & jnp.all(proposal[:,None,:] == s["decoy_pos"][None,:,:],axis=2) & (s["team"][:,None] != s["decoy_team"][None,:]) & (s["decoy_ttl"] > 0)[None,:]
        trapped = jnp.any(decoy_hit,axis=1)
        s["agent_pos"] = jnp.where((chosen & ~trapped)[:,None],proposal,s["agent_pos"])
        s["stun"] = jnp.where(trapped,1,s["stun"]).astype(jnp.int32)
        s["stamina"] = jnp.maximum(s["stamina"]-trapped.astype(jnp.int32),0)
        s["decoy_ttl"] = jnp.where(jnp.any(decoy_hit,axis=0),0,s["decoy_ttl"]).astype(jnp.int32)
        # A charged beacon supplies team radar and consumable capture wards; opponents can drain it.
        for offset in range(4):
            i = jnp.mod(state["t"]+offset,4)
            bd = jnp.sum(jnp.abs(s["beacon_pos"]-s["agent_pos"][i]),axis=1)
            b = jnp.argmin(bd)
            channel = (a[i] == 6) & (bd[b] <= 1)
            friendly = (s["beacon_owner"][b] < 0) | (s["beacon_owner"][b] == s["team"][i])
            charge = jnp.where(friendly,jnp.minimum(4,s["beacon_charge"][b]+1),jnp.maximum(0,s["beacon_charge"][b]-1)).astype(jnp.int32)
            owner = jnp.where(friendly,s["team"][i],jnp.where(charge == 0,-1,s["beacon_owner"][b])).astype(jnp.int32)
            s["beacon_charge"] = s["beacon_charge"].at[b].set(jnp.where(channel,charge,s["beacon_charge"][b]))
            s["beacon_owner"] = s["beacon_owner"].at[b].set(jnp.where(channel,owner,s["beacon_owner"][b]))
        distance = jnp.sum(jnp.abs(s["agent_pos"][:,None,:]-s["agent_pos"][None,:,:]),axis=2)
        attacks = (a == 5)[:,None] & (s["team"][:,None] != s["team"][None,:]) & (distance == 1) & ~state["captured"][None,:]
        target = jnp.argmin(jnp.where(attacks,rank[None,:],99),axis=1)
        aimed = attacks & (ids[None,:] == target[:,None])
        pending = jnp.any(aimed,axis=0)
        cover = jnp.any((a == 13)[:,None] & (s["team"][:,None] == s["team"][None,:]) & (ids[:,None] != ids[None,:]) & (distance == 1),axis=0)
        hits = pending & ~cover
        for offset in range(4):
            victim = jnp.mod(state["t"]+offset,4)
            bd = jnp.sum(jnp.abs(s["beacon_pos"]-s["agent_pos"][victim]),axis=1)
            dd = jnp.sum(jnp.abs(s["decoy_pos"]-s["agent_pos"][victim]),axis=1)
            jam = jnp.any((s["decoy_ttl"] > 0) & (s["decoy_team"] != s["team"][victim]) & (dd <= 2))
            ward = (s["beacon_owner"] == s["team"][victim]) & (s["beacon_charge"] > 0) & (bd <= 2) & ~jam
            b = jnp.argmin(jnp.where(ward,bd*3+jnp.arange(2),99))
            protect = hits[victim] & jnp.any(ward)
            s["beacon_charge"] = s["beacon_charge"].at[b].add(-protect.astype(jnp.int32))
            hits = hits.at[victim].set(hits[victim] & ~protect)
        # Capture decisions are simultaneous; a newly captured rescuer cannot free a teammate.
        rescue = jnp.zeros(4,jnp.bool_)
        mate = jnp.array([1,0,3,2],jnp.int32)
        for i in range(4):
            j = mate[i]
            free = (a[i] == 12) & ~state["captured"][i] & ~hits[i] & state["captured"][j] & (distance[i,j] == 1)
            rescue = rescue.at[j].set(rescue[j] | free)
        s["captured"] = (state["captured"] | hits) & ~rescue
        s["capture_event"] = hits
        s["rescue_event"] = rescue
        gain = jnp.array([jnp.sum(hits & (s["team"] == 1),dtype=jnp.int32),jnp.sum(hits & (s["team"] == 0),dtype=jnp.int32)],jnp.int32)
        s["score"] = state["score"]+gain
        down0 = jnp.all(s["captured"][:2])
        down1 = jnp.all(s["captured"][2:])
        s["winner"] = jnp.where(down0 & ~down1,1,jnp.where(down1 & ~down0,0,-1)).astype(jnp.int32)
        s["t"] = state["t"]+jnp.array(1,jnp.int32)
        s["terminated"] = down0 | down1
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        reward = (gain[s["team"]]-gain[1-s["team"]]).astype(jnp.float32)
        s = jax.tree.map(lambda old,new:jnp.where(state["episode_done"],old,new),state,s)
        reward = jnp.where(state["episode_done"],jnp.zeros(4,jnp.float32),reward)
        return self.observe(s),s,reward,jnp.full(4,s["episode_done"],jnp.bool_),{"done_all":s["episode_done"],"terminated":s["terminated"],"truncated":s["truncated"]}

    def observe(self, s):
        core = jnp.stack([jnp.arange(4),s["team"],s["agent_pos"][:,0],s["agent_pos"][:,1],s["stamina"],s["stun"],s["captured"],jnp.full(4,s["t"])],axis=1)
        public = jnp.concatenate([s["score"],s["captured_count"],jnp.array([s["winner"],s["episode_done"]])])
        av = s["agent_visible"]
        ad = jnp.concatenate([s["agent_pos"],s["team"][:,None],s["stamina"][:,None],s["stun"][:,None],s["captured"][:,None],s["facing"][:,None]],axis=1)
        agents = jnp.concatenate([jnp.where(av[:,:,None],ad[None,:,:],-1),av[:,:,None]],axis=2).reshape(4,32)
        beacons = jnp.concatenate([s["beacon_pos"],s["beacon_owner"][:,None],s["beacon_charge"][:,None]],axis=1).reshape(8)
        dv = s["decoy_visible"]
        dd = jnp.concatenate([s["decoy_pos"],s["decoy_team"][:,None],s["decoy_ttl"][:,None]],axis=1)
        decoys = jnp.concatenate([jnp.where(dv[:,:,None],dd[None,:,:],-1),dv[:,:,None]],axis=2).reshape(4,10)
        events = jnp.concatenate([s["capture_event"],s["rescue_event"]])
        return jnp.concatenate([core,jnp.broadcast_to(public[None,:],(4,6)),agents,jnp.broadcast_to(beacons[None,:],(4,8)),decoys,jnp.broadcast_to(events[None,:],(4,8)),jnp.broadcast_to(s["map"].reshape(1,99),(4,99))],axis=1).astype(jnp.float32)
