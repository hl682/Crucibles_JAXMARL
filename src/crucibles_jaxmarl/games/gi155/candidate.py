"""Echo Tow: asymmetric sonar and synchronized salvage, authored but not executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 165
    num_actions = 14
    max_steps = 144

    def refresh(self, s):
        s = dict(s)
        s["sight"] = jnp.array([jnp.where(s["ping"] > 0,6,4),2,2,2],jnp.int32)
        s["agent_visible"] = jnp.sum(jnp.abs(s["agent_pos"][:,None,:]-s["agent_pos"][None,:,:]),axis=2) <= s["sight"][:,None]
        s["cargo_active"] = ~s["salvaged"]
        s["cargo_visible"] = (jnp.sum(jnp.abs(s["agent_pos"][:,None,:]-s["cargo_pos"][None,:,:]),axis=2) <= s["sight"][:,None]) & s["cargo_active"][None,:]
        s["current_visible"] = jnp.sum(jnp.abs(s["agent_pos"][:,None,:]-s["current_pos"][None,:,:]),axis=2) <= s["sight"][:,None]
        s["flowing"] = s["divert_ttl"] == 0
        s["marker_active"] = s["marker_ttl"] > 0
        s["marker_visible"] = jnp.broadcast_to(s["marker_active"][None,:],(4,1))
        return s

    def reset(self, key):
        board = jnp.array([[1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,1],[1,0,2,0,0,0,0,0,0,0,1],
            [1,0,3,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,0,1],
            [1,0,3,0,0,0,0,0,0,0,1],[1,0,2,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1,1,1,1]],jnp.int32)
        second_x = jax.random.randint(key,(),7,9,dtype=jnp.int32)
        s = {"t":jnp.array(0,jnp.int32),"episode_done":jnp.array(False),"terminated":jnp.array(False),"truncated":jnp.array(False),"abort":jnp.array(False),
             "map":board,"agent_pos":jnp.array([[4,2],[5,3],[7,3],[4,4]],jnp.int32),"agent_active":jnp.ones(4,jnp.bool_),"agent_visible":jnp.ones((4,4),jnp.bool_),
             "oxygen":jnp.array([24,20,20,24],jnp.int32),"hook":jnp.full(4,-1,jnp.int32),"ping":jnp.array(0,jnp.int32),"sight":jnp.array([4,2,2,2],jnp.int32),
             "cargo_pos":jnp.array([[6,3],[second_x,6]],jnp.int32),"cargo_active":jnp.ones(2,jnp.bool_),"cargo_visible":jnp.zeros((4,2),jnp.bool_),"salvaged":jnp.zeros(2,jnp.bool_),
             "current_pos":jnp.array([[4,3],[6,5]],jnp.int32),"current_active":jnp.ones(2,jnp.bool_),"current_visible":jnp.zeros((4,2),jnp.bool_),"divert_ttl":jnp.zeros(2,jnp.int32),"flowing":jnp.ones(2,jnp.bool_),
             "dock_pos":jnp.array([[2,2],[2,6]],jnp.int32),"dock_active":jnp.ones(2,jnp.bool_),"dock_visible":jnp.ones((4,2),jnp.bool_),"dock_stock":jnp.full(2,18,jnp.int32),
             "berth_pos":jnp.array([[2,3],[2,5]],jnp.int32),"berth_active":jnp.ones(2,jnp.bool_),"berth_visible":jnp.ones((4,2),jnp.bool_),
             "marker_pos":jnp.array([[1,1]],jnp.int32),"marker_active":jnp.zeros(1,jnp.bool_),"marker_visible":jnp.zeros((4,1),jnp.bool_),"marker_kind":jnp.zeros(1,jnp.int32),"marker_ref":jnp.zeros(1,jnp.int32),"marker_age":jnp.zeros(1,jnp.int32),"marker_ttl":jnp.zeros(1,jnp.int32),
             "salvage_event":jnp.zeros(2,jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s),s

    def legal_actions(self, s):
        ids = jnp.arange(4,dtype=jnp.int32)
        d = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        p = jnp.clip(s["agent_pos"][:,None,:]+d[None,:,:],jnp.array([0,0],jnp.int32),jnp.array([10,8],jnp.int32))
        mask = jnp.ones((4,14),jnp.bool_).at[:,:5].set(s["map"][p[:,:,1],p[:,:,0]] != 1).at[:,0].set(True)
        mask = mask.at[:,5].set((ids == 1) | (ids == 2))
        mask = mask.at[:,6].set((ids == 0) & (s["oxygen"] >= 4))
        mask = mask.at[:,7].set((ids == 0) & (s["oxygen"] >= 3))
        mask = mask.at[:,8].set((ids == 3) & (s["oxygen"] >= 4))
        mask = mask.at[:,10:14].set((ids[:,None] != ids[None,:]) & (s["oxygen"] >= 5)[:,None])
        return mask & (~s["episode_done"] | (jnp.arange(14)[None,:] == 0))

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions,0,13)
        mask = self.legal_actions(state)
        a = jnp.where((actions >= 0) & (actions < 14) & mask[jnp.arange(4),safe],safe,0)
        ids = jnp.arange(4,dtype=jnp.int32)
        s = dict(state)
        extra = jnp.zeros(4,jnp.int32)
        s["ping"] = jnp.where(a[0] == 6,3,jnp.maximum(state["ping"]-1,0)).astype(jnp.int32)
        extra = extra.at[0].add(2*(a[0] == 6).astype(jnp.int32))
        s["divert_ttl"] = jnp.maximum(state["divert_ttl"]-1,0)
        distance = jnp.sum(jnp.abs(s["current_pos"]-s["agent_pos"][3]),axis=1)
        c = jnp.argmin(distance*3+jnp.arange(2))
        divert = (a[3] == 8) & (distance[c] <= 1)
        s["divert_ttl"] = s["divert_ttl"].at[c].set(jnp.where(divert,4,s["divert_ttl"][c]))
        extra = extra.at[3].add(2*divert.astype(jnp.int32))
        # Only the two tugs have fittings; each must occupy an opposite horizontal flank.
        for i in range(1,3):
            offset = s["agent_pos"][i]-s["cargo_pos"]
            eligible = (jnp.abs(offset[:,0]) == 1) & (offset[:,1] == 0) & ~s["salvaged"]
            p = jnp.argmax(eligible)
            attach = (a[i] == 5) & jnp.any(eligible) & (state["hook"][i] < 0)
            detach = (a[i] == 5) & (state["hook"][i] >= 0)
            s["hook"] = s["hook"].at[i].set(jnp.where(detach,-1,jnp.where(attach,p,state["hook"][i])).astype(jnp.int32))
        p = jnp.clip(s["hook"][1],0,1)
        relative = s["agent_pos"][1:3]-s["cargo_pos"][p]
        aligned = jnp.all(relative[:,1] == 0) & jnp.all(jnp.abs(relative[:,0]) == 1) & (relative[0,0] != relative[1,0])
        together = (s["hook"][1] >= 0) & (s["hook"][2] == s["hook"][1]) & aligned & (a[1] == a[2]) & (a[1] >= 1) & (a[1] <= 4) & jnp.all(s["oxygen"][1:3] >= 3)
        directions = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        step = directions[jnp.clip(a[1],0,4)]
        body = jnp.concatenate([s["cargo_pos"][p][None,:],s["agent_pos"][1:3]],axis=0)+step
        clipped = jnp.clip(body,jnp.array([0,0],jnp.int32),jnp.array([10,8],jnp.int32))
        clear = jnp.all(body == clipped) & jnp.all(s["map"][clipped[:,1],clipped[:,0]] != 1)
        other_crew = jnp.any(jnp.all(body[:,None,:] == s["agent_pos"][None,:,:],axis=2) & ((ids == 0) | (ids == 3))[None,:])
        other_cargo = jnp.any(jnp.all(body[:,None,:] == s["cargo_pos"][None,:,:],axis=2) & (jnp.arange(2) != p)[None,:] & ~s["salvaged"][None,:])
        current_block = jnp.any(jnp.all(body[0] == s["current_pos"],axis=1) & (s["divert_ttl"] == 0))
        tow = together & clear & ~other_crew & ~other_cargo & ~current_block
        s["cargo_pos"] = s["cargo_pos"].at[p].set(jnp.where(tow,body[0],s["cargo_pos"][p]))
        s["agent_pos"] = s["agent_pos"].at[1:3].set(jnp.where(tow,body[1:3],s["agent_pos"][1:3]))
        extra = extra.at[1:3].add(tow.astype(jnp.int32))
        # Unhooked divers move around the whole translated tow body; swaps are disallowed.
        proposal = s["agent_pos"] + directions[jnp.clip(a,0,4)]*((a >= 1) & (a <= 4))[:,None]
        proposal = jnp.clip(proposal,jnp.array([0,0],jnp.int32),jnp.array([10,8],jnp.int32))
        occupied = jnp.any((jnp.all(proposal[:,None,:] == state["agent_pos"][None,:,:],axis=2) | jnp.all(proposal[:,None,:] == s["agent_pos"][None,:,:],axis=2)) & (ids[:,None] != ids[None,:]),axis=1)
        cargo_block = jnp.any(jnp.all(proposal[:,None,:] == s["cargo_pos"][None,:,:],axis=2) & ~s["salvaged"][None,:],axis=1)
        can = (a >= 1) & (a <= 4) & (s["hook"] < 0) & ~occupied & ~cargo_block & (s["map"][proposal[:,1],proposal[:,0]] != 1)
        rank = jnp.mod(ids-state["t"],4)
        loser = jnp.any(jnp.all(proposal[:,None,:] == proposal[None,:,:],axis=2) & can[None,:] & (rank[None,:] < rank[:,None]),axis=1)
        moved = can & ~loser
        s["agent_pos"] = jnp.where(moved[:,None],proposal,s["agent_pos"])
        extra = extra + moved.astype(jnp.int32)
        # The radio publishes a true snapshot, not a continuously privileged cargo track.
        s["marker_ttl"] = jnp.maximum(state["marker_ttl"]-1,0)
        s["marker_age"] = jnp.minimum(state["marker_age"]+1,self.max_steps)
        radius = jnp.where(s["ping"] > 0,6,4)
        cd = jnp.sum(jnp.abs(s["cargo_pos"]-s["agent_pos"][0]),axis=1)
        hd = jnp.sum(jnp.abs(s["current_pos"]-s["agent_pos"][0]),axis=1)
        known_cargo = (cd <= radius) & ~s["salvaged"]
        known_current = hd <= radius
        cp = jnp.argmin(jnp.where(known_cargo,cd*3+jnp.arange(2),99))
        hp = jnp.argmin(jnp.where(known_current,hd*3+jnp.arange(2),99))
        cargo_first = jnp.any(known_cargo)
        broadcast = (a[0] == 7) & (cargo_first | jnp.any(known_current))
        point = jnp.where(cargo_first,s["cargo_pos"][cp],s["current_pos"][hp])
        s["marker_pos"] = jnp.where(broadcast,point[None,:],s["marker_pos"])
        s["marker_kind"] = jnp.where(broadcast,jnp.full(1,jnp.where(cargo_first,0,1),jnp.int32),s["marker_kind"])
        s["marker_ref"] = jnp.where(broadcast,jnp.full(1,jnp.where(cargo_first,cp,hp),jnp.int32),s["marker_ref"])
        s["marker_ttl"] = jnp.where(broadcast,jnp.full(1,5,jnp.int32),s["marker_ttl"])
        s["marker_age"] = jnp.where(broadcast,jnp.zeros(1,jnp.int32),s["marker_age"])
        extra = extra.at[0].add(broadcast.astype(jnp.int32))
        for offset in range(4):
            i = jnp.mod(state["t"]+offset,4)
            j = jnp.clip(a[i]-10,0,3)
            share = (a[i] >= 10) & (a[i] <= 13) & (i != j) & (s["oxygen"][i] >= 5) & (s["oxygen"][j] <= 21) & (jnp.sum(jnp.abs(s["agent_pos"][i]-s["agent_pos"][j])) == 1)
            s["oxygen"] = s["oxygen"].at[i].add(-3*share.astype(jnp.int32)).at[j].add(3*share.astype(jnp.int32))
        s["dock_stock"] = jnp.minimum(s["dock_stock"]+1,18)
        used = jnp.zeros(2,jnp.bool_)
        for offset in range(4):
            i = jnp.mod(state["t"]+offset,4)
            dd = jnp.sum(jnp.abs(s["dock_pos"]-s["agent_pos"][i]),axis=1)
            d = jnp.argmin(dd)
            recharge = (a[i] == 9) & (dd[d] <= 1) & ~used[d]
            amount = jnp.where(recharge,jnp.minimum(6,jnp.minimum(s["dock_stock"][d],24-s["oxygen"][i])),0).astype(jnp.int32)
            s["oxygen"] = s["oxygen"].at[i].add(amount)
            s["dock_stock"] = s["dock_stock"].at[d].add(-amount)
            used = used.at[d].set(used[d] | (amount > 0))
        drag = jnp.any(jnp.all(s["agent_pos"][:,None,:] == s["current_pos"][None,:,:],axis=2) & (s["divert_ttl"] == 0)[None,:],axis=1)
        s["oxygen"] = jnp.maximum(s["oxygen"]-1-extra-drag.astype(jnp.int32),0)
        delivered = jnp.all(s["cargo_pos"] == s["berth_pos"],axis=1) & ~state["salvaged"]
        s["salvaged"] = state["salvaged"] | delivered
        s["salvage_event"] = delivered
        s["hook"] = jnp.where((s["hook"] >= 0) & s["salvaged"][jnp.clip(s["hook"],0,1)],-1,s["hook"]).astype(jnp.int32)
        complete = jnp.all(s["salvaged"])
        s["abort"] = jnp.any(s["oxygen"] == 0) & ~complete
        s["t"] = state["t"]+jnp.array(1,jnp.int32)
        s["terminated"] = complete | s["abort"]
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        reward = jnp.full(4,-0.001+jnp.sum(delivered,dtype=jnp.int32)-0.5*s["abort"].astype(jnp.float32),jnp.float32)
        s = jax.tree.map(lambda old,new:jnp.where(state["episode_done"],old,new),state,s)
        reward = jnp.where(state["episode_done"],jnp.zeros(4,jnp.float32),reward)
        return self.observe(s),s,reward,jnp.full(4,s["episode_done"],jnp.bool_),{"done_all":s["episode_done"],"terminated":s["terminated"],"truncated":s["truncated"]}

    def observe(self, s):
        core = jnp.stack([jnp.arange(4),s["agent_pos"][:,0],s["agent_pos"][:,1],s["oxygen"],s["hook"],jnp.full(4,s["ping"]),jnp.full(4,s["t"]),jnp.full(4,s["episode_done"])],axis=1)
        public = jnp.concatenate([s["salvaged"],s["salvage_event"],jnp.array([s["abort"]])])
        av = s["agent_visible"]
        ad = jnp.concatenate([s["agent_pos"],s["oxygen"][:,None],s["hook"][:,None]],axis=1)
        agents = jnp.concatenate([jnp.where(av[:,:,None],ad[None,:,:],-1),av[:,:,None]],axis=2).reshape(4,20)
        cv = s["cargo_visible"]
        cd = jnp.concatenate([s["cargo_pos"],s["salvaged"][:,None]],axis=1)
        cargo = jnp.concatenate([jnp.where(cv[:,:,None],cd[None,:,:],-1),cv[:,:,None]],axis=2).reshape(4,8)
        hv = s["current_visible"]
        hd = jnp.concatenate([s["current_pos"],s["divert_ttl"][:,None]],axis=1)
        currents = jnp.concatenate([jnp.where(hv[:,:,None],hd[None,:,:],-1),hv[:,:,None]],axis=2).reshape(4,8)
        docks = jnp.concatenate([s["dock_pos"],s["dock_stock"][:,None]],axis=1).reshape(6)
        md = jnp.concatenate([s["marker_pos"].reshape(2),s["marker_kind"],s["marker_ref"],s["marker_age"],s["marker_ttl"]])
        marker = jnp.concatenate([jnp.where(s["marker_active"][0],md,-1),s["marker_active"]])
        return jnp.concatenate([core,jnp.broadcast_to(public[None,:],(4,5)),agents,cargo,currents,jnp.broadcast_to(docks[None,:],(4,6)),jnp.broadcast_to(marker[None,:],(4,7)),jnp.broadcast_to(s["berth_pos"].reshape(1,4),(4,4)),jnp.broadcast_to(s["map"].reshape(1,99),(4,99))],axis=1).astype(jnp.float32)
