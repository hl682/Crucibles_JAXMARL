"""Island Spark: fictional cooperative microgrid game, authored but not executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 150
    num_actions = 14
    max_steps = 160

    def refresh(self, s):
        s = dict(s)
        ready = s["closed"] & (s["health"] == 2) & (s["cool"] == 0)
        source = s["started"] & (s["fuel"] > 0)
        trunk = source & ready[0]
        s["line_live"] = jnp.array([trunk,trunk & ready[1],trunk & ready[2]],jnp.bool_)
        s["powered"] = jnp.concatenate([jnp.array([source],jnp.bool_),s["line_live"]])
        s["agent_visible"] = jnp.sum(jnp.abs(s["agent_pos"][:,None,:]-s["agent_pos"][None,:,:]),axis=2) <= 3
        s["line_visible"] = jnp.sum(jnp.abs(s["agent_pos"][:,None,:]-s["line_pos"][None,:,:]),axis=2) <= 3
        return s

    def reset(self, key):
        board = jnp.array([[1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,5,0,4,0,1],
            [1,0,2,0,5,3,0,0,0,0,1],[1,0,0,0,0,0,5,0,4,0,1],
            [1,0,0,0,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1,1,1,1]],jnp.int32)
        s = {"t":jnp.array(0,jnp.int32),"episode_done":jnp.array(False),"terminated":jnp.array(False),"truncated":jnp.array(False),
             "map":board,"agent_pos":jnp.array([[3,2],[4,4],[2,3]],jnp.int32),"agent_active":jnp.ones(3,jnp.bool_),"agent_visible":jnp.ones((3,3),jnp.bool_),
             "charge":jnp.array([3,4,6],jnp.int32),"personal":jnp.zeros(3,jnp.int32),
             "line_pos":jnp.array([[4,3],[6,2],[6,4]],jnp.int32),"line_active":jnp.ones(3,jnp.bool_),"line_visible":jnp.ones((3,3),jnp.bool_),
             "health":jnp.array([1,0,0],jnp.int32),"closed":jnp.zeros(3,jnp.bool_),"cool":jnp.zeros(3,jnp.int32),"line_live":jnp.zeros(3,jnp.bool_),
             "node_pos":jnp.array([[2,3],[5,3],[8,2],[8,4]],jnp.int32),"node_active":jnp.ones(4,jnp.bool_),"node_visible":jnp.ones((3,4),jnp.bool_),
             "powered":jnp.zeros(4,jnp.bool_),"delivered":jnp.zeros(4,jnp.int32),"fuel":jnp.array(120,jnp.int32),"started":jnp.array(False),
             "trip_event":jnp.zeros(3,jnp.bool_),"boot_event":jnp.array(False)}
        s = self.refresh(s)
        return self.observe(s),s

    def legal_actions(self, s):
        ids = jnp.arange(3,dtype=jnp.int32)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        target = jnp.clip(s["agent_pos"][:,None,:]+delta[None,:,:],jnp.array([0,0],jnp.int32),jnp.array([10,6],jnp.int32))
        live_cell = jnp.any(jnp.all(target[:,:,None,:] == s["line_pos"][None,None,:,:],axis=3) & s["line_live"][None,None,:],axis=2)
        walk = (s["map"][target[:,:,1],target[:,:,0]] != 1) & ~((ids == 0)[:,None] & live_cell)
        mask = jnp.ones((3,14),jnp.bool_).at[:,:5].set(walk).at[:,0].set(True)
        mask = mask.at[:,5].set((ids == 0) & (s["charge"] > 0))
        mask = mask.at[:,6].set((ids == 1) & (s["charge"] > 0))
        mask = mask.at[:,7].set((ids == 2) & (s["charge"] >= 2) & ~s["started"])
        mask = mask.at[:,8].set(s["charge"] > 0)
        mask = mask.at[:,9].set(s["charge"] < 6)
        mask = mask.at[:,10].set((ids == 2) & (s["charge"] >= 2))
        mask = mask.at[:,11:14].set((ids[:,None] != ids[None,:]) & (s["charge"] > 0)[:,None])
        return mask & (~s["episode_done"] | (jnp.arange(14)[None,:] == 0))

    def movement(self, s, a):
        ids = jnp.arange(3,dtype=jnp.int32)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        p = s["agent_pos"] + delta[jnp.clip(a,0,4)] * ((a >= 1) & (a <= 4))[:,None]
        p = jnp.clip(p,jnp.array([0,0],jnp.int32),jnp.array([10,6],jnp.int32))
        occupied = jnp.any(jnp.all(p[:,None,:] == s["agent_pos"][None,:,:],axis=2) & (ids[:,None] != ids[None,:]),axis=1)
        can = (a >= 1) & (a <= 4) & ~occupied
        rank = jnp.mod(ids-s["t"],3)
        loser = jnp.any(jnp.all(p[:,None,:] == p[None,:,:],axis=2) & can[None,:] & (rank[None,:] < rank[:,None]),axis=1)
        return jnp.where((can & ~loser)[:,None],p,s["agent_pos"])

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions,0,13)
        mask = self.legal_actions(state)
        a = jnp.where((actions >= 0) & (actions < 14) & mask[jnp.arange(3),safe],safe,0)
        s = dict(state)
        s["agent_pos"] = self.movement(state,a)
        s["cool"] = jnp.maximum(state["cool"]-1,0)
        reward = jnp.full(3,-0.001,jnp.float32)
        # Portable tool charge is conserved by donations; proximity is checked after movement.
        for offset in range(3):
            i = jnp.mod(state["t"]+offset,3)
            j = jnp.clip(a[i]-11,0,2)
            give = (a[i] >= 11) & (a[i] <= 13) & (i != j) & (s["charge"][i] > 0) & (s["charge"][j] < 6) & (jnp.sum(jnp.abs(s["agent_pos"][i]-s["agent_pos"][j])) == 1)
            s["charge"] = s["charge"].at[i].add(-give.astype(jnp.int32)).at[j].add(give.astype(jnp.int32))
        distance = jnp.sum(jnp.abs(s["line_pos"]-s["agent_pos"][0]),axis=1)
        eligible = (distance <= 1) & (s["health"] < 2) & ~state["line_live"] & (state["cool"] == 0)
        p = jnp.argmin(jnp.where(eligible,distance*4+jnp.arange(3),99))
        repair = (a[0] == 5) & (s["charge"][0] > 0) & jnp.any(eligible)
        s["health"] = s["health"].at[p].add(repair.astype(jnp.int32))
        s["charge"] = s["charge"].at[0].add(-repair.astype(jnp.int32))
        reward = reward + 0.05*repair.astype(jnp.float32)
        distance = jnp.sum(jnp.abs(s["line_pos"]-s["agent_pos"][1]),axis=1)
        p = jnp.argmin(distance*4+jnp.arange(3))
        toggle = (a[1] == 6) & (distance[p] <= 1) & (s["charge"][1] > 0)
        s["closed"] = s["closed"].at[p].set(s["closed"][p] ^ toggle)
        s["charge"] = s["charge"].at[1].add(-toggle.astype(jnp.int32))
        boot = (a[2] == 7) & ~s["started"] & (s["charge"][2] >= 2) & (jnp.sum(jnp.abs(s["agent_pos"][2]-s["node_pos"][0])) <= 1)
        s["started"] = s["started"] | boot
        s["boot_event"] = boot
        s["charge"] = s["charge"].at[2].add(-2*boot.astype(jnp.int32))
        s = self.refresh(s)
        # All demands are admitted together; shared line limits are tested before fulfillment.
        distances = jnp.sum(jnp.abs(s["agent_pos"][:,None,:]-s["node_pos"][None,:,:]),axis=2)
        nearest = jnp.argmin(distances,axis=1)
        local = distances[jnp.arange(3),nearest] <= 1
        service = (a == 8) & (nearest >= 2) & (distances[jnp.arange(3),nearest] == 0) & (s["charge"] > 0) & (s["delivered"][nearest] < 3)
        charging = (a == 9) & local & (s["charge"] < 6)
        request = (service | charging) & s["powered"][nearest]
        load = jnp.array([jnp.sum(request & (nearest >= 1),dtype=jnp.int32),jnp.sum(request & (nearest == 2),dtype=jnp.int32),jnp.sum(request & (nearest == 3),dtype=jnp.int32)],jnp.int32)
        trip = s["line_live"] & (load > jnp.array([2,1,1],jnp.int32))
        s["closed"] = s["closed"] & ~trip
        s["health"] = jnp.maximum(s["health"]-trip.astype(jnp.int32),0)
        s["cool"] = jnp.where(trip,3,s["cool"]).astype(jnp.int32)
        s["trip_event"] = trip
        reward = reward - 0.2*jnp.sum(trip,dtype=jnp.int32)
        s = self.refresh(s)
        for offset in range(3):
            i = jnp.mod(state["t"]+offset,3)
            n = nearest[i]
            served = service[i] & s["powered"][n] & (s["delivered"][n] < 3)
            charged = charging[i] & s["powered"][n]
            s["charge"] = s["charge"].at[i].add(charged.astype(jnp.int32)-served.astype(jnp.int32))
            s["delivered"] = s["delivered"].at[n].add(served.astype(jnp.int32))
            s["personal"] = s["personal"].at[i].add(served.astype(jnp.int32))
            reward = reward + 0.2*served.astype(jnp.float32)
            reward = reward.at[i].add(0.6*served.astype(jnp.float32))
        # The battery crew may bypass the grid at twice the portable-charge cost.
        n = nearest[2]
        portable = (a[2] == 10) & (n >= 2) & (distances[2,n] == 0) & (s["charge"][2] >= 2) & (s["delivered"][n] < 3)
        s["charge"] = s["charge"].at[2].add(-2*portable.astype(jnp.int32))
        s["delivered"] = s["delivered"].at[n].add(portable.astype(jnp.int32))
        s["personal"] = s["personal"].at[2].add(portable.astype(jnp.int32))
        reward = reward + 0.2*portable.astype(jnp.float32)
        reward = reward.at[2].add(0.6*portable.astype(jnp.float32))
        s["fuel"] = jnp.maximum(s["fuel"]-s["started"].astype(jnp.int32),0)
        complete = jnp.all(s["delivered"][2:4] == 3)
        failed = s["started"] & (s["fuel"] == 0) & ~complete
        reward = reward - failed.astype(jnp.float32)
        s["t"] = state["t"]+jnp.array(1,jnp.int32)
        s["terminated"] = complete | failed
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        s = jax.tree.map(lambda old,new:jnp.where(state["episode_done"],old,new),state,s)
        reward = jnp.where(state["episode_done"],jnp.zeros(3,jnp.float32),reward).astype(jnp.float32)
        return self.observe(s),s,reward,jnp.full(3,s["episode_done"],jnp.bool_),{"done_all":s["episode_done"],"terminated":s["terminated"],"truncated":s["truncated"]}

    def observe(self, s):
        core = jnp.stack([jnp.arange(3),s["agent_pos"][:,0],s["agent_pos"][:,1],s["charge"],jnp.full(3,s["t"]),jnp.full(3,s["started"]),s["personal"]],axis=1)
        public = jnp.concatenate([jnp.array([s["fuel"]]),s["delivered"],s["powered"],s["line_live"],s["trip_event"],jnp.array([s["boot_event"],s["episode_done"]])])
        av = s["agent_visible"]
        ad = jnp.concatenate([s["agent_pos"],s["charge"][:,None],s["personal"][:,None]],axis=1)
        agents = jnp.concatenate([jnp.where(av[:,:,None],ad[None,:,:],-1),av[:,:,None]],axis=2).reshape(3,15)
        lv = s["line_visible"]
        ld = jnp.concatenate([s["line_pos"],s["health"][:,None],s["closed"][:,None],s["cool"][:,None]],axis=1)
        lines = jnp.concatenate([jnp.where(lv[:,:,None],ld[None,:,:],-1),lv[:,:,None]],axis=2).reshape(3,18)
        nodes = jnp.concatenate([s["node_pos"],s["powered"][:,None],s["delivered"][:,None]],axis=1).reshape(16)
        return jnp.concatenate([core,jnp.broadcast_to(public[None,:],(3,17)),agents,lines,jnp.broadcast_to(nodes[None,:],(3,16)),jnp.broadcast_to(s["map"].reshape(1,77),(3,77))],axis=1).astype(jnp.float32)
