"""Common Meadow: original three-herder renewable commons; authored, unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 138
    num_actions = 12
    max_steps = 144

    def refresh(self, s):
        s = dict(s)
        s["steward"] = jnp.mod(s["t"] // 8, 3).astype(jnp.int32)
        s["steward_display"] = jnp.full(1,s["steward"],jnp.int32)
        s["agent_visible"] = jnp.sum(jnp.abs(s["agent_pos"][:,None,:] - s["agent_pos"][None,:,:]),axis=2) <= 2
        s["patch_visible"] = jnp.sum(jnp.abs(s["agent_pos"][:,None,:] - s["patch_pos"][None,:,:]),axis=2) <= 2
        return s

    def reset(self, key):
        board = jnp.array([
            [1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,1],
            [1,0,2,0,2,0,2,0,1],
            [1,0,0,0,3,0,0,0,1],
            [1,0,2,0,2,0,2,0,1],
            [1,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1]], jnp.int32)
        s = {
            "t": jnp.array(0,jnp.int32), "episode_done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "collapse": jnp.array(False), "sustained": jnp.array(False),
            "map": board, "agent_pos": jnp.array([[1,2],[3,2],[7,2]],jnp.int32),
            "agent_active": jnp.ones(3,jnp.bool_), "agent_visible": jnp.ones((3,3),jnp.bool_),
            "bag": jnp.zeros(3,jnp.int32), "delivered": jnp.zeros(3,jnp.int32),
            "patch_pos": jnp.array([[2,2],[4,2],[6,2],[2,4],[4,4],[6,4]],jnp.int32),
            "patch_active": jnp.ones(6,jnp.bool_), "patch_visible": jnp.zeros((3,6),jnp.bool_),
            "biomass": jnp.full(6,4,jnp.int32), "regrow": jnp.zeros(6,jnp.int32),
            "closed": jnp.zeros(6,jnp.int32), "bare_age": jnp.zeros(6,jnp.int32),
            "roots": jnp.ones(6,jnp.bool_),
            "depot_pos": jnp.array([[4,3]],jnp.int32), "depot_active": jnp.ones(1,jnp.bool_),
            "depot_visible": jnp.ones((3,1),jnp.bool_), "steward_display": jnp.zeros(1,jnp.int32),
            "steward": jnp.array(0,jnp.int32), "banked_step": jnp.array(0,jnp.int32),
            "harvested_total": jnp.array(0,jnp.int32), "seeded_total": jnp.array(0,jnp.int32),
            "grown_total": jnp.array(0,jnp.int32)}
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        ids = jnp.arange(3,dtype=jnp.int32)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        target = jnp.clip(s["agent_pos"][:,None,:] + delta[None,:,:],jnp.array([0,0],jnp.int32),jnp.array([8,6],jnp.int32))
        mask = jnp.ones((3,12),jnp.bool_).at[:,:5].set(s["map"][target[:,:,1],target[:,:,0]] != 1)
        near = jnp.sum(jnp.abs(s["agent_pos"][:,None,:] - s["patch_pos"][None,:,:]),axis=2) <= 1
        harvest = near & s["roots"][None,:] & (s["biomass"][None,:] > 0) & (s["closed"][None,:] == 0)
        sow = near & s["roots"][None,:] & (s["biomass"][None,:] < 5)
        mask = mask.at[:,5].set((s["bag"] < 3) & jnp.any(harvest,axis=1))
        mask = mask.at[:,6].set((s["bag"] > 0) & jnp.any(sow,axis=1))
        at_depot = jnp.all(s["agent_pos"] == s["depot_pos"][0],axis=1)
        mask = mask.at[:,7].set(at_depot & (s["bag"] > 0) & (s["delivered"] < 3))
        mask = mask.at[:,8].set((ids == s["steward"]) & jnp.any(near & s["roots"][None,:],axis=1))
        adjacent = jnp.sum(jnp.abs(s["agent_pos"][:,None,:] - s["agent_pos"][None,:,:]),axis=2) == 1
        mask = mask.at[:,9:12].set(adjacent & (s["bag"] > 0)[:,None] & (s["bag"] < 3)[None,:] & (ids[:,None] != ids[None,:]))
        return mask & (~s["episode_done"] | (jnp.arange(12)[None,:] == 0))

    def move(self, s, actions):
        ids = jnp.arange(3,dtype=jnp.int32)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        proposed = s["agent_pos"] + delta[jnp.clip(actions,0,4)] * ((actions >= 1) & (actions <= 4))[:,None]
        proposed = jnp.clip(proposed,jnp.array([0,0],jnp.int32),jnp.array([8,6],jnp.int32))
        occupied = jnp.any(jnp.all(proposed[:,None,:] == s["agent_pos"][None,:,:],axis=2) & (ids[:,None] != ids[None,:]),axis=1)
        possible = (actions >= 1) & (actions <= 4) & (s["map"][proposed[:,1],proposed[:,0]] != 1) & ~occupied
        rank = jnp.mod(ids - s["t"],3)
        same = jnp.all(proposed[:,None,:] == proposed[None,:,:],axis=2)
        loses = jnp.any(same & possible[None,:] & (rank[None,:] < rank[:,None]),axis=1)
        return jnp.where((possible & ~loses)[:,None],proposed,s["agent_pos"])

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions,0,11)
        mask = self.legal_actions(state)
        actions = jnp.where((actions >= 0) & (actions < 12) & mask[jnp.arange(3),safe],safe,0)
        s = dict(state)
        s["agent_pos"] = self.move(state,actions)
        s["regrow"] = jnp.maximum(state["regrow"] - 1,0)
        s["closed"] = jnp.maximum(state["closed"] - 1,0)
        s["banked_step"] = jnp.array(0,jnp.int32)
        reward = jnp.full(3,-0.001,jnp.float32)
        # Gifts are first, and each donor can only spend pre-step inventory once.
        for offset in range(3):
            i = jnp.mod(state["t"] + offset,3)
            target = jnp.clip(actions[i] - 9,0,2)
            gift = (actions[i] >= 9) & (actions[i] <= 11) & (target != i) & (state["bag"][i] > 0) & (s["bag"][i] > 0) & (s["bag"][target] < 3) & (jnp.sum(jnp.abs(s["agent_pos"][i] - s["agent_pos"][target])) == 1)
            amount = gift.astype(jnp.int32)
            s["bag"] = s["bag"].at[i].add(-amount).at[target].add(amount)
        # Harvest contention resolves in rotating order; nearest then patch-id breaks ties.
        for offset in range(3):
            i = jnp.mod(state["t"] + offset,3)
            distance = jnp.sum(jnp.abs(s["patch_pos"] - s["agent_pos"][i]),axis=1)
            available = (distance <= 1) & s["roots"] & (s["biomass"] > 0) & (state["closed"] == 0)
            p = jnp.argmin(jnp.where(available,distance * 8 + jnp.arange(6),99))
            harvest = (actions[i] == 5) & (s["bag"][i] < 3) & jnp.any(available)
            amount = harvest.astype(jnp.int32)
            s["biomass"] = s["biomass"].at[p].add(-amount)
            s["bag"] = s["bag"].at[i].add(amount)
            s["regrow"] = s["regrow"].at[p].set(jnp.where(harvest,3,s["regrow"][p]))
            s["harvested_total"] = s["harvested_total"] + amount
            reward = reward.at[i].add(0.05 * amount)
        # A seed is spent irreversibly; stewardship accelerates recovery, never creates inventory.
        for offset in range(3):
            i = jnp.mod(state["t"] + offset,3)
            distance = jnp.sum(jnp.abs(s["patch_pos"] - s["agent_pos"][i]),axis=1)
            available = (distance <= 1) & s["roots"] & (s["biomass"] < 5)
            p = jnp.argmin(jnp.where(available,distance * 8 + jnp.arange(6),99))
            sow = (actions[i] == 6) & (s["bag"][i] > 0) & jnp.any(available)
            amount = sow.astype(jnp.int32)
            s["bag"] = s["bag"].at[i].add(-amount)
            s["biomass"] = s["biomass"].at[p].add(amount)
            s["regrow"] = s["regrow"].at[p].set(jnp.where(sow,jnp.where(i == state["steward"],1,2),s["regrow"][p]))
            s["seeded_total"] = s["seeded_total"] + amount
            reward = reward.at[i].add(-0.03 * amount)
            protected = (distance <= 1) & s["roots"]
            q = jnp.argmin(jnp.where(protected,distance * 8 + jnp.arange(6),99))
            fallow = (actions[i] == 8) & (i == state["steward"]) & jnp.any(protected)
            s["closed"] = s["closed"].at[q].set(jnp.where(fallow,4,s["closed"][q]))
            bank = (actions[i] == 7) & (s["bag"][i] > 0) & (s["delivered"][i] < 3) & jnp.all(s["agent_pos"][i] == s["depot_pos"][0])
            deposit = bank.astype(jnp.int32)
            s["bag"] = s["bag"].at[i].add(-deposit)
            s["delivered"] = s["delivered"].at[i].add(deposit)
            s["banked_step"] = s["banked_step"] + deposit
            reward = reward.at[i].add(deposit.astype(jnp.float32))
        # Two retained tufts seed regrowth. Crowding by two or more herders suppresses it.
        density = jnp.sum(jnp.sum(jnp.abs(s["agent_pos"][:,None,:] - s["patch_pos"][None,:,:]),axis=2) <= 1,axis=0)
        growth = s["roots"] & (s["biomass"] >= 2) & (s["biomass"] < 5) & (s["regrow"] == 0) & (density <= 1)
        s["biomass"] = s["biomass"] + growth.astype(jnp.int32)
        s["regrow"] = jnp.where(growth,3,s["regrow"]).astype(jnp.int32)
        s["grown_total"] = s["grown_total"] + jnp.sum(growth.astype(jnp.int32),dtype=jnp.int32)
        s["bare_age"] = jnp.where(s["roots"] & (s["biomass"] == 0),jnp.minimum(state["bare_age"] + 1,6),jnp.where(s["roots"],0,state["bare_age"])).astype(jnp.int32)
        s["roots"] = s["roots"] & (s["bare_age"] < 6)
        s["collapse"] = ~jnp.any(s["roots"])
        s["sustained"] = jnp.all(s["delivered"] == 3) & jnp.all(s["roots"]) & (jnp.sum(s["biomass"]) >= 12)
        reward = reward - 0.5 * s["collapse"].astype(jnp.float32) + 0.25 * s["sustained"].astype(jnp.float32)
        s["t"] = state["t"] + jnp.array(1,jnp.int32)
        s["terminated"] = s["collapse"] | s["sustained"]
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        s = jax.tree.map(lambda old,new: jnp.where(state["episode_done"],old,new),state,s)
        reward = jnp.where(state["episode_done"],jnp.zeros(3,jnp.float32),reward).astype(jnp.float32)
        return self.observe(s),s,reward,jnp.full(3,s["episode_done"],jnp.bool_),{"done_all":s["episode_done"],"terminated":s["terminated"],"truncated":s["truncated"]}

    def observe(self, s):
        core = jnp.concatenate([s["agent_pos"],s["bag"][:,None],s["delivered"][:,None]],axis=1)
        public = jnp.broadcast_to(jnp.array([s["t"],s["steward"],8-jnp.mod(s["t"],8),s["banked_step"],s["collapse"],s["sustained"]],jnp.float32),(3,6))
        deliveries = jnp.broadcast_to(s["delivered"][None,:],(3,3))
        av = s["agent_visible"]
        agent_data = jnp.concatenate([s["agent_pos"],s["bag"][:,None]],axis=1)
        agents = jnp.concatenate([jnp.where(av[:,:,None],agent_data[None,:,:],-1),av[:,:,None]],axis=2).reshape(3,12)
        pv = s["patch_visible"]
        patch_data = jnp.concatenate([s["patch_pos"],s["biomass"][:,None],s["regrow"][:,None],s["closed"][:,None],s["bare_age"][:,None],s["roots"][:,None]],axis=1)
        patches = jnp.concatenate([jnp.where(pv[:,:,None],patch_data[None,:,:],-1),pv[:,:,None]],axis=2).reshape(3,48)
        depot = jnp.broadcast_to(s["depot_pos"],(3,2))
        board = jnp.broadcast_to(s["map"].reshape(1,63),(3,63))
        return jnp.concatenate([core,public,deliveries,agents,patches,depot,board],axis=1).astype(jnp.float32)
