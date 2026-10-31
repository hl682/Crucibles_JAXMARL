"""Leeward Caravan: movable shelter and interchangeable hauling roles; unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 159
    num_actions = 15
    max_steps = 128

    def shade(self, positions, shelter, wind):
        relative = positions-shelter
        forward = jnp.sum(relative*wind,axis=1)
        sideways = jnp.abs(relative[:,0]*wind[1]-relative[:,1]*wind[0])
        return (forward >= 1) & (forward <= 3) & (sideways <= 1)

    def refresh(self, s):
        s = dict(s)
        weather = jnp.array([[1,0],[0,1],[-1,0],[0,-1]],jnp.int32)
        s["wind_dir"] = weather[jnp.mod(s["t"]//8,4)][None,:]
        s["sheltered"] = self.shade(s["agent_pos"],s["shelter_pos"][0],s["wind_dir"][0])
        s["load_sheltered"] = self.shade(s["load_pos"],s["shelter_pos"][0],s["wind_dir"][0])
        s["load_grips"] = jnp.array([jnp.sum(s["grip"],dtype=jnp.int32)],jnp.int32)
        return s

    def reset(self, key):
        board = jnp.array([[1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,1],[1,0,2,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,1,1,0,0,3,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,1],[1,0,2,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1,1,1,1,1,1]],jnp.int32)
        s = {"t":jnp.array(0,jnp.int32),"episode_done":jnp.array(False),"terminated":jnp.array(False),"truncated":jnp.array(False),
             "map":board,"agent_pos":jnp.array([[3,3],[3,5],[1,4]],jnp.int32),"agent_active":jnp.ones(3,jnp.bool_),"agent_visible":jnp.ones((3,3),jnp.bool_),
             "energy":jnp.full(3,10,jnp.int32),"water":jnp.full(3,2,jnp.int32),"grip":jnp.zeros(3,jnp.bool_),"sheltered":jnp.zeros(3,jnp.bool_),
             "load_pos":jnp.array([[3,4]],jnp.int32),"load_active":jnp.ones(1,jnp.bool_),"load_visible":jnp.ones((3,1),jnp.bool_),"load_grips":jnp.zeros(1,jnp.int32),"load_sheltered":jnp.zeros(1,jnp.bool_),
             "shelter_pos":jnp.array([[2,4]],jnp.int32),"shelter_active":jnp.ones(1,jnp.bool_),"shelter_visible":jnp.ones((3,1),jnp.bool_),"wind_dir":jnp.array([[1,0]],jnp.int32),"wind_strength":jnp.array([2],jnp.int32),
             "well_pos":jnp.array([[2,2],[2,6]],jnp.int32),"well_active":jnp.ones(2,jnp.bool_),"well_visible":jnp.ones((3,2),jnp.bool_),"well_stock":jnp.full(2,4,jnp.int32),
             "goal_pos":jnp.array([[10,4]],jnp.int32),"goal_active":jnp.ones(1,jnp.bool_),"goal_visible":jnp.ones((3,1),jnp.bool_),
             "water_used":jnp.array(0,jnp.int32),"delivered":jnp.array(False),"haul_event":jnp.array(False),"shelter_event":jnp.array(False)}
        s = self.refresh(s)
        return self.observe(s),s

    def legal_actions(self, s):
        ids = jnp.arange(3,dtype=jnp.int32)
        d = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        p = jnp.clip(s["agent_pos"][:,None,:]+d[None,:,:],jnp.array([0,0],jnp.int32),jnp.array([12,8],jnp.int32))
        walk = (s["map"][p[:,:,1],p[:,:,0]] != 1) & (s["energy"] >= jnp.where(s["grip"],2,1))[:,None]
        mask = jnp.ones((3,15),jnp.bool_).at[:,:5].set(walk).at[:,0].set(True)
        mask = mask.at[:,6:10].set(jnp.broadcast_to(((s["energy"] >= 2) & ~s["grip"])[:,None],(3,4)))
        mask = mask.at[:,10].set((s["water"] > 0) & (s["energy"] < 12))
        mask = mask.at[:,11:14].set((ids[:,None] != ids[None,:]) & (s["water"] > 0)[:,None])
        mask = mask.at[:,14].set(s["water"] < 3)
        return mask & (~s["episode_done"] | (jnp.arange(15)[None,:] == 0))

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions,0,14)
        mask = self.legal_actions(state)
        a = jnp.where((actions >= 0) & (actions < 15) & mask[jnp.arange(3),safe],safe,0)
        ids = jnp.arange(3,dtype=jnp.int32)
        direction = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        s = dict(state)
        s["shelter_event"] = jnp.array(False)
        # One physical windbreak shift per tick; it changes other actors costs immediately.
        for offset in range(3):
            i = jnp.mod(state["t"]+offset,3)
            d = direction[jnp.clip(a[i]-5,0,4)]
            target = s["shelter_pos"][0]+d
            clipped = jnp.clip(target,jnp.array([0,0],jnp.int32),jnp.array([12,8],jnp.int32))
            occupied = jnp.any(jnp.all(target == s["agent_pos"],axis=1)) | jnp.all(target == s["load_pos"][0])
            push = (a[i] >= 6) & (a[i] <= 9) & ~s["grip"][i] & (s["energy"][i] >= 2) & ~s["shelter_event"] & (jnp.sum(jnp.abs(s["agent_pos"][i]-s["shelter_pos"][0])) == 1) & jnp.all(target == clipped) & (s["map"][clipped[1],clipped[0]] != 1) & ~occupied
            s["shelter_pos"] = jnp.where(push,target[None,:],s["shelter_pos"])
            s["energy"] = s["energy"].at[i].add(-2*push.astype(jnp.int32))
            s["shelter_event"] = s["shelter_event"] | push
        relative = s["agent_pos"]-s["load_pos"][0]
        flank = (relative[:,0] == 0) & (jnp.abs(relative[:,1]) == 1)
        s["grip"] = jnp.where(a == 5,~state["grip"] & flank,state["grip"])
        # Water transfers and drinking use real finite units, with no automatic replenishment.
        for offset in range(3):
            i = jnp.mod(state["t"]+offset,3)
            j = jnp.clip(a[i]-11,0,2)
            give = (a[i] >= 11) & (a[i] <= 13) & (i != j) & (s["water"][i] > 0) & (s["water"][j] < 3) & (jnp.sum(jnp.abs(s["agent_pos"][i]-s["agent_pos"][j])) == 1)
            s["water"] = s["water"].at[i].add(-give.astype(jnp.int32)).at[j].add(give.astype(jnp.int32))
            drink = (a[i] == 10) & (s["water"][i] > 0)
            s["water"] = s["water"].at[i].add(-drink.astype(jnp.int32))
            s["energy"] = s["energy"].at[i].set(jnp.minimum(12,s["energy"][i]+4*drink.astype(jnp.int32)))
            s["water_used"] = s["water_used"]+drink.astype(jnp.int32)
            distance = jnp.sum(jnp.abs(s["well_pos"]-s["agent_pos"][i]),axis=1)
            w = jnp.argmin(distance)
            fill = (a[i] == 14) & (distance[w] <= 1) & (s["water"][i] < 3) & (s["well_stock"][w] > 0)
            s["water"] = s["water"].at[i].add(fill.astype(jnp.int32))
            s["well_stock"] = s["well_stock"].at[w].add(-fill.astype(jnp.int32))
        covered = self.shade(s["agent_pos"],s["shelter_pos"][0],state["wind_dir"][0])
        s["energy"] = jnp.minimum(12,s["energy"]+2*((a == 0) & covered).astype(jnp.int32))
        against = jnp.sum(direction[jnp.clip(a,0,4)]*state["wind_dir"][0],axis=1) < 0
        cost = (jnp.where(s["grip"],2,1)+jnp.where(covered,0,state["wind_strength"][0])+against.astype(jnp.int32)).astype(jnp.int32)
        # Any pair can haul, but must occupy opposite north/south handles and choose one direction.
        i = jnp.argmax(s["grip"])
        j = jnp.argmax(s["grip"] & (ids != i))
        count = jnp.sum(s["grip"],dtype=jnp.int32)
        handles = jnp.stack([s["agent_pos"][i],s["agent_pos"][j]])-s["load_pos"][0]
        aligned = jnp.all(handles[:,0] == 0) & jnp.all(jnp.abs(handles[:,1]) == 1) & (handles[0,1] != handles[1,1])
        together = (count == 2) & aligned & (a[i] == a[j]) & (a[i] >= 1) & (a[i] <= 4) & (s["energy"][i] >= cost[i]) & (s["energy"][j] >= cost[j])
        d = direction[jnp.clip(a[i],0,4)]
        body = jnp.stack([s["load_pos"][0],s["agent_pos"][i],s["agent_pos"][j]])+d
        clipped = jnp.clip(body,jnp.array([0,0],jnp.int32),jnp.array([12,8],jnp.int32))
        blocked = jnp.any(jnp.all(body == s["shelter_pos"][0],axis=1)) | jnp.any(jnp.all(body[:,None,:] == s["agent_pos"][None,:,:],axis=2) & ((ids != i) & (ids != j))[None,:])
        haul = together & jnp.all(body == clipped) & jnp.all(s["map"][clipped[:,1],clipped[:,0]] != 1) & ~blocked
        s["load_pos"] = jnp.where(haul,body[0][None,:],s["load_pos"])
        s["agent_pos"] = s["agent_pos"].at[i].set(jnp.where(haul,body[1],s["agent_pos"][i])).at[j].set(jnp.where(haul,body[2],s["agent_pos"][j]))
        s["energy"] = s["energy"].at[i].add(-cost[i]*haul.astype(jnp.int32)).at[j].add(-cost[j]*haul.astype(jnp.int32))
        s["haul_event"] = haul
        # Unattached travelers cannot walk through the load, windbreak or another traveler's old cell.
        proposal = s["agent_pos"]+direction[jnp.clip(a,0,4)]*((a >= 1) & (a <= 4))[:,None]
        proposal = jnp.clip(proposal,jnp.array([0,0],jnp.int32),jnp.array([12,8],jnp.int32))
        occupied = jnp.any((jnp.all(proposal[:,None,:] == state["agent_pos"][None,:,:],axis=2) | jnp.all(proposal[:,None,:] == s["agent_pos"][None,:,:],axis=2)) & (ids[:,None] != ids[None,:]),axis=1)
        objects = jnp.all(proposal == s["load_pos"][0],axis=1) | jnp.all(proposal == s["shelter_pos"][0],axis=1)
        can = (a >= 1) & (a <= 4) & ~s["grip"] & (s["energy"] >= cost) & ~occupied & ~objects & (s["map"][proposal[:,1],proposal[:,0]] != 1)
        rank = jnp.mod(ids-state["t"],3)
        loser = jnp.any(jnp.all(proposal[:,None,:] == proposal[None,:,:],axis=2) & can[None,:] & (rank[None,:] < rank[:,None]),axis=1)
        moved = can & ~loser
        s["agent_pos"] = jnp.where(moved[:,None],proposal,s["agent_pos"])
        s["energy"] = s["energy"]-cost*moved.astype(jnp.int32)
        s["delivered"] = jnp.all(s["load_pos"][0] == s["goal_pos"][0])
        s["t"] = state["t"]+jnp.array(1,jnp.int32)
        s["terminated"] = s["delivered"]
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        reward = jnp.full(3,-0.001+s["delivered"].astype(jnp.float32),jnp.float32)
        s = jax.tree.map(lambda old,new:jnp.where(state["episode_done"],old,new),state,s)
        reward = jnp.where(state["episode_done"],jnp.zeros(3,jnp.float32),reward)
        return self.observe(s),s,reward,jnp.full(3,s["episode_done"],jnp.bool_),{"done_all":s["episode_done"],"terminated":s["terminated"],"truncated":s["truncated"]}

    def observe(self, s):
        public = jnp.array([s["t"],s["episode_done"],s["water_used"],s["haul_event"],s["shelter_event"],s["delivered"]],jnp.float32)
        agents = jnp.concatenate([s["agent_pos"],s["energy"][:,None],s["water"][:,None],s["grip"][:,None],s["sheltered"][:,None]],axis=1).reshape(18)
        load = jnp.concatenate([s["load_pos"].reshape(2),s["load_grips"],s["load_sheltered"]])
        shelter = jnp.concatenate([s["shelter_pos"].reshape(2),s["wind_dir"].reshape(2),s["wind_strength"]])
        wells = jnp.concatenate([s["well_pos"],s["well_stock"][:,None]],axis=1).reshape(6)
        data = jnp.concatenate([public,agents,load,shelter,wells,s["goal_pos"].reshape(2),s["map"].reshape(117)]).astype(jnp.float32)
        return jnp.concatenate([jnp.arange(3,dtype=jnp.float32)[:,None],jnp.broadcast_to(data[None,:],(3,158))],axis=1)
