"""Original Copper Relay territory game. Authored, never imported or executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 393
    num_actions = 9
    max_steps = 80

    def network(self, owner, contested, active, positions):
        adjacent = jnp.sum(jnp.abs(positions[:,None,:] - positions[None,:,:]),axis=2) == 1
        linked = jnp.zeros(35,dtype=jnp.bool_)
        homes = jnp.array([[1,3],[7,3]],dtype=jnp.int32)
        for team in range(2):
            eligible = active & (owner == team) & ~contested
            reached = eligible & jnp.all(positions == homes[team],axis=1)
            for hop in range(35):
                reached = eligible & (reached | jnp.any(adjacent & reached[None,:],axis=1))
            linked = linked | reached
        return linked

    def refresh(self, s):
        s = dict(s)
        present = jnp.all(s["cell_pos"][:,None,:] == s["agent_pos"][None,:,:],axis=2)
        s["contested"] = s["cell_active"] & (s["owner"] >= 0) & jnp.any(
            present & (s["owner"][:,None] != s["team"][None,:]),axis=1)
        s["linked"] = self.network(s["owner"],s["contested"],s["cell_active"],s["cell_pos"])
        return s

    def reset(self, key):
        terrain = jnp.array([[1,1,1,1,1,1,1,1,1],
                             [1,0,0,0,1,0,0,0,1],
                             [1,0,0,0,0,0,0,0,1],
                             [1,0,0,0,0,0,0,0,1],
                             [1,0,0,0,0,0,0,0,1],
                             [1,0,0,0,1,0,0,0,1],
                             [1,1,1,1,1,1,1,1,1]],dtype=jnp.int32)
        positions = jnp.array([[x,y] for y in range(1,6) for x in range(1,8)],dtype=jnp.int32)
        active = terrain[positions[:,1],positions[:,0]] == 0
        owner = jnp.full(35,-1,dtype=jnp.int32).at[14].set(0).at[20].set(1)
        charge = jnp.where(owner >= 0,3,0).astype(jnp.int32)
        relay = jnp.all(positions == jnp.array([3,2]),axis=1) | jnp.all(positions == jnp.array([5,4]),axis=1)
        s = {"t": jnp.array(0,dtype=jnp.int32), "episode_done":jnp.array(False),
             "terminated":jnp.array(False), "truncated":jnp.array(False), "map":terrain,
             "agent_pos":jnp.array([[1,2],[1,4],[7,2],[7,4]],dtype=jnp.int32),
             "agent_active":jnp.ones(4,dtype=jnp.bool_), "agent_visible":jnp.ones((4,4),dtype=jnp.bool_),
             "team":jnp.array([0,0,1,1],dtype=jnp.int32), "pack":jnp.full(4,4,dtype=jnp.int32),
             "blocked":jnp.zeros(4,dtype=jnp.bool_),
             "cell_pos":positions, "cell_active":active, "cell_visible":jnp.broadcast_to(active,(4,35)),
             "owner":owner, "charge":charge, "linked":jnp.zeros(35,dtype=jnp.bool_),
             "contested":jnp.zeros(35,dtype=jnp.bool_), "relay":relay,
             "home_pos":jnp.array([[1,3],[7,3]],dtype=jnp.int32),
             "home_active":jnp.ones(2,dtype=jnp.bool_), "home_visible":jnp.ones((4,2),dtype=jnp.bool_),
             "stock":jnp.full(2,16,dtype=jnp.int32), "score":jnp.zeros(2,dtype=jnp.int32),
             "built_now":jnp.zeros(4,dtype=jnp.bool_), "cut_now":jnp.zeros(4,dtype=jnp.bool_),
             "transfer_now":jnp.zeros(4,dtype=jnp.bool_), "scored_now":jnp.zeros(2,dtype=jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        flat = jnp.concatenate([s["agent_pos"].reshape(-1),s["team"],s["pack"],s["blocked"],
                                s["cell_pos"].reshape(-1),s["cell_active"],s["owner"],s["charge"],
                                s["linked"],s["contested"],s["relay"],s["home_pos"].reshape(-1),
                                s["stock"],s["score"],s["map"].reshape(-1),s["built_now"],
                                s["cut_now"],s["transfer_now"],s["scored_now"],
                                jnp.stack([s["t"],s["episode_done"],s["terminated"],s["truncated"]])]).astype(jnp.float32)
        return jnp.concatenate([jnp.broadcast_to(flat,(4,389)),jnp.eye(4,dtype=jnp.float32)],axis=1)

    def move_agents(self, state, actions):
        offsets = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        step = offsets[jnp.where((actions >= 1) & (actions <= 4), actions, 0)]
        p = state["agent_pos"]
        q = jnp.clip(p + step, jnp.array([0, 0], dtype=jnp.int32), jnp.array([8, 6], dtype=jnp.int32))
        terrain_ok = state["map"][q[:, 1], q[:, 0]] == 0
        occupied = jnp.any(jnp.all(q[:, None, :] == p[None, :, :], axis=2) &
                           ~jnp.eye(self.num_agents, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1) > 1
        moving = jnp.any(step != 0, axis=1)
        blocked = moving & (~terrain_ok | occupied | contested)
        return jnp.where(blocked[:, None], p, q), blocked

    def advance(self, key, state, actions):
        a = jnp.where((actions >= 0) & (actions < 9), actions, 0).astype(jnp.int32)
        s = dict(state)
        p, blocked = self.move_agents(state,a)
        owner, charge, pack, stock = s["owner"],s["charge"],s["pack"],s["stock"]
        built = jnp.zeros(4,dtype=jnp.bool_)
        cut = jnp.zeros(4,dtype=jnp.bool_)
        transfer = jnp.zeros(4,dtype=jnp.bool_)
        used = jnp.zeros(4,dtype=jnp.bool_)
        home_cell = jnp.any(jnp.all(s["cell_pos"][:,None,:] == s["home_pos"][None,:,:],axis=2),axis=1)
        # Occupation itself severs an enemy conduit, even without attacking its durability.
        s["agent_pos"] = p
        s = self.refresh(s)
        for offset in range(4):
            i = jnp.mod(s["t"] + offset,4)
            team = s["team"][i]
            here = jnp.argmax(jnp.all(s["cell_pos"] == p[i],axis=1))
            building = (a[i] == 5) & (owner[here] < 0) & (pack[i] >= 2)
            repair = (a[i] == 5) & (owner[here] == team) & (charge[here] < 3) & (pack[i] >= 1)
            owner = owner.at[here].set(jnp.where(building,team,owner[here]))
            charge = charge.at[here].set(jnp.where(building,3,jnp.where(repair,charge[here]+1,charge[here])))
            pack = pack.at[i].set(pack[i] - 2*building.astype(jnp.int32) - repair.astype(jnp.int32))
            built = built.at[i].set(building | repair)
            # Choose the lowest indexed adjacent enemy conduit; home hardware cannot be destroyed.
            targets = ((owner >= 0) & (owner != team) & ~home_cell & s["cell_active"] &
                       (jnp.sum(jnp.abs(s["cell_pos"] - p[i]),axis=1) <= 1))
            target = jnp.argmax(targets)
            attack = (a[i] == 6) & (pack[i] > 0) & jnp.any(targets)
            charge = charge.at[target].set(jnp.where(attack,jnp.maximum(charge[target]-1,0),charge[target]))
            owner = owner.at[target].set(jnp.where(attack & (charge[target] == 0),-1,owner[target]))
            pack = pack.at[i].set(pack[i] - attack.astype(jnp.int32))
            cut = cut.at[i].set(attack)
            # Paired GIVE actions transfer one unit from the richer teammate, once per participant.
            partners = ((s["team"] == team) & (a == 7) & ~used &
                        (jnp.sum(jnp.abs(p-p[i]),axis=1) == 1) & (pack < pack[i]) & (pack < 6))
            partner = jnp.argmax(partners)
            give = (a[i] == 7) & ~used[i] & (pack[i] > 0) & jnp.any(partners)
            pack = pack.at[i].add(-give.astype(jnp.int32))
            pack = pack.at[partner].add(give.astype(jnp.int32))
            used = used.at[i].set(used[i] | give).at[partner].set(used[partner] | give)
            transfer = transfer.at[i].set(transfer[i] | give).at[partner].set(transfer[partner] | give)
            # Restocking uses the beginning-of-operation network and the team's shared finite stock.
            take = (a[i] == 8) & (s["owner"][here] == team) & s["linked"][here] & (pack[i] < 6) & (stock[team] > 0)
            amount = jnp.where(take,jnp.minimum(2,jnp.minimum(6-pack[i],stock[team])),0)
            pack = pack.at[i].add(amount)
            stock = stock.at[team].add(-amount)
        s["owner"],s["charge"],s["pack"] = owner,charge,pack
        s = self.refresh(s)
        t = s["t"] + 1
        decay = s["cell_active"] & (owner >= 0) & ~s["linked"] & ~home_cell & (jnp.mod(t,3) == 0)
        charge = jnp.maximum(charge-decay.astype(jnp.int32),0)
        owner = jnp.where((charge == 0) & ~home_cell,-1,owner)
        s["owner"],s["charge"] = owner,charge
        s = self.refresh(s)
        pulses = jnp.stack([jnp.sum((s["relay"] & s["linked"] & (owner == team)).astype(jnp.int32),dtype=jnp.int32)
                            for team in range(2)])
        pulses = jnp.where(jnp.mod(t,4) == 0,pulses,jnp.zeros(2,dtype=jnp.int32))
        s["score"] = s["score"] + pulses
        s["stock"] = jnp.minimum(stock + (jnp.mod(t,4) == 0).astype(jnp.int32),24)
        s["built_now"],s["cut_now"],s["transfer_now"],s["scored_now"] = built,cut,transfer,pulses > 0
        s["blocked"],s["t"] = blocked,t
        s["terminated"] = jnp.any(s["score"] >= 12)
        s["truncated"] = (t >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        delta = pulses.astype(jnp.float32) * 0.5
        rewards = delta[s["team"]] - delta[1-s["team"]]
        dones = jnp.full((4,),s["episode_done"],dtype=jnp.bool_)
        info = {"done_all":s["episode_done"],"terminated":s["terminated"],"truncated":s["truncated"]}
        return self.observe(s), s, rewards, dones, info

    def legal_actions(self, state):
        live = jnp.broadcast_to(~state["episode_done"], (self.num_agents, self.num_actions))
        return live.at[:, 0].set(True)

    def absorb(self, key, state, actions):
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        dones = jnp.ones((self.num_agents,), dtype=jnp.bool_)
        info = {"done_all": state["episode_done"], "terminated": state["terminated"],
                "truncated": state["truncated"]}
        return self.observe(state), state, rewards, dones, info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["episode_done"], self.absorb, self.advance, key, state, actions)
