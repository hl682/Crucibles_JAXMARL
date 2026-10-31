"""Breathshare Claims. Original finite-resource mining rules, author-unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 153
    num_actions = 10
    max_steps = 80

    def reset(self, key):
        terrain = jnp.zeros((7,11), jnp.int32).at[2,5].set(1).at[4,5].set(1)
        state = {
            "t": jnp.array(0, jnp.int32),
            "episode_done": jnp.array(False, jnp.bool_),
            "terminated": jnp.array(False, jnp.bool_),
            "truncated": jnp.array(False, jnp.bool_),
            "terrain": terrain,
            "agent_pos": jnp.array([[2,3],[2,2],[2,4]], jnp.int32),
            "agent_active": jnp.ones((3,), jnp.bool_),
            "agent_visible": jnp.ones((3,3), jnp.bool_),
            "oxygen": jnp.full((3,), 12, jnp.int32),
            "ore": jnp.full((3,), -1, jnp.int32),
            "carrying_canister": jnp.full((3,), -1, jnp.int32),
            "claim": jax.random.permutation(key, jnp.arange(3, dtype=jnp.int32)).astype(jnp.int32),
            "claim_visible": jnp.eye(3, dtype=jnp.bool_),
            "memorial_active": jnp.zeros((3,), jnp.bool_),
            "vein_pos": jnp.array([[7,1],[8,3],[7,5]], jnp.int32),
            "vein_active": jnp.ones((3,), jnp.bool_),
            "vein_visible": jnp.ones((3,3), jnp.bool_),
            "vein_type": jnp.arange(3, dtype=jnp.int32),
            "vein_remaining": jnp.full((3,), 2, jnp.int32),
            "canister_pos": jnp.array([[2,2],[2,4]], jnp.int32),
            "canister_active": jnp.ones((2,), jnp.bool_),
            "canister_visible": jnp.ones((3,2), jnp.bool_),
            "canister_owner": jnp.full((2,), -1, jnp.int32),
            "canister_volume": jnp.full((2,), 6, jnp.int32),
            "station_pos": jnp.array([[1,3]], jnp.int32),
            "station_active": jnp.ones((1,), jnp.bool_),
            "station_visible": jnp.ones((3,1), jnp.bool_),
            "station_stock": jnp.array([24], jnp.int32),
            "station_occupant": jnp.array([-1], jnp.int32),
            "delivered_counts": jnp.zeros((3,), jnp.int32),
            "lost_counts": jnp.zeros((3,), jnp.int32),
            "mined": jnp.zeros((3,), jnp.bool_),
            "delivery_event": jnp.zeros((3,), jnp.bool_),
            "transferred": jnp.zeros((2,), jnp.bool_),
            "used": jnp.zeros((2,), jnp.bool_),
            "died": jnp.zeros((3,), jnp.bool_),
            "refilled": jnp.zeros((3,), jnp.bool_),
            "picked": jnp.zeros((2,), jnp.bool_),
            "dropped": jnp.zeros((2,), jnp.bool_),
        }
        return self.observe(state), state

    def legal_actions(self, state):
        pos = state["agent_pos"]
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]], jnp.int32)
        target = pos[:,None,:] + delta[None,:,:]
        inside = jnp.all((target >= jnp.array([0,0])) & (target <= jnp.array([10,6])), axis=2)
        safe = jnp.clip(target, jnp.array([0,0]), jnp.array([10,6]))
        floor = state["terrain"][safe[:,:,1],safe[:,:,0]] == 0
        vein_distance = jnp.sum(jnp.abs(pos[:,None,:] - state["vein_pos"][None,:,:]), axis=2)
        mine = (state["ore"] < 0) & jnp.any((vein_distance <= 1) & (state["vein_remaining"] > 0)[None,:], axis=1)
        can_distance = jnp.sum(jnp.abs(pos[:,None,:] - state["canister_pos"][None,:,:]), axis=2)
        has_can = state["carrying_canister"] >= 0
        floor_can = (state["canister_owner"] < 0)[None,:] & (can_distance <= 1)
        handle = has_can | jnp.any(floor_can, axis=1)
        distance = jnp.sum(jnp.abs(pos[:,None,:] - pos[None,:,:]), axis=2)
        other = ~jnp.eye(3, dtype=jnp.bool_)
        donor = jnp.any((distance == 1) & other & has_can[None,:] & state["agent_active"][None,:], axis=1)
        safe_can = jnp.maximum(state["carrying_canister"], 0)
        use_accept = (has_can & (state["canister_volume"][safe_can] > 0)) | (~has_can & donor)
        at_station = jnp.all(pos == state["station_pos"][0], axis=1)
        deficit = (state["oxygen"] < 12) | (has_can & (state["canister_volume"][safe_can] < 6))
        refill = at_station & (state["station_stock"][0] > 0) & deficit
        mask = jnp.concatenate((inside & floor, mine[:,None], handle[:,None], has_can[:,None],
                                use_accept[:,None], refill[:,None]), axis=1)
        enabled = state["agent_active"] & ~state["episode_done"]
        return (mask & enabled[:,None]).at[:,0].set(True)

    def observe(self, state):
        agents = jnp.concatenate((state["agent_pos"].astype(jnp.float32) / jnp.array([10.0,6.0]),
                                  state["agent_active"][:,None].astype(jnp.float32), state["oxygen"][:,None] / 12.0,
                                  (state["ore"][:,None] + 1) / 3.0,
                                  (state["carrying_canister"][:,None] + 1) / 2.0), axis=1).reshape(-1)
        veins = jnp.concatenate((state["vein_pos"].astype(jnp.float32) / jnp.array([10.0,6.0]),
                                 state["vein_type"][:,None] / 2.0,
                                 state["vein_remaining"][:,None] / 2.0), axis=1).reshape(-1)
        cans = jnp.concatenate((state["canister_pos"].astype(jnp.float32) / jnp.array([10.0,6.0]),
                                (state["canister_owner"][:,None] + 1) / 3.0,
                                state["canister_volume"][:,None] / 6.0), axis=1).reshape(-1)
        station = jnp.concatenate((state["station_pos"][0].astype(jnp.float32) / jnp.array([10.0,6.0]),
                                   state["station_stock"].astype(jnp.float32) / 24.0,
                                   (state["station_occupant"].astype(jnp.float32) + 1.0) / 3.0))
        common = jnp.concatenate((agents, veins, cans, station, state["delivered_counts"] / 6.0,
                                  state["lost_counts"] / 6.0))
        tail = jnp.concatenate((jnp.array([state["t"] / 80.0, state["episode_done"]], jnp.float32),
                                state["terrain"].reshape(-1).astype(jnp.float32),
                                state["mined"].astype(jnp.float32), state["delivery_event"].astype(jnp.float32),
                                state["transferred"].astype(jnp.float32), state["used"].astype(jnp.float32),
                                state["died"].astype(jnp.float32), state["refilled"].astype(jnp.float32),
                                state["picked"].astype(jnp.float32), state["dropped"].astype(jnp.float32)))
        return jnp.concatenate((jnp.broadcast_to(common, (3,48)), jax.nn.one_hot(state["claim"], 3),
                                jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(tail, (3,99))), axis=1).astype(jnp.float32)

    def advance(self, state, actions):
        clipped = jnp.clip(actions, 0, 9)
        mask = self.legal_actions(state)
        actions = jnp.where((actions >= 0) & (actions < 10) & mask[jnp.arange(3),clipped], clipped, 0)
        alive = state["agent_active"]
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0],[0,0],[0,0],[0,0],[0,0],[0,0]], jnp.int32)
        proposed = state["agent_pos"] + delta[actions]
        others = ~jnp.eye(3, dtype=jnp.bool_)
        occupied = jnp.any(jnp.all(proposed[:,None,:] == state["agent_pos"][None,:,:], axis=2)
                           & others & alive[None,:], axis=1)
        contested = jnp.any(jnp.all(proposed[:,None,:] == proposed[None,:,:], axis=2)
                            & others & alive[None,:], axis=1)
        pos = jnp.where((occupied | contested | ~alive)[:,None], state["agent_pos"], proposed)
        owner = state["canister_owner"]
        can_pos = state["canister_pos"]
        volume = state["canister_volume"]
        picked = jnp.zeros((2,), jnp.bool_)
        dropped = jnp.zeros((2,), jnp.bool_)
        transferred = jnp.zeros((2,), jnp.bool_)
        used = jnp.zeros((2,), jnp.bool_)
        oxygen = state["oxygen"]
        for c in range(2):
            prior = owner[c]
            safe_owner = jnp.maximum(prior, 0)
            current_pos = jnp.where(prior >= 0, pos[safe_owner], can_pos[c])
            dropping = (prior >= 0) & (actions[safe_owner] == 6)
            busy = jnp.any(owner[:,None] == jnp.arange(3)[None,:], axis=0)
            near = jnp.sum(jnp.abs(pos - current_pos[None,:]), axis=1) <= 1
            eligible = near & alive & (actions == 6) & ~busy & (state["carrying_canister"] < 0)
            chosen = jnp.min(jnp.where(eligible, jnp.arange(3), 3))
            picking = (prior < 0) & (chosen < 3)
            current = jnp.where(dropping, -1, jnp.where(picking, chosen, prior))
            current_pos = jnp.where(current >= 0, pos[jnp.maximum(current, 0)], current_pos)
            owner = owner.at[c].set(current)
            can_pos = can_pos.at[c].set(current_pos)
            picked = picked.at[c].set(picking)
            dropped = dropped.at[c].set(dropping)
        for c in range(2):
            donor = owner[c]
            safe_donor = jnp.maximum(donor, 0)
            busy = jnp.any(owner[:,None] == jnp.arange(3)[None,:], axis=0)
            near = jnp.sum(jnp.abs(pos - pos[safe_donor][None,:]), axis=1) == 1
            eligible = near & alive & ~busy & (actions == 8)
            recipient = jnp.min(jnp.where(eligible, jnp.arange(3), 3))
            consent = (donor >= 0) & (actions[safe_donor] == 7) & (recipient < 3)
            current = jnp.where(consent, recipient, donor)
            owner = owner.at[c].set(current)
            can_pos = can_pos.at[c].set(jnp.where(current >= 0, pos[jnp.maximum(current, 0)], can_pos[c]))
            transferred = transferred.at[c].set(consent)
            safe_current = jnp.maximum(current, 0)
            drawing = (current >= 0) & (actions[safe_current] == 8)
            amount = jnp.where(drawing, jnp.minimum(volume[c], 12 - oxygen[safe_current]), 0).astype(jnp.int32)
            oxygen = oxygen.at[safe_current].add(amount)
            volume = volume.at[c].add(-amount)
            used = used.at[c].set(amount > 0)
        carrying = jnp.max(jnp.where(owner[:,None] == jnp.arange(3)[None,:], jnp.arange(2)[:,None], -1), axis=0).astype(jnp.int32)
        at_station = jnp.all(pos == state["station_pos"][0], axis=1) & alive
        stock = state["station_stock"][0]
        refilled = jnp.zeros((3,), jnp.bool_)
        for a in range(3):
            refill = at_station[a] & (actions[a] == 9)
            budget = jnp.where(refill, jnp.minimum(stock, 4), 0)
            personal = jnp.minimum(budget, 12 - oxygen[a])
            c = jnp.maximum(carrying[a], 0)
            portable = jnp.where(carrying[a] >= 0, jnp.minimum(budget - personal, 6 - volume[c]), 0)
            oxygen = oxygen.at[a].add(personal)
            volume = volume.at[c].add(portable)
            stock = stock - personal - portable
            refilled = refilled.at[a].set((personal + portable) > 0)
        ore = state["ore"]
        remaining = state["vein_remaining"]
        mined = jnp.zeros((3,), jnp.bool_)
        for order in range(3):
            a = jnp.mod(state["t"] + order, 3)
            near = jnp.sum(jnp.abs(state["vein_pos"] - pos[a][None,:]), axis=1) <= 1
            available = near & (remaining > 0)
            vein = jnp.min(jnp.where(available, jnp.arange(3), 3))
            extraction = alive[a] & (actions[a] == 5) & (ore[a] < 0) & (vein < 3)
            safe_vein = jnp.minimum(vein, 2)
            ore = ore.at[a].set(jnp.where(extraction, state["vein_type"][safe_vein], ore[a]))
            remaining = remaining.at[safe_vein].add(-extraction.astype(jnp.int32))
            mined = mined.at[a].set(extraction)
        delivery = at_station & (ore >= 0)
        matching = ore == state["claim"]
        delivered_add = jnp.sum(((ore[:,None] == jnp.arange(3)[None,:]) & delivery[:,None]).astype(jnp.int32), axis=0)
        ore = jnp.where(delivery, -1, ore).astype(jnp.int32)
        oxygen = jnp.maximum(oxygen - (alive & ~at_station).astype(jnp.int32), 0).astype(jnp.int32)
        died = alive & (oxygen == 0)
        lost_add = jnp.sum(((ore[:,None] == jnp.arange(3)[None,:]) & died[:,None]).astype(jnp.int32), axis=0)
        ore = jnp.where(died, -1, ore).astype(jnp.int32)
        active = alive & ~died
        for c in range(2):
            carrier_died = (owner[c] >= 0) & died[jnp.maximum(owner[c], 0)]
            owner = owner.at[c].set(jnp.where(carrier_died, -1, owner[c]))
            dropped = dropped.at[c].set(dropped[c] | carrier_died)
        carrying = jnp.max(jnp.where(owner[:,None] == jnp.arange(3)[None,:], jnp.arange(2)[:,None], -1), axis=0).astype(jnp.int32)
        occupant = jnp.min(jnp.where(at_station & active, jnp.arange(3), 3))
        occupant = jnp.where(occupant < 3, occupant, -1).astype(jnp.int32)
        ns = dict(state)
        ns["t"] = state["t"] + jnp.array(1, jnp.int32)
        ns["agent_pos"] = pos
        ns["agent_active"] = active
        ns["memorial_active"] = ~active
        ns["oxygen"] = oxygen
        ns["ore"] = ore
        ns["carrying_canister"] = carrying
        ns["vein_remaining"] = remaining
        ns["canister_pos"] = can_pos
        ns["canister_owner"] = owner
        ns["canister_volume"] = volume
        ns["station_stock"] = jnp.array([stock], jnp.int32)
        ns["station_occupant"] = jnp.array([occupant], jnp.int32)
        ns["delivered_counts"] = state["delivered_counts"] + delivered_add
        ns["lost_counts"] = state["lost_counts"] + lost_add
        ns["mined"] = mined
        ns["delivery_event"] = delivery
        ns["transferred"] = transferred
        ns["used"] = used
        ns["died"] = died
        ns["refilled"] = refilled
        ns["picked"] = picked
        ns["dropped"] = dropped
        ns["terminated"] = (~jnp.any(active)) | ((jnp.sum(remaining) == 0) & jnp.all(ore < 0))
        ns["truncated"] = (ns["t"] >= 80) & ~ns["terminated"]
        ns["episode_done"] = ns["terminated"] | ns["truncated"]
        reward = jnp.full((3,), 0.1 * jnp.sum(delivery.astype(jnp.float32)) - 0.25 * jnp.sum(died.astype(jnp.float32)), jnp.float32)
        reward = reward + jnp.where(delivery, jnp.where(matching, 1.0, 0.25), 0.0)
        return ns, reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        proposed, reward = self.advance(state, actions)
        ns = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, proposed)
        reward = jnp.where(state["episode_done"], jnp.zeros((3,), jnp.float32), reward)
        dones = jnp.full((3,), ns["episode_done"], jnp.bool_)
        info = {"done_all": ns["episode_done"], "terminated": ns["terminated"], "truncated": ns["truncated"]}
        return self.observe(ns), ns, reward.astype(jnp.float32), dones, info
