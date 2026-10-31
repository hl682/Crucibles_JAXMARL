"""Phase-Lock Relay. Original gate scheduling rules; not executed by author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 148
    num_actions = 8
    max_steps = 96

    def reset(self, key):
        terrain = jnp.zeros((7,11), jnp.int32)
        terrain = terrain.at[:,4].set(1).at[:,7].set(1)
        terrain = terrain.at[1,4].set(2).at[5,4].set(2).at[1,7].set(2).at[5,7].set(2)
        phase = jnp.array([0,3,2,5], jnp.int32)
        state = {
            "t": jnp.array(0, jnp.int32),
            "episode_done": jnp.array(False, jnp.bool_),
            "terminated": jnp.array(False, jnp.bool_),
            "truncated": jnp.array(False, jnp.bool_),
            "terrain": terrain,
            "agent_pos": jnp.array([[1,1],[2,2],[6,2]], jnp.int32),
            "agent_active": jnp.ones((3,), jnp.bool_),
            "agent_visible": jnp.ones((3,3), jnp.bool_),
            "lock_timer": jnp.zeros((3,), jnp.int32),
            "carrying": jnp.full((3,), -1, jnp.int32),
            "cargo_pos": jnp.array([[1,1],[1,5]], jnp.int32),
            "cargo_owner": jnp.full((2,), -1, jnp.int32),
            "cargo_active": jnp.ones((2,), jnp.bool_),
            "cargo_visible": jnp.ones((3,2), jnp.bool_),
            "delivered": jnp.zeros((2,), jnp.bool_),
            "cargo_id": jnp.array([0,1], jnp.int32),
            "gate_pos": jnp.array([[4,1],[4,5],[7,1],[7,5]], jnp.int32),
            "gate_active": jnp.ones((4,), jnp.bool_),
            "gate_visible": jnp.ones((3,4), jnp.bool_),
            "gate_phase": phase,
            "gate_offset": phase,
            "gate_override": jnp.zeros((4,), jnp.int32),
            "gate_open": phase < 2,
            "switch_pos": jnp.array([[2,2],[2,4],[6,2],[6,4]], jnp.int32),
            "switch_active": jnp.ones((4,), jnp.bool_),
            "switch_visible": jnp.ones((3,4), jnp.bool_),
            "switch_gate": jnp.arange(4, dtype=jnp.int32),
            "dock_pos": jnp.array([[9,1],[9,5]], jnp.int32),
            "dock_active": jnp.ones((2,), jnp.bool_),
            "dock_visible": jnp.ones((3,2), jnp.bool_),
            "gate_changed": jnp.zeros((4,), jnp.bool_),
            "commanded": jnp.zeros((4,), jnp.bool_),
            "delivery_event": jnp.zeros((2,), jnp.bool_),
            "picked": jnp.zeros((2,), jnp.bool_),
            "dropped": jnp.zeros((2,), jnp.bool_),
        }
        return self.observe(state), state

    def legal_actions(self, state):
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]], jnp.int32)
        target = state["agent_pos"][:,None,:] + delta[None,:,:]
        inside = jnp.all((target >= jnp.array([0,0])) & (target <= jnp.array([10,6])), axis=2)
        safe = jnp.clip(target, jnp.array([0,0]), jnp.array([10,6]))
        passable = state["terrain"][safe[:,:,1],safe[:,:,0]] != 1
        grounded = (state["cargo_owner"] < 0) & state["cargo_active"]
        on_cargo = jnp.any(jnp.all(state["agent_pos"][:,None,:] == state["cargo_pos"][None,:,:], axis=2)
                          & grounded[None,:], axis=1)
        handle = (state["carrying"] >= 0) | on_cargo
        station = jnp.any(jnp.all(state["agent_pos"][:,None,:] == state["switch_pos"][None,:,:], axis=2), axis=1)
        mask = jnp.concatenate((inside & passable, handle[:,None], station[:,None], station[:,None]), axis=1)
        free = (state["lock_timer"] == 0) & ~state["episode_done"]
        return (mask & free[:,None]).at[:,0].set(True)

    def observe(self, state):
        agents = jnp.concatenate((state["agent_pos"].astype(jnp.float32) / jnp.array([10.0,6.0]),
                                  state["lock_timer"][:,None] / 2.0,
                                  (state["carrying"][:,None] + 1) / 2.0), axis=1).reshape(-1)
        cargo = jnp.concatenate((state["cargo_pos"].astype(jnp.float32) / jnp.array([10.0,6.0]),
                                 (state["cargo_owner"][:,None] + 1) / 3.0,
                                 state["delivered"][:,None].astype(jnp.float32)), axis=1).reshape(-1)
        gates = jnp.concatenate((state["gate_pos"].astype(jnp.float32) / jnp.array([10.0,6.0]),
                                 state["gate_phase"][:,None] / 5.0,
                                 state["gate_override"][:,None] / 3.0,
                                 state["gate_open"][:,None].astype(jnp.float32)), axis=1).reshape(-1)
        common = jnp.concatenate((agents, cargo, gates,
                                  (state["switch_pos"].astype(jnp.float32) / jnp.array([10.0,6.0])).reshape(-1),
                                  (state["dock_pos"].astype(jnp.float32) / jnp.array([10.0,6.0])).reshape(-1)))
        tail = jnp.concatenate((jnp.array([state["t"] / 96.0, state["episode_done"]], jnp.float32),
                                state["terrain"].reshape(-1).astype(jnp.float32) / 2.0,
                                state["gate_changed"].astype(jnp.float32), state["commanded"].astype(jnp.float32),
                                state["delivery_event"].astype(jnp.float32), state["picked"].astype(jnp.float32),
                                state["dropped"].astype(jnp.float32)))
        return jnp.concatenate((jnp.broadcast_to(common, (3,52)), jnp.eye(3, dtype=jnp.float32),
                                jnp.broadcast_to(tail, (3,93))), axis=1).astype(jnp.float32)

    def advance(self, state, actions):
        clipped = jnp.clip(actions, 0, 7)
        mask = self.legal_actions(state)
        actions = jnp.where((actions >= 0) & (actions < 8) & mask[jnp.arange(3),clipped], clipped, 0)
        at_switch = jnp.all(state["agent_pos"][:,None,:] == state["switch_pos"][None,:,:], axis=2)
        hold = jnp.any(at_switch & (actions == 6)[:,None], axis=0)
        shift = jnp.any(at_switch & (actions == 7)[:,None], axis=0)
        offset = jnp.mod(state["gate_offset"] + 3 * shift.astype(jnp.int32), 6).astype(jnp.int32)
        override = jnp.where(hold, 3, jnp.maximum(state["gate_override"] - 1, 0)).astype(jnp.int32)
        t = state["t"] + jnp.array(1, jnp.int32)
        phase = jnp.mod(t + offset, 6).astype(jnp.int32)
        gate_open = (phase < 2) | (override > 0)
        lock = jnp.maximum(state["lock_timer"] - 1, 0)
        lock = jnp.where(actions == 6, 2, jnp.where(actions == 7, 1, lock)).astype(jnp.int32)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0],[0,0],[0,0],[0,0]], jnp.int32)
        proposed = state["agent_pos"] + delta[actions]
        others = ~jnp.eye(3, dtype=jnp.bool_)
        occupied = jnp.any(jnp.all(proposed[:,None,:] == state["agent_pos"][None,:,:], axis=2) & others, axis=1)
        contested = jnp.any(jnp.all(proposed[:,None,:] == proposed[None,:,:], axis=2) & others, axis=1)
        gate_block = jnp.any(jnp.all(proposed[:,None,:] == state["gate_pos"][None,:,:], axis=2) & ~gate_open[None,:], axis=1)
        pos = jnp.where((occupied | contested | gate_block)[:,None], state["agent_pos"], proposed)
        owner = state["cargo_owner"]
        cargo_pos = state["cargo_pos"]
        delivered = state["delivered"]
        picked = jnp.zeros((2,), jnp.bool_)
        dropped = jnp.zeros((2,), jnp.bool_)
        delivery = jnp.zeros((2,), jnp.bool_)
        for c in range(2):
            prior_owner = owner[c]
            safe_owner = jnp.maximum(prior_owner, 0)
            current_pos = jnp.where(prior_owner >= 0, pos[safe_owner], cargo_pos[c])
            dropping = (prior_owner >= 0) & (actions[safe_owner] == 5)
            busy = jnp.any(owner[:,None] == jnp.arange(3)[None,:], axis=0)
            eligible = jnp.all(pos == current_pos[None,:], axis=1) & (actions == 5) & ~busy
            chosen = jnp.min(jnp.where(eligible, jnp.arange(3), 3))
            picking = (prior_owner < 0) & ~delivered[c] & (chosen < 3)
            current_owner = jnp.where(dropping, -1, jnp.where(picking, chosen, prior_owner))
            arriving = (current_owner >= 0) & jnp.all(current_pos == state["dock_pos"][c]) & ~delivered[c]
            current_owner = jnp.where(arriving, -1, current_owner)
            owner = owner.at[c].set(current_owner)
            cargo_pos = cargo_pos.at[c].set(current_pos)
            delivered = delivered.at[c].set(delivered[c] | arriving)
            picked = picked.at[c].set(picking)
            dropped = dropped.at[c].set(dropping)
            delivery = delivery.at[c].set(arriving)
        carrying = jnp.max(jnp.where(owner[:,None] == jnp.arange(3)[None,:], jnp.arange(2)[:,None], -1), axis=0).astype(jnp.int32)
        ns = dict(state)
        ns["t"] = t
        ns["agent_pos"] = pos
        ns["lock_timer"] = lock
        ns["carrying"] = carrying
        ns["cargo_pos"] = cargo_pos
        ns["cargo_owner"] = owner
        ns["delivered"] = delivered
        ns["cargo_active"] = ~delivered
        ns["gate_offset"] = offset
        ns["gate_phase"] = phase
        ns["gate_override"] = override
        ns["gate_open"] = gate_open
        ns["gate_changed"] = gate_open != state["gate_open"]
        ns["commanded"] = hold | shift
        ns["delivery_event"] = delivery
        ns["picked"] = picked
        ns["dropped"] = dropped
        ns["terminated"] = jnp.all(delivered)
        ns["truncated"] = (t >= 96) & ~ns["terminated"]
        ns["episode_done"] = ns["terminated"] | ns["truncated"]
        reward = jnp.full((3,), jnp.sum(delivery.astype(jnp.float32)), jnp.float32)
        reward = reward - ((actions == 6) | (actions == 7)).astype(jnp.float32) * 0.005 - 0.001
        return ns, reward

    def raw_step(self, key, state, actions):
        proposed, reward = self.advance(state, actions)
        ns = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, proposed)
        reward = jnp.where(state["episode_done"], jnp.zeros((3,), jnp.float32), reward)
        dones = jnp.full((3,), ns["episode_done"], jnp.bool_)
        info = {"done_all": ns["episode_done"], "terminated": ns["terminated"], "truncated": ns["truncated"]}
        return self.observe(ns), ns, reward.astype(jnp.float32), dones, info
