"""Original scout-relay corridor contest with expiring reports and public lures."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 145
    num_actions = 9
    max_steps = 80

    def refresh(self, state):
        nxt = dict(state)
        pos = state["agent_pos"]
        radius = jnp.array([3, 1, 3, 1], dtype=jnp.int32)
        distance = jnp.sum(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=2)
        visible = distance <= radius[:, None]
        gate_visible = jnp.sum(jnp.abs(pos[:, None, :] - state["gate_pos"][None, :, :]), axis=2) <= radius[:, None]
        scouts = jnp.array([0, 2], dtype=jnp.int32)
        targets = jnp.array([3, 1], dtype=jnp.int32)
        seen = visible[scouts, targets]
        age = jnp.where(seen, 0, jnp.minimum(state["memory_target_age"] + 1, 9))
        target = jnp.where(seen[:, None], pos[targets], state["memory_target"])
        target = jnp.where((age < 9)[:, None], target, -1)
        gate_seen = gate_visible[scouts]
        gate_age = jnp.where(gate_seen, 0, jnp.minimum(state["memory_gate_age"] + 1, 9))
        gate_memory = jnp.where(gate_seen, state["gate_open"][None, :].astype(jnp.int32), state["memory_gate"])
        gate_memory = jnp.where(gate_age < 9, gate_memory, -1)
        nxt["agent_visible"] = visible
        nxt["gate_visible"] = gate_visible
        nxt["parcel_visible"] = visible[:, 1:2]
        nxt["memory_target"] = target
        nxt["memory_target_age"] = age.astype(jnp.int32)
        nxt["memory_gate"] = gate_memory.astype(jnp.int32)
        nxt["memory_gate_age"] = gate_age.astype(jnp.int32)
        return nxt

    def reset(self, key):
        x = jnp.arange(9)[None, :]
        y = jnp.arange(7)[:, None]
        corridor = (y == 1) | (y == 3) | (y == 5)
        terrain = jnp.where((x == 4) & ~corridor, 1, 0).astype(jnp.int32)
        terrain = jnp.where((x == 4) & corridor, 2, terrain).astype(jnp.int32)
        closed = jax.random.randint(key, (), 0, 3, dtype=jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "episode_done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
            "terrain": terrain,
            "agent_pos": jnp.array([[2, 1], [0, 3], [6, 5], [8, 3]], dtype=jnp.int32),
            "agent_active": jnp.ones((4,), dtype=jnp.bool_),
            "agent_visible": jnp.eye(4, dtype=jnp.bool_),
            "role": jnp.arange(4, dtype=jnp.int32),
            "energy": jnp.array([3, 0, 3, 0], dtype=jnp.int32),
            "gate_pos": jnp.array([[4, 1], [4, 3], [4, 5]], dtype=jnp.int32),
            "gate_open": jnp.arange(3) != closed,
            "gate_cooldown": jnp.zeros((3,), dtype=jnp.int32),
            "gate_active": jnp.ones((3,), dtype=jnp.bool_),
            "gate_visible": jnp.zeros((4, 3), dtype=jnp.bool_),
            "parcel_pos": jnp.array([[0, 3]], dtype=jnp.int32),
            "parcel_status": jnp.zeros((1,), dtype=jnp.int32),
            "parcel_active": jnp.ones((1,), dtype=jnp.bool_),
            "parcel_visible": jnp.zeros((4, 1), dtype=jnp.bool_),
            "exit_pos": jnp.array([[8, 1], [8, 5]], dtype=jnp.int32),
            "exit_active": jnp.ones((2,), dtype=jnp.bool_),
            "exit_visible": jnp.ones((4, 2), dtype=jnp.bool_),
            "memory_target": jnp.full((2, 2), -1, dtype=jnp.int32),
            "memory_target_age": jnp.full((2,), 9, dtype=jnp.int32),
            "memory_gate": jnp.full((2, 3), -1, dtype=jnp.int32),
            "memory_gate_age": jnp.full((2, 3), 9, dtype=jnp.int32),
            "report_pos": jnp.array([[2, 1], [6, 5]], dtype=jnp.int32),
            "report_target": jnp.full((2, 2), -1, dtype=jnp.int32),
            "report_target_age": jnp.full((2,), 9, dtype=jnp.int32),
            "report_gate": jnp.full((2, 3), -1, dtype=jnp.int32),
            "report_gate_age": jnp.full((2, 3), 9, dtype=jnp.int32),
            "report_ttl": jnp.zeros((2,), dtype=jnp.int32),
            "report_listeners": jnp.zeros((4, 2), dtype=jnp.bool_),
            "report_active": jnp.zeros((2,), dtype=jnp.bool_),
            "report_visible": jnp.zeros((4, 2), dtype=jnp.bool_),
            "beacon_pos": jnp.array([[0, 0], [8, 6]], dtype=jnp.int32),
            "beacon_ttl": jnp.zeros((2,), dtype=jnp.int32),
            "beacon_claim": jnp.array([0, 1], dtype=jnp.int32),
            "beacon_active": jnp.zeros((2,), dtype=jnp.bool_),
            "beacon_visible": jnp.ones((4, 2), dtype=jnp.bool_),
            "decoy_stock": jnp.full((2,), 2, dtype=jnp.int32),
            "outcome": jnp.array(0, dtype=jnp.int32),
            "relay_event": jnp.zeros((2,), dtype=jnp.int32),
            "toggle_event": jnp.zeros((3,), dtype=jnp.int32),
            "beacon_event": jnp.zeros((2,), dtype=jnp.int32)
        }
        state = self.refresh(state)
        return self.observe(state), state

    def observe(self, state):
        own = jnp.concatenate((jnp.eye(4, dtype=jnp.float32), state["agent_pos"] / 8.0,
                               state["energy"][:, None] / 3.0), axis=1)
        agents = jnp.where(state["agent_visible"][:, :, None], state["agent_pos"][None, :, :] / 8.0, -1.0).reshape(4, 8)
        local_agents = jnp.concatenate((agents, state["agent_visible"].astype(jnp.float32)), axis=1)
        gates = jnp.concatenate((jnp.where(state["gate_visible"], state["gate_open"][None, :].astype(jnp.float32), -1.0),
                                jnp.where(state["gate_visible"], state["gate_cooldown"][None, :] / 4.0, -1.0),
                                state["gate_visible"].astype(jnp.float32)), axis=1)
        parcel = jnp.concatenate((jnp.where(state["parcel_visible"][:, :, None], state["parcel_pos"][None, :, :] / 8.0, -1.0).reshape(4, 2),
                                 jnp.where(state["parcel_visible"], state["parcel_status"][None, :] / 2.0, -1.0)), axis=1)
        mt = jnp.where(state["memory_target"] >= 0, state["memory_target"] / 8.0, -1.0)
        memory = jnp.concatenate((mt, state["memory_target_age"][:, None] / 9.0,
                                  state["memory_gate"].astype(jnp.float32), state["memory_gate_age"] / 9.0), axis=1)
        notes = jnp.full((4, 9), -1.0, dtype=jnp.float32).at[0].set(memory[0]).at[2].set(memory[1])
        rv = state["report_visible"]
        rt = jnp.where(state["report_target"] >= 0, state["report_target"] / 8.0, -1.0)
        reports = jnp.concatenate((
            jnp.where(rv[:, :, None], state["report_pos"][None, :, :] / 8.0, -1.0).reshape(4, 4),
            jnp.where(rv[:, :, None], rt[None, :, :], -1.0).reshape(4, 4),
            jnp.where(rv, state["report_target_age"][None, :] / 9.0, -1.0),
            jnp.where(rv[:, :, None], state["report_gate"][None, :, :].astype(jnp.float32), -1.0).reshape(4, 6),
            jnp.where(rv[:, :, None], state["report_gate_age"][None, :, :] / 9.0, -1.0).reshape(4, 6),
            jnp.where(rv, state["report_ttl"][None, :] / 4.0, -1.0), rv.astype(jnp.float32)
        ), axis=1)
        public = jnp.concatenate((
            jnp.where(state["beacon_active"][:, None], state["beacon_pos"] / 8.0, -1.0).reshape(-1),
            state["beacon_ttl"] / 5.0, state["beacon_active"].astype(jnp.float32), state["decoy_stock"] / 2.0,
            jnp.reshape(state["t"] / 80.0, (1,)), state["exit_pos"].reshape(-1) / 8.0,
            jnp.reshape(state["outcome"] / 2.0, (1,)), state["terrain"].reshape(-1) / 2.0
        )).astype(jnp.float32)
        return jnp.concatenate((own, local_agents, gates, parcel, notes, reports,
                                jnp.broadcast_to(public, (4, 79))), axis=1).astype(jnp.float32)

    def legal_actions(self, state):
        return jnp.where(state["episode_done"], jnp.arange(9)[None, :] == 0,
                         jnp.ones((4, 9), dtype=jnp.bool_))

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions, 0, 8)
        a = jnp.where((actions >= 0) & (actions < 9) &
                      self.legal_actions(state)[jnp.arange(4), safe], safe, 0)
        scout_role = jnp.array([True, False, True, False])
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)
        proposed = jnp.clip(state["agent_pos"] + delta[a], jnp.array([0, 0]), jnp.array([8, 6]))
        tile = state["terrain"][proposed[:, 1], proposed[:, 0]]
        gate_here = jnp.all(proposed[:, None, :] == state["gate_pos"][None, :, :], axis=2)
        passable = (tile != 1) & ((tile != 2) | scout_role | jnp.any(gate_here & state["gate_open"][None, :], axis=1))
        pos = jnp.where(passable[:, None], proposed, state["agent_pos"])
        conflict = jnp.all(pos[1] == pos[3])
        courier_blocked = jnp.all(pos[1] == state["agent_pos"][3]) | conflict
        interceptor_blocked = jnp.all(pos[3] == state["agent_pos"][1]) | conflict
        pos = pos.at[1].set(jnp.where(courier_blocked, state["agent_pos"][1], pos[1]))
        pos = pos.at[3].set(jnp.where(interceptor_blocked, state["agent_pos"][3], pos[3]))
        near_gate = jnp.sum(jnp.abs(pos[:, None, :] - state["gate_pos"][None, :, :]), axis=2) <= 1
        eligible = near_gate & (state["gate_cooldown"][None, :] == 0)
        operate = (a == 6) & scout_role & (state["energy"] > 0) & jnp.any(eligible, axis=1)
        target = jnp.argmax(eligible, axis=1)
        toggles = jnp.sum((jnp.arange(3)[None, :] == target[:, None]) * operate[:, None], axis=0).astype(jnp.int32)
        gate_open = jnp.where(toggles % 2 == 1, ~state["gate_open"], state["gate_open"])
        cooldown = jnp.where(toggles > 0, 4, jnp.maximum(state["gate_cooldown"] - 1, 0))
        energy = jnp.clip(state["energy"] + ((a == 0) & scout_role).astype(jnp.int32) - operate.astype(jnp.int32), 0, 3)
        nxt = dict(state)
        nxt["agent_pos"] = pos
        nxt["gate_open"] = gate_open
        nxt["gate_cooldown"] = cooldown.astype(jnp.int32)
        nxt["energy"] = energy
        nxt["parcel_pos"] = pos[1:2]
        nxt = self.refresh(nxt)
        scouts = jnp.array([0, 2], dtype=jnp.int32)
        broadcast = a[scouts] == 5
        ttl = jnp.where(broadcast, 4, jnp.maximum(state["report_ttl"] - 1, 0))
        report_pos = jnp.where(broadcast[:, None], pos[scouts], state["report_pos"])
        report_target = jnp.where(broadcast[:, None], nxt["memory_target"], state["report_target"])
        report_age = jnp.where(broadcast, nxt["memory_target_age"], jnp.minimum(state["report_target_age"] + 1, 9))
        report_gate = jnp.where(broadcast[:, None], nxt["memory_gate"], state["report_gate"])
        report_gate_age = jnp.where(broadcast[:, None], nxt["memory_gate_age"], jnp.minimum(state["report_gate_age"] + 1, 9))
        team = jnp.arange(4) // 2
        teammates = team[:, None] == jnp.arange(2)[None, :]
        nearby_listener = jnp.sum(jnp.abs(pos[:, None, :] - pos[scouts][None, :, :]), axis=2) <= 2
        listeners = jnp.where(broadcast[None, :], teammates | nearby_listener, state["report_listeners"])
        report_visible = listeners & (ttl > 0)[None, :]
        beacon_pos = state["beacon_pos"]
        beacon_ttl = jnp.maximum(state["beacon_ttl"] - 1, 0)
        stock = state["decoy_stock"]
        beacon_event = jnp.zeros((2,), dtype=jnp.int32)
        for side in range(2):
            emitters = (a == 8) & (team == side)
            emit = jnp.any(emitters) & (stock[side] > 0)
            emitter = jnp.argmax(emitters)
            beacon_pos = beacon_pos.at[side].set(jnp.where(emit, pos[emitter], beacon_pos[side]))
            beacon_ttl = beacon_ttl.at[side].set(jnp.where(emit, 5, beacon_ttl[side]))
            stock = stock.at[side].add(-emit.astype(jnp.int32))
            beacon_event = beacon_event.at[side].set(emit.astype(jnp.int32))
        caught = (a[3] == 7) & (jnp.sum(jnp.abs(pos[3] - pos[1])) <= 1)
        delivered = (a[1] == 7) & jnp.any(jnp.all(pos[1] == state["exit_pos"], axis=1)) & ~caught
        outcome = jnp.where(caught, 2, jnp.where(delivered, 1, 0)).astype(jnp.int32)
        t = state["t"] + jnp.array(1, dtype=jnp.int32)
        terminated = caught | delivered
        truncated = (t >= self.max_steps) & ~terminated
        nxt["t"] = t
        nxt["report_pos"] = report_pos
        nxt["report_target"] = report_target
        nxt["report_target_age"] = report_age.astype(jnp.int32)
        nxt["report_gate"] = report_gate
        nxt["report_gate_age"] = report_gate_age.astype(jnp.int32)
        nxt["report_ttl"] = ttl.astype(jnp.int32)
        nxt["report_listeners"] = listeners
        nxt["report_active"] = ttl > 0
        nxt["report_visible"] = report_visible
        nxt["beacon_pos"] = beacon_pos
        nxt["beacon_ttl"] = beacon_ttl
        nxt["beacon_active"] = beacon_ttl > 0
        nxt["decoy_stock"] = stock
        nxt["outcome"] = outcome
        nxt["parcel_status"] = jnp.reshape(outcome, (1,))
        nxt["relay_event"] = broadcast.astype(jnp.int32)
        nxt["toggle_event"] = toggles
        nxt["beacon_event"] = beacon_event
        nxt["terminated"] = terminated
        nxt["truncated"] = truncated
        nxt["episode_done"] = terminated | truncated
        advantage = delivered.astype(jnp.float32) - caught.astype(jnp.float32)
        reward = jnp.array([1.0, 1.0, -1.0, -1.0], dtype=jnp.float32) * advantage
        nxt = jax.tree.map(lambda new, old: jnp.where(state["episode_done"], old, new), nxt, state)
        reward = jnp.where(state["episode_done"], jnp.zeros((4,), dtype=jnp.float32), reward).astype(jnp.float32)
        dones = jnp.full((4,), nxt["episode_done"], dtype=jnp.bool_)
        info = {"done_all": nxt["episode_done"], "terminated": nxt["terminated"], "truncated": nxt["truncated"]}
        return self.observe(nxt), nxt, reward, dones, info
