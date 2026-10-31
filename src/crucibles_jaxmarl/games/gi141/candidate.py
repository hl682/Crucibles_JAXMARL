"""Lantern Strait: original private manifests, public leases and tidal passages."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 168
    num_actions = 9
    max_steps = 80

    def refresh(self, state):
        n = dict(state)
        phase = jnp.mod(state["t"], 8)
        n["tide_phase"] = phase
        n["gate_open"] = jnp.stack((jnp.asarray(True), phase < 4))
        n["agent_active"] = ~state["finished"]
        n["grounded"] = jnp.all(state["agent_pos"] == state["gate_pos"][1], axis=1) & ~(phase < 4) & ~state["finished"]
        n["cargo_pos"] = state["agent_pos"]
        n["cargo_active"] = ~state["finished"] & ~state["stranded"]
        n["cargo_status"] = jnp.where(state["finished"], 1, jnp.where(state["stranded"], 2, 0)).astype(jnp.int32)
        n["goal_active"] = ~state["finished"] & ~state["stranded"]
        tiles = jnp.zeros((7, 9), dtype=jnp.int32).at[:, 4].set(1)
        tiles = tiles.at[2, 4].set(0).at[4, 4].set(jnp.where(phase < 4, 2, 3))
        for b in range(6):
            tiles = tiles.at[state["berth_pos"][b, 1], state["berth_pos"][b, 0]].set(4)
        n["tiles"] = tiles
        return n

    def reset(self, key):
        berths = jnp.array([[1, 1], [1, 3], [1, 5], [7, 1], [7, 3], [7, 5]], dtype=jnp.int32)
        goal_id = jnp.array([3, 0, 3], dtype=jnp.int32) + jax.random.randint(key, (3,), 0, 3, dtype=jnp.int32)
        state = {
            "t": jnp.asarray(0, dtype=jnp.int32),
            "episode_done": jnp.asarray(False), "terminated": jnp.asarray(False), "truncated": jnp.asarray(False),
            "agent_pos": jnp.array([[2, 2], [6, 2], [2, 4]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_), "agent_visibility": jnp.ones((3, 3), dtype=jnp.bool_),
            "fuel": jnp.full((3,), 28, dtype=jnp.int32),
            "signal": jnp.zeros((3,), dtype=jnp.int32), "signal_ttl": jnp.zeros((3,), dtype=jnp.int32),
            "service": jnp.zeros((3,), dtype=jnp.int32),
            "finished": jnp.zeros((3,), dtype=jnp.bool_), "stranded": jnp.zeros((3,), dtype=jnp.bool_),
            "grounded": jnp.zeros((3,), dtype=jnp.bool_),
            "goal_id": goal_id, "goal_pos": berths[goal_id],
            "goal_active": jnp.ones((3,), dtype=jnp.bool_), "goal_visibility": jnp.eye(3, dtype=jnp.bool_),
            "cargo_pos": jnp.array([[2, 2], [6, 2], [2, 4]], dtype=jnp.int32),
            "cargo_active": jnp.ones((3,), dtype=jnp.bool_), "cargo_visibility": jnp.ones((3, 3), dtype=jnp.bool_),
            "cargo_owner": jnp.arange(3, dtype=jnp.int32), "cargo_status": jnp.zeros((3,), dtype=jnp.int32),
            "berth_pos": berths, "berth_active": jnp.ones((6,), dtype=jnp.bool_),
            "berth_visibility": jnp.ones((3, 6), dtype=jnp.bool_),
            "berth_owner": jnp.full((6,), -1, dtype=jnp.int32), "berth_timer": jnp.zeros((6,), dtype=jnp.int32),
            "gate_pos": jnp.array([[4, 2], [4, 4]], dtype=jnp.int32),
            "gate_active": jnp.ones((2,), dtype=jnp.bool_), "gate_visibility": jnp.ones((3, 2), dtype=jnp.bool_),
            "lease_owner": jnp.full((2,), -1, dtype=jnp.int32), "lease_ttl": jnp.zeros((2,), dtype=jnp.int32),
            "gate_open": jnp.ones((2,), dtype=jnp.bool_), "tide_phase": jnp.asarray(0, dtype=jnp.int32),
            "tiles": jnp.zeros((7, 9), dtype=jnp.int32),
            "reservation_event": jnp.zeros((2,), dtype=jnp.bool_),
            "blocked_event": jnp.zeros((3,), dtype=jnp.bool_), "collision_event": jnp.zeros((3,), dtype=jnp.bool_),
            "entered_event": jnp.zeros((3,), dtype=jnp.bool_), "dock_event": jnp.zeros((3,), dtype=jnp.bool_),
            "completed_event": jnp.zeros((3,), dtype=jnp.bool_), "stranded_event": jnp.zeros((3,), dtype=jnp.bool_),
            "grounding_event": jnp.zeros((3,), dtype=jnp.bool_),
        }
        state = self.refresh(state)
        return self.observe(state), state

    def legal_actions(self, state):
        available = ~state["finished"] & ~state["stranded"] & (state["service"] == 0) & (state["fuel"] > 0) & ~state["episode_done"]
        # Maneuver intents may be blocked by simultaneous requests, tides or traffic.
        return available[:, None] | (jnp.arange(9)[None, :] == 0)

    def advance(self, state, actions):
        n = dict(state)
        ids = jnp.arange(3, dtype=jnp.int32)
        clipped = jnp.clip(actions, 0, 8)
        valid = (actions >= 0) & (actions < 9) & self.legal_actions(state)[ids, clipped]
        a = jnp.where(valid, clipped, 0)
        rank = jnp.mod(ids - state["t"], 3)
        n["service"] = jnp.maximum(state["service"] - 1, 0)
        completed = (state["service"] > 0) & (n["service"] == 0)
        n["finished"] = state["finished"] | completed
        n["berth_timer"] = jnp.maximum(state["berth_timer"] - 1, 0)
        n["berth_owner"] = jnp.where(n["berth_timer"] == 0, -1, state["berth_owner"])
        n["signal_ttl"] = jnp.maximum(state["signal_ttl"] - 1, 0)
        n["signal"] = jnp.where(n["signal_ttl"] > 0, state["signal"], 0)
        request = (a == 5) | (a == 6)
        n["signal"] = jnp.where(a == 5, 1, jnp.where(a == 6, 2, jnp.where(a == 7, 3, n["signal"])))
        n["signal_ttl"] = jnp.where(request, 4, jnp.where(a == 7, 2, n["signal_ttl"]))
        n["fuel"] = state["fuel"] - ((a >= 1) & (a <= 7)).astype(jnp.int32)
        n["lease_ttl"] = jnp.maximum(state["lease_ttl"] - 1, 0)
        n["lease_owner"] = jnp.where(n["lease_ttl"] > 0, state["lease_owner"], -1)
        n["reservation_event"] = jnp.zeros((2,), dtype=jnp.bool_)

        # Requests precede movement. One leased entrance is an actual scarce opportunity.
        for g in range(2):
            old_owner = n["lease_owner"][g]
            release = (old_owner >= 0) & ((a[jnp.maximum(old_owner, 0)] == 7) | n["finished"][jnp.maximum(old_owner, 0)])
            owner = jnp.where(release, -1, old_owner)
            ttl = jnp.where(release, 0, n["lease_ttl"][g])
            same_row = state["agent_pos"][:, 1] == state["gate_pos"][g, 1]
            request_east = (a == 5) & (state["agent_pos"][:, 0] == 3)
            request_west = (a == 6) & (state["agent_pos"][:, 0] == 5)
            occupied_gate = jnp.any(jnp.all(state["agent_pos"] == state["gate_pos"][g], axis=1) & ~n["finished"])
            eligible = same_row & (request_east | request_west) & state["gate_open"][g] & ~occupied_gate & ((owner < 0) | (ids == owner))
            winner = jnp.argmin(jnp.where(eligible, rank, 9)).astype(jnp.int32)
            grant = jnp.any(eligible)
            n["lease_owner"] = n["lease_owner"].at[g].set(jnp.where(grant, winner, owner))
            n["lease_ttl"] = n["lease_ttl"].at[g].set(jnp.where(grant, 3, ttl))
            n["reservation_event"] = n["reservation_event"].at[g].set(grant)

        p = state["agent_pos"]
        moving = (a >= 1) & (a <= 4)
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        q = p + directions[jnp.where(moving, a, 0)]
        inside = (q[:, 0] >= 0) & (q[:, 0] <= 8) & (q[:, 1] >= 0) & (q[:, 1] <= 6)
        wall = (q[:, 0] == 4) & (q[:, 1] != 2) & (q[:, 1] != 4)
        at_gate = jnp.all(q[:, None, :] == state["gate_pos"][None, :, :], axis=2)
        allowed_gate = state["gate_open"][None, :] & ((n["lease_owner"][None, :] < 0) | (n["lease_owner"][None, :] == ids[:, None]))
        denied_gate = jnp.any(at_gate & ~allowed_gate, axis=1)
        physical = ~n["finished"]
        occupied = jnp.any(jnp.all(q[:, None, :] == p[None, :, :], axis=2) & physical[None, :] & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        passable = inside & ~wall & ~denied_gate & ~state["grounded"]
        q = jnp.where((moving & passable & ~occupied)[:, None], q, p)
        same_target = jnp.all(q[:, None, :] == q[None, :, :], axis=2)
        conflict = jnp.any(same_target & physical[None, :] & ~jnp.eye(3, dtype=jnp.bool_), axis=1) & moving
        n["agent_pos"] = jnp.where(conflict[:, None], p, q)
        n["blocked_event"] = moving & (~passable | occupied | conflict)
        n["collision_event"] = moving & (occupied | conflict)
        newly_in_gate = jnp.all(n["agent_pos"][:, None, :] == state["gate_pos"][None, :, :], axis=2)
        was_in_gate = jnp.all(p[:, None, :] == state["gate_pos"][None, :, :], axis=2)
        n["entered_event"] = jnp.any(newly_in_gate & ~was_in_gate, axis=1)
        for g in range(2):
            owner = n["lease_owner"][g]
            oi = jnp.maximum(owner, 0)
            exited = (owner >= 0) & was_in_gate[oi, g] & ~newly_in_gate[oi, g]
            n["lease_owner"] = n["lease_owner"].at[g].set(jnp.where(exited, -1, owner))
            n["lease_ttl"] = n["lease_ttl"].at[g].set(jnp.where(exited, 0, n["lease_ttl"][g]))

        # A destination is private; only a successful berth match starts two-step service.
        docked = jnp.zeros((3,), dtype=jnp.bool_)
        for offset in range(3):
            i = jnp.mod(state["t"] + offset, 3)
            berth = state["goal_id"][i]
            at_destination = jnp.all(n["agent_pos"][i] == state["berth_pos"][berth])
            dock = (a[i] == 8) & at_destination & (n["berth_owner"][berth] < 0) & (n["service"][i] == 0) & ~n["finished"][i] & ~state["stranded"][i]
            n["service"] = n["service"].at[i].set(jnp.where(dock, 2, n["service"][i]))
            n["berth_owner"] = n["berth_owner"].at[berth].set(jnp.where(dock, i, n["berth_owner"][berth]))
            n["berth_timer"] = n["berth_timer"].at[berth].set(jnp.where(dock, 2, n["berth_timer"][berth]))
            docked = docked.at[i].set(dock)
        stranded_now = (n["fuel"] == 0) & ~n["finished"] & (n["service"] == 0) & ~state["stranded"]
        n["stranded"] = state["stranded"] | stranded_now
        n["dock_event"] = docked
        n["completed_event"] = completed
        n["stranded_event"] = stranded_now
        n["t"] = state["t"] + 1
        n["terminated"] = jnp.all(n["finished"] | n["stranded"])
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["episode_done"] = n["terminated"] | n["truncated"]
        n = self.refresh(n)
        n["grounding_event"] = n["grounded"] & ~state["grounded"]
        working = ~state["finished"] & ~state["stranded"]
        rewards = (0.25 * jnp.sum(completed.astype(jnp.float32)) + 0.75 * completed.astype(jnp.float32)
                   - 0.5 * stranded_now.astype(jnp.float32) - 0.125 * n["collision_event"].astype(jnp.float32)
                   - working.astype(jnp.float32) / 64.0)
        return n, rewards.astype(jnp.float32)

    def observe(self, state):
        own = jnp.concatenate((jnp.arange(3, dtype=jnp.float32)[:, None] / 4.0,
                                state["goal_id"][:, None] / 8.0, state["goal_pos"] / 16.0), axis=1)
        agents = jnp.concatenate((state["agent_pos"] / 16.0, state["fuel"][:, None] / 32.0,
                                  state["signal"][:, None] / 4.0, state["signal_ttl"][:, None] / 4.0,
                                  state["service"][:, None] / 4.0, state["finished"][:, None].astype(jnp.float32),
                                  state["stranded"][:, None].astype(jnp.float32), state["grounded"][:, None].astype(jnp.float32)), axis=1).reshape((27,))
        cargo = jnp.concatenate((state["cargo_pos"] / 16.0, state["cargo_status"][:, None] / 4.0,
                                 state["cargo_active"][:, None].astype(jnp.float32)), axis=1).reshape((12,))
        berths = jnp.concatenate((state["berth_pos"] / 16.0, (state["berth_owner"][:, None] + 1) / 4.0,
                                  state["berth_timer"][:, None] / 4.0), axis=1).reshape((24,))
        gates = jnp.concatenate((state["gate_pos"] / 16.0, (state["lease_owner"][:, None] + 1) / 4.0,
                                 state["lease_ttl"][:, None] / 4.0, state["gate_open"][:, None].astype(jnp.float32)), axis=1).reshape((10,))
        events = jnp.concatenate((state["reservation_event"], state["blocked_event"], state["collision_event"],
                                  state["entered_event"], state["dock_event"], state["completed_event"],
                                  state["stranded_event"], state["grounding_event"])).astype(jnp.float32)
        public = jnp.concatenate((jnp.array([state["t"] / 128.0, state["tide_phase"] / 8.0], dtype=jnp.float32),
                                  agents, cargo, berths, gates, events, state["tiles"].reshape((63,)) / 4.0))
        return jnp.concatenate((own, jnp.broadcast_to(public, (3, 161)), jnp.zeros((3, 3), dtype=jnp.float32)), axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        next_state, rewards = jax.lax.cond(state["episode_done"],
            lambda s, a: (s, jnp.zeros((3,), dtype=jnp.float32)), self.advance, state, actions)
        dones = jnp.full((3,), next_state["episode_done"], dtype=jnp.bool_)
        info = {"done_all": next_state["episode_done"], "terminated": next_state["terminated"], "truncated": next_state["truncated"]}
        return self.observe(next_state), next_state, rewards, dones, info
