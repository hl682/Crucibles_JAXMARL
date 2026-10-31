"""Shepherd Circles: shared pressure fields, costly calls and timed gates."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 12
    observation_size = 414
    max_steps = 96

    def refresh(self, s):
        s = dict(s)
        s["flock_active"] = s["flock_pen"] < 0
        s["call_active"] = s["call_ttl"] > 0
        s["agent_visible"] = jnp.ones((3, 3), dtype=jnp.bool_)
        s["flock_visible"] = jnp.broadcast_to(s["flock_active"], (3, 6))
        s["call_visible"] = jnp.broadcast_to(s["call_active"], (3, 3))
        s["field_active"] = jnp.any(s["field_direction"] != 0, axis=1)
        s["field_visible"] = jnp.broadcast_to(s["field_active"], (3, 99))
        s["gate_open"] = s["gate_timer"] > 0
        s["pen_count"] = jnp.sum(s["flock_pen"][None, :] == jnp.arange(2)[:, None], axis=1, dtype=jnp.int32)
        return s

    def reset(self, key):
        coords = jnp.stack([jnp.arange(99) % 11, jnp.arange(99) // 11], axis=1).astype(jnp.int32)
        wall = (jnp.arange(11)[None, :] == 8) & ~((jnp.arange(9)[:, None] == 2) | (jnp.arange(9)[:, None] == 6))
        gate = (jnp.arange(11)[None, :] == 8) & ((jnp.arange(9)[:, None] == 2) | (jnp.arange(9)[:, None] == 6))
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "episode_done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "agent_pos": jnp.array([[1, 2], [1, 6], [6, 4]], dtype=jnp.int32),
            "agent_energy": jnp.full(3, 4, dtype=jnp.int32),
            "agent_pressure": jnp.zeros(3, dtype=jnp.int32),
            "agent_active": jnp.ones(3, dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "flock_pos": jnp.array([[3, 2], [4, 2], [3, 3], [3, 5], [4, 6], [3, 6]], dtype=jnp.int32),
            "flock_target": jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32),
            "flock_pen": jnp.full(6, -1, dtype=jnp.int32),
            "flock_correct": jnp.zeros(6, dtype=jnp.bool_),
            "flock_drive": jnp.zeros((6, 2), dtype=jnp.int32),
            "flock_active": jnp.ones(6, dtype=jnp.bool_),
            "flock_visible": jnp.ones((3, 6), dtype=jnp.bool_),
            "gate_pos": jnp.array([[8, 2], [8, 6]], dtype=jnp.int32),
            "gate_timer": jnp.zeros(2, dtype=jnp.int32),
            "gate_open": jnp.zeros(2, dtype=jnp.bool_),
            "gate_active": jnp.ones(2, dtype=jnp.bool_),
            "gate_visible": jnp.ones((3, 2), dtype=jnp.bool_),
            "pen_pos": jnp.array([[10, 2], [10, 6]], dtype=jnp.int32),
            "pen_count": jnp.zeros(2, dtype=jnp.int32),
            "pen_active": jnp.ones(2, dtype=jnp.bool_),
            "pen_visible": jnp.ones((3, 2), dtype=jnp.bool_),
            "call_pos": jnp.zeros((3, 2), dtype=jnp.int32),
            "call_ttl": jnp.zeros(3, dtype=jnp.int32),
            "call_owner": jnp.arange(3, dtype=jnp.int32),
            "call_active": jnp.zeros(3, dtype=jnp.bool_),
            "call_visible": jnp.zeros((3, 3), dtype=jnp.bool_),
            "field_pos": coords, "field_direction": jnp.zeros((99, 2), dtype=jnp.int32),
            "field_active": jnp.zeros(99, dtype=jnp.bool_),
            "field_visible": jnp.zeros((3, 99), dtype=jnp.bool_),
            "call_event": jnp.zeros(3, dtype=jnp.bool_),
            "gate_event": jnp.zeros(2, dtype=jnp.bool_),
            "arrival_event": jnp.zeros(6, dtype=jnp.bool_),
            "wrong_event": jnp.zeros(6, dtype=jnp.bool_),
            "blocked_event": jnp.zeros(3, dtype=jnp.bool_),
            "flock_block_event": jnp.zeros(6, dtype=jnp.bool_),
            "map": wall.astype(jnp.int32) + 2 * gate.astype(jnp.int32),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        fields = ("agent_pos", "agent_energy", "agent_pressure", "flock_pos", "flock_target",
                  "flock_pen", "flock_correct", "flock_drive", "flock_active", "gate_pos",
                  "gate_timer", "gate_open", "pen_pos", "pen_count", "call_pos", "call_ttl",
                  "call_active", "field_direction", "field_active", "call_event", "gate_event",
                  "arrival_event", "wrong_event", "blocked_event", "flock_block_event")
        pub = jnp.concatenate([s[f].ravel().astype(jnp.float32) / 16.0 for f in fields] +
                              [jnp.array([s["t"] / 128.0, s["episode_done"]], dtype=jnp.float32)])
        return jnp.concatenate([jnp.eye(3, dtype=jnp.float32),
                                jnp.broadcast_to(pub, (3, pub.shape[0]))], axis=1)

    def legal_actions(self, s):
        return jnp.broadcast_to((jnp.arange(12) == 0) | ~s["episode_done"], (3, 12))

    def traversable(self, pos, s, timers):
        q = jnp.clip(pos, jnp.array([0, 0]), jnp.array([10, 8]))
        inside = jnp.all(pos == q, axis=1)
        tile = s["map"][q[:, 1], q[:, 0]]
        at_gate = jnp.all(q[:, None, :] == s["gate_pos"][None, :, :], axis=2)
        opened = jnp.any(at_gate & (timers > 0)[None, :], axis=1)
        return inside & (tile != 1) & ((tile != 2) | opened)

    def raw_step(self, key, state, actions):
        s = dict(state)
        a = jnp.where((actions >= 0) & (actions < 12), actions, 0)
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        moving = (a >= 1) & (a <= 4)
        proposal = s["agent_pos"] + jnp.where(moving[:, None], dirs[jnp.clip(a - 1, 0, 3)], 0)
        # Previous frame's gate state controls shepherd movement; fresh pulls will affect flock movement.
        free = self.traversable(proposal, s, s["gate_timer"])
        occupied_agent = jnp.any(jnp.all(proposal[:, None, :] == s["agent_pos"][None, :, :], axis=2) &
                                 (jnp.arange(3)[:, None] != jnp.arange(3)[None, :]), axis=1)
        occupied_flock = jnp.any(jnp.all(proposal[:, None, :] == s["flock_pos"][None, :, :], axis=2) & s["flock_active"][None, :], axis=1)
        possible = moving & free & ~occupied_agent & ~occupied_flock
        priority = (jnp.arange(3) - s["t"]) % 3
        beaten = jnp.any(jnp.all(proposal[:, None, :] == proposal[None, :, :], axis=2) &
                         possible[None, :] & (priority[None, :] < priority[:, None]), axis=1)
        moved = possible & ~beaten
        pos = jnp.where(moved[:, None], proposal, s["agent_pos"])
        energy = jnp.minimum(4, s["agent_energy"] + (a == 0).astype(jnp.int32))
        calling = (a >= 5) & (a <= 8) & (energy > 0)
        pressing = (a == 9) & (energy > 0)
        near_gate = jnp.sum(jnp.abs(pos[:, None, :] - s["gate_pos"][None, :, :]), axis=2) <= 1
        pulling = (a == 10) & (energy > 0) & jnp.any(near_gate, axis=1)
        energy = energy - (calling | pressing | pulling).astype(jnp.int32)
        ct = jnp.where(calling, 2, jnp.maximum(0, s["call_ttl"] - 1))
        call_target = jnp.clip(pos + 2 * dirs[jnp.clip(a - 5, 0, 3)], jnp.array([0, 0]), jnp.array([10, 8]))
        cp = jnp.where(calling[:, None], call_target, s["call_pos"])
        # Multiple gate pulls refresh one timer, never add unbounded duration.
        gate_event = jnp.any(near_gate & pulling[:, None], axis=0)
        gt = jnp.where(gate_event, 3, jnp.maximum(0, s["gate_timer"] - 1))
        pressure = jnp.where(pressing, 2, moving.astype(jnp.int32))
        # Field vectors are the actual bounded inputs to flock motion, also bound to native animation.
        delta = s["field_pos"][:, None, :] - pos[None, :, :]
        distance = jnp.sum(jnp.abs(delta), axis=2)
        radius = jnp.where(pressing, 3, 1)
        repulsion = jnp.sum(jnp.sign(delta) * ((distance <= radius[None, :]) & (distance > 0))[:, :, None] * pressure[None, :, None], axis=1)
        call_delta = cp[None, :, :] - s["field_pos"][:, None, :]
        call_distance = jnp.sum(jnp.abs(call_delta), axis=2)
        attraction = jnp.sum(jnp.sign(call_delta) * ((call_distance <= 3) & (ct > 0)[None, :])[:, :, None], axis=1)
        field = (repulsion + attraction).astype(jnp.int32)
        fp = s["flock_pos"]
        drive = field[fp[:, 1] * 11 + fp[:, 0]]
        neighbour_delta = fp[None, :, :] - fp[:, None, :]
        neighbour_dist = jnp.sum(jnp.abs(neighbour_delta), axis=2)
        neighbours = (neighbour_dist <= 2) & (neighbour_dist > 0) & s["flock_active"][None, :]
        cohesion = jnp.sign(jnp.sum(neighbour_delta * neighbours[:, :, None], axis=1))
        force = drive + jnp.where(jnp.any(drive != 0, axis=1)[:, None], cohesion, 0)
        prefer_x = (jnp.abs(force[:, 0]) > jnp.abs(force[:, 1])) | ((jnp.abs(force[:, 0]) == jnp.abs(force[:, 1])) & ((jnp.arange(6) + s["t"]) % 2 == 0))
        direction = jnp.stack([jnp.where(prefer_x, jnp.sign(force[:, 0]), 0),
                                jnp.where(~prefer_x, jnp.sign(force[:, 1]), 0)], axis=1).astype(jnp.int32)
        direction = jnp.where(s["flock_active"][:, None], direction, 0)
        proposed_flock = fp + direction
        free = self.traversable(proposed_flock, s, gt)
        occupied = jnp.any(jnp.all(proposed_flock[:, None, :] == fp[None, :, :], axis=2) &
                           (jnp.arange(6)[:, None] != jnp.arange(6)[None, :]) & s["flock_active"][None, :], axis=1)
        shepherd = jnp.any(jnp.all(proposed_flock[:, None, :] == pos[None, :, :], axis=2), axis=1)
        can_move = free & ~occupied & ~shepherd & s["flock_active"] & jnp.any(direction != 0, axis=1)
        sheep_priority = (jnp.arange(6) - s["t"]) % 6
        beaten = jnp.any(jnp.all(proposed_flock[:, None, :] == proposed_flock[None, :, :], axis=2) &
                         can_move[None, :] & (sheep_priority[None, :] < sheep_priority[:, None]), axis=1)
        flock_moves = can_move & ~beaten
        new_fp = jnp.where(flock_moves[:, None], proposed_flock, fp)
        top = (new_fp[:, 0] >= 9) & (new_fp[:, 1] >= 1) & (new_fp[:, 1] <= 3)
        bottom = (new_fp[:, 0] >= 9) & (new_fp[:, 1] >= 5) & (new_fp[:, 1] <= 7)
        arrival = s["flock_active"] & (top | bottom)
        pen = jnp.where(arrival, bottom.astype(jnp.int32), s["flock_pen"])
        correct_arrival = arrival & (pen == s["flock_target"])
        wrong = arrival & ~correct_arrival
        s["agent_pos"], s["agent_energy"], s["agent_pressure"] = pos, energy, pressure
        s["flock_pos"], s["flock_pen"] = new_fp, pen
        s["flock_correct"] = s["flock_correct"] | correct_arrival
        s["flock_drive"] = direction
        s["gate_timer"], s["call_pos"], s["call_ttl"] = gt, cp, ct
        s["field_direction"] = field
        s["call_event"], s["gate_event"], s["arrival_event"], s["wrong_event"] = calling, gate_event, arrival, wrong
        s["blocked_event"] = moving & ~moved
        s["flock_block_event"] = s["flock_active"] & jnp.any(direction != 0, axis=1) & ~flock_moves
        s["t"] = state["t"] + jnp.array(1, dtype=jnp.int32)
        s["terminated"] = jnp.all(pen >= 0)
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        good, bad = correct_arrival.astype(jnp.float32), wrong.astype(jnp.float32)
        common = jnp.sum(good) - 0.5 * jnp.sum(bad)
        specialist = jnp.arange(3)[:, None] == s["flock_target"][None, :]
        reward = jnp.full(3, common, dtype=jnp.float32) + 0.25 * jnp.sum(specialist.astype(jnp.float32) * good[None, :], axis=1)
        s = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, s)
        reward = jnp.where(state["episode_done"], jnp.zeros(3, dtype=jnp.float32), reward)
        info = {"done_all": s["episode_done"], "terminated": s["terminated"], "truncated": s["truncated"]}
        return self.observe(s), s, reward.astype(jnp.float32), jnp.full((3,), s["episode_done"], dtype=jnp.bool_), info
