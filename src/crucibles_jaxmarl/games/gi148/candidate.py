"""Echo hunt and distributed local sensing. Original author kernel; unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 160
    num_actions = 12
    max_steps = 72

    def reset(self, key):
        # Deterministic setup; no hidden RNG stored in the state.
        xy = jnp.stack((jnp.arange(77) % 11, jnp.arange(77) // 11), axis=1)
        x, y = xy[:, 0], xy[:, 1]
        wall = (x == 0) | (x == 10) | (y == 0) | (y == 6)
        wall = wall | ((x == 5) & ((y == 1) | (y == 2) | (y == 4) | (y == 5)))
        wall = wall | ((x == 7) & ((y == 2) | (y == 4)))
        tiles = jnp.where(wall, 1, 0).reshape(7, 11).astype(jnp.int32)
        tiles = tiles.at[2, 4].set(2).at[4, 6].set(2).at[1, 9].set(3).at[5, 9].set(3)
        p = jnp.array([[3, 3], [3, 4], [5, 3], [7, 3]], jnp.int32)
        s = {
            "t": jnp.array(0, jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "tiles": tiles, "agent_pos": p,
            "agent_active": jnp.ones(4, jnp.bool_), "agent_vis": jnp.zeros((4, 4), jnp.bool_),
            "team": jnp.array([0, 0, 1, 1], jnp.int32),
            "tokens": jnp.full(4, 3, jnp.int32), "cooldown": jnp.zeros(4, jnp.int32),
            "cloak": jnp.zeros(4, jnp.int32), "status": jnp.zeros(4, jnp.int32),
            "beacon_pos": jnp.array([[4, 2], [6, 4]], jnp.int32),
            "beacon_active": jnp.ones(2, jnp.bool_), "beacon_vis": jnp.ones((4, 2), jnp.bool_),
            "beacon_ttl": jnp.zeros(2, jnp.int32), "beacon_jam": jnp.zeros(2, jnp.int32),
            "beacon_charge": jnp.full(2, 3, jnp.int32),
            "signal_pos": p, "signal_active": jnp.zeros(4, jnp.bool_),
            "signal_vis": jnp.zeros((4, 4), jnp.bool_),
            "signal_ttl": jnp.zeros(4, jnp.int32),
            "signal_kind": jnp.array([0, 0, 1, 1], jnp.int32),
            "exit_pos": jnp.array([[9, 1], [9, 5]], jnp.int32),
            "exit_active": jnp.ones(2, jnp.bool_), "exit_vis": jnp.ones((4, 2), jnp.bool_),
            "track_ttl": jnp.zeros(2, jnp.int32),
            "captured": jnp.array(0, jnp.int32), "escaped": jnp.array(0, jnp.int32),
            "capture_event": jnp.zeros(2, jnp.bool_), "escape_event": jnp.zeros(2, jnp.bool_),
            "pulse_event": jnp.zeros(4, jnp.bool_), "beacon_event": jnp.zeros(2, jnp.bool_),
            "jam_event": jnp.zeros(2, jnp.bool_), "tag_event": jnp.zeros(2, jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def sensing(self, state):
        s = state
        p = s["agent_pos"]
        dx = jnp.abs(p[:, None, 0] - p[None, :, 0])
        dy = jnp.abs(p[:, None, 1] - p[None, :, 1])
        ids = jnp.arange(4)
        teammate = s["team"][:, None] == s["team"][None, :]
        # SCOUT_0 senses horizontally; SCOUT_1 vertically; runners use radius two.
        footprint = (dx + dy <= 1) | ((ids[:, None] == 0) & (dy == 0) & (dx <= 3)) | ((ids[:, None] == 1) & (dx == 0) & (dy <= 3))
        footprint = jnp.where(ids[:, None] >= 2, dx + dy <= 2, footprint)
        cloaked_far = (s["cloak"][None, :] > 0) & (dx + dy > 1) & (ids[:, None] < 2)
        local = footprint & ~cloaked_far
        beacon_range = jnp.sum(jnp.abs(s["beacon_pos"][:, None, :] - p[None, :, :]), axis=2) <= 2
        beacon_seen = jnp.any(beacon_range & (s["beacon_ttl"][:, None] > 0), axis=0)
        # Beacons pierce smoke; a prior pulse tracks live positions until its timer expires.
        tracked = jnp.concatenate((jnp.zeros(2, jnp.bool_), s["track_ttl"] > 0))
        network = (ids[:, None] < 2) & (beacon_seen[None, :] | (tracked[None, :] & (s["cloak"][None, :] == 0)))
        return teammate | local | network

    def refresh(self, state):
        s = dict(state)
        s["agent_active"] = s["status"] == 0
        s["agent_vis"] = self.sensing(s)
        same_team = s["team"][:, None] == s["team"][None, :]
        close_signal = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["signal_pos"][None, :, :]), axis=2) <= 2
        s["signal_vis"] = same_team | close_signal
        s["signal_active"] = s["signal_ttl"] > 0
        s["captured"] = jnp.sum((s["status"][2:] == 1).astype(jnp.int32))
        s["escaped"] = jnp.sum((s["status"][2:] == 2).astype(jnp.int32))
        return s

    def records(self, position, attributes, visibility):
        values = jnp.concatenate((position, attributes), axis=1)
        values = jnp.where(visibility[:, :, None], values[None, :, :], -1)
        return jnp.concatenate((visibility[:, :, None].astype(jnp.int32), values), axis=2).reshape(4, -1)

    def observe(self, state):
        s = state
        own = jnp.concatenate((jnp.full((4, 1), s["t"], jnp.int32), jnp.arange(4)[:, None],
                               s["team"][:, None], s["agent_pos"], s["tokens"][:, None],
                               s["cooldown"][:, None], s["cloak"][:, None],
                               jnp.full((4, 1), s["done"], jnp.int32)), axis=1)
        count = jnp.broadcast_to(jnp.stack((s["captured"], s["escaped"])), (4, 2))
        agents = self.records(s["agent_pos"], jnp.stack((s["team"], s["tokens"], s["cooldown"], s["cloak"], s["status"]), axis=1), s["agent_vis"])
        beacons = self.records(s["beacon_pos"], jnp.stack((s["beacon_ttl"], s["beacon_jam"], s["beacon_charge"]), axis=1), s["beacon_vis"])
        signals = self.records(s["signal_pos"], jnp.stack((s["signal_ttl"], s["signal_kind"]), axis=1), s["signal_vis"])
        exits = self.records(s["exit_pos"], jnp.zeros((2, 0), jnp.int32), s["exit_vis"])
        tracks = jnp.where(jnp.arange(4)[:, None] < 2, s["track_ttl"][None, :], -1)
        return jnp.concatenate((own, count, jnp.broadcast_to(s["tiles"].reshape(1, 77), (4, 77)),
                                agents, beacons, signals, exits, tracks), axis=1).astype(jnp.float32)

    def legal_actions(self, state):
        s = state
        p = s["agent_pos"]
        ids = jnp.arange(4)
        d = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]], jnp.int32)
        one = p[:, None, :] + d[None, :, :]
        two = one + d[None, :, :]
        x1, y1 = jnp.clip(one[:, :, 0], 0, 10), jnp.clip(one[:, :, 1], 0, 6)
        x2, y2 = jnp.clip(two[:, :, 0], 0, 10), jnp.clip(two[:, :, 1], 0, 6)
        inside1 = (one[:, :, 0] > 0) & (one[:, :, 0] < 10) & (one[:, :, 1] > 0) & (one[:, :, 1] < 6)
        inside2 = (two[:, :, 0] > 0) & (two[:, :, 0] < 10) & (two[:, :, 1] > 0) & (two[:, :, 1] < 6)
        move = inside1 & (s["tiles"][y1, x1] != 1)
        ready = (s["tokens"] > 0) & (s["cooldown"] == 0)
        dash = move & inside2 & (s["tiles"][y2, x2] != 1) & ready[:, None]
        near = jnp.sum(jnp.abs(p[:, None, :] - s["beacon_pos"][None, :, :]), axis=2) <= 1
        can_activate = jnp.any(near & (s["beacon_jam"][None, :] == 0) & (s["beacon_charge"][None, :] > 0), axis=1)
        can_jam = jnp.any(near, axis=1) & ready
        beacon = jnp.where(ids < 2, can_activate, can_jam)
        exit_here = jnp.any(jnp.all(p[:, None, :] == s["exit_pos"][None, :, :], axis=2), axis=1)
        tag_or_escape = jnp.where(ids < 2, s["cooldown"] == 0, exit_here)
        mask = jnp.concatenate((jnp.ones((4, 1), jnp.bool_), move,
                                jnp.stack((ready, beacon, tag_or_escape), axis=1), dash), axis=1)
        return jnp.where((s["done"] | ~s["agent_active"])[:, None], jnp.arange(12)[None, :] == 0, mask)

    def raw_step(self, key, state, actions):
        actions = jnp.asarray(actions, jnp.int32)
        clipped = jnp.clip(actions, 0, 11)
        valid = (actions == clipped) & self.legal_actions(state)[jnp.arange(4), clipped]
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, jnp.where(valid, clipped, 0)))

    def absorb(self, bundle):
        key, s, a = bundle
        return self.observe(s), s, jnp.zeros(4, jnp.float32), jnp.ones(4, jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def advance(self, bundle):
        key, old, a = bundle
        s = dict(old)
        ids = jnp.arange(4)
        rank = (ids - old["t"]) % 4
        p = old["agent_pos"]
        normal = (a >= 1) & (a <= 4)
        dash = a >= 8
        direction_id = jnp.where(dash, a - 7, jnp.where(normal, a, 0))
        delta = jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], jnp.int32)[direction_id]
        still_moving = normal | dash
        # MOVEMENT: two simultaneous substeps; blocked first dash hop cancels its second.
        for hop in range(2):
            wants = still_moving & ((hop == 0) | dash)
            target = p + delta * wants[:, None]
            occupied = jnp.any(jnp.all(target[:, None, :] == p[None, :, :], axis=2) & old["agent_active"][None, :] & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
            contested = jnp.sum((jnp.all(target[:, None, :] == target[None, :, :], axis=2) & old["agent_active"][None, :]).astype(jnp.int32), axis=1) > 1
            ok = wants & ~occupied & ~contested
            p = jnp.where(ok[:, None], target, p)
            still_moving = still_moving & ok
        s["agent_pos"] = p
        pulse = a == 5
        jam_action = (a == 6) & (ids >= 2)
        tag = (a == 7) & (ids < 2)
        cost = (pulse | dash | jam_action).astype(jnp.int32)
        s["tokens"] = old["tokens"] - cost
        s["cooldown"] = jnp.where(pulse | dash | jam_action | tag, 2, jnp.maximum(old["cooldown"] - 1, 0))
        s["cloak"] = jnp.where(pulse & (ids >= 2), 2, jnp.maximum(old["cloak"] - 1, 0))
        s["signal_ttl"] = jnp.where(pulse, 2, jnp.maximum(old["signal_ttl"] - 1, 0))
        s["signal_pos"] = jnp.where(pulse[:, None], p, old["signal_pos"])
        s["pulse_event"] = pulse
        s["tag_event"] = tag[:2]
        # BEACON: one activation consumes one charge; concurrent jamming wins and spends no charge.
        near = jnp.sum(jnp.abs(p[:, None, :] - old["beacon_pos"][None, :, :]), axis=2) <= 1
        activation_eligible = near & (old["beacon_jam"][None, :] == 0) & (old["beacon_charge"][None, :] > 0)
        activate_choice = jnp.argmax(activation_eligible, axis=1)
        jam_choice = jnp.argmax(near, axis=1)
        ttl = jnp.maximum(old["beacon_ttl"] - 1, 0)
        jam = jnp.maximum(old["beacon_jam"] - 1, 0)
        charge = old["beacon_charge"]
        activated = jnp.zeros(2, jnp.bool_)
        jammed = jnp.zeros(2, jnp.bool_)
        for b in range(2):
            activators = (a == 6) & (ids < 2) & activation_eligible[:, b] & (activate_choice == b)
            jammers = jam_action & near[:, b] & (jam_choice == b)
            jam_now = jnp.any(jammers)
            activate = jnp.any(activators) & ~jam_now
            ttl = ttl.at[b].set(jnp.where(jam_now, 0, jnp.where(activate, 4, ttl[b])))
            jam = jam.at[b].set(jnp.where(jam_now, 2, jam[b]))
            charge = charge.at[b].add(-activate.astype(jnp.int32))
            activated = activated.at[b].set(activate)
            jammed = jammed.at[b].set(jam_now)
        s["beacon_ttl"], s["beacon_jam"], s["beacon_charge"] = ttl, jam, charge
        s["beacon_event"], s["jam_event"] = activated, jammed
        # DISTRIBUTED_SCAN: pulse sees its author's directional footprint at resolved positions.
        dx = jnp.abs(p[:2, None, 0] - p[None, 2:, 0])
        dy = jnp.abs(p[:2, None, 1] - p[None, 2:, 1])
        footprint = (dx + dy <= 1) | ((jnp.arange(2)[:, None] == 0) & (dy == 0) & (dx <= 3)) | ((jnp.arange(2)[:, None] == 1) & (dx == 0) & (dy <= 3))
        detectable = footprint & ((s["cloak"][None, 2:] == 0) | (dx + dy <= 1))
        found = jnp.any(detectable & pulse[:2, None], axis=0) & old["agent_active"][2:]
        track = jnp.where(found, 3, jnp.maximum(old["track_ttl"] - 1, 0))
        s["track_ttl"] = jnp.where(s["cloak"][2:] > 0, 0, track)
        s = self.refresh(s)
        # CAPTURE_BEFORE_ESCAPE: tags require adjacency and current legitimate visibility.
        distance = jnp.sum(jnp.abs(p[:2, None, :] - p[None, 2:, :]), axis=2)
        captured = jnp.any((distance <= 1) & tag[:2, None] & s["agent_vis"][:2, 2:], axis=0) & old["agent_active"][2:]
        at_exit = jnp.any(jnp.all(p[2:, None, :] == old["exit_pos"][None, :, :], axis=2), axis=1)
        escaped = (a[2:] == 7) & at_exit & old["agent_active"][2:] & ~captured
        s["status"] = s["status"].at[2:].set(jnp.where(captured, 1, jnp.where(escaped, 2, old["status"][2:])))
        s["capture_event"], s["escape_event"] = captured, escaped
        s["track_ttl"] = jnp.where(captured | escaped, 0, s["track_ttl"])
        s = self.refresh(s)
        s["t"] = old["t"] + 1
        s["terminated"] = jnp.all(s["status"][2:] != 0)
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        margin = 6 * (jnp.sum(captured.astype(jnp.int32)) - jnp.sum(escaped.astype(jnp.int32)))
        reward = jnp.where(s["team"] == 0, margin, -margin).astype(jnp.float32)
        dones = (s["status"] != 0) | s["done"]
        return self.observe(s), s, reward, dones, {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

