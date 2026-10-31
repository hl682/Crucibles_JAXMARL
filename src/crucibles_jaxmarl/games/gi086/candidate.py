"""Lockstep Gantry: two-end cargo, a shared hydraulic pool, and a timed gate."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 40
    num_actions = 8
    max_steps = 80

    def direction(self, a):
        table = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)
        return table[a]

    def cells(self, center, axis):
        unit = jnp.stack((1 - axis, axis))
        return center + jnp.arange(-1, 2, dtype=jnp.int32)[:, None] * unit

    def handles(self, center, axis):
        unit = jnp.stack((1 - axis, axis))
        return center + jnp.array([-2, 2], dtype=jnp.int32)[:, None] * unit

    def clear(self, s, points, gate_open):
        x = jnp.clip(points[:, 0], 0, 14)
        y = jnp.clip(points[:, 1], 0, 8)
        inside = (points[:, 0] >= 1) & (points[:, 0] <= 13)
        inside = inside & (points[:, 1] >= 1) & (points[:, 1] <= 7)
        gate = jnp.all(points == s["gate_pos"][0], axis=1)
        return inside & (s["tiles"][y, x] != 1) & (~gate | gate_open)

    def reset(self, key):
        x = jnp.arange(15, dtype=jnp.int32)[None, :]
        y = jnp.arange(9, dtype=jnp.int32)[:, None]
        walls = (x == 0) | (x == 14) | (y == 0) | (y == 8)
        walls = walls | ((x == 6) & (y >= 2) & (y <= 7) & (y != 4))
        tiles = walls.astype(jnp.int32)
        tiles = tiles.at[4, 6].set(2)
        tiles = tiles.at[4, 9:12].set(3)
        center = jnp.array([[3, 4]], dtype=jnp.int32)
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "tiles": tiles,
            "agent_pos": jnp.array([[1, 4], [5, 4], [5, 2]], dtype=jnp.int32),
            "agent_active": jnp.ones(3, dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "grip": jnp.full(3, -1, dtype=jnp.int32),
            "cargo_center": center,
            "cargo_pos": self.cells(center[0], jnp.array(0, dtype=jnp.int32)),
            "cargo_axis": jnp.zeros(3, dtype=jnp.int32),
            "cargo_active": jnp.ones(3, dtype=jnp.bool_),
            "cargo_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "gate_pos": jnp.array([[6, 4]], dtype=jnp.int32),
            "gate_active": jnp.ones(1, dtype=jnp.bool_),
            "gate_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "gate_open": jnp.zeros(1, dtype=jnp.bool_),
            "gate_timer": jnp.zeros(1, dtype=jnp.int32),
            "pump_pos": jnp.array([[5, 2]], dtype=jnp.int32),
            "pump_active": jnp.ones(1, dtype=jnp.bool_),
            "pump_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "pressure": jnp.array([6], dtype=jnp.int32),
            "dock_pos": jnp.array([[10, 4]], dtype=jnp.int32),
            "dock_active": jnp.ones(1, dtype=jnp.bool_),
            "dock_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "pump_event": jnp.zeros(3, dtype=jnp.bool_),
            "cargo_moved": jnp.array(False), "cargo_rotated": jnp.array(False),
            "move_count": jnp.array(0, dtype=jnp.int32),
            "jammed": jnp.array(False), "success": jnp.array(False),
            "done": jnp.array(False), "terminated": jnp.array(False),
            "truncated": jnp.array(False),
        }
        return self.observe(s), s

    def observe(self, s):
        scale = jnp.array([14.0, 8.0], dtype=jnp.float32)
        # Public fields 0:37, followed by the three-way observer identity.
        public = jnp.concatenate((
            (s["agent_pos"] / scale).reshape(-1),
            (s["grip"].astype(jnp.float32) + 1.0) / 2.0,
            (s["cargo_pos"] / scale).reshape(-1),
            s["cargo_axis"][:1].astype(jnp.float32),
            (s["dock_pos"] / scale).reshape(-1),
            (s["pump_pos"] / scale).reshape(-1),
            (s["gate_pos"] / scale).reshape(-1),
            s["pressure"].astype(jnp.float32) / 10.0,
            s["gate_timer"].astype(jnp.float32) / 4.0,
            s["gate_open"].astype(jnp.float32),
            jnp.stack((s["t"] / 80.0, s["done"], s["success"], s["jammed"],
                       s["cargo_moved"], s["cargo_rotated"], s["move_count"] / 80.0,
                       s["terminated"], s["truncated"])).astype(jnp.float32),
            s["pump_event"].astype(jnp.float32),
        )).astype(jnp.float32)
        return jnp.concatenate((jnp.broadcast_to(public, (3, 37)),
                                jax.nn.one_hot(jnp.arange(3), 3)), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        pads = self.handles(s["cargo_center"][0], s["cargo_axis"][0])
        at_pad = jnp.any(jnp.all(s["agent_pos"][:, None, :] == pads[None, :, :], axis=2), axis=1)
        gripped = s["grip"] >= 0
        at_pump = jnp.all(s["agent_pos"] == s["pump_pos"][0], axis=1)
        mask = jnp.ones((3, 8), dtype=jnp.bool_)
        mask = mask.at[:, 5].set(gripped | at_pad)
        mask = mask.at[:, 6].set(gripped)
        mask = mask.at[:, 7].set(~gripped & at_pump)
        noop = jnp.broadcast_to(jnp.arange(8) == 0, (3, 8))
        return jnp.where(s["done"], noop, mask)

    def advance(self, payload):
        key, s, actions = payload
        # Every invalid intent is a deterministic WAIT; blocked movement is a legal attempt.
        safe = jnp.clip(actions, 0, 7)
        valid = (actions >= 0) & (actions < 8) & self.legal_actions(s)[jnp.arange(3), safe]
        a = jnp.where(valid, safe, 0)
        pads = self.handles(s["cargo_center"][0], s["cargo_axis"][0])
        at_pad = jnp.all(s["agent_pos"][:, None, :] == pads[None, :, :], axis=2)
        attach_slot = jnp.argmax(at_pad, axis=1).astype(jnp.int32)
        grip = jnp.where(a == 5, jnp.where(s["grip"] >= 0, -1, attach_slot), s["grip"])
        crew = grip >= 0
        pump = a == 7
        pressure = jnp.minimum(10, s["pressure"][0] + 3 * jnp.sum(pump.astype(jnp.int32)))
        timer = jnp.where(jnp.any(pump), 4, jnp.maximum(0, s["gate_timer"][0] - 1))
        gate_open = timer > 0
        # The two ends must agree. A single carrier, release, or disagreeing direction holds the beam.
        ready = (jnp.sum(crew.astype(jnp.int32)) == 2)
        ready = ready & jnp.any(grip == 0) & jnp.any(grip == 1)
        command = jnp.max(jnp.where(crew, a, 0))
        agreement = jnp.all(~crew | (a == command))
        translate = (command >= 1) & (command <= 4)
        rotate = command == 6
        center = s["cargo_center"][0] + jnp.where(translate, self.direction(command), 0)
        axis = jnp.where(rotate, 1 - s["cargo_axis"][0], s["cargo_axis"][0])
        cargo = self.cells(center, axis)
        new_pads = self.handles(center, axis)
        carried_pos = new_pads[jnp.clip(grip, 0, 1)]
        occupied = jnp.concatenate((cargo, new_pads), axis=0)
        other_block = jnp.any(jnp.all(occupied[:, None, :] == s["agent_pos"][None, :, :], axis=2)
                              & (~crew)[None, :])
        clearance = jnp.all(self.clear(s, occupied, gate_open)) & ~other_block
        # Rotation also needs the entire swept 5x5 square to be physically clear.
        sx = jnp.tile(jnp.arange(-2, 3, dtype=jnp.int32), 5)
        sy = jnp.repeat(jnp.arange(-2, 3, dtype=jnp.int32), 5)
        sweep = s["cargo_center"][0] + jnp.stack((sx, sy), axis=1)
        sweep_block = jnp.any(jnp.all(sweep[:, None, :] == s["agent_pos"][None, :, :], axis=2)
                             & (~crew)[None, :])
        rotation_clear = jnp.all(self.clear(s, sweep, gate_open)) & ~sweep_block
        cost = jnp.where(rotate, 2, 1)
        operated = ready & agreement & (translate | rotate) & (pressure >= cost)
        operated = operated & clearance & (~rotate | rotation_clear)
        center = jnp.where(operated, center, s["cargo_center"][0])
        axis = jnp.where(operated, axis, s["cargo_axis"][0])
        cargo = self.cells(center, axis)
        pos = jnp.where((crew & operated)[:, None], carried_pos, s["agent_pos"])
        pressure = pressure - jnp.where(operated, cost, 0)
        # Noncarriers move after cargo; no swaps, entry into occupied cells, or shared destinations.
        proposed = pos + self.direction(a)
        walking = (~crew) & (a >= 1) & (a <= 4)
        hit_agent = jnp.any(jnp.all(proposed[:, None, :] == pos[None, :, :], axis=2)
                            & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        hit_cargo = jnp.any(jnp.all(proposed[:, None, :] == cargo[None, :, :], axis=2), axis=1)
        eligible = walking & self.clear(s, proposed, gate_open) & ~hit_agent & ~hit_cargo
        same = jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2)
        clash = jnp.any(same & eligible[None, :] & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        pos = jnp.where((eligible & ~clash)[:, None], proposed, pos)
        # Expiring the gate while anyone or any beam segment occupies it is a terminal jam.
        in_gate = jnp.any(jnp.all(pos == s["gate_pos"][0], axis=1))
        in_gate = in_gate | jnp.any(jnp.all(cargo == s["gate_pos"][0], axis=1))
        jammed = (~gate_open) & in_gate
        success = jnp.all(center == s["dock_pos"][0]) & (axis == 0) & ~jammed
        t = s["t"] + 1
        terminated = success | jammed
        truncated = (t >= self.max_steps) & ~terminated
        ns = dict(s)
        ns["t"] = t
        ns["agent_pos"] = pos
        ns["grip"] = grip.astype(jnp.int32)
        ns["cargo_center"] = center[None, :]
        ns["cargo_pos"] = cargo
        ns["cargo_axis"] = jnp.full(3, axis, dtype=jnp.int32)
        ns["pressure"] = pressure[None].astype(jnp.int32)
        ns["gate_timer"] = timer[None].astype(jnp.int32)
        ns["gate_open"] = gate_open[None]
        ns["pump_event"] = pump
        ns["cargo_moved"] = operated & translate
        ns["cargo_rotated"] = operated & rotate
        ns["move_count"] = s["move_count"] + (operated & translate).astype(jnp.int32)
        ns["jammed"] = jammed
        ns["success"] = success
        ns["terminated"] = terminated
        ns["truncated"] = truncated
        ns["done"] = terminated | truncated
        reward = jnp.full(3, -0.01, dtype=jnp.float32) - 0.02 * pump.astype(jnp.float32)
        reward = reward + 8.0 * success.astype(jnp.float32) - 5.0 * jammed.astype(jnp.float32)
        info = {"done_all": ns["done"], "terminated": terminated, "truncated": truncated}
        return self.observe(ns), ns, reward.astype(jnp.float32), jnp.full(3, ns["done"]), info

    def absorb(self, payload):
        key, s, actions = payload
        info = {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}
        return self.observe(s), s, jnp.zeros(3, dtype=jnp.float32), jnp.ones(3, dtype=jnp.bool_), info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, actions))
