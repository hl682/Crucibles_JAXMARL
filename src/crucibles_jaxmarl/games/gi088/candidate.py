"""Borrowed Current: route a finite supply while crossing gates spends service power."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 54
    num_actions = 11
    max_steps = 96

    def direction(self, a):
        return jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                          [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)[a]

    def refresh(self, s):
        ns = dict(s)
        distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2)
        ns["agent_visible"] = (distance <= 3) | (s["call_ttl"][None, :] > 0)
        station_distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["station_pos"][None, :, :]), axis=2)
        ns["station_visible"] = station_distance <= 3
        occupied = jnp.any(jnp.all(s["agent_pos"][:, None, :] == s["gate_pos"][None, :, :], axis=2), axis=0)
        ns["gate_open"] = (s["capacitor"] > 0) | occupied
        return ns

    def reset(self, key):
        x = jnp.arange(13, dtype=jnp.int32)[None, :]
        y = jnp.arange(9, dtype=jnp.int32)[:, None]
        walls = (x == 0) | (x == 12) | (y == 0) | (y == 8)
        walls = walls | ((x == 6) & (y >= 1) & (y <= 7) & (y != 2) & (y != 6))
        tiles = walls.astype(jnp.int32)
        tiles = tiles.at[2, 6].set(2)
        tiles = tiles.at[6, 6].set(2)
        tiles = tiles.at[2, 9].set(3)
        tiles = tiles.at[6, 9].set(3)
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "tiles": tiles,
            "agent_pos": jnp.array([[4, 4], [9, 2], [9, 6]], dtype=jnp.int32),
            "agent_active": jnp.ones(3, dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "handcell": jnp.ones(3, dtype=jnp.int32),
            "order": jnp.array([0, 0, 1], dtype=jnp.int32),
            "completed": jnp.zeros(3, dtype=jnp.int32),
            "call_target": jnp.zeros(3, dtype=jnp.int32),
            "call_ttl": jnp.zeros(3, dtype=jnp.int32),
            "generator_pos": jnp.array([[2, 4]], dtype=jnp.int32),
            "generator_active": jnp.ones(1, dtype=jnp.bool_),
            "generator_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "buffer": jnp.array([5], dtype=jnp.int32),
            "console_pos": jnp.array([[4, 4]], dtype=jnp.int32),
            "console_active": jnp.ones(1, dtype=jnp.bool_),
            "console_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "route": jnp.array([2], dtype=jnp.int32),
            "cooldown": jnp.zeros(1, dtype=jnp.int32),
            "station_pos": jnp.array([[9, 2], [9, 6]], dtype=jnp.int32),
            "station_active": jnp.ones(2, dtype=jnp.bool_),
            "station_visible": jnp.ones((3, 2), dtype=jnp.bool_),
            "capacitor": jnp.array([2, 2], dtype=jnp.int32),
            "gate_pos": jnp.array([[6, 2], [6, 6]], dtype=jnp.int32),
            "gate_active": jnp.ones(2, dtype=jnp.bool_),
            "gate_visible": jnp.ones((3, 2), dtype=jnp.bool_),
            "gate_open": jnp.ones(2, dtype=jnp.bool_),
            "switch_event": jnp.array(False),
            "power_event": jnp.zeros(2, dtype=jnp.int32),
            "service_total": jnp.array(0, dtype=jnp.int32),
            "success": jnp.array(False), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        scale = jnp.array([12.0, 8.0], dtype=jnp.float32)
        agent_data = jnp.concatenate((s["agent_pos"] / scale, s["handcell"][:, None] / 2.0,
                                      s["completed"][:, None] / 2.0, s["call_target"][:, None],
                                      s["call_ttl"][:, None] / 3.0), axis=1)
        agents = jnp.where(s["agent_visible"][:, :, None], agent_data[None, :, :], 0.0)
        agents = jnp.concatenate((agents, s["agent_visible"][:, :, None].astype(jnp.float32)), axis=2)
        station_data = jnp.concatenate((s["station_pos"] / scale, s["capacitor"][:, None] / 4.0), axis=1)
        stations = jnp.where(s["station_visible"][:, :, None], station_data[None, :, :], 0.0)
        stations = jnp.concatenate((stations, s["station_visible"][:, :, None].astype(jnp.float32)), axis=2)
        public = jnp.concatenate((jnp.stack((
            s["t"] / 96.0, s["done"], s["success"], s["route"][0] / 2.0,
            s["cooldown"][0] / 2.0, s["buffer"][0] / 10.0,
            s["switch_event"], s["service_total"] / 6.0,
        )), s["gate_open"].astype(jnp.float32), s["power_event"].astype(jnp.float32),
            (s["generator_pos"] / scale).reshape(-1), (s["console_pos"] / scale).reshape(-1),
            (s["gate_pos"] / scale).reshape(-1))).astype(jnp.float32)
        private = jnp.stack((s["order"], s["completed"] / 2.0), axis=1).astype(jnp.float32)
        return jnp.concatenate((jax.nn.one_hot(jnp.arange(3), 3), private,
                                jnp.broadcast_to(public, (3, 20)),
                                agents.reshape(3, 21), stations.reshape(3, 8)), axis=1).astype(jnp.float32)

    def local_station(self, s):
        distances = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["station_pos"][None, :, :]), axis=2)
        return jnp.argmin(distances, axis=1).astype(jnp.int32), jnp.min(distances, axis=1) <= 1

    def legal_actions(self, s):
        station, near = self.local_station(s)
        generator = jnp.sum(jnp.abs(s["agent_pos"] - s["generator_pos"][0]), axis=1) <= 1
        console = jnp.sum(jnp.abs(s["agent_pos"] - s["console_pos"][0]), axis=1) <= 1
        mask = jnp.ones((3, 11), dtype=jnp.bool_)
        mask = mask.at[:, 5].set(console)
        mask = mask.at[:, 6].set(generator)
        # Service/donation/charge are legal requests; simultaneous available energy decides realization.
        mask = mask.at[:, 7].set(near & (station == s["order"]) & (s["completed"] < 2))
        mask = mask.at[:, 9].set(near & (s["handcell"] > 0))
        mask = mask.at[:, 10].set(generator & (s["handcell"] < 2))
        return jnp.where(s["done"], jnp.broadcast_to(jnp.arange(11) == 0, (3, 11)), mask)

    def select_local(self, requests, station, t):
        rank = jnp.mod(jnp.arange(3) - t, 3)
        competition = (station[:, None] == station[None, :]) & requests[None, :]
        beaten = jnp.any(competition & (rank[None, :] < rank[:, None]), axis=1)
        return requests & ~beaten

    def advance(self, payload):
        key, s, actions = payload
        safe = jnp.clip(actions, 0, 10)
        valid = (actions >= 0) & (actions < 11) & self.legal_actions(s)[jnp.arange(3), safe]
        a = jnp.where(valid, safe, 0)
        switched = jnp.any(a == 5)
        route = jnp.where(switched, jnp.mod(s["route"][0] + 1, 3), s["route"][0])
        cooldown = jnp.where(switched, 2, jnp.maximum(0, s["cooldown"][0] - 1))
        cranked = a == 6
        buffer = jnp.clip(s["buffer"][0] + 2 * jnp.sum(cranked.astype(jnp.int32)) - switched.astype(jnp.int32), 0, 10)
        # One handcell charging socket uses two units; the priority rotates every turn.
        charging = self.select_local(a == 10, jnp.zeros(3, dtype=jnp.int32), s["t"]) & (buffer >= 2)
        buffer = buffer - 2 * jnp.sum(charging.astype(jnp.int32))
        handcell = s["handcell"] + charging.astype(jnp.int32)
        # Route 2 alternates north/south with the old clock. Supply is at most one unit per turn.
        destination = jnp.where(route < 2, route, jnp.mod(s["t"], 2))
        powered = (cooldown == 0) & (buffer > 0) & (s["capacitor"][destination] < 4)
        power_event = jax.nn.one_hot(destination, 2, dtype=jnp.int32) * powered.astype(jnp.int32)
        capacitor = s["capacitor"] + power_event
        buffer = buffer - powered.astype(jnp.int32)
        # Gate entry consumes that route's capacitor, reducing what a distant service partner can use.
        pos = s["agent_pos"]
        proposed = pos + self.direction(a)
        gate_hit = jnp.all(proposed[:, None, :] == s["gate_pos"][None, :, :], axis=2)
        gate_clear = jnp.all(~gate_hit | (capacitor[None, :] > 0), axis=1)
        px = jnp.clip(proposed[:, 0], 0, 12)
        py = jnp.clip(proposed[:, 1], 0, 8)
        inside = (proposed[:, 0] > 0) & (proposed[:, 0] < 12) & (proposed[:, 1] > 0) & (proposed[:, 1] < 8)
        occupied = jnp.any(jnp.all(proposed[:, None, :] == pos[None, :, :], axis=2)
                           & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        eligible = (a >= 1) & (a <= 4) & inside & (s["tiles"][py, px] != 1) & gate_clear & ~occupied
        same = jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2)
        clash = jnp.any(same & eligible[None, :] & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        moved = eligible & ~clash
        pos = jnp.where(moved[:, None], proposed, pos)
        crossing = jnp.sum((gate_hit & moved[:, None]).astype(jnp.int32), axis=0)
        capacitor = capacitor - crossing
        station, near = self.local_station(s)
        at = jax.nn.one_hot(station, 2, dtype=jnp.int32)
        donating = self.select_local((a == 9) & (capacitor[station] < 4), station, s["t"])
        capacitor = capacitor + jnp.sum(at * donating[:, None].astype(jnp.int32), axis=0)
        handcell = handcell - donating.astype(jnp.int32)
        serving = self.select_local((a == 7) & (capacitor[station] >= 2), station, s["t"])
        capacitor = capacitor - 2 * jnp.sum(at * serving[:, None].astype(jnp.int32), axis=0)
        completed = s["completed"] + serving.astype(jnp.int32)
        order = jnp.where(serving, 1 - s["order"], s["order"])
        t = s["t"] + 1
        capacitor = jnp.maximum(0, capacitor - (jnp.mod(t, 4) == 0).astype(jnp.int32))
        # A radio call costs a turn and tracks the caller for three turns. Its requested port can become stale.
        call = a == 8
        call_target = jnp.where(call, s["order"], s["call_target"])
        call_ttl = jnp.where(call, 3, jnp.maximum(0, s["call_ttl"] - 1))
        success = jnp.all(completed == 2)
        terminated = success
        truncated = (t >= self.max_steps) & ~terminated
        ns = dict(s)
        ns["t"] = t
        ns["agent_pos"] = pos
        ns["handcell"] = handcell.astype(jnp.int32)
        ns["order"] = order.astype(jnp.int32)
        ns["completed"] = completed.astype(jnp.int32)
        ns["call_target"] = call_target.astype(jnp.int32)
        ns["call_ttl"] = call_ttl.astype(jnp.int32)
        ns["buffer"] = buffer[None].astype(jnp.int32)
        ns["route"] = route[None].astype(jnp.int32)
        ns["cooldown"] = cooldown[None].astype(jnp.int32)
        ns["capacitor"] = capacitor.astype(jnp.int32)
        ns["switch_event"] = switched
        ns["power_event"] = power_event
        ns["service_total"] = jnp.sum(completed, dtype=jnp.int32)
        ns["success"] = success
        ns["terminated"] = terminated
        ns["truncated"] = truncated
        ns["done"] = terminated | truncated
        ns = self.refresh(ns)
        reward = -0.01 - 0.03 * cranked.astype(jnp.float32) - 0.02 * (a == 5).astype(jnp.float32)
        reward = reward + 2.0 * serving.astype(jnp.float32) + 0.2 * jnp.sum(serving.astype(jnp.float32))
        reward = reward + 6.0 * success.astype(jnp.float32)
        info = {"done_all": ns["done"], "terminated": terminated, "truncated": truncated}
        return self.observe(ns), ns, reward.astype(jnp.float32), jnp.full(3, ns["done"]), info

    def absorb(self, payload):
        key, s, actions = payload
        info = {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}
        return self.observe(s), s, jnp.zeros(3, dtype=jnp.float32), jnp.ones(3, dtype=jnp.bool_), info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, actions))
