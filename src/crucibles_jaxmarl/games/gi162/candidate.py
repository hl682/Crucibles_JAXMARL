"""Canal Commons: finite packets, spatial sluices and delayed crop dependence."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 118
    num_actions = 10
    max_steps = 64

    def reset(self, key):
        gate_pos = jnp.array([[3, 2], [7, 2], [11, 2]], jnp.int32)
        garden_pos = jnp.array([[3, 5], [7, 5], [11, 5]], jnp.int32)
        terrain = jnp.zeros((7, 13), jnp.int32).at[2, :].set(1)
        terrain = terrain.at[3:6, 3].set(2).at[3:6, 7].set(2).at[3:6, 11].set(2)
        terrain = terrain.at[5, jnp.array([3, 7, 11], jnp.int32)].set(3)
        s = {
            "t": jnp.array(0, jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "terrain": terrain, "agent_pos": gate_pos + jnp.array([0, 1], jnp.int32),
            "agent_active": jnp.ones(3, jnp.bool_), "agent_visible": jnp.ones((3, 3), jnp.bool_),
            "tank": jnp.zeros(3, jnp.int32), "gate_pos": gate_pos,
            "gate_active": jnp.ones(3, jnp.bool_), "gate_visible": jnp.ones((3, 3), jnp.bool_),
            "gate_open": jnp.zeros(3, jnp.bool_), "integrity": jnp.ones(3, jnp.int32),
            "garden_pos": garden_pos, "garden_active": jnp.ones(3, jnp.bool_),
            "garden_visible": jnp.ones((3, 3), jnp.bool_), "moisture": jnp.full(3, 4, jnp.int32),
            "growth": jnp.zeros(3, jnp.int32), "dry": jnp.zeros(3, jnp.int32),
            "harvested": jnp.zeros(3, jnp.bool_), "dead": jnp.zeros(3, jnp.bool_),
            "reservoir_pos": jnp.array([[0, 2]], jnp.int32),
            "reservoir_active": jnp.ones(1, jnp.bool_), "reservoir_visible": jnp.ones((3, 1), jnp.bool_),
            "reservoir": jnp.array([40], jnp.int32),
            "water_pos": jnp.zeros((16, 2), jnp.int32), "water_active": jnp.zeros(16, jnp.bool_),
            "water_visible": jnp.zeros((3, 16), jnp.bool_), "water_route": jnp.zeros(16, jnp.int32),
            "arrived": jnp.zeros(3, jnp.int32), "maintained": jnp.zeros(3, jnp.bool_),
            "toggled": jnp.zeros(3, jnp.bool_), "picked": jnp.zeros(3, jnp.bool_),
            "wilted": jnp.zeros(3, jnp.bool_), "lost": jnp.array(0, jnp.int32)
        }
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate((
            s["t"].reshape(1), s["agent_pos"].reshape(-1), s["tank"], s["gate_open"],
            s["integrity"], s["moisture"], s["growth"], s["dry"], s["harvested"], s["dead"],
            s["reservoir"], s["water_pos"].reshape(-1), s["water_active"], s["water_route"],
            s["arrived"], s["maintained"], s["toggled"], s["picked"], s["wilted"],
            s["lost"].reshape(1), s["done"].reshape(1), s["terminated"].reshape(1), s["truncated"].reshape(1)
        )).astype(jnp.float32)
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(public, (3, 115))), axis=1)

    def legal_actions(self, s):
        p = s["agent_pos"]
        dg = jnp.sum(jnp.abs(p[:, None, :] - s["gate_pos"][None, :, :]), axis=2)
        dp = jnp.sum(jnp.abs(p[:, None, :] - s["garden_pos"][None, :, :]), axis=2)
        own = jnp.sum(jnp.abs(p - s["garden_pos"]), axis=1) <= 1
        nearest = jnp.argmin(dp, axis=1)
        mask = jnp.stack((
            jnp.ones(3, jnp.bool_), p[:, 1] > 0, p[:, 0] < 12, p[:, 1] < 6, p[:, 0] > 0,
            jnp.min(dg, axis=1) <= 1, jnp.min(dg, axis=1) <= 1,
            own & (s["moisture"] > 0) & (s["tank"] < 4) & ~s["harvested"] & ~s["dead"],
            (jnp.min(dp, axis=1) <= 1) & (s["tank"] > 0) & (s["moisture"][nearest] < 6)
            & ~s["harvested"][nearest] & ~s["dead"][nearest],
            own & (s["growth"] >= 5) & ~s["harvested"] & ~s["dead"]
        ), axis=1)
        return mask & (~s["done"] | (jnp.arange(10)[None, :] == 0))

    def raw_step(self, key, s, actions):
        ac = jnp.clip(actions, 0, 9)
        valid = (actions >= 0) & (actions < 10) & self.legal_actions(s)[jnp.arange(3), ac]
        ac = jnp.where(valid, ac, 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        pos = s["agent_pos"] + delta[ac]
        dg = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["gate_pos"][None, :, :]), axis=2)
        dp = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["garden_pos"][None, :, :]), axis=2)
        gate = jnp.argmin(dg, axis=1)
        garden = jnp.argmin(dp, axis=1)
        toggles = jnp.sum(jax.nn.one_hot(gate, 3, dtype=jnp.int32) * (ac == 5)[:, None], axis=0)
        maintained = jnp.any(jax.nn.one_hot(gate, 3, dtype=jnp.bool_) & (ac == 6)[:, None], axis=0)
        gate_open = s["gate_open"] ^ ((toggles % 2) == 1)
        t = s["t"] + 1
        integrity = jnp.where(maintained, 3, jnp.maximum(0, s["integrity"] - ((t % 12) == 0)))

        # Existing packets move one real canal cell before a new packet is released.
        route = s["water_route"]
        wp = s["water_pos"] + jnp.stack(((route == 0).astype(jnp.int32), (route > 0).astype(jnp.int32)), axis=1)
        at_mouth = jnp.all(wp[:, None, :] == s["gate_pos"][None, :, :], axis=2) & s["water_active"][:, None]
        blocked = jnp.any(at_mouth & (integrity == 0)[None, :], axis=1)
        divert = at_mouth & gate_open[None, :] & (integrity > 0)[None, :]
        turning = jnp.any(divert, axis=1)
        route = jnp.where(turning, jnp.argmax(divert, axis=1) + 1, route)
        wp = wp.at[:, 1].set(jnp.where(turning, 3, wp[:, 1]))
        arriving = s["water_active"] & (route > 0) & (wp[:, 1] >= 5) & ~blocked
        arrivals = jnp.sum(jax.nn.one_hot(jnp.maximum(route - 1, 0), 3, dtype=jnp.int32) * arriving[:, None], axis=0)
        drain = s["water_active"] & (wp[:, 0] > 12)
        live = s["water_active"] & ~arriving & ~blocked & ~drain
        wp = jnp.clip(wp, jnp.array([0, 0]), jnp.array([12, 6]))
        slot = s["t"] % 16
        release = (s["reservoir"][0] > 0) & ~live[slot]
        wp = wp.at[slot].set(jnp.where(release, s["reservoir_pos"][0], wp[slot]))
        live = live.at[slot].set(live[slot] | release)
        route = route.at[slot].set(jnp.where(release, 0, route[slot]))
        wp = jnp.where(live[:, None], wp, 0)
        route = jnp.where(live, route, 0)

        # Storage transfers use pre-step legality; arrivals and pours feed this tick's crops.
        draw = ac == 7
        pour = ac == 8
        poured = jnp.sum(jax.nn.one_hot(garden, 3, dtype=jnp.int32) * pour[:, None], axis=0)
        tank = s["tank"] + draw.astype(jnp.int32) - pour.astype(jnp.int32)
        unresolved = ~s["harvested"] & ~s["dead"]
        moisture = jnp.clip(s["moisture"] - draw + arrivals + poured, 0, 6)
        wet_tick = ((t % 4) == 0) & (moisture > 0) & unresolved
        growth = jnp.minimum(5, s["growth"] + wet_tick)
        moisture = moisture - wet_tick
        dry = jnp.where(unresolved & (moisture == 0), jnp.minimum(12, s["dry"] + 1), 0)
        picked = ac == 9
        wilted = unresolved & (dry >= 12) & ~picked
        harvested = s["harvested"] | picked
        dead = s["dead"] | wilted
        term = jnp.all(harvested | dead)
        trunc = (t >= self.max_steps) & ~term
        rewards = (4.0 * picked + 0.25 * jnp.sum(picked) - wilted
                   - 0.125 * ((ac == 5) | draw | pour) - 0.25 * (ac == 6) - 0.015625).astype(jnp.float32)
        ns = {**s, "t": t, "done": term | trunc, "terminated": term, "truncated": trunc,
              "agent_pos": pos, "tank": tank, "gate_open": gate_open, "integrity": integrity,
              "water_pos": wp, "water_active": live, "water_visible": jnp.broadcast_to(live, (3, 16)),
              "water_route": route, "reservoir": s["reservoir"] - release,
              "moisture": moisture, "growth": growth, "dry": dry, "harvested": harvested, "dead": dead,
              "arrived": arrivals, "maintained": maintained, "toggled": (toggles % 2) == 1,
              "picked": picked, "wilted": wilted, "lost": jnp.sum(blocked | drain, dtype=jnp.int32)}
        ns = jax.tree.map(lambda old, new: jnp.where(s["done"], old, new), s, ns)
        rewards = jnp.where(s["done"], jnp.zeros(3, jnp.float32), rewards)
        info = {"done_all": ns["done"], "terminated": ns["terminated"], "truncated": ns["truncated"]}
        return self.observe(ns), ns, rewards, jnp.full(3, ns["done"], jnp.bool_), info
