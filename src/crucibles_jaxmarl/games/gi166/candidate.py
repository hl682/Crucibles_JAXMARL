"""Switchyard Claims: original unexecuted author kernel."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 51
    num_actions = 10
    max_steps = 80

    def refresh(self, s):
        s = dict(s)
        s["agent_active"] = ~s["delivered"]
        s["agent_vis"] = jnp.broadcast_to(s["agent_active"][None, :], (3, 3))
        return s

    def reset(self, key):
        x = jnp.arange(9)[None, :]
        y = jnp.arange(7)[:, None]
        tracks = (((y == 1) | (y == 3) | (y == 5)) & (x >= 1) & (x <= 7)) | (((x == 1) | (x == 4) | (x == 7)) & (y >= 1) & (y <= 5))
        s = {
            "t": jnp.array(0, jnp.int32),
            "done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
            "map": jnp.where(tracks, 0, 1).astype(jnp.int32),
            "pos": jnp.array([[3, 3], [4, 4], [7, 5]], jnp.int32),
            "urgency": jax.random.randint(key, (3,), 1, 4, dtype=jnp.int32),
            "delivered": jnp.zeros(3, jnp.bool_), "commit": jnp.zeros(3, jnp.int32),
            "agent_active": jnp.ones(3, jnp.bool_), "agent_vis": jnp.ones((3, 3), jnp.bool_),
            "block_pos": jnp.array([[4, 1], [4, 3], [4, 5]], jnp.int32),
            "block_active": jnp.ones(3, jnp.bool_), "block_vis": jnp.ones((3, 3), jnp.bool_),
            "owner": jnp.full(3, -1, jnp.int32), "ttl": jnp.zeros(3, jnp.int32),
            "axis": jnp.zeros(3, jnp.int32),
            "depot_pos": jnp.array([[7, 3]], jnp.int32), "depot_active": jnp.ones(1, jnp.bool_),
            "depot_vis": jnp.ones((3, 1), jnp.bool_), "cooldown": jnp.zeros(1, jnp.int32),
            "moved": jnp.zeros(3, jnp.bool_), "claimed": jnp.zeros(3, jnp.bool_),
            "blocked": jnp.zeros(3, jnp.bool_), "delivery_event": jnp.zeros(3, jnp.bool_)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def destination_ok(self, s, i, p, direction):
        inside = (p[0] >= 0) & (p[0] <= 8) & (p[1] >= 0) & (p[1] <= 6)
        q = jnp.clip(p, jnp.array([0, 0]), jnp.array([8, 6]))
        track = s["map"][q[1], q[0]] == 0
        match = jnp.all(s["block_pos"] == p[None, :], axis=1)
        horizontal = (direction == 2) | (direction == 4)
        axis_ok = jnp.all(~match | (s["axis"] == jnp.where(horizontal, 0, 1)))
        reservation_ok = jnp.all(~match | (s["owner"] < 0) | (s["owner"] == i))
        occupied = jnp.any(jnp.all(s["pos"] == p[None, :], axis=1) & ~s["delivered"] & (jnp.arange(3) != i))
        return inside & track & axis_ok & reservation_ok & ~occupied

    def legal_actions(self, s):
        moves = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        mask = jnp.zeros((3, 10), jnp.bool_).at[:, 0].set(True)
        for i in range(3):
            live = ~s["done"] & ~s["delivered"][i]
            for a in range(1, 5):
                ok = self.destination_ok(s, i, s["pos"][i] + moves[a], a)
                mask = mask.at[i, a].set(live & (s["commit"][i] == 0) & ok)
            d = jnp.sum(jnp.abs(s["block_pos"] - s["pos"][i]), axis=1)
            b = jnp.argmin(d)
            access = (d[b] <= 1) & ((s["owner"][b] < 0) | (s["owner"][b] == i))
            mask = mask.at[i, 5].set(live & access)
            mask = mask.at[i, 6].set(live & jnp.any(s["owner"] == i))
            mask = mask.at[i, 7].set(live & access)
            mask = mask.at[i, 8].set(live & jnp.all(s["pos"][i] == s["depot_pos"][0]) & (s["cooldown"][0] == 0))
            mask = mask.at[i, 9].set(live & (s["commit"][i] == 0))
        return mask

    def observe(self, s):
        private = jnp.stack((s["pos"][:, 0], s["pos"][:, 1], s["urgency"], s["delivered"].astype(jnp.int32), s["commit"], jnp.full(3, s["t"])), axis=1)
        public = jnp.concatenate((s["pos"].reshape(-1), s["delivered"].astype(jnp.int32), s["commit"], s["block_pos"].reshape(-1), s["owner"], s["ttl"], s["axis"], s["depot_pos"].reshape(-1), s["cooldown"], s["moved"].astype(jnp.int32), s["claimed"].astype(jnp.int32), s["blocked"].astype(jnp.int32), s["delivery_event"].astype(jnp.int32)))
        return jnp.concatenate((private, jnp.broadcast_to(public[None, :], (3, 42)), jnp.eye(3, dtype=jnp.int32)), axis=1).astype(jnp.float32)

    def raw_step(self, key, s, actions):
        clipped = jnp.clip(actions, 0, 9)
        legal = self.legal_actions(s)
        a = jnp.where((actions >= 0) & (actions < 10) & legal[jnp.arange(3), clipped], clipped, 0)
        n = dict(s)
        n["t"] = s["t"] + jnp.array(1, jnp.int32)
        n["ttl"] = jnp.maximum(s["ttl"] - 1, 0)
        n["owner"] = jnp.where(n["ttl"] > 0, s["owner"], -1)
        n["commit"] = jnp.maximum(s["commit"] - 1, 0)
        n["cooldown"] = jnp.maximum(s["cooldown"] - 1, 0)
        n["moved"] = jnp.zeros(3, jnp.bool_)
        n["claimed"] = jnp.zeros(3, jnp.bool_)
        n["blocked"] = jnp.zeros(3, jnp.bool_)
        n["delivery_event"] = jnp.zeros(3, jnp.bool_)
        # CONTROL: expiry, release and binding yield precede rotating claims.
        for i in range(3):
            release = (a[i] == 6) | (a[i] == 9)
            owned = n["owner"] == i
            n["owner"] = jnp.where(owned & release, -1, n["owner"])
            n["ttl"] = jnp.where(owned & release, 0, n["ttl"])
            n["commit"] = n["commit"].at[i].set(jnp.where(a[i] == 9, 2, n["commit"][i]))
        for k in range(3):
            i = (s["t"] + k) % 3
            d = jnp.sum(jnp.abs(n["block_pos"] - n["pos"][i]), axis=1)
            b = jnp.argmin(d)
            available = (d[b] <= 1) & ((n["owner"][b] < 0) | (n["owner"][b] == i))
            claim = (a[i] == 5) & available
            n["owner"] = n["owner"].at[b].set(jnp.where(claim, i, n["owner"][b]))
            n["ttl"] = n["ttl"].at[b].set(jnp.where(claim, 4, n["ttl"][b]))
            n["claimed"] = n["claimed"].at[i].set(claim)
            toggle = (a[i] == 7) & available
            n["axis"] = n["axis"].at[b].set(jnp.where(toggle, 1 - n["axis"][b], n["axis"][b]))
        # MOVEMENT: old occupancy and updated occupancy both block entry.
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        for k in range(3):
            i = (s["t"] + k) % 3
            p = s["pos"][i] + delta[jnp.minimum(a[i], 4)]
            moving = (a[i] >= 1) & (a[i] <= 4)
            ok = moving & self.destination_ok(n, i, p, a[i])
            n["pos"] = n["pos"].at[i].set(jnp.where(ok, p, n["pos"][i]))
            n["moved"] = n["moved"].at[i].set(ok)
            n["blocked"] = n["blocked"].at[i].set(moving & ~ok)
        # DELIVERY: one physical service slot, two subsequent tick starts of cooldown.
        for k in range(3):
            i = (s["t"] + k) % 3
            deliver = (a[i] == 8) & (n["cooldown"][0] == 0) & ~n["delivered"][i] & jnp.all(n["pos"][i] == n["depot_pos"][0])
            n["delivered"] = n["delivered"].at[i].set(n["delivered"][i] | deliver)
            n["cooldown"] = n["cooldown"].at[0].set(jnp.where(deliver, 2, n["cooldown"][0]))
            n["delivery_event"] = n["delivery_event"].at[i].set(deliver)
        reward = -(~s["delivered"]).astype(jnp.int32) * s["urgency"] - n["claimed"].astype(jnp.int32) + n["delivery_event"].astype(jnp.int32) * (12 + 2 * s["urgency"])
        n["terminated"] = jnp.all(n["delivered"])
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["done"] = n["terminated"] | n["truncated"]
        n = self.refresh(n)
        # ABSORPTION: preserve state, clock, observation and actual terminal causes.
        n = {k: jnp.where(s["done"], s[k], n[k]) for k in s}
        reward = jnp.where(s["done"], 0, reward).astype(jnp.float32)
        return self.observe(n), n, reward, jnp.full(3, n["done"], jnp.bool_), {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
