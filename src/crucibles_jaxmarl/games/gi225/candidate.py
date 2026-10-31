"""Seasonal Pasture Commons. Original source; unexecuted author proposal."""
import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 64
    observation_size = 1026
    num_actions = 19
    max_steps = 768
    width = 34
    height = 34
    groups = ("plot", "trough", "herd", "gate", "reservoir")

    def pick(self, p, points, eligible, radius):
        d = jnp.sum(jnp.abs(points - p), axis=1)
        d = jnp.where(eligible, d, 10000)
        q = jnp.argmin(d)
        return q, d[q] <= radius

    def masks(self, s, rows):
        return jnp.where(s["done"], jnp.arange(self.num_actions)[None, :] == 0, rows)

    def mask_row(self, i, carry):
        s, rows = carry
        return s, rows.at[i].set(self.choices(s, i))

    def legal_actions(self, s):
        unused, rows = jax.lax.fori_loop(0, self.num_agents, self.mask_row, (s, jnp.zeros((self.num_agents, self.num_actions), dtype=jnp.bool_)))
        return self.masks(s, rows)

    def refresh(self, s):
        s = dict(s)
        p = s["pos"]
        s["agent_visible"] = jnp.max(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2) <= 3
        yy = jnp.arange(self.height)[:, None]
        xx = jnp.arange(self.width)[None, :]
        s["map_visible"] = (jnp.abs(xx[None, :, :] - p[:, 0, None, None]) <= 3) & (jnp.abs(yy[None, :, :] - p[:, 1, None, None]) <= 3)
        for name in self.groups:
            s[name + "_visible"] = (jnp.max(jnp.abs(p[:, None, :] - s[name + "_pos"][None, :, :]), axis=2) <= 3) & s[name + "_active"][None, :]
        return s

    def visible_rows(self, s, name, values):
        v = s[name + "_visible"]
        return jnp.concatenate((v[:, :, None].astype(jnp.float32), jnp.where(v[:, :, None], values[None, :, :], 0).astype(jnp.float32)), axis=2).reshape((self.num_agents, -1))

    def patch(self, s, field):
        dx = jnp.tile(jnp.arange(-3, 4), 7)
        dy = jnp.repeat(jnp.arange(-3, 4), 7)
        x = s["pos"][:, 0, None] + dx
        y = s["pos"][:, 1, None] + dy
        inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        return jnp.where(inside, s[field][jnp.clip(y, 0, self.height - 1), jnp.clip(x, 0, self.width - 1)], -1).astype(jnp.float32)

    def move_actor(self, s, i, a):
        n = dict(s)
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        face = jnp.clip(a - 1, 0, 3)
        moving = (a >= 1) & (a <= 4)
        q = s["pos"][i] + dirs[face]
        q = jnp.clip(q, jnp.array([0, 0]), jnp.array([self.width - 1, self.height - 1]))
        free = ~jnp.any(jnp.all(s["pos"] == q, axis=1) & (jnp.arange(self.num_agents) != i))
        allowed = moving & free & self.passable(s, q, i)
        n["pos"] = s["pos"].at[i].set(jnp.where(allowed, q, s["pos"][i]))
        n["face"] = s["face"].at[i].set(jnp.where(moving, face, s["face"][i]))
        return n

    def end_step(self, old, n, rewards, natural):
        n["t"] = old["t"] + jnp.asarray(1, dtype=jnp.int32)
        n["terminated"] = natural
        n["truncated"] = (n["t"] >= self.max_steps) & ~natural
        n["done"] = n["terminated"] | n["truncated"]
        n = self.refresh(n)
        n = jax.tree.map(lambda before, after: jnp.where(old["done"], before, after), old, n)
        rewards = jnp.where(old["done"], jnp.zeros_like(rewards), rewards).astype(jnp.float32)
        dones = jnp.full((self.num_agents,), n["done"], dtype=jnp.bool_)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, rewards, dones, info

    def reset(self, key):
        ids = jnp.arange(64, dtype=jnp.int32)
        crew = ids // 4
        g = jnp.arange(16, dtype=jnp.int32)
        centers = jnp.stack((4 + 8 * (g % 4), 4 + 8 * (g // 4)), axis=1)
        offset = jnp.array([[-1, 0], [0, 0], [1, 0], [0, 1]], dtype=jnp.int32)
        x = jnp.arange(34)[None, :]
        y = jnp.arange(34)[:, None]
        wall = (x == 0) | (x == 33) | (y == 0) | (y == 33) | ((x % 8 == 0) & (x > 0))
        gate_pos = centers + jnp.array([4, 0], dtype=jnp.int32)
        terrain = jnp.where(wall, 1, 0).astype(jnp.int32)
        terrain = terrain.at[gate_pos[:, 1], gate_pos[:, 0]].set(0)
        s = {"t": jnp.asarray(0, dtype=jnp.int32), "done": jnp.asarray(False, dtype=jnp.bool_), "terminated": jnp.asarray(False, dtype=jnp.bool_), "truncated": jnp.asarray(False, dtype=jnp.bool_),
             "pos": centers[crew] + offset[ids % 4], "crew": crew, "tool": ids % 4,
             "seed": jnp.full((64,), 2, dtype=jnp.int32), "grain": jnp.zeros((64,), dtype=jnp.int32),
             "compost": jnp.ones((64,), dtype=jnp.int32), "face": jnp.zeros((64,), dtype=jnp.int32),
             "agent_active": jnp.ones((64,), dtype=jnp.bool_), "agent_visible": jnp.zeros((64, 64), dtype=jnp.bool_),
             "map_visible": jnp.zeros((64, 34, 34), dtype=jnp.bool_), "base": terrain, "terrain": terrain,
             "grass": jnp.where(terrain == 0, 3, 0).astype(jnp.int32), "reserve": jnp.zeros((34, 34), dtype=jnp.bool_),
             "plot_pos": centers + jnp.array([0, -2], dtype=jnp.int32), "plot_active": jnp.ones((16,), dtype=jnp.bool_), "plot_visible": jnp.zeros((64, 16), dtype=jnp.bool_),
             "trough_pos": centers + jnp.array([2, 0], dtype=jnp.int32), "trough_active": jnp.ones((16,), dtype=jnp.bool_), "trough_visible": jnp.zeros((64, 16), dtype=jnp.bool_),
             "crop": (g % 2).astype(jnp.int32), "growth": jnp.zeros((16,), dtype=jnp.int32), "moisture": jnp.zeros((16,), dtype=jnp.int32),
             "pantry": jnp.zeros((16,), dtype=jnp.int32), "harvests": jnp.zeros((16,), dtype=jnp.int32),
             "herd_pos": centers + jnp.array([0, 2], dtype=jnp.int32), "herd_active": jnp.ones((16,), dtype=jnp.bool_), "herd_visible": jnp.zeros((64, 16), dtype=jnp.bool_),
             "hunger": jnp.zeros((16,), dtype=jnp.int32), "health": jnp.full((16,), 6, dtype=jnp.int32),
             "disease": g == 0, "manure": jnp.ones((16,), dtype=jnp.int32),
             "gate_pos": gate_pos, "gate_active": jnp.ones((16,), dtype=jnp.bool_), "gate_visible": jnp.zeros((64, 16), dtype=jnp.bool_),
             "gate_open": jnp.ones((16,), dtype=jnp.bool_), "wear": jnp.full((16,), 3, dtype=jnp.int32),
             "reservoir_pos": jnp.stack((jnp.ones((4,), dtype=jnp.int32), 4 + 8 * jnp.arange(4, dtype=jnp.int32)), axis=1),
             "reservoir_active": jnp.ones((4,), dtype=jnp.bool_), "reservoir_visible": jnp.zeros((64, 4), dtype=jnp.bool_),
             "water_stock": jnp.full((4,), 8, dtype=jnp.int32), "recharges": jnp.full((4,), 32, dtype=jnp.int32),
             "total_harvests": jnp.asarray(0, dtype=jnp.int32), "stored_total": jnp.asarray(0, dtype=jnp.int32),
             "withered": jnp.zeros((16,), dtype=jnp.bool_), "new_infection": jnp.zeros((16,), dtype=jnp.bool_)}
        s = self.sync_map(s)
        s = self.refresh(s)
        return self.observe(s), s

    def sync_map(self, s):
        n = dict(s)
        tile = jnp.where(s["base"] == 1, 1, jnp.where(s["reserve"], 3, 4 + s["grass"])).astype(jnp.int32)
        tile = tile.at[s["gate_pos"][:, 1], s["gate_pos"][:, 0]].set(jnp.where(s["gate_open"], 2, 1).astype(jnp.int32))
        tile = tile.at[s["trough_pos"][:, 1], s["trough_pos"][:, 0]].set(jnp.asarray(9, dtype=jnp.int32))
        n["terrain"] = tile
        return n

    def context(self, s, i):
        p = s["pos"][i]
        f, nf = self.pick(p, s["plot_pos"], s["plot_active"], 3)
        trough = s["trough_pos"]
        w, nw = self.pick(p, trough, s["plot_active"], 1)
        h, nh = self.pick(p, s["herd_pos"], s["herd_active"], 2)
        g, ng = self.pick(p, s["gate_pos"], s["gate_active"], 1)
        j, nj = self.pick(p, s["pos"], jnp.arange(64) != i, 1)
        center = s["plot_pos"][s["crew"][i]] + jnp.array([0, 2], dtype=jnp.int32)
        home = jnp.sum(jnp.abs(p - center)) <= 1
        return f, nf, w, nw, h, nh, g, ng, j, nj, home

    def passable(self, s, q, i):
        closed = jnp.any(jnp.all(s["gate_pos"] == q, axis=1) & ~s["gate_open"])
        return (s["base"][q[1], q[0]] == 0) & ~closed

    def choices(self, s, i):
        f, nf, w, nw, h, nh, g, ng, j, nj, home = self.context(s, i)
        p = s["pos"][i]
        return jnp.array([True, True, True, True, True,
            nf & (s["tool"][i] == 0) & (s["seed"][i] > 0) & (s["crop"][f] == 0),
            nw & (s["tool"][i] == 2) & (s["water_stock"][w // 4] >= 2) & (s["moisture"][w] < 3),
            nf & (s["crop"][f] == 3) & (s["grain"][i] <= 2),
            nh & (s["grain"][i] > 0) & (s["hunger"][h] > 0),
            nh & (s["tool"][i] == 1),
            ng & (s["wear"][g] > 0),
            (s["tool"][i] == 3) & (s["seed"][i] > 0) & ~s["reserve"][p[1], p[0]],
            (s["compost"][i] > 0) & (s["grass"][p[1], p[0]] < 4),
            nh & (s["tool"][i] == 2) & (s["compost"][i] > 0) & (s["disease"][h] | (s["health"][h] < 6)),
            nj, home, nh & (s["manure"][h] > 0) & (s["compost"][i] < 3),
            ng & (s["compost"][i] > 0) & (s["wear"][g] < 3),
            home & (s["grain"][i] > 0) & (s["pantry"][s["crew"][i]] < 8)], dtype=jnp.bool_)

    def actor(self, k, carry):
        n, actions, reward = carry
        i = (k + n["t"]) % 64
        a = actions[i]
        a = jnp.where(self.choices(n, i)[a], a, 0)
        f, nf, w, nw, h, nh, g, ng, j, nj, home = self.context(n, i)
        p = n["pos"][i]
        n = self.move_actor(n, i, a)
        plant = a == 5
        n["crop"] = n["crop"].at[f].set(jnp.where(plant, 1, n["crop"][f]))
        n["growth"] = n["growth"].at[f].set(jnp.where(plant, 0, n["growth"][f]))
        n["seed"] = n["seed"].at[i].add(-(plant | (a == 11)).astype(jnp.int32))
        irrigate = a == 6
        down = jnp.where(w % 4 < 3, w + 1, w)
        share = irrigate & (w % 4 < 3) & n["gate_open"][w]
        n["water_stock"] = n["water_stock"].at[w // 4].add(-2 * irrigate.astype(jnp.int32))
        n["moisture"] = n["moisture"].at[w].set(jnp.where(irrigate, jnp.minimum(n["moisture"][w] + 2, 3), n["moisture"][w]))
        n["moisture"] = n["moisture"].at[down].set(jnp.minimum(n["moisture"][down] + share.astype(jnp.int32), 3))
        harvest = a == 7
        n["crop"] = n["crop"].at[f].set(jnp.where(harvest, 0, n["crop"][f]))
        n["grain"] = n["grain"].at[i].add(2 * harvest.astype(jnp.int32) - ((a == 8) | (a == 18)).astype(jnp.int32))
        n["seed"] = n["seed"].at[i].set(jnp.minimum(n["seed"][i] + harvest.astype(jnp.int32), 4))
        n["harvests"] = n["harvests"].at[f].add(harvest.astype(jnp.int32))
        n["total_harvests"] = n["total_harvests"] + harvest.astype(jnp.int32)
        n["hunger"] = n["hunger"].at[h].set(jnp.where(a == 8, jnp.maximum(n["hunger"][h] - 4, 0), n["hunger"][h]))
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        hp = n["herd_pos"][h]
        q = jnp.clip(hp + dirs[n["face"][i]], jnp.array([0, 0]), jnp.array([33, 33]))
        free = ~jnp.any(jnp.all(n["herd_pos"] == q, axis=1) & (jnp.arange(16) != h))
        drive = (a == 9) & free & self.passable(n, q, i)
        n["herd_pos"] = n["herd_pos"].at[h].set(jnp.where(drive, q, hp))
        crossed = drive & jnp.all(n["gate_pos"] == q, axis=1)
        n["wear"] = jnp.maximum(n["wear"] - crossed.astype(jnp.int32), 0)
        occupied_gate = jnp.any(jnp.all(n["pos"][None, :, :] == n["gate_pos"][:, None, :], axis=2), axis=1) | jnp.any(jnp.all(n["herd_pos"][None, :, :] == n["gate_pos"][:, None, :], axis=2), axis=1)
        toggle = (a == 10) & ~(n["gate_open"][g] & occupied_gate[g])
        n["gate_open"] = n["gate_open"].at[g].set(n["gate_open"][g] ^ toggle)
        n["reserve"] = n["reserve"].at[p[1], p[0]].set(n["reserve"][p[1], p[0]] | (a == 11))
        fert = a == 12
        n["grass"] = n["grass"].at[p[1], p[0]].set(jnp.minimum(n["grass"][p[1], p[0]] + 2 * fert.astype(jnp.int32), 4))
        tend = a == 13
        n["disease"] = n["disease"].at[h].set(n["disease"][h] & ~tend)
        n["health"] = n["health"].at[h].set(jnp.where(tend, jnp.minimum(n["health"][h] + 2, 6), n["health"][h]))
        n["compost"] = n["compost"].at[i].add(-((a == 12) | (a == 13) | (a == 17)).astype(jnp.int32) + (a == 16).astype(jnp.int32))
        ti = n["tool"][i]
        tj = n["tool"][j]
        n["tool"] = n["tool"].at[i].set(jnp.where(a == 14, tj, ti)).at[j].set(jnp.where(a == 14, ti, tj))
        n["tool"] = n["tool"].at[i].set(jnp.where(a == 15, (n["tool"][i] + 1) % 4, n["tool"][i]))
        n["manure"] = n["manure"].at[h].add(-(a == 16).astype(jnp.int32))
        n["wear"] = n["wear"].at[g].set(jnp.where(a == 17, 3, n["wear"][g]))
        n["pantry"] = n["pantry"].at[n["crew"][i]].add((a == 18).astype(jnp.int32))
        n["stored_total"] = n["stored_total"] + (a == 18).astype(jnp.int32)
        reward = reward + 0.5 * harvest.astype(jnp.float32) * (n["crew"] == f).astype(jnp.float32) + 0.01 * harvest.astype(jnp.float32)
        reward = reward.at[i].add(0.25 * (a == 18).astype(jnp.float32) - 0.02 * ((a == 5) | (a == 6) | (a == 11) | (a == 12) | (a == 13) | (a == 17)).astype(jnp.float32))
        return n, actions, reward

    def graze(self, h, n):
        p = n["herd_pos"][h]
        meal = ((n["t"] + 1) % 4 == 0) & (n["grass"][p[1], p[0]] > 0) & ~n["reserve"][p[1], p[0]]
        hungry_tick = (n["t"] + 1) % 4 == 0
        n["grass"] = n["grass"].at[p[1], p[0]].add(-meal.astype(jnp.int32))
        n["hunger"] = n["hunger"].at[h].set(jnp.clip(n["hunger"][h] + hungry_tick.astype(jnp.int32) - 3 * meal.astype(jnp.int32), 0, 8))
        n["manure"] = n["manure"].at[h].set(jnp.minimum(n["manure"][h] + meal.astype(jnp.int32), 3))
        return n

    def raw_step(self, key, s, actions):
        a = jnp.where((actions >= 0) & (actions < 19), actions, 0).astype(jnp.int32)
        n, a, reward = jax.lax.fori_loop(0, 64, self.actor, (dict(s), a, jnp.zeros((64,), dtype=jnp.float32)))
        n = jax.lax.fori_loop(0, 16, self.graze, n)
        season = ((s["t"] + 1) // 64) % 4
        regrow = ((s["t"] + 1) % 12 == 0) & ((season == 0) | (season == 2))
        protected = ((s["t"] + 1) % 8 == 0) & n["reserve"]
        n["grass"] = jnp.where(n["base"] == 0, jnp.minimum(n["grass"] + (regrow | protected).astype(jnp.int32), 4), 0)
        tick = (s["t"] + 1) % 4 == 0
        grow = tick & (n["crop"] > 0) & (n["crop"] < 3) & (n["moisture"] > 0)
        n["growth"] = jnp.minimum(n["growth"] + grow.astype(jnp.int32), 6)
        n["moisture"] = jnp.maximum(n["moisture"] - grow.astype(jnp.int32), 0)
        n["crop"] = jnp.where((n["crop"] > 0) & (n["growth"] >= 6), 3, jnp.where((n["crop"] > 0) & (n["growth"] >= 3), 2, n["crop"])).astype(jnp.int32)
        wither = (season == 1) & ((s["t"] + 1) % 16 == 0) & (n["moisture"] == 0) & (n["crop"] > 0) & (n["crop"] < 3)
        n["crop"] = jnp.where(wither, 0, n["crop"]).astype(jnp.int32)
        n["withered"] = wither
        near = jnp.sum(jnp.abs(n["herd_pos"][:, None, :] - n["herd_pos"][None, :, :]), axis=2) <= 2
        infection = ((s["t"] + 1) % 8 == 0) & jnp.any(near & n["disease"][None, :], axis=1) & ~n["disease"]
        n["new_infection"] = infection
        n["disease"] = n["disease"] | infection
        damage = ((s["t"] + 1) % 8 == 0) & ((n["hunger"] == 8) | n["disease"])
        n["health"] = jnp.maximum(n["health"] - damage.astype(jnp.int32), 1)
        recharge = ((s["t"] + 1) % 16 == 0) & (n["recharges"] > 0)
        n["water_stock"] = jnp.minimum(n["water_stock"] + jnp.where(season == 1, 2, 6) * recharge.astype(jnp.int32), 8)
        n["recharges"] = n["recharges"] - recharge.astype(jnp.int32)
        occupied = jnp.any(jnp.all(n["gate_pos"][:, None, :] == n["pos"][None, :, :], axis=2), axis=1) | jnp.any(jnp.all(n["gate_pos"][:, None, :] == n["herd_pos"][None, :, :], axis=2), axis=1)
        n["gate_open"] = n["gate_open"] & ((n["wear"] > 0) | occupied)
        welfare = ((s["t"] + 1) % 32 == 0) & (n["health"] >= 4) & (n["hunger"] <= 3) & ~n["disease"]
        reward = reward + 0.4 * welfare[n["crew"]].astype(jnp.float32) - 0.1 * damage[n["crew"]].astype(jnp.float32) - 0.2 * wither[n["crew"]].astype(jnp.float32)
        reward = reward + 0.02 * jnp.mean(welfare.astype(jnp.float32))
        n = self.sync_map(n)
        natural = jnp.all(n["pantry"] >= 2) & jnp.all(n["health"] >= 4) & ~jnp.any(n["disease"])
        return self.end_step(s, n, reward, natural)

    def observe(self, s):
        own = jnp.stack((s["pos"][:, 0], s["pos"][:, 1], s["crew"], s["tool"], s["seed"], s["grain"], s["compost"], s["face"]), axis=1)
        public = jnp.broadcast_to(jnp.array([s["t"], (s["t"] // 64) % 4, s["total_harvests"], s["stored_total"]], dtype=jnp.float32), (64, 4))
        av = jnp.stack((s["pos"][:, 0], s["pos"][:, 1], s["crew"], s["tool"], s["seed"], s["grain"], s["compost"]), axis=1)
        pv = jnp.concatenate((s["plot_pos"], jnp.stack((s["crop"], s["growth"], s["moisture"], s["pantry"], s["harvests"]), axis=1)), axis=1)
        tv = jnp.concatenate((s["trough_pos"], s["moisture"][:, None]), axis=1)
        hv = jnp.concatenate((s["herd_pos"], jnp.stack((s["hunger"], s["health"], s["disease"], s["manure"]), axis=1)), axis=1)
        gv = jnp.concatenate((s["gate_pos"], jnp.stack((s["gate_open"], s["wear"]), axis=1)), axis=1)
        rv = jnp.concatenate((s["reservoir_pos"], jnp.stack((s["water_stock"], s["recharges"]), axis=1)), axis=1)
        return jnp.concatenate((own.astype(jnp.float32), public, self.visible_rows(s, "agent", av), self.visible_rows(s, "plot", pv), self.visible_rows(s, "trough", tv), self.visible_rows(s, "herd", hv), self.visible_rows(s, "gate", gv), self.visible_rows(s, "reservoir", rv), self.patch(s, "terrain"), self.patch(s, "grass")), axis=1).astype(jnp.float32)
