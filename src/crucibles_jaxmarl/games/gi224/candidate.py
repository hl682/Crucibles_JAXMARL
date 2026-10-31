"""Archipelago Councils. Original source; unexecuted author proposal."""
import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 48
    observation_size = 681
    num_actions = 18
    max_steps = 640
    width = 34
    height = 15
    groups = ("town", "crossing", "ferry", "grove", "farm")

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
        ids = jnp.arange(48, dtype=jnp.int32)
        crew = ids // 6
        g = jnp.arange(8, dtype=jnp.int32)
        xx = 2 + 4 * g
        x = jnp.arange(34)[None, :]
        y = jnp.arange(15)[:, None]
        terrain = jnp.where((x == 0) | (x == 33) | (y == 0) | (y == 14), 1, jnp.where((x % 4 == 0) & (x < 32) & (y < 11), 2, 0)).astype(jnp.int32)
        s = {"t": jnp.asarray(0, dtype=jnp.int32), "done": jnp.asarray(False, dtype=jnp.bool_), "terminated": jnp.asarray(False, dtype=jnp.bool_), "truncated": jnp.asarray(False, dtype=jnp.bool_),
             "pos": jnp.stack((1 + 4 * crew + ids % 6 % 3, 6 + ids % 6 // 3), axis=1).astype(jnp.int32),
             "crew": crew, "tool": ids % 6, "wood": jnp.full((48,), 2, dtype=jnp.int32),
             "grain": jnp.zeros((48,), dtype=jnp.int32), "fuel_pack": jnp.ones((48,), dtype=jnp.int32),
             "face": jnp.zeros((48,), dtype=jnp.int32),
             "agent_active": jnp.ones((48,), dtype=jnp.bool_), "agent_visible": jnp.zeros((48, 48), dtype=jnp.bool_),
             "map_visible": jnp.zeros((48, 15, 34), dtype=jnp.bool_), "terrain": terrain,
             "town_pos": jnp.stack((xx, jnp.full((8,), 5)), axis=1), "town_active": jnp.ones((8,), dtype=jnp.bool_), "town_visible": jnp.zeros((48, 8), dtype=jnp.bool_),
             "timber_delivered": jnp.zeros((8,), dtype=jnp.int32), "grain_delivered": jnp.zeros((8,), dtype=jnp.int32),
             "fuel_store": jnp.full((8,), 8, dtype=jnp.int32), "open_port": jnp.ones((8,), dtype=jnp.bool_),
             "signal": jnp.full((8,), -1, dtype=jnp.int32), "signal_age": jnp.zeros((8,), dtype=jnp.int32),
             "crossing_pos": jnp.stack((4 + 4 * jnp.arange(7, dtype=jnp.int32), jnp.full((7,), 6)), axis=1),
             "crossing_active": jnp.ones((7,), dtype=jnp.bool_), "crossing_visible": jnp.zeros((48, 7), dtype=jnp.bool_),
             "ferry_pos": jnp.stack((4 + 4 * jnp.arange(7, dtype=jnp.int32), jnp.full((7,), 9)), axis=1),
             "ferry_active": jnp.ones((7,), dtype=jnp.bool_), "ferry_visible": jnp.zeros((48, 7), dtype=jnp.bool_),
             "bridge": jnp.zeros((7,), dtype=jnp.int32), "canal": jnp.zeros((7,), dtype=jnp.bool_),
             "ferry_fuel": jnp.full((7,), 3, dtype=jnp.int32), "cooldown": jnp.zeros((7,), dtype=jnp.int32),
             "reef": jax.random.randint(key, (7,), 0, 2, dtype=jnp.int32),
             "used_crossing": jnp.zeros((7,), dtype=jnp.bool_),
             "grove_pos": jnp.stack((xx, jnp.full((8,), 3)), axis=1), "grove_active": jnp.ones((8,), dtype=jnp.bool_), "grove_visible": jnp.zeros((48, 8), dtype=jnp.bool_),
             "timber_stock": jnp.full((8,), 6, dtype=jnp.int32), "renewals": jnp.full((8,), 8, dtype=jnp.int32),
             "farm_pos": jnp.stack((xx, jnp.full((8,), 8)), axis=1), "farm_active": jnp.ones((8,), dtype=jnp.bool_), "farm_visible": jnp.zeros((48, 8), dtype=jnp.bool_),
             "grain_stock": jnp.full((8,), 4, dtype=jnp.int32), "water": jnp.full((8,), 3, dtype=jnp.int32),
             "damaged": jnp.zeros((7,), dtype=jnp.bool_), "developed_count": jnp.asarray(0, dtype=jnp.int32), "crossings_count": jnp.asarray(0, dtype=jnp.int32)}
        s = self.refresh(s)
        return self.observe(s), s

    def context(self, s, i):
        p = s["pos"][i]
        t, nt = self.pick(p, s["town_pos"], s["town_active"], 1)
        c, nc = self.pick(p, s["crossing_pos"], s["crossing_active"], 1)
        fp = s["ferry_pos"]
        f, nf = self.pick(p, fp, s["crossing_active"], 1)
        g, ng = self.pick(p, s["grove_pos"], s["grove_active"], 1)
        h, nh = self.pick(p, s["farm_pos"], s["farm_active"], 1)
        j, nj = self.pick(p, s["pos"], jnp.arange(48) != i, 1)
        return t, nt, c, nc, f, nf, g, ng, h, nh, j, nj

    def passable(self, s, q, i):
        tile = s["terrain"][q[1], q[0]]
        bridge = jnp.any(jnp.all(s["crossing_pos"] == q, axis=1) & (s["bridge"] >= 2))
        return (tile == 0) | bridge

    def choices(self, s, i):
        t, nt, c, nc, f, nf, g, ng, h, nh, j, nj = self.context(s, i)
        owner = f
        access = s["open_port"][owner] | (s["crew"][i] == owner)
        can_give = ((s["wood"][i] > 0) & (s["wood"][j] < 4)) | ((s["grain"][i] > 0) & (s["grain"][j] < 4)) | ((s["fuel_pack"][i] > 0) & (s["fuel_pack"][j] < 3))
        return jnp.array([True, True, True, True, True,
            ng & (s["tool"][i] == 0) & (s["timber_stock"][g] > 0) & (s["wood"][i] < 4),
            nh & (s["tool"][i] == 1) & (s["grain_stock"][h] > 0) & (s["grain"][i] < 4),
            nj & can_give,
            nt & (s["wood"][i] > 0) & (s["timber_delivered"][t] < 2),
            nt & (s["grain"][i] > 0) & (s["grain_delivered"][t] < 2),
            nc & (s["tool"][i] == 0) & (s["wood"][i] > 0) & (s["bridge"][c] < 2),
            nc & (s["tool"][i] == 0) & (s["wood"][i] > 0) & ~s["canal"][c],
            nf & access & (s["cooldown"][f] == 0) & (s["ferry_fuel"][f] > 0) & (s["pos"][i, 1] == 9),
            nf & (s["fuel_pack"][i] > 0) & (s["ferry_fuel"][f] < 8),
            nt & (s["tool"][i] == 3) & (s["fuel_pack"][i] > 0),
            nt & (s["fuel_pack"][i] < 3) & (s["fuel_store"][t] > 0),
            nt & (t == s["crew"][i]),
            nt & (t == s["crew"][i]) & (s["tool"][i] == 4)], dtype=jnp.bool_)

    def actor(self, k, carry):
        n, actions, reward = carry
        i = (k + n["t"]) % 48
        a = actions[i]
        a = jnp.where(self.choices(n, i)[a], a, 0)
        t, nt, c, nc, f, nf, g, ng, h, nh, j, nj = self.context(n, i)
        n = self.move_actor(n, i, a)
        cut = a == 5
        reap = a == 6
        n["wood"] = n["wood"].at[i].add(cut.astype(jnp.int32))
        n["timber_stock"] = n["timber_stock"].at[g].add(-cut.astype(jnp.int32))
        n["grain"] = n["grain"].at[i].add(reap.astype(jnp.int32))
        n["grain_stock"] = n["grain_stock"].at[h].add(-reap.astype(jnp.int32))
        gw = (a == 7) & (n["wood"][i] > 0) & (n["wood"][j] < 4)
        gg = (a == 7) & ~gw & (n["grain"][i] > 0) & (n["grain"][j] < 4)
        gf = (a == 7) & ~gw & ~gg & (n["fuel_pack"][i] > 0) & (n["fuel_pack"][j] < 3)
        n["wood"] = n["wood"].at[i].add(-gw.astype(jnp.int32)).at[j].add(gw.astype(jnp.int32))
        n["grain"] = n["grain"].at[i].add(-gg.astype(jnp.int32)).at[j].add(gg.astype(jnp.int32))
        n["fuel_pack"] = n["fuel_pack"].at[i].add(-gf.astype(jnp.int32)).at[j].add(gf.astype(jnp.int32))
        dw = a == 8
        dg = a == 9
        n["timber_delivered"] = n["timber_delivered"].at[t].add(dw.astype(jnp.int32))
        n["grain_delivered"] = n["grain_delivered"].at[t].add(dg.astype(jnp.int32))
        n["wood"] = n["wood"].at[i].add(-(dw | (a == 10) | (a == 11)).astype(jnp.int32))
        n["grain"] = n["grain"].at[i].add(-dg.astype(jnp.int32))
        n["bridge"] = n["bridge"].at[c].add((a == 10).astype(jnp.int32))
        n["canal"] = n["canal"].at[c].set(n["canal"][c] | (a == 11))
        p = n["pos"][i]
        dest = jnp.array([2 * n["crossing_pos"][f, 0] - p[0], 9], dtype=jnp.int32)
        free = ~jnp.any(jnp.all(n["pos"] == dest, axis=1) & (jnp.arange(48) != i))
        tide_safe = ((n["t"] // 16) % 3 != 2) | (n["reef"][f] == 0) | (n["tool"][i] == 2)
        sail = (a == 12) & free & tide_safe
        first = sail & ~n["used_crossing"][f]
        n["pos"] = n["pos"].at[i].set(jnp.where(sail, dest, p))
        n["ferry_fuel"] = n["ferry_fuel"].at[f].add(-sail.astype(jnp.int32) + (a == 13).astype(jnp.int32))
        n["cooldown"] = n["cooldown"].at[f].set(jnp.where(sail, 4, n["cooldown"][f]))
        n["used_crossing"] = n["used_crossing"].at[f].set(n["used_crossing"][f] | sail)
        n["crossings_count"] = n["crossings_count"] + first.astype(jnp.int32)
        n["fuel_pack"] = n["fuel_pack"].at[i].add(-((a == 13) | (a == 14)).astype(jnp.int32) + (a == 15).astype(jnp.int32))
        n["fuel_store"] = n["fuel_store"].at[t].add(-(a == 15).astype(jnp.int32))
        n["signal"] = n["signal"].at[t].set(jnp.where(a == 14, 3 * n["reef"][jnp.minimum(t, 6)] + (n["t"] // 16) % 3, n["signal"][t]))
        n["signal_age"] = n["signal_age"].at[t].set(jnp.where(a == 14, 0, n["signal_age"][t]))
        n["tool"] = n["tool"].at[i].set(jnp.where(a == 16, (n["tool"][i] + 1) % 6, n["tool"][i]))
        n["open_port"] = n["open_port"].at[t].set(n["open_port"][t] ^ (a == 17))
        reward = reward + 0.8 * (dw | dg).astype(jnp.float32) * (n["crew"] == t).astype(jnp.float32) + 0.1 * first.astype(jnp.float32)
        reward = reward.at[i].add(0.3 * (dw | dg).astype(jnp.float32) - 0.04 * ((a == 10) | (a == 11) | (a == 13) | (a == 14)).astype(jnp.float32))
        return n, actions, reward

    def raw_step(self, key, s, actions):
        a = jnp.where((actions >= 0) & (actions < 18), actions, 0).astype(jnp.int32)
        n, a, reward = jax.lax.fori_loop(0, 48, self.actor, (dict(s), a, jnp.zeros((48,), dtype=jnp.float32)))
        n["cooldown"] = jnp.maximum(n["cooldown"] - 1, 0)
        n["signal_age"] = jnp.minimum(n["signal_age"] + 1, 17)
        n["signal"] = jnp.where(n["signal_age"] > 16, -1, n["signal"]).astype(jnp.int32)
        surge = (s["t"] + 1) % 48 == 32
        damage = surge & (n["bridge"] > 0) & ~n["canal"]
        n["damaged"] = damage
        occupied = jnp.any(jnp.all(n["crossing_pos"][:, None, :] == n["pos"][None, :, :], axis=2), axis=1)
        n["bridge"] = jnp.maximum(n["bridge"] - (damage & ~occupied).astype(jnp.int32), 0)
        downstream_loss = jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), ((n["bridge"] > 0) & ~n["canal"]).astype(jnp.int32)))
        season = (s["t"] + 1) % 32 == 0
        n["water"] = jnp.where(season, jnp.maximum(3 - 2 * downstream_loss, 0), n["water"]).astype(jnp.int32)
        renew = season & (n["renewals"] > 0)
        n["timber_stock"] = jnp.minimum(n["timber_stock"] + 2 * renew.astype(jnp.int32), 8)
        n["grain_stock"] = jnp.minimum(n["grain_stock"] + n["water"] * renew.astype(jnp.int32), 8)
        n["renewals"] = n["renewals"] - renew.astype(jnp.int32)
        developed = (n["timber_delivered"] == 2) & (n["grain_delivered"] == 2)
        n["developed_count"] = jnp.sum(developed.astype(jnp.int32))
        harm = jnp.concatenate((jnp.zeros((1,), dtype=jnp.bool_), damage))
        reward = reward - 0.2 * harm[n["crew"]].astype(jnp.float32)
        return self.end_step(s, n, reward, jnp.all(developed))

    def observe(self, s):
        own = jnp.stack((s["pos"][:, 0], s["pos"][:, 1], s["crew"], s["tool"], s["wood"], s["grain"], s["fuel_pack"], s["face"]), axis=1)
        public = jnp.broadcast_to(jnp.concatenate((jnp.array([s["t"], (s["t"] // 16) % 3, s["developed_count"], s["crossings_count"]], dtype=jnp.float32), s["signal"].astype(jnp.float32), s["signal_age"].astype(jnp.float32))), (48, 20))
        av = jnp.stack((s["pos"][:, 0], s["pos"][:, 1], s["crew"], s["tool"], s["wood"], s["grain"], s["fuel_pack"]), axis=1)
        tv = jnp.concatenate((s["town_pos"], jnp.stack((s["timber_delivered"], s["grain_delivered"], s["fuel_store"], s["open_port"]), axis=1)), axis=1)
        cv = jnp.concatenate((s["crossing_pos"], jnp.stack((s["bridge"], s["canal"]), axis=1)), axis=1)
        bv = jnp.concatenate((s["ferry_pos"], jnp.stack((s["ferry_fuel"], s["cooldown"], s["reef"], s["used_crossing"]), axis=1)), axis=1)
        gv = jnp.concatenate((s["grove_pos"], jnp.stack((s["timber_stock"], s["renewals"]), axis=1)), axis=1)
        fv = jnp.concatenate((s["farm_pos"], jnp.stack((s["grain_stock"], s["water"]), axis=1)), axis=1)
        return jnp.concatenate((own.astype(jnp.float32), public, self.visible_rows(s, "agent", av), self.visible_rows(s, "town", tv), self.visible_rows(s, "crossing", cv), self.visible_rows(s, "ferry", bv), self.visible_rows(s, "grove", gv), self.visible_rows(s, "farm", fv), self.patch(s, "terrain")), axis=1).astype(jnp.float32)
