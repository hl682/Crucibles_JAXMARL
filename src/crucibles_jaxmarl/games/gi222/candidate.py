"""Cavern Rescue Guilds. Original source; unexecuted author proposal."""
import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 20
    observation_size = 344
    num_actions = 18
    max_steps = 512
    width = 27
    height = 15
    groups = ("depot", "casualty")

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
        ids = jnp.arange(20, dtype=jnp.int32)
        crew = ids // 4
        x = jnp.arange(27)[None, :]
        y = jnp.arange(15)[:, None]
        border = (x == 0) | (x == 26) | (y == 0) | (y == 14)
        open_cell = ((y >= 6) & (y <= 8)) | (((x - 3) % 5 == 0) & (y >= 3) & (y <= 11))
        terrain = jnp.where(border, 2, jnp.where(open_cell, 0, 1)).astype(jnp.int32)
        dp = jnp.stack((3 + 5 * jnp.arange(5), jnp.full((5,), 8)), axis=1).astype(jnp.int32)
        cp = jnp.stack((3 + 5 * jnp.arange(5), jnp.full((5,), 2)), axis=1).astype(jnp.int32)
        s = {"t": jnp.asarray(0, dtype=jnp.int32), "done": jnp.asarray(False, dtype=jnp.bool_), "terminated": jnp.asarray(False, dtype=jnp.bool_), "truncated": jnp.asarray(False, dtype=jnp.bool_),
             "pos": jnp.stack((3 + 5 * crew, 6 + ids % 4), axis=1).astype(jnp.int32),
             "crew": crew, "tool": ids % 4, "wood": jnp.full((20,), 2, dtype=jnp.int32),
             "med": jnp.ones((20,), dtype=jnp.int32), "health": jnp.full((20,), 5, dtype=jnp.int32),
             "face": jnp.zeros((20,), dtype=jnp.int32), "carrying": jnp.zeros((20,), dtype=jnp.bool_),
             "agent_active": jnp.ones((20,), dtype=jnp.bool_), "agent_visible": jnp.zeros((20, 20), dtype=jnp.bool_),
             "map_visible": jnp.zeros((20, 15, 27), dtype=jnp.bool_), "terrain": terrain,
             "dust": jnp.zeros((15, 27), dtype=jnp.int32),
             "depot_pos": dp, "depot_active": jnp.ones((5,), dtype=jnp.bool_), "depot_visible": jnp.zeros((20, 5), dtype=jnp.bool_),
             "depot_wood": jnp.full((5,), 12, dtype=jnp.int32), "depot_med": jnp.full((5,), 4, dtype=jnp.int32),
             "depot_aid": jnp.zeros((5,), dtype=jnp.int32),
             "casualty_pos": cp, "casualty_active": jnp.ones((5,), dtype=jnp.bool_), "casualty_visible": jnp.zeros((20, 5), dtype=jnp.bool_),
             "bearer": jnp.full((5,), -1, dtype=jnp.int32), "rescued": jnp.asarray(0, dtype=jnp.int32),
             "dug": jnp.zeros((20,), dtype=jnp.bool_), "pumped": jnp.zeros((20,), dtype=jnp.bool_),
             "injured": jnp.zeros((20,), dtype=jnp.bool_), "evacuated": jnp.zeros((5,), dtype=jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def context(self, s, i):
        p = s["pos"][i]
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        q = jnp.clip(p + dirs[s["face"][i]], jnp.array([0, 0]), jnp.array([26, 14]))
        d, nd = self.pick(p, s["depot_pos"], s["depot_active"], 1)
        c, nc = self.pick(p, s["casualty_pos"], s["casualty_active"] & (s["bearer"] < 0), jnp.where(s["terrain"][p[1], p[0]] == 6, 3, 1))
        j, nj = self.pick(p, s["pos"], jnp.arange(20) != i, 1)
        return p, q, d, nd, c, nc, j, nj

    def passable(self, s, q, i):
        tile = s["terrain"][q[1], q[0]]
        return (tile != 1) & (tile != 2) & ((s["health"][i] > 2) | (s["t"] % 2 == 0))

    def choices(self, s, i):
        p, q, d, nd, c, nc, j, nj = self.context(s, i)
        tile = s["terrain"][p[1], p[0]]
        wood = s["wood"][i] > 0
        builder = (s["tool"][i] == 1) & wood & (tile == 0)
        vent = s["tool"][i] == 2
        medic = s["tool"][i] == 3
        return jnp.array([True, True, True, True, True,
            (s["tool"][i] == 0) & (s["terrain"][q[1], q[0]] == 1) & (s["health"][i] > 1),
            vent & wood & (tile == 0), builder, builder, builder,
            vent & (tile == 4), (tile == 5) | (tile == 6),
            medic & nc & ~s["carrying"][i] & (s["terrain"][s["casualty_pos"][c, 1], s["casualty_pos"][c, 0]] != 1),
            nj & (((s["wood"][i] > 0) & (s["wood"][j] < 4)) | ((s["med"][i] > 0) & (s["med"][j] < 2))),
            medic & nj & (s["med"][i] > 0) & (s["health"][j] < 5),
            nd & (((s["wood"][i] < 4) & (s["depot_wood"][d] > 0)) | ((s["med"][i] < 2) & (s["depot_med"][d] > 0))),
            nd, nd & (d != s["crew"][i]) & wood & (s["depot_aid"][d] < 2)], dtype=jnp.bool_)

    def actor(self, k, carry):
        n, actions, reward = carry
        i = (k + n["t"]) % 20
        a = actions[i]
        a = jnp.where(self.choices(n, i)[a], a, 0)
        p, q, d, nd, c, nc, j, nj = self.context(n, i)
        n = self.move_actor(n, i, a)
        dig = a == 5
        n["terrain"] = n["terrain"].at[q[1], q[0]].set(jnp.where(dig, 0, n["terrain"][q[1], q[0]]))
        n["dust"] = n["dust"].at[q[1], q[0]].set(jnp.where(dig, 8, n["dust"][q[1], q[0]]))
        n["dug"] = n["dug"].at[i].set(dig)
        build = (a >= 6) & (a <= 9)
        code = jnp.where(a == 6, 4, jnp.where(a == 7, 3, jnp.where(a == 8, 5, 6)))
        n["terrain"] = n["terrain"].at[p[1], p[0]].set(jnp.where(build, code, n["terrain"][p[1], p[0]]))
        n["wood"] = n["wood"].at[i].add(-build.astype(jnp.int32))
        yy = jnp.arange(15)[:, None]
        xx = jnp.arange(27)[None, :]
        pumped = (a == 10) & (jnp.abs(xx - p[0]) + jnp.abs(yy - p[1]) <= 3)
        n["dust"] = jnp.maximum(n["dust"] - 4 * pumped.astype(jnp.int32), 0)
        n["pumped"] = n["pumped"].at[i].set(a == 10)
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        dest = jnp.clip(p + 2 * dirs[n["face"][i]], jnp.array([0, 0]), jnp.array([26, 14]))
        rail = (n["terrain"][q[1], q[0]] >= 5) & (n["terrain"][dest[1], dest[0]] >= 5)
        clear = ~jnp.any(jnp.all(n["pos"] == dest, axis=1) & (jnp.arange(20) != i))
        clear = clear & ~jnp.any(jnp.all(n["pos"] == q, axis=1) & (jnp.arange(20) != i))
        n["pos"] = n["pos"].at[i].set(jnp.where((a == 11) & rail & clear, dest, n["pos"][i]))
        n["bearer"] = n["bearer"].at[c].set(jnp.where(a == 12, i, n["bearer"][c]))
        n["carrying"] = n["carrying"].at[i].set(n["carrying"][i] | (a == 12))
        givew = (a == 13) & (n["wood"][i] > 0) & (n["wood"][j] < 4)
        givem = (a == 13) & ~givew & (n["med"][i] > 0) & (n["med"][j] < 2)
        n["wood"] = n["wood"].at[i].add(-givew.astype(jnp.int32)).at[j].add(givew.astype(jnp.int32))
        n["med"] = n["med"].at[i].add(-givem.astype(jnp.int32)).at[j].add(givem.astype(jnp.int32))
        aid = a == 14
        n["health"] = n["health"].at[j].set(jnp.where(aid, 5, n["health"][j]))
        n["med"] = n["med"].at[i].add(-aid.astype(jnp.int32))
        lw = (a == 15) & (n["wood"][i] < 4) & (n["depot_wood"][d] > 0)
        lm = (a == 15) & (n["med"][i] < 2) & (n["depot_med"][d] > 0)
        n["wood"] = n["wood"].at[i].add(lw.astype(jnp.int32))
        n["med"] = n["med"].at[i].add(lm.astype(jnp.int32))
        n["depot_wood"] = n["depot_wood"].at[d].add(-lw.astype(jnp.int32))
        n["depot_med"] = n["depot_med"].at[d].add(-lm.astype(jnp.int32))
        n["tool"] = n["tool"].at[i].set(jnp.where(a == 16, (n["tool"][i] + 1) % 4, n["tool"][i]))
        donate = a == 17
        n["wood"] = n["wood"].at[i].add(-donate.astype(jnp.int32))
        n["depot_wood"] = n["depot_wood"].at[d].add(donate.astype(jnp.int32))
        n["depot_aid"] = n["depot_aid"].at[d].add(donate.astype(jnp.int32))
        reward = reward.at[i].add(-0.02 * (build | aid).astype(jnp.float32) + 0.4 * donate.astype(jnp.float32))
        return n, actions, reward

    def raw_step(self, key, s, actions):
        a = jnp.where((actions >= 0) & (actions < 18), actions, 0).astype(jnp.int32)
        n = dict(s)
        n["dug"] = jnp.zeros((20,), dtype=jnp.bool_)
        n["pumped"] = jnp.zeros((20,), dtype=jnp.bool_)
        n, a, reward = jax.lax.fori_loop(0, 20, self.actor, (n, a, jnp.zeros((20,), dtype=jnp.float32)))
        attached = n["bearer"] >= 0
        carrier = jnp.clip(n["bearer"], 0, 19)
        n["casualty_pos"] = jnp.where(attached[:, None], n["pos"][carrier], n["casualty_pos"])
        close = jnp.min(jnp.sum(jnp.abs(n["casualty_pos"][:, None, :] - n["depot_pos"][None, :, :]), axis=2), axis=1) <= 1
        evacuated = attached & close & n["casualty_active"]
        n["evacuated"] = evacuated
        n["casualty_active"] = n["casualty_active"] & ~evacuated
        n["bearer"] = jnp.where(evacuated, -2, n["bearer"]).astype(jnp.int32)
        n["carrying"] = jnp.any(n["bearer"][:, None] == jnp.arange(20)[None, :], axis=0)
        n["rescued"] = s["rescued"] + jnp.sum(evacuated.astype(jnp.int32))
        crew_credit = jnp.sum(evacuated[:, None] & (s["crew"][carrier][:, None] == s["crew"][None, :]), axis=0)
        reward = reward + 5.0 * crew_credit.astype(jnp.float32) + 0.4 * jnp.sum(evacuated.astype(jnp.float32))
        open_cell = (n["terrain"] != 1) & (n["terrain"] != 2)
        dust = jnp.where(open_cell, n["dust"], 0)
        spread = jnp.maximum(jnp.maximum(jnp.roll(dust, 1, 0), jnp.roll(dust, -1, 0)), jnp.maximum(jnp.roll(dust, 1, 1), jnp.roll(dust, -1, 1)))
        n["dust"] = jnp.where(open_cell, jnp.maximum(jnp.maximum(dust - 1, spread - 2), 0), 0).astype(jnp.int32)
        occupied = jnp.any((jnp.arange(27)[None, None, :] == n["pos"][:, 0, None, None]) & (jnp.arange(15)[None, :, None] == n["pos"][:, 1, None, None]), axis=0)
        collapse = (n["terrain"] == 0) & (n["dust"] >= 5) & ((s["t"] + 1) % 24 == 0) & ~occupied
        n["terrain"] = jnp.where(collapse, 1, n["terrain"]).astype(jnp.int32)
        injury = n["dust"][n["pos"][:, 1], n["pos"][:, 0]] >= 5
        n["injured"] = injury
        n["health"] = jnp.maximum(n["health"] - injury.astype(jnp.int32), 1)
        reward = reward - 0.1 * injury.astype(jnp.float32) - 0.005 * jnp.sum(injury.astype(jnp.float32))
        return self.end_step(s, n, reward, n["rescued"] == 5)

    def observe(self, s):
        own = jnp.stack((s["pos"][:, 0], s["pos"][:, 1], s["crew"], s["tool"], s["wood"], s["med"], s["health"], s["carrying"], s["face"]), axis=1)
        public = jnp.broadcast_to(jnp.array([s["t"], s["rescued"]], dtype=jnp.float32), (20, 2))
        av = jnp.stack((s["pos"][:, 0], s["pos"][:, 1], s["crew"], s["tool"], s["wood"], s["med"], s["health"], s["carrying"]), axis=1)
        dv = jnp.stack((s["depot_pos"][:, 0], s["depot_pos"][:, 1], s["depot_wood"], s["depot_med"], s["depot_aid"]), axis=1)
        cv = jnp.stack((s["casualty_pos"][:, 0], s["casualty_pos"][:, 1], s["bearer"], s["casualty_active"]), axis=1)
        return jnp.concatenate((own.astype(jnp.float32), public, self.visible_rows(s, "agent", av), self.visible_rows(s, "depot", dv), self.visible_rows(s, "casualty", cv), self.patch(s, "terrain"), self.patch(s, "dust")), axis=1).astype(jnp.float32)
