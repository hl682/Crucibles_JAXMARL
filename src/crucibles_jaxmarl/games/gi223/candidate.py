"""Harbor Banquet. Original source; unexecuted author proposal."""
import jax
import jax.numpy as jnp

class GeneratedEnv:
    num_agents = 40
    observation_size = 576
    num_actions = 18
    max_steps = 512
    width = 34
    height = 13
    groups = ("station", "dock", "table", "boat", "junction")

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
        ids = jnp.arange(40, dtype=jnp.int32)
        crew = ids // 5
        g = jnp.arange(8, dtype=jnp.int32)
        xx = 2 + 4 * g
        x = jnp.arange(34)[None, :]
        y = jnp.arange(13)[:, None]
        terrain = jnp.where((x == 0) | (x == 33) | (y == 0) | (y == 12), 1, jnp.where((y == 8) | (y == 9), 2, 0)).astype(jnp.int32)
        s = {"t": jnp.asarray(0, dtype=jnp.int32), "done": jnp.asarray(False, dtype=jnp.bool_), "terminated": jnp.asarray(False, dtype=jnp.bool_), "truncated": jnp.asarray(False, dtype=jnp.bool_),
             "pos": jnp.stack((1 + 4 * crew + ids % 5 % 3, 5 + ids % 5 // 3), axis=1).astype(jnp.int32),
             "crew": crew, "tool": ids % 5, "item": jnp.where(ids % 5 == 1, 1 + crew % 2, 0).astype(jnp.int32),
             "warmth": jnp.zeros((40,), dtype=jnp.int32), "face": jnp.zeros((40,), dtype=jnp.int32),
             "agent_active": jnp.ones((40,), dtype=jnp.bool_), "agent_visible": jnp.zeros((40, 40), dtype=jnp.bool_),
             "map_visible": jnp.zeros((40, 13, 34), dtype=jnp.bool_), "terrain": terrain,
             "station_pos": jnp.stack((xx, jnp.full((8,), 4)), axis=1), "station_active": jnp.ones((8,), dtype=jnp.bool_), "station_visible": jnp.zeros((40, 8), dtype=jnp.bool_),
             "specialty": g % 2 + 1, "fuel": jnp.full((8,), 3, dtype=jnp.int32), "mess": jnp.zeros((8,), dtype=jnp.int32),
             "timer": jnp.zeros((8,), dtype=jnp.int32), "output": jnp.zeros((8,), dtype=jnp.int32), "age": jnp.zeros((8,), dtype=jnp.int32),
             "dock_pos": jnp.stack((xx + 1, jnp.full((8,), 5)), axis=1), "dock_active": jnp.ones((8,), dtype=jnp.bool_), "dock_visible": jnp.zeros((40, 8), dtype=jnp.bool_),
             "stock": jnp.full((8,), 4, dtype=jnp.int32), "assortment": g % 2 + 1, "deliveries_left": jnp.full((8,), 8, dtype=jnp.int32),
             "table_pos": jnp.stack((xx, jnp.full((8,), 7)), axis=1), "table_active": jnp.ones((8,), dtype=jnp.bool_), "table_visible": jnp.zeros((40, 8), dtype=jnp.bool_),
             "trays": jnp.full((8,), 2, dtype=jnp.int32), "fish": jnp.zeros((8,), dtype=jnp.bool_), "stew": jnp.zeros((8,), dtype=jnp.bool_),
             "feast_age": jnp.zeros((8,), dtype=jnp.int32), "served": jnp.zeros((8,), dtype=jnp.int32), "feasts": jnp.zeros((8,), dtype=jnp.int32),
             "boat_pos": jnp.stack((xx, jnp.full((8,), 8)), axis=1), "boat_active": jnp.zeros((8,), dtype=jnp.bool_), "boat_visible": jnp.zeros((40, 8), dtype=jnp.bool_),
             "boat_ticks": jnp.zeros((8,), dtype=jnp.int32), "boat_ready": jnp.zeros((8,), dtype=jnp.bool_),
             "junction_pos": jnp.array([[16, 7]], dtype=jnp.int32), "junction_active": jnp.ones((1,), dtype=jnp.bool_), "junction_visible": jnp.zeros((40, 1), dtype=jnp.bool_),
             "divert": jnp.zeros((1,), dtype=jnp.bool_), "blocked": jnp.zeros((1,), dtype=jnp.bool_),
             "total_meals": jnp.asarray(0, dtype=jnp.int32), "total_feasts": jnp.asarray(0, dtype=jnp.int32), "spoiled": jnp.zeros((8,), dtype=jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def context(self, s, i):
        p = s["pos"][i]
        st, ns = self.pick(p, s["station_pos"], s["station_active"], 1)
        d, nd = self.pick(p, s["dock_pos"], s["dock_active"], 1)
        tb, nt = self.pick(p, s["table_pos"], s["table_active"], 1)
        j, nj = self.pick(p, s["pos"], (jnp.arange(40) != i) & (s["item"] == 0), 1)
        b, nb = self.pick(p, s["boat_pos"], s["boat_active"] & s["boat_ready"], 1)
        lever = jnp.sum(jnp.abs(p - s["junction_pos"][0])) <= 1
        return st, ns, d, nd, tb, nt, j, nj, b, nb, lever

    def passable(self, s, q, i):
        return s["terrain"][q[1], q[0]] != 1

    def choices(self, s, i):
        st, ns, d, nd, tb, nt, j, nj, b, nb, lever = self.context(s, i)
        item = s["item"][i]
        cooked = (item == 3) | (item == 4)
        clear = ~jnp.any(s["boat_active"] & jnp.all(s["boat_pos"] == s["table_pos"][tb] + jnp.array([0, 1], dtype=jnp.int32), axis=1))
        plate = clear & nt & cooked & (s["trays"][tb] > 0) & ~s["boat_active"][tb] & ~s["fish"][tb] & ~s["stew"][tb]
        composite = clear & nt & cooked & ~s["boat_active"][tb] & (((~s["fish"][tb] & ~s["stew"][tb]) & (s["trays"][tb] > 0)) | s["fish"][tb] | s["stew"][tb])
        composite = composite & jnp.where(item == 3, ~s["fish"][tb], ~s["stew"][tb])
        return jnp.array([True, True, True, True, True,
            nd & (item == 0) & (s["stock"][d] > 0),
            ns & (s["tool"][i] == 1) & (item == s["specialty"][st]) & (s["timer"][st] == 0) & (s["output"][st] == 0) & (s["fuel"][st] > 0) & (s["mess"][st] < 3),
            ns & (item == 0) & (s["output"][st] > 0), plate,
            ns & (s["mess"][st] > 0), lever, lever & (s["tool"][i] == 4) & s["blocked"][0],
            ns & (item == 5) & (s["fuel"][st] <= 8), item > 0,
            nj & (item > 0), composite, nb, nt & (tb == s["crew"][i])], dtype=jnp.bool_)

    def actor(self, k, carry):
        n, actions, reward = carry
        i = (k + n["t"]) % 40
        a = actions[i]
        a = jnp.where(self.choices(n, i)[a], a, 0)
        st, ns, d, nd, tb, nt, j, nj, b, nb, lever = self.context(n, i)
        item = n["item"][i]
        n = self.move_actor(n, i, a)
        pickup = a == 5
        n["item"] = n["item"].at[i].set(jnp.where(pickup, n["assortment"][d], item))
        n["stock"] = n["stock"].at[d].add(-pickup.astype(jnp.int32))
        cook = a == 6
        n["item"] = n["item"].at[i].set(jnp.where(cook, 0, n["item"][i]))
        n["timer"] = n["timer"].at[st].set(jnp.where(cook, 4, n["timer"][st]))
        n["fuel"] = n["fuel"].at[st].add(-cook.astype(jnp.int32))
        take = a == 7
        n["item"] = n["item"].at[i].set(jnp.where(take, n["output"][st], n["item"][i]))
        n["warmth"] = n["warmth"].at[i].set(jnp.where(take, 16, n["warmth"][i]))
        n["output"] = n["output"].at[st].set(jnp.where(take, 0, n["output"][st]))
        n["mess"] = n["mess"].at[st].set(jnp.where(a == 9, jnp.maximum(n["mess"][st] - 1, 0), n["mess"][st]))
        n["divert"] = n["divert"].at[0].set(n["divert"][0] ^ (a == 10))
        n["blocked"] = n["blocked"].at[0].set(n["blocked"][0] & ~(a == 11))
        n["fuel"] = n["fuel"].at[st].add(3 * (a == 12).astype(jnp.int32))
        n["item"] = n["item"].at[i].set(jnp.where((a == 12) | (a == 13), 0, n["item"][i]))
        give = a == 14
        n["item"] = n["item"].at[j].set(jnp.where(give, item, n["item"][j])).at[i].set(jnp.where(give, 0, n["item"][i]))
        n["warmth"] = n["warmth"].at[j].set(jnp.where(give, n["warmth"][i], n["warmth"][j]))
        composite = a == 15
        first = composite & ~n["fish"][tb] & ~n["stew"][tb]
        n["fish"] = n["fish"].at[tb].set(n["fish"][tb] | (composite & (item == 3)))
        n["stew"] = n["stew"].at[tb].set(n["stew"][tb] | (composite & (item == 4)))
        n["feast_age"] = n["feast_age"].at[tb].set(jnp.where(first, 0, n["feast_age"][tb]))
        complete = composite & n["fish"][tb] & n["stew"][tb]
        serve = a == 8
        launch = serve | complete
        n["trays"] = n["trays"].at[tb].add(-(serve | first).astype(jnp.int32))
        n["fish"] = n["fish"].at[tb].set(n["fish"][tb] & ~complete)
        n["stew"] = n["stew"].at[tb].set(n["stew"][tb] & ~complete)
        n["item"] = n["item"].at[i].set(jnp.where(serve | composite, 0, n["item"][i]))
        n["boat_active"] = n["boat_active"].at[tb].set(n["boat_active"][tb] | launch)
        n["boat_pos"] = n["boat_pos"].at[tb].set(jnp.where(launch, n["table_pos"][tb] + jnp.array([0, 1], dtype=jnp.int32), n["boat_pos"][tb]))
        n["boat_ticks"] = n["boat_ticks"].at[tb].set(jnp.where(launch, 0, n["boat_ticks"][tb]))
        n["boat_ready"] = n["boat_ready"].at[tb].set(n["boat_ready"][tb] & ~launch)
        n["served"] = n["served"].at[tb].add(serve.astype(jnp.int32))
        n["feasts"] = n["feasts"].at[tb].add(complete.astype(jnp.int32))
        n["total_meals"] = n["total_meals"] + serve.astype(jnp.int32)
        n["total_feasts"] = n["total_feasts"] + complete.astype(jnp.int32)
        preferred = item == 3 + ((n["t"] // 32 + tb) % 2)
        value = serve.astype(jnp.float32) * jnp.where(preferred, 1.0, 0.5) + 2.0 * complete.astype(jnp.float32)
        reward = reward + value * (n["crew"] == tb).astype(jnp.float32) + 0.1 * complete.astype(jnp.float32)
        unload = a == 16
        n["trays"] = n["trays"].at[b].add(unload.astype(jnp.int32))
        n["boat_active"] = n["boat_active"].at[b].set(n["boat_active"][b] & ~unload)
        n["boat_ready"] = n["boat_ready"].at[b].set(n["boat_ready"][b] & ~unload)
        n["tool"] = n["tool"].at[i].set(jnp.where(a == 17, (n["tool"][i] + 1) % 5, n["tool"][i]))
        reward = reward.at[i].add(-0.01 * ((a >= 6) & (a != 7) & (a != 14)).astype(jnp.float32))
        return n, actions, reward

    def sail(self, b, n):
        p = n["boat_pos"][b]
        nx = 1 + p[0] % 32
        lane = ((p[0] == 16) & n["divert"][0]) | ((p[1] == 9) & (nx < 19))
        q = jnp.array([nx, jnp.where(lane, 9, 8)], dtype=jnp.int32)
        jam = (nx == 17) & ~lane & n["blocked"][0]
        occupied = jnp.any(n["boat_active"] & (jnp.arange(8) != b) & jnp.all(n["boat_pos"] == q, axis=1))
        moving = n["boat_active"][b] & ~n["boat_ready"][b] & ~jam & ~occupied
        n["boat_pos"] = n["boat_pos"].at[b].set(jnp.where(moving, q, p))
        n["boat_ticks"] = n["boat_ticks"].at[b].add(moving.astype(jnp.int32))
        arrived = n["boat_active"][b] & (n["boat_ticks"][b] >= 32) & (n["boat_pos"][b, 0] == n["table_pos"][b, 0]) & (n["boat_pos"][b, 1] == 8)
        n["boat_ready"] = n["boat_ready"].at[b].set(n["boat_ready"][b] | arrived)
        return n

    def raw_step(self, key, s, actions):
        a = jnp.where((actions >= 0) & (actions < 18), actions, 0).astype(jnp.int32)
        n, a, reward = jax.lax.fori_loop(0, 40, self.actor, (dict(s), a, jnp.zeros((40,), dtype=jnp.float32)))
        finish = n["timer"] == 1
        n["timer"] = jnp.maximum(n["timer"] - 1, 0)
        n["output"] = jnp.where(finish, n["specialty"] + 2, n["output"]).astype(jnp.int32)
        n["mess"] = jnp.minimum(n["mess"] + finish.astype(jnp.int32), 3)
        n["age"] = jnp.where(finish, 0, jnp.where(n["output"] > 0, n["age"] + 1, 0)).astype(jnp.int32)
        spoiled = (n["output"] > 0) & (n["age"] >= 12)
        n["output"] = jnp.where(spoiled, 0, n["output"]).astype(jnp.int32)
        n["spoiled"] = spoiled
        reward = reward - 0.25 * spoiled[n["crew"]].astype(jnp.float32)
        n["warmth"] = jnp.maximum(n["warmth"] - 1, 0)
        cold = ((n["item"] == 3) | (n["item"] == 4)) & (n["warmth"] == 0)
        n["item"] = jnp.where(cold, 6, n["item"]).astype(jnp.int32)
        reward = reward - 0.1 * cold.astype(jnp.float32)
        waiting = n["fish"] | n["stew"]
        n["feast_age"] = jnp.where(waiting, n["feast_age"] + 1, 0).astype(jnp.int32)
        timeout = waiting & (n["feast_age"] >= 24)
        n["fish"] = n["fish"] & ~timeout
        n["stew"] = n["stew"] & ~timeout
        n["trays"] = n["trays"] + timeout.astype(jnp.int32)
        reward = reward - 0.3 * timeout[n["crew"]].astype(jnp.float32)
        arrival = ((s["t"] + 1) % 32 == 0) & (n["deliveries_left"] > 0)
        n["stock"] = jnp.where(arrival, 3, n["stock"]).astype(jnp.int32)
        next_type = jnp.where(((s["t"] + 1) // 32 + jnp.arange(8)) % 3 == 2, 5, 1 + ((s["t"] + 1) // 32 + jnp.arange(8)) % 3)
        n["assortment"] = jnp.where(arrival, next_type, n["assortment"]).astype(jnp.int32)
        n["deliveries_left"] = n["deliveries_left"] - arrival.astype(jnp.int32)
        n["blocked"] = n["blocked"] | jnp.full((1,), (s["t"] + 1) % 48 == 0, dtype=jnp.bool_)
        n = jax.lax.fori_loop(0, 8, self.sail, n)
        return self.end_step(s, n, reward, jnp.all(n["feasts"] >= 1))

    def observe(self, s):
        own = jnp.stack((s["pos"][:, 0], s["pos"][:, 1], s["crew"], s["tool"], s["item"], s["warmth"], s["face"]), axis=1)
        public = jnp.broadcast_to(jnp.array([s["t"], s["total_meals"], s["total_feasts"]], dtype=jnp.float32), (40, 3))
        av = jnp.stack((s["pos"][:, 0], s["pos"][:, 1], s["crew"], s["tool"], s["item"], s["warmth"]), axis=1)
        st = jnp.concatenate((s["station_pos"], jnp.stack((s["specialty"], s["fuel"], s["mess"], s["timer"], s["output"], s["age"]), axis=1)), axis=1)
        dock = jnp.concatenate((s["dock_pos"], jnp.stack((s["stock"], s["assortment"], s["deliveries_left"]), axis=1)), axis=1)
        table = jnp.concatenate((s["table_pos"], jnp.stack((s["trays"], s["fish"], s["stew"], s["feast_age"], s["served"], s["feasts"]), axis=1)), axis=1)
        boat = jnp.concatenate((s["boat_pos"], jnp.stack((s["boat_ticks"], s["boat_ready"]), axis=1)), axis=1)
        junction = jnp.concatenate((s["junction_pos"], jnp.stack((s["divert"], s["blocked"]), axis=1)), axis=1)
        return jnp.concatenate((own.astype(jnp.float32), public, self.visible_rows(s, "agent", av), self.visible_rows(s, "station", st), self.visible_rows(s, "dock", dock), self.visible_rows(s, "table", table), self.visible_rows(s, "boat", boat), self.visible_rows(s, "junction", junction), self.patch(s, "terrain")), axis=1).astype(jnp.float32)
