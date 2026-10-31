"""Skybridge Quarry Cooperative. Original authored rules; not executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 160
    observation_size = 154
    num_actions = 17
    max_steps = 768

    def directions(self):
        return jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)

    def target(self, s, i):
        return jnp.clip(s["pos"][i] + self.directions()[s["face"][i]],
                        jnp.array([0, 0], jnp.int32), jnp.array([59, 39], jnp.int32))

    def context(self, s, i):
        p = s["pos"][i]
        q = self.target(s, i)
        j = jnp.argmin(jnp.sum(jnp.abs(s["pos"] - q), axis=1))
        near = (j != i) & s["active"][j] & jnp.all(s["pos"][j] == q)
        l = jnp.argmin(jnp.sum(jnp.abs(s["lift_pos"] - p), axis=1))
        on = (p[1] == s["lift_pos"][l, 1]) & (jnp.abs(p[0] - s["lift_pos"][l, 0]) == 1)
        end = jnp.stack([2 * s["lift_pos"][l, 0] - p[0], p[1]])
        d = jnp.argmin(jnp.sum(jnp.abs(s["depot_pos"] - p), axis=1))
        return q, j, near, l, on, end, d

    def legal_row(self, s, i):
        p = s["pos"][i]
        q, j, near, l, on, end, d = self.context(s, i)
        tiles = s["tiles"]
        tile = tiles[q[1], q[0]]
        grounded = (tiles[p[1], p[0]] != 0)
        steps = p[None, :] + self.directions()
        inside = jnp.all((steps >= jnp.array([0, 0], jnp.int32)) & (steps <= jnp.array([59, 39], jnp.int32)), axis=1)
        c = jnp.clip(steps, jnp.array([0, 0], jnp.int32), jnp.array([59, 39], jnp.int32))
        v = tiles[c[:, 1], c[:, 0]]
        occupied = jnp.any(jnp.all(c[:, None, :] == s["pos"][None, :, :], axis=2) & s["active"][None, :], axis=1)
        moves = inside & ((v == 1) | (v == 4) | (v == 5)) & ~occupied & grounded
        blocked_end = jnp.any(jnp.all(s["pos"] == end, axis=1) & s["active"])
        carrying = (s["stone"][i] > 0) | (s["ore"][i] > 0)
        give = near & (((s["stone"][i] > 0) & (s["stone"][j] < 4)) |
                       ((s["ore"][i] > 0) & (s["ore"][j] < 1)))
        depot = jnp.sum(jnp.abs(s["depot_pos"][d] - p)) <= 1
        mask = jnp.concatenate([
            jnp.ones((1,), jnp.bool_), moves, jnp.ones((4,), jnp.bool_),
            jnp.stack([
                (s["tool"][i] == 0) & ((tile == 2) | (tile == 3)) & (s["stone"][i] < 4) & (s["ore"][i] < 1),
                (s["tool"][i] == 1) & (s["stone"][i] > 0) & (tile == 0),
                depot & (s["ore"][i] > 0), give,
                on & grounded & (s["weight"][l] >= 1 + carrying.astype(jnp.int32)) & ~blocked_end,
                on & (s["tool"][i] == 2) & (s["stone"][i] > 0) & (s["weight"][l] < 4),
                near, jnp.asarray(True)
            ])])
        return jnp.where(s["active"][i] & ~s["done"], mask,
                         jnp.arange(self.num_actions) == 0)

    def legal_actions(self, s):
        return jax.vmap(lambda i: self.legal_row(s, i))(jnp.arange(self.num_agents))

    def refresh(self, s):
        s = dict(s)
        p = s["pos"]
        s["active"] = s["health"] > 0
        distances = jnp.max(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2)
        local = distances <= 3
        other = local & ~jnp.eye(160, dtype=jnp.bool_) & s["active"][None, :]
        nearest = jnp.argsort(jnp.where(other, distances, 999), axis=1)[:, :4]
        detail = jnp.sum(jax.nn.one_hot(nearest, 160, dtype=jnp.int32), axis=1) > 0
        s["agent_vis"] = local & (detail | jnp.eye(160, dtype=jnp.bool_)) & s["active"][None, :]
        x = jnp.arange(60)[None, None, :]
        y = jnp.arange(40)[None, :, None]
        s["map_vis"] = (jnp.abs(x - p[:, 0, None, None]) <= 3) & (jnp.abs(y - p[:, 1, None, None]) <= 3)
        s["resource_active"] = (s["tiles"][s["resource_pos"][:, 1], s["resource_pos"][:, 0]] == s["resource_kind"])
        for name in ("resource", "lift", "depot"):
            s[name + "_vis"] = (jnp.max(jnp.abs(p[:, None, :] - s[name + "_pos"][None, :, :]), axis=2) <= 3) & s[name + "_active"][None, :]
        return s

    def reset(self, key):
        g = jnp.arange(20, dtype=jnp.int32)
        base = jnp.stack([(g % 5) * 12, (g // 5) * 10], axis=1)
        offsets = jnp.array([[5, 3], [5, 4], [6, 4], [5, 6], [2, 3], [4, 4], [3, 5], [4, 6]], jnp.int32)
        pos = (base[:, None, :] + offsets[None, :, :]).reshape((160, 2))
        x = jnp.arange(60)[None, :]
        y = jnp.arange(40)[:, None]
        tiles = jnp.where(((x % 12) == 6) & ((y % 10) > 0) & ((y % 10) < 9), 0, 1).astype(jnp.int32)
        bridge = base + jnp.array([6, 4], jnp.int32)
        support = base + jnp.array([6, 3], jnp.int32)
        ore = base + jnp.array([3, 3], jnp.int32)
        depot = base + jnp.array([2, 5], jnp.int32)
        tiles = tiles.at[bridge[:, 1], bridge[:, 0]].set(4)
        tiles = tiles.at[support[:, 1], support[:, 0]].set(3)
        tiles = tiles.at[ore[:, 1], ore[:, 0]].set(2)
        tiles = tiles.at[depot[:, 1], depot[:, 0]].set(5)
        s = {
            "t": jnp.asarray(0, jnp.int32), "done": jnp.asarray(False), "terminated": jnp.asarray(False), "truncated": jnp.asarray(False),
            "tiles": tiles, "pos": pos, "active": jnp.ones((160,), jnp.bool_),
            "face": jnp.ones((160,), jnp.int32), "crew": jnp.arange(160, dtype=jnp.int32) // 8,
            "tool": jnp.tile(jnp.array([0, 1, 3, 2, 0, 1, 3, 3], jnp.int32), 20),
            "stone": jnp.tile(jnp.array([0, 1, 0, 1, 0, 1, 0, 0], jnp.int32), 20),
            "ore": jnp.zeros((160,), jnp.int32), "health": jnp.full((160,), 4, jnp.int32),
            "message": jnp.zeros((160,), jnp.int32), "heard": jnp.zeros((160,), jnp.int32),
            "bridge_pos": bridge, "support_pos": support, "strength": jnp.full((20,), 3, jnp.int32),
            "resource_pos": jnp.concatenate([ore, support], axis=0),
            "resource_kind": jnp.concatenate([jnp.full((20,), 2, jnp.int32), jnp.full((20,), 3, jnp.int32)]),
            "resource_active": jnp.ones((40,), jnp.bool_), "resource_vis": jnp.zeros((160, 40), jnp.bool_),
            "lift_pos": base + jnp.array([6, 6], jnp.int32), "lift_active": jnp.ones((20,), jnp.bool_),
            "lift_vis": jnp.zeros((160, 20), jnp.bool_), "weight": jnp.full((20,), 2, jnp.int32),
            "depot_pos": depot, "depot_active": jnp.ones((20,), jnp.bool_),
            "depot_vis": jnp.zeros((160, 20), jnp.bool_), "delivered": jnp.zeros((20,), jnp.int32),
            "agent_vis": jnp.zeros((160, 160), jnp.bool_), "map_vis": jnp.zeros((160, 40, 60), jnp.bool_),
            "mined": jnp.zeros((160,), jnp.bool_), "built": jnp.zeros((160,), jnp.bool_),
            "rescued": jnp.zeros((160,), jnp.bool_), "fell": jnp.zeros((160,), jnp.bool_),
            "unloaded": jnp.zeros((160,), jnp.bool_), "rode": jnp.zeros((160,), jnp.bool_),
            "reward_parts": jnp.zeros((160, 5), jnp.float32)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        ids = jnp.arange(160)
        p = s["pos"]
        self_fields = jnp.stack([p[:, 0] / 59.0, p[:, 1] / 39.0, s["face"] / 3.0,
            s["stone"] / 4.0, s["ore"] * 1.0, s["tool"] / 3.0, s["health"] / 4.0,
            s["crew"] / 19.0, jnp.full((160,), s["t"] / 768.0), s["heard"] / 3.0,
            (s["tiles"][p[:, 1], p[:, 0]] != 0) * 1.0,
            s["mined"] * 1.0, s["built"] * 1.0, s["rescued"] * 1.0], axis=1)
        k = jnp.arange(49)
        off = jnp.stack([k % 7 - 3, k // 7 - 3], axis=1)
        q = p[:, None, :] + off[None, :, :]
        inside = jnp.all((q >= jnp.array([0, 0], jnp.int32)) & (q <= jnp.array([59, 39], jnp.int32)), axis=2)
        qc = jnp.clip(q, jnp.array([0, 0], jnp.int32), jnp.array([59, 39], jnp.int32))
        terrain = jnp.where(inside, s["tiles"][qc[:, :, 1], qc[:, :, 0]] / 5.0, -1.0)
        occupied = jnp.any(jnp.all(q[:, :, None, :] == p[None, None, :, :], axis=3) & s["active"][None, None, :], axis=2)
        patch = jnp.stack([terrain, occupied * 1.0], axis=2).reshape((160, 98))
        dist = jnp.max(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2)
        nv = s["agent_vis"] & (ids[:, None] != ids[None, :])
        ni = jnp.argsort(jnp.where(nv, dist, 999), axis=1)[:, :4]
        visible = jnp.take_along_axis(nv, ni, axis=1)
        rel = (p[ni] - p[:, None, :]) / 3.0
        neigh = jnp.stack([visible * 1.0, rel[:, :, 0], rel[:, :, 1], s["tool"][ni] / 3.0,
                           s["stone"][ni] / 4.0, s["ore"][ni] * 1.0, s["message"][ni] / 3.0,
                           s["crew"][ni] / 19.0], axis=2)
        neigh = jnp.where(visible[:, :, None], neigh, 0.0).reshape((160, 32))
        li = jnp.argmin(jnp.sum(jnp.abs(p[:, None, :] - s["lift_pos"][None, :, :]), axis=2), axis=1)
        lv = s["lift_vis"][ids, li]
        lr = (s["lift_pos"][li] - p) / 3.0
        lift = jnp.stack([lv * 1.0, lr[:, 0], lr[:, 1], s["weight"][li] / 4.0, (s["weight"][li] > 0) * 1.0], axis=1)
        di = jnp.argmin(jnp.sum(jnp.abs(p[:, None, :] - s["depot_pos"][None, :, :]), axis=2), axis=1)
        dv = s["depot_vis"][ids, di]
        dr = (s["depot_pos"][di] - p) / 3.0
        depot = jnp.stack([dv * 1.0, dr[:, 0], dr[:, 1], s["delivered"][di] / 20.0, di / 19.0], axis=1)
        return jnp.concatenate([self_fields, patch, neigh, jnp.where(lv[:, None], lift, 0.0),
                                jnp.where(dv[:, None], depot, 0.0)], axis=1).astype(jnp.float32)

    def act(self, i, s, a):
        i = jnp.asarray(i, jnp.int32)
        z = dict(s)
        legal = self.legal_row(z, i)
        action = jnp.where((a >= 0) & (a < 17) & legal[jnp.clip(a, 0, 16)], a, 0)
        p = z["pos"][i]
        q, j, near, l, on, end, d = self.context(z, i)
        tile = z["tiles"][q[1], q[0]]
        move = (action >= 1) & (action <= 4)
        direction = jnp.clip(action - 1, 0, 3)
        turn = (action >= 5) & (action <= 8)
        face = jnp.where(move, direction, jnp.where(turn, action - 5, z["face"][i]))
        z["face"] = z["face"].at[i].set(face)
        z["pos"] = z["pos"].at[i].set(jnp.where(move, p + self.directions()[direction], p))
        mine = action == 9
        build = action == 10
        ore_gain = mine & (tile == 2)
        rock_gain = mine & (tile == 3)
        z["ore"] = z["ore"].at[i].add(ore_gain.astype(jnp.int32))
        z["stone"] = z["stone"].at[i].add(rock_gain.astype(jnp.int32) - build.astype(jnp.int32))
        z["tiles"] = z["tiles"].at[q[1], q[0]].set(jnp.where(mine, jnp.where(tile == 2, 1, 0), jnp.where(build, 4, tile)))
        hit_support = jnp.all(z["support_pos"] == q, axis=1) & rock_gain
        hit_bridge = jnp.all(z["bridge_pos"] == q, axis=1) & build
        btiles = z["tiles"][z["bridge_pos"][:, 1], z["bridge_pos"][:, 0]]
        z["tiles"] = z["tiles"].at[z["bridge_pos"][:, 1], z["bridge_pos"][:, 0]].set(jnp.where(hit_support, 0, btiles))
        z["strength"] = jnp.where(hit_support, 0, jnp.where(hit_bridge, 3, z["strength"]))
        victim = jnp.any(jnp.all(z["pos"][:, None, :] == z["bridge_pos"][None, :, :], axis=2) & hit_support[None, :], axis=1) & z["active"]
        rescue = jnp.all(z["pos"] == q, axis=1) & z["active"] & build & (s["tiles"][q[1], q[0]] == 0)
        z["health"] = jnp.maximum(0, z["health"] - victim.astype(jnp.int32))
        z["fell"] = z["fell"] | victim
        z["rescued"] = z["rescued"] | rescue
        unload = action == 11
        z["ore"] = z["ore"].at[i].add(-unload.astype(jnp.int32))
        z["delivered"] = z["delivered"].at[d].add(unload.astype(jnp.int32))
        give_ore = (action == 12) & (z["ore"][i] > 0) & (z["ore"][j] < 1)
        give_rock = (action == 12) & ~give_ore & (z["stone"][i] > 0) & (z["stone"][j] < 4)
        z["ore"] = z["ore"].at[i].add(-give_ore.astype(jnp.int32)).at[j].add(give_ore.astype(jnp.int32))
        z["stone"] = z["stone"].at[i].add(-give_rock.astype(jnp.int32)).at[j].add(give_rock.astype(jnp.int32))
        ride = action == 13
        load = 1 + ((z["stone"][i] > 0) | (z["ore"][i] > 0)).astype(jnp.int32)
        pair = jnp.where(l % 2 == 0, l + 1, l - 1)
        z["pos"] = z["pos"].at[i].set(jnp.where(ride, end, z["pos"][i]))
        z["weight"] = z["weight"].at[l].add(-ride.astype(jnp.int32) * load + (action == 14).astype(jnp.int32))
        z["weight"] = z["weight"].at[pair].set(jnp.minimum(4, z["weight"][pair] + ride.astype(jnp.int32) * load))
        z["stone"] = z["stone"].at[i].add(-(action == 14).astype(jnp.int32))
        old_i = z["tool"][i]
        old_j = z["tool"][j]
        z["tool"] = z["tool"].at[i].set(jnp.where(action == 15, old_j, old_i))
        z["tool"] = z["tool"].at[j].set(jnp.where(action == 15, old_i, old_j))
        z["message"] = z["message"].at[i].set(jnp.where(action == 16, 1 + (s["stone"][i] > 0).astype(jnp.int32) + (s["ore"][i] > 0).astype(jnp.int32), z["message"][i]))
        z["mined"] = z["mined"].at[i].set(mine)
        z["built"] = z["built"].at[i].set(build)
        z["unloaded"] = z["unloaded"].at[i].set(unload)
        z["rode"] = z["rode"].at[i].set(ride)
        rp = z["reward_parts"]
        rp = rp.at[i, 0].add(-0.01 * move * (1 + s["ore"][i]) - 0.02 * (mine | build | (action == 14)))
        rp = rp.at[i, 1].add(0.4 * unload)
        rp = rp.at[:, 2].add(0.125 * unload * (z["crew"] == d))
        rp = rp.at[:, 3].add(0.2 * rescue).at[i, 3].add(0.4 * jnp.sum(rescue))
        rp = rp.at[:, 4].add(-0.5 * victim).at[i, 4].add(-0.2 * jnp.sum(victim))
        z["reward_parts"] = rp
        z["active"] = z["health"] > 0
        return z

    def advance(self, s, actions):
        z = dict(s)
        for name in ("mined", "built", "rescued", "fell", "unloaded", "rode"):
            z[name] = jnp.zeros((160,), jnp.bool_)
        z["message"] = jnp.zeros((160,), jnp.int32)
        z["reward_parts"] = jnp.zeros((160, 5), jnp.float32)
        z = jax.lax.fori_loop(0, 160, lambda k, w: self.act((k + s["t"]) % 160, w, actions[(k + s["t"]) % 160]), z)
        weather = ((s["t"] + 1) % 32) == 0
        supported = z["tiles"][z["support_pos"][:, 1], z["support_pos"][:, 0]] == 3
        old = z["strength"]
        z["strength"] = jnp.maximum(0, old - (weather & ~supported).astype(jnp.int32))
        collapse = (old > 0) & (z["strength"] == 0)
        bt = z["tiles"][z["bridge_pos"][:, 1], z["bridge_pos"][:, 0]]
        z["tiles"] = z["tiles"].at[z["bridge_pos"][:, 1], z["bridge_pos"][:, 0]].set(jnp.where(collapse, 0, bt))
        victims = jnp.any(jnp.all(z["pos"][:, None, :] == z["bridge_pos"][None, :, :], axis=2) & collapse[None, :], axis=1) & z["active"]
        z["health"] = jnp.maximum(0, z["health"] - victims.astype(jnp.int32))
        z["fell"] = z["fell"] | victims
        z["reward_parts"] = z["reward_parts"].at[:, 4].add(-0.5 * victims)
        hearing = jnp.max(jnp.abs(z["pos"][:, None, :] - z["pos"][None, :, :]), axis=2) <= 6
        z["heard"] = jnp.max(jnp.where(hearing, z["message"][None, :], 0), axis=1)
        z["t"] = s["t"] + jnp.asarray(1, jnp.int32)
        z["terminated"] = (~jnp.any(z["tiles"] == 2) & (jnp.sum(z["ore"]) == 0)) | ~jnp.any(z["health"] > 0)
        z["truncated"] = (z["t"] >= 768) & ~z["terminated"]
        z["done"] = z["terminated"] | z["truncated"]
        return self.refresh(z)

    def raw_step(self, key, state, actions):
        ns = jax.lax.cond(state["done"], lambda _: state, lambda _: self.advance(state, actions), operand=None)
        reward = jnp.where(state["done"], jnp.zeros((160,), jnp.float32), jnp.sum(ns["reward_parts"], axis=1))
        dones = (~ns["active"]) | ns["done"]
        info = {"done_all": ns["done"], "terminated": ns["terminated"], "truncated": ns["truncated"]}
        return self.observe(ns), ns, reward, dones, info

