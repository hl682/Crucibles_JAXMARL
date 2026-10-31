"""Festival Steam Kitchens. Original authored rules; not executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 24
    observation_size = 160
    num_actions = 23
    max_steps = 512

    def directions(self):
        return jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)

    def context(self, s, i):
        p = s["pos"][i]
        q = jnp.clip(p + self.directions()[s["face"][i]], jnp.array([0, 0], jnp.int32), jnp.array([29, 17], jnp.int32))
        k = jnp.argmin(jnp.sum(jnp.abs(s["station_pos"] - q), axis=1))
        at = jnp.all(s["station_pos"][k] == q)
        team = k // 7
        kind = k % 7
        j = jnp.argmin(jnp.sum(jnp.abs(s["pos"] - q), axis=1))
        near = (j != i) & jnp.all(s["pos"][j] == q)
        c = jnp.argmin(jnp.where(s["cart_active"], jnp.sum(jnp.abs(s["cart_pos"] - q), axis=1), 999))
        cart = s["cart_active"][c] & jnp.all(s["cart_pos"][c] == q)
        return q, team, kind, at, j, near, c, cart

    def free(self, s, xy):
        bounded = jnp.all((xy >= jnp.array([0, 0], jnp.int32)) & (xy <= jnp.array([29, 17], jnp.int32)))
        q = jnp.clip(xy, jnp.array([0, 0], jnp.int32), jnp.array([29, 17], jnp.int32))
        occupied = jnp.any(jnp.all(s["pos"] == q, axis=1))
        cart = jnp.any(jnp.all(s["cart_pos"] == q, axis=1) & s["cart_active"])
        return bounded & (s["tiles"][q[1], q[0]] == 0) & ~occupied & ~cart

    def legal_row(self, s, i):
        q, g, kind, at, j, near, c, cart = self.context(s, i)
        carry = s["carry"][i]
        tool = s["tool"][i]
        moves = jax.vmap(lambda xy: self.free(s, xy))(s["pos"][i][None, :] + self.directions())
        into = q + self.directions()[s["face"][i]]
        load = (((carry == 3) | (carry == 4)) & s["plate"][g]) | ((carry == 10) & ~s["plate"][g])
        mask = jnp.concatenate([jnp.ones((1,), jnp.bool_), moves, jnp.ones((4,), jnp.bool_), jnp.stack([
            at & (kind == 0) & (carry == 0) & (s["stock"][g, 0] > 0),
            at & (kind == 0) & (carry == 0) & (s["stock"][g, 1] > 0),
            at & (kind == 1) & (tool == 0) & ((carry == 1) | (carry == 2)),
            at & (kind == 2) & (s["recipe"][g] == 0) & load,
            at & (kind == 2) & (carry == 0) & ((tool == 1) | (tool == 2)) & (s["recipe"][g] > 0) & (s["progress"][g] == 4),
            at & (kind == 3) & ((carry == 5) | (carry == 6)) & (s["temperature"][i] == 0),
            at & (kind == 4) & ((carry == 7) | (carry == 8)) & ~s["cart_active"][g] & (s["served"][g] < 2),
            at & (kind == 5) & (tool == 3),
            near & (carry > 0) & (s["carry"][j] == 0),
            near,
            at & (kind == 6) & (carry == 9) & (s["water"][g] > 0),
            cart & (carry == 0),
            cart & self.free(s, into),
            jnp.asarray(True)
        ])])
        return jnp.where(~s["done"], mask, jnp.arange(23) == 0)

    def legal_actions(self, s):
        return jax.vmap(lambda i: self.legal_row(s, i))(jnp.arange(24))

    def refresh(self, s):
        z = dict(s)
        p = z["pos"]
        distances = jnp.max(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2)
        local = distances <= 3
        other = local & ~jnp.eye(24, dtype=jnp.bool_)
        nearest = jnp.argsort(jnp.where(other, distances, 999), axis=1)[:, :3]
        detail = jnp.sum(jax.nn.one_hot(nearest, 24, dtype=jnp.int32), axis=1) > 0
        z["agent_vis"] = local & (detail | jnp.eye(24, dtype=jnp.bool_))
        z["map_vis"] = (jnp.abs(jnp.arange(30)[None, None, :] - p[:, 0, None, None]) <= 3) & (jnp.abs(jnp.arange(18)[None, :, None] - p[:, 1, None, None]) <= 3)
        for name in ("station", "pot", "valve", "cart"):
            z[name + "_vis"] = (jnp.max(jnp.abs(p[:, None, :] - z[name + "_pos"][None, :, :]), axis=2) <= 3) & z[name + "_active"][None, :]
        z["menu_vis"] = jnp.sum(jnp.abs(p[:, None, :] - z["menu_pos"][None, :, :]), axis=2) <= 1
        return z

    def reset(self, key):
        g = jnp.arange(6, dtype=jnp.int32)
        base = jnp.stack([(g % 3) * 10, (g // 3) * 9], axis=1)
        stations = (base[:, None, :] + jnp.array([[1, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3]], jnp.int32)[None, :, :]).reshape((42, 2))
        tiles = jnp.zeros((18, 30), jnp.int32)
        tiles = tiles.at[stations[:, 1], stations[:, 0]].set(jnp.tile(jnp.arange(1, 8, dtype=jnp.int32), 6))
        pos = (base[:, None, :] + jnp.array([[2, 3], [4, 4], [6, 4], [7, 4]], jnp.int32)[None, :, :]).reshape((24, 2))
        s = {
            "t": jnp.asarray(0, jnp.int32), "done": jnp.asarray(False), "terminated": jnp.asarray(False), "truncated": jnp.asarray(False),
            "tiles": tiles, "map_vis": jnp.zeros((24, 18, 30), jnp.bool_),
            "pos": pos, "active": jnp.ones((24,), jnp.bool_), "agent_vis": jnp.zeros((24, 24), jnp.bool_),
            "team": jnp.arange(24, dtype=jnp.int32) // 4, "tool": jnp.tile(jnp.arange(4, dtype=jnp.int32), 6),
            "face": jnp.tile(jnp.array([3, 0, 0, 0], jnp.int32), 6),
            "carry": jnp.zeros((24,), jnp.int32), "temperature": jnp.zeros((24,), jnp.int32),
            "message": jnp.zeros((24,), jnp.int32), "heard": jnp.zeros((24,), jnp.int32),
            "station_pos": stations, "station_active": jnp.ones((42,), jnp.bool_), "station_vis": jnp.zeros((24, 42), jnp.bool_),
            "station_kind": jnp.tile(jnp.arange(7, dtype=jnp.int32), 6),
            "stock": jnp.full((6, 2), 8, jnp.int32), "water": jnp.full((6,), 4, jnp.int32),
            "pot_pos": base + jnp.array([4, 3], jnp.int32), "pot_active": jnp.ones((6,), jnp.bool_), "pot_vis": jnp.zeros((24, 6), jnp.bool_),
            "recipe": 1 + g % 2, "progress": jnp.zeros((6,), jnp.int32), "burn": jnp.zeros((6,), jnp.int32),
            "plate": jnp.ones((6,), jnp.bool_), "heat": jnp.zeros((6,), jnp.int32),
            "valve_pos": base + jnp.array([7, 3], jnp.int32), "valve_active": jnp.ones((6,), jnp.bool_), "valve_vis": jnp.zeros((24, 6), jnp.bool_),
            "valve": jnp.ones((6,), jnp.bool_), "steam": jnp.full((6, 2), 2, jnp.int32),
            "menu_pos": base + jnp.array([6, 3], jnp.int32), "menu_active": jnp.ones((6,), jnp.bool_), "menu_vis": jnp.zeros((24, 6), jnp.bool_),
            "wanted": 1 + g % 2, "deadline": jnp.full((6,), 80, jnp.int32), "served": jnp.zeros((6,), jnp.int32),
            "cart_pos": base + jnp.array([6, 4], jnp.int32), "cart_active": jnp.zeros((6,), jnp.bool_), "cart_vis": jnp.zeros((24, 6), jnp.bool_),
            "cart_kind": jnp.full((6,), 9, jnp.int32),
            "served_event": jnp.zeros((24,), jnp.bool_), "washed": jnp.zeros((24,), jnp.bool_),
            "handoff": jnp.zeros((24,), jnp.bool_), "toggled": jnp.zeros((24,), jnp.bool_),
            "spoiled": jnp.zeros((6,), jnp.bool_), "missed": jnp.zeros((6,), jnp.bool_),
            "reward_parts": jnp.zeros((24, 5), jnp.float32)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        p = s["pos"]
        ids = jnp.arange(24)
        own = jnp.stack([p[:, 0] / 29.0, p[:, 1] / 17.0, s["face"] / 3.0, s["tool"] / 3.0,
            s["carry"] / 10.0, s["temperature"] / 4.0, s["team"] / 5.0, s["heard"] / 3.0,
            jnp.full((24,), s["t"] / 512.0), s["served_event"] * 1.0, s["washed"] * 1.0,
            s["handoff"] * 1.0, s["toggled"] * 1.0, jnp.ones((24,))], axis=1)
        k = jnp.arange(49)
        q = p[:, None, :] + jnp.stack([k % 7 - 3, k // 7 - 3], axis=1)[None, :, :]
        inside = jnp.all((q >= jnp.array([0, 0], jnp.int32)) & (q <= jnp.array([29, 17], jnp.int32)), axis=2)
        c = jnp.clip(q, jnp.array([0, 0], jnp.int32), jnp.array([29, 17], jnp.int32))
        terrain = jnp.where(inside, s["tiles"][c[:, :, 1], c[:, :, 0]] / 7.0, -1.0)
        actor = jnp.any(jnp.all(q[:, :, None, :] == p[None, None, :, :], axis=3), axis=2)
        cart = jnp.any(jnp.all(q[:, :, None, :] == s["cart_pos"][None, None, :, :], axis=3) & s["cart_active"][None, None, :], axis=2)
        patch = jnp.stack([terrain, jnp.where(cart, 1.0, actor * 0.5)], axis=2).reshape((24, 98))
        dist = jnp.max(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2)
        nv = s["agent_vis"] & (ids[:, None] != ids[None, :])
        ni = jnp.argsort(jnp.where(nv, dist, 999), axis=1)[:, :3]
        valid = jnp.take_along_axis(nv, ni, axis=1)
        rel = (p[ni] - p[:, None, :]) / 3.0
        nb = jnp.stack([valid * 1.0, rel[:, :, 0], rel[:, :, 1], s["tool"][ni] / 3.0,
                        s["carry"][ni] / 10.0, s["temperature"][ni] / 4.0,
                        s["message"][ni] / 3.0, s["team"][ni] / 5.0], axis=2)
        nb = jnp.where(valid[:, :, None], nb, 0.0).reshape((24, 24))
        pi = jnp.argmin(jnp.sum(jnp.abs(p[:, None, :] - s["pot_pos"][None, :, :]), axis=2), axis=1)
        pv = s["pot_vis"][ids, pi]
        pr = (s["pot_pos"][pi] - p) / 3.0
        pot = jnp.stack([pv * 1.0, pr[:, 0], pr[:, 1], s["recipe"][pi] / 2.0, s["progress"][pi] / 4.0,
                        s["burn"][pi] / 3.0, s["plate"][pi] * 1.0, s["heat"][pi] / 2.0], axis=1)
        vi = jnp.argmin(jnp.sum(jnp.abs(p[:, None, :] - s["valve_pos"][None, :, :]), axis=2), axis=1)
        vv = s["valve_vis"][ids, vi]
        vr = (s["valve_pos"][vi] - p) / 3.0
        valve = jnp.stack([vv * 1.0, vr[:, 0], vr[:, 1], s["valve"][vi] * 1.0,
                          s["steam"][vi, 0] / 6.0, s["steam"][vi, 1] / 6.0,
                          s["water"][vi] / 8.0], axis=1)
        mi = jnp.argmin(jnp.sum(jnp.abs(p[:, None, :] - s["menu_pos"][None, :, :]), axis=2), axis=1)
        mv = s["menu_vis"][ids, mi]
        menu = jnp.stack([mv * 1.0, s["wanted"][mi] / 2.0, s["deadline"][mi] / 80.0,
                         s["served"][mi] / 2.0, mi / 5.0], axis=1)
        stock_visible = jnp.sum(jnp.abs(p - s["station_pos"][pi * 7]), axis=1) <= 3
        stock = jnp.where(stock_visible[:, None], s["stock"][pi] / 8.0, 0.0)
        return jnp.concatenate([own, patch, nb, jnp.where(pv[:, None], pot, 0.0),
             jnp.where(vv[:, None], valve, 0.0), jnp.where(mv[:, None], menu, 0.0), stock,
             stock_visible[:, None] * 1.0, jnp.zeros((24, 1))], axis=1).astype(jnp.float32)

    def act(self, i, s, a):
        i = jnp.asarray(i, jnp.int32)
        z = dict(s)
        mask = self.legal_row(z, i)
        action = jnp.where((a >= 0) & (a < 23) & mask[jnp.clip(a, 0, 22)], a, 0)
        q, g, kind, at, j, near, c, cart = self.context(z, i)
        old_carry = z["carry"][i]
        old_temp = z["temperature"][i]
        move = (action >= 1) & (action <= 4)
        direction = jnp.clip(action - 1, 0, 3)
        face = jnp.where(move, direction, jnp.where((action >= 5) & (action <= 8), action - 5, z["face"][i]))
        z["face"] = z["face"].at[i].set(face)
        z["pos"] = z["pos"].at[i].set(z["pos"][i] + jnp.where(move, self.directions()[direction], jnp.zeros((2,), jnp.int32)))
        root = action == 9
        bean = action == 10
        chop = action == 11
        load = action == 12
        take = action == 13
        cool = action == 14
        serve = action == 15
        toggle = action == 16
        give = action == 17
        swap = action == 18
        wash = action == 19
        pick = action == 20
        push = action == 21
        z["stock"] = z["stock"].at[g, 0].add(-root.astype(jnp.int32)).at[g, 1].add(-bean.astype(jnp.int32))
        carry = jnp.where(root, 1, jnp.where(bean, 2, jnp.where(chop, old_carry + 2, old_carry)))
        carry = jnp.where(load | serve | give, 0, jnp.where(take, z["recipe"][g] + 4,
                  jnp.where(cool, old_carry + 2, jnp.where(wash, 10, jnp.where(pick, z["cart_kind"][c], carry)))))
        z["carry"] = z["carry"].at[i].set(carry)
        z["carry"] = z["carry"].at[j].set(jnp.where(give, old_carry, z["carry"][j]))
        z["temperature"] = z["temperature"].at[i].set(jnp.where(take, 4, jnp.where(give | serve | load, 0, old_temp)))
        z["temperature"] = z["temperature"].at[j].set(jnp.where(give, old_temp, z["temperature"][j]))
        fill = load & (old_carry != 10)
        z["recipe"] = z["recipe"].at[g].set(jnp.where(fill, old_carry - 2, jnp.where(take, 0, z["recipe"][g])))
        z["progress"] = z["progress"].at[g].set(jnp.where(fill | take, 0, z["progress"][g]))
        z["burn"] = z["burn"].at[g].set(jnp.where(fill | take, 0, z["burn"][g]))
        z["plate"] = z["plate"].at[g].set(jnp.where(take, False, jnp.where(load & (old_carry == 10), True, z["plate"][g])))
        quality = 1 + ((old_carry - 6) == z["wanted"][g]).astype(jnp.int32)
        z["served"] = z["served"].at[g].add(serve.astype(jnp.int32))
        z["wanted"] = z["wanted"].at[g].set(jnp.where(serve, 3 - z["wanted"][g], z["wanted"][g]))
        z["deadline"] = z["deadline"].at[g].set(jnp.where(serve, 80, z["deadline"][g]))
        z["cart_pos"] = z["cart_pos"].at[g].set(jnp.where(serve, z["menu_pos"][g] + jnp.array([0, 1], jnp.int32), z["cart_pos"][g]))
        z["cart_active"] = z["cart_active"].at[g].set(z["cart_active"][g] | serve)
        z["cart_active"] = z["cart_active"].at[c].set(jnp.where(pick, False, z["cart_active"][c]))
        z["cart_pos"] = z["cart_pos"].at[c].set(jnp.where(push, q + self.directions()[s["face"][i]], z["cart_pos"][c]))
        z["valve"] = z["valve"].at[g].set(z["valve"][g] ^ toggle)
        z["water"] = z["water"].at[g].add(-wash.astype(jnp.int32))
        ti, tj = z["tool"][i], z["tool"][j]
        z["tool"] = z["tool"].at[i].set(jnp.where(swap, tj, ti)).at[j].set(jnp.where(swap, ti, tj))
        menu = jnp.argmin(jnp.sum(jnp.abs(z["menu_pos"] - z["pos"][i]), axis=1))
        readable = jnp.sum(jnp.abs(z["menu_pos"][menu] - z["pos"][i])) <= 1
        msg = jnp.where(readable, z["wanted"][menu], 3)
        z["message"] = z["message"].at[i].set(jnp.where(action == 22, msg, z["message"][i]))
        z["served_event"] = z["served_event"].at[i].set(serve)
        z["washed"] = z["washed"].at[i].set(wash)
        z["handoff"] = z["handoff"].at[i].set(give)
        z["toggled"] = z["toggled"].at[i].set(toggle)
        rp = z["reward_parts"]
        rp = rp.at[i, 0].add(-0.01 * (move | chop | wash | push))
        rp = rp.at[i, 1].add(0.4 * quality * serve)
        rp = rp.at[:, 2].add(0.25 * quality * serve * (z["team"] == g))
        z["reward_parts"] = rp
        return z

    def advance(self, s, actions):
        z = dict(s)
        for name in ("served_event", "washed", "handoff", "toggled"):
            z[name] = jnp.zeros((24,), jnp.bool_)
        z["message"] = jnp.zeros((24,), jnp.int32)
        z["reward_parts"] = jnp.zeros((24, 5), jnp.float32)
        z = jax.lax.fori_loop(0, 24, lambda k, w: self.act((k + s["t"]) % 24, w, actions[(k + s["t"]) % 24]), z)
        draw = jnp.where(z["valve"], jnp.minimum(2, z["steam"][:, 1]), 0)
        stage0 = jnp.roll(z["steam"][:, 1] - draw, 1)
        stage0 = stage0.at[0].set(jnp.minimum(6, stage0[0] + 2))
        z["steam"] = jnp.stack([stage0, z["steam"][:, 0]], axis=1)
        z["heat"] = draw
        cooking = z["recipe"] > 0
        ready = z["progress"] == 4
        z["progress"] = jnp.where(cooking, jnp.minimum(4, z["progress"] + draw), 0)
        z["burn"] = jnp.where(cooking & ready, jnp.minimum(3, z["burn"] + (draw > 0).astype(jnp.int32)), z["burn"])
        z["spoiled"] = cooking & (z["burn"] == 3)
        z["recipe"] = jnp.where(z["spoiled"], 0, z["recipe"])
        z["progress"] = jnp.where(z["spoiled"], 0, z["progress"])
        z["burn"] = jnp.where(z["spoiled"], 0, z["burn"])
        z["temperature"] = jnp.maximum(0, z["temperature"] - ((z["carry"] == 5) | (z["carry"] == 6)).astype(jnp.int32))
        deadline = jnp.maximum(0, z["deadline"] - (z["served"] < 2).astype(jnp.int32))
        z["missed"] = (deadline == 0) & (z["served"] < 2)
        z["deadline"] = jnp.where(z["missed"], 80, deadline)
        z["water"] = jnp.minimum(8, z["water"] + (((s["t"] + 1) % 8 == 0) & (draw > 0)).astype(jnp.int32))
        team = z["team"]
        z["reward_parts"] = z["reward_parts"].at[:, 3].add(-0.2 * z["missed"][team] - 0.075 * z["spoiled"][team])
        z["reward_parts"] = z["reward_parts"].at[:, 4].add(-0.005 * draw[team])
        near = jnp.sum(jnp.abs(z["pos"][:, None, :] - z["pos"][None, :, :]), axis=2) <= 6
        z["heard"] = jnp.max(jnp.where(near, z["message"][None, :], 0), axis=1)
        z["t"] = s["t"] + jnp.asarray(1, jnp.int32)
        z["terminated"] = jnp.all(z["served"] >= 2)
        z["truncated"] = (z["t"] >= 512) & ~z["terminated"]
        z["done"] = z["terminated"] | z["truncated"]
        return self.refresh(z)

    def raw_step(self, key, state, actions):
        ns = jax.lax.cond(state["done"], lambda _: state, lambda _: self.advance(state, actions), operand=None)
        rewards = jnp.where(state["done"], jnp.zeros((24,), jnp.float32), jnp.sum(ns["reward_parts"], axis=1))
        dones = jnp.full((24,), ns["done"], jnp.bool_)
        return self.observe(ns), ns, rewards, dones, {"done_all": ns["done"], "terminated": ns["terminated"], "truncated": ns["truncated"]}

