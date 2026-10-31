"""Tidal Canal Federation: original, unexecuted native spatial game source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 96
    observation_size = 184
    num_actions = 19
    max_steps = 768

    def canal(self):
        x = jnp.arange(111, dtype=jnp.int32)
        px = jnp.where(x <= 30, 5 + x, jnp.where(x <= 40, 35,
             jnp.where(x <= 70, 75 - x, jnp.where(x <= 80, 5, x - 75))))
        py = jnp.where(x <= 30, 5, jnp.where(x <= 40, x - 25,
             jnp.where(x <= 70, 15, jnp.where(x <= 80, x - 55, 25))))
        return jnp.stack((px, py), axis=1)

    def sight(self, pos, target, active, count):
        distance = jnp.sum(jnp.abs(pos[:, None, :] - target[None, :, :]), axis=2)
        ids = jnp.arange(target.shape[0], dtype=jnp.int32)
        score = jnp.where(active[None, :] & (distance <= 5),
                          distance * (target.shape[0] + 1) + ids[None, :], 100000 + ids[None, :])
        chosen = jnp.argsort(score, axis=1)[:, :count]
        valid = jnp.take_along_axis(score, chosen, axis=1) < 100000
        return jnp.zeros(distance.shape, dtype=jnp.bool_).at[jnp.arange(96)[:, None], chosen].set(valid)

    def refresh(self, state):
        s = dict(state)
        s["boat_pos"] = s["path"][s["boat_index"]]
        s["agent_visible"] = self.sight(s["pos"], s["pos"], s["active"], 8)
        s["gate_visible"] = self.sight(s["pos"], s["gate_pos"], s["district_active"], 3)
        s["basin_visible"] = self.sight(s["pos"], s["node_pos"], s["district_active"], 3)
        s["field_visible"] = self.sight(s["pos"], s["field_pos"], s["district_active"], 3)
        s["store_visible"] = self.sight(s["pos"], s["store_pos"], s["district_active"], 3)
        s["boat_visible"] = self.sight(s["pos"], s["boat_pos"], s["district_active"], 3)
        return s

    def reset(self, key):
        d = jnp.arange(12, dtype=jnp.int32)
        ids = jnp.arange(96, dtype=jnp.int32)
        path = self.canal()
        nodes = path[10 * d]
        tangent = jnp.array([[1, 0], [1, 0], [1, 0], [0, 1], [-1, 0], [-1, 0],
                             [-1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]], dtype=jnp.int32)
        normal = jnp.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
        gates = path[jnp.minimum(10 * d + 5, 110)]
        stores = nodes - 2 * tangent - 2 * normal
        fields = nodes + 2 * tangent + 2 * normal
        positions = jnp.stack((gates - normal, gates + normal, nodes - normal, fields,
                              nodes - 2 * normal, stores, nodes + tangent - normal,
                              fields + tangent), axis=1)
        # The west-to-south bend has water on its east side: use its dry north bank.
        positions = positions.at[7, 2].set(jnp.array([6, 14], dtype=jnp.int32))
        positions = positions.at[7, 4].set(jnp.array([7, 14], dtype=jnp.int32))
        positions = positions.at[7, 6].set(jnp.array([6, 16], dtype=jnp.int32))
        positions = positions.at[8, 2].set(jnp.array([4, 25], dtype=jnp.int32))
        positions = positions.at[8, 4].set(jnp.array([4, 23], dtype=jnp.int32))
        # Separate the outlet keeper from the last basin hauler.
        positions = positions.at[11, 2].set(jnp.array([34, 24], dtype=jnp.int32))
        tile = jnp.zeros((31, 41), dtype=jnp.int32)
        tile = tile.at[path[:, 1], path[:, 0]].set(1)
        tile = tile.at[stores[:, 1], stores[:, 0]].set(2)
        tile = tile.at[fields[:, 1], fields[:, 0]].set(3)
        tile = tile.at[gates[:, 1], gates[:, 0]].set(4)
        s = {"t": jnp.asarray(0, dtype=jnp.int32), "done": jnp.asarray(False),
             "terminated": jnp.asarray(False), "truncated": jnp.asarray(False),
             "map": tile, "path": path, "pos": positions.reshape((96, 2)),
             "active": jnp.ones((96,), dtype=jnp.bool_), "crew": ids // 8,
             "profession": ids % 4, "tool": jnp.array([1, 1, 3, 2], dtype=jnp.int32)[ids % 4],
             "carry": jnp.where(ids % 4 == 1, 1, 0).astype(jnp.int32),
             "node_pos": nodes, "gate_pos": gates, "store_pos": stores, "field_pos": fields,
             "district_active": jnp.ones((12,), dtype=jnp.bool_),
             "water": jnp.where(d == 0, 6, 1).astype(jnp.int32),
             "transit": jnp.zeros((12, 3), dtype=jnp.int32),
             "gate_open": jnp.zeros((12,), dtype=jnp.bool_),
             "bridge": jnp.where(d % 3 == 0, 0, 2).astype(jnp.int32),
             "soil": jnp.zeros((12,), dtype=jnp.int32), "growth": jnp.full((12,), 2, dtype=jnp.int32),
             "wood": jnp.full((12,), 8, dtype=jnp.int32), "food": jnp.zeros((12,), dtype=jnp.int32),
             "levee": jnp.zeros((12,), dtype=jnp.int32), "orders": jnp.full((12,), 6, dtype=jnp.int32),
             "score": jnp.zeros((12,), dtype=jnp.int32), "report": jnp.zeros((12,), dtype=jnp.int32),
             "report_age": jnp.full((12,), 768, dtype=jnp.int32),
             "flood": jnp.zeros((12,), dtype=jnp.bool_), "boat_index": 10 * d,
             "boat_owner": d, "boat_load": 1 + d % 2,
             "boat_dest": jnp.where(d < 11, d + 1, 10).astype(jnp.int32),
             "ev_gate": jnp.zeros((12,), dtype=jnp.bool_), "ev_arrival": jnp.zeros((12,), dtype=jnp.int32),
             "ev_tow": jnp.zeros((12,), dtype=jnp.bool_), "ev_delivery": jnp.zeros((12,), dtype=jnp.bool_),
             "ev_signal": jnp.zeros((12,), dtype=jnp.bool_), "ev_flood": jnp.zeros((12,), dtype=jnp.bool_),
             "ev_harvest": jnp.zeros((96,), dtype=jnp.bool_), "ev_build": jnp.zeros((96,), dtype=jnp.bool_),
             "ev_trade": jnp.zeros((96,), dtype=jnp.bool_), "ev_move": jnp.zeros((96,), dtype=jnp.bool_),
             "reward_components": jnp.zeros((96, 6), dtype=jnp.float32)}
        s = self.refresh(s)
        return self.observe(s), s

    def block(self, s, position, visible, count, values):
        dist = jnp.sum(jnp.abs(s["pos"][:, None, :] - position[None, :, :]), axis=2)
        ids = jnp.arange(position.shape[0], dtype=jnp.int32)
        order = jnp.argsort(jnp.where(visible, dist * (position.shape[0] + 1) + ids,
                                     100000 + ids), axis=1)[:, :count]
        seen = jnp.take_along_axis(visible, order, axis=1)
        rows = jnp.concatenate((seen[:, :, None], values[order] * seen[:, :, None]), axis=2)
        return rows.reshape((96, -1)).astype(jnp.float32)

    def observe(self, s):
        ids = jnp.arange(96, dtype=jnp.int32)
        d = jnp.arange(12, dtype=jnp.int32)
        forecast = (s["profession"] == 0).astype(jnp.float32)
        own = jnp.stack((jnp.full((96,), s["t"] / 768.0), s["pos"][:, 0] / 40.0,
                         s["pos"][:, 1] / 30.0, s["crew"] / 11.0, s["profession"] / 3.0,
                         s["tool"] / 3.0, s["carry"] / 2.0, forecast,
                         forecast * jnp.sum(s["transit"][s["crew"]], axis=1) / 3.0), axis=1)
        agent = jnp.stack((ids / 95.0, s["pos"][:, 0] / 40.0, s["pos"][:, 1] / 30.0,
                           s["crew"] / 11.0, s["profession"] / 3.0, s["tool"] / 3.0, s["carry"] / 2.0), axis=1)
        gate = jnp.stack((d / 11.0, s["gate_pos"][:, 0] / 40.0, s["gate_pos"][:, 1] / 30.0,
                          s["gate_open"], s["bridge"] / 3.0), axis=1)
        basin = jnp.stack((d / 11.0, s["node_pos"][:, 0] / 40.0, s["node_pos"][:, 1] / 30.0,
                           s["water"] / 8.0, s["flood"]), axis=1)
        field = jnp.stack((d / 11.0, s["field_pos"][:, 0] / 40.0, s["field_pos"][:, 1] / 30.0,
                           s["soil"] / 3.0, s["growth"] / 6.0), axis=1)
        store = jnp.stack((d / 11.0, s["store_pos"][:, 0] / 40.0, s["store_pos"][:, 1] / 30.0,
                           s["wood"] / 32.0, s["food"] / 32.0, s["levee"] / 2.0,
                           s["orders"] / 6.0, s["score"] / 32.0,
                           s["report"] / 8.0, s["report_age"] / 768.0), axis=1)
        boat = jnp.stack((d / 11.0, s["boat_pos"][:, 0] / 40.0, s["boat_pos"][:, 1] / 30.0,
                          s["boat_load"] / 2.0, s["boat_owner"] / 11.0, s["boat_dest"] / 11.0,
                          s["boat_index"] / 110.0), axis=1)
        return jnp.concatenate((own, self.block(s, s["pos"], s["agent_visible"], 8, agent),
            self.block(s, s["gate_pos"], s["gate_visible"], 3, gate),
            self.block(s, s["node_pos"], s["basin_visible"], 3, basin),
            self.block(s, s["field_pos"], s["field_visible"], 3, field),
            self.block(s, s["store_pos"], s["store_visible"], 3, store),
            self.block(s, s["boat_pos"], s["boat_visible"], 3, boat)), axis=1).astype(jnp.float32)

    def nearest(self, positions, objects):
        distance = jnp.sum(jnp.abs(positions[:, None, :] - objects[None, :, :]), axis=2)
        return jnp.argmin(distance, axis=1), jnp.min(distance, axis=1)

    def walkable(self, s, positions):
        x = jnp.clip(positions[..., 0], 0, 40)
        y = jnp.clip(positions[..., 1], 0, 30)
        bridge = jnp.any(jnp.all(positions[..., None, :] == s["gate_pos"], axis=-1)
                         & (s["bridge"] > 0), axis=-1)
        tile = s["map"][y, x]
        return (positions[..., 0] == x) & (positions[..., 1] == y) & (((tile != 1) & (tile != 4)) | bridge)

    def boat_allowed(self, s, direction):
        idx = s["boat_index"]
        target = jnp.clip(idx + direction, 0, 110)
        here = jnp.minimum((idx + 5) // 10, 11)
        there = jnp.minimum((target + 5) // 10, 11)
        gate = jnp.minimum(here, there)
        passage = (here == there) | s["gate_open"][gate]
        occupied = jnp.sum((target[:, None] == idx[None, :]) & ~jnp.eye(12, dtype=jnp.bool_), axis=1)
        capacity = jnp.where(target % 10 == 0, 2, 1)
        return (target != idx) & passage & (s["water"][here] >= 2) & (s["water"][there] >= 2) & (occupied < capacity)

    def legal_actions(self, s):
        moves = s["pos"][:, None, :] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        occupied = jnp.any(jnp.all(moves[:, :, None, :] == s["pos"][None, None, :, :], axis=3), axis=2)
        moving = self.walkable(s, moves) & ~occupied
        g, gd = self.nearest(s["pos"], s["gate_pos"])
        f, fd = self.nearest(s["pos"], s["field_pos"])
        h, hd = self.nearest(s["pos"], s["store_pos"])
        b, bd = self.nearest(s["pos"], s["boat_pos"])
        adjacent = (jnp.sum(jnp.abs(s["pos"][:, None, :] - s["pos"][None, :, :]), axis=2) == 1)
        empty_neighbor = jnp.any(adjacent & (s["carry"][None, :] == 0), axis=1)
        wrench = s["tool"] == 1
        extras = jnp.stack(((hd <= 1) & (s["carry"] == 0) & (s["wood"][h] > 0),
            (hd <= 1) & (s["carry"] > 0) & jnp.where(s["carry"] == 1, s["wood"][h] < 32, s["food"][h] < 32),
            jnp.any(adjacent, axis=1), (s["carry"] > 0) & empty_neighbor,
            (gd <= 1) & wrench & ~s["gate_open"][g], (gd <= 1) & wrench & s["gate_open"][g],
            (gd <= 1) & wrench & (s["carry"] == 1) & (s["bridge"][g] < 3),
            (fd <= 1) & (s["tool"] == 2) & (s["soil"][f] < 3) & (s["water"][f] > 0),
            (fd <= 1) & (s["carry"] == 0) & (s["growth"][f] == 6),
            (bd <= 1) & (s["tool"] == 3) & self.boat_allowed(s, 1)[b],
            (bd <= 1) & (s["tool"] == 3) & self.boat_allowed(s, -1)[b],
            (bd <= 1) & (s["carry"] > 0) & (s["boat_load"][b] == 0),
            (gd <= 1) & wrench,
            (hd <= 1) & wrench & (s["carry"] == 1) & (s["wood"][h] >= 2) & (s["levee"][h] < 2)), axis=1)
        live = jnp.concatenate((jnp.ones((96, 1), dtype=jnp.bool_), moving, extras), axis=1)
        return jnp.where(s["done"], jnp.arange(19)[None, :] == 0, live)

    def actor_turn(self, k, carry):
        s, actions = carry
        n = dict(s)
        i = (s["t"] - 1 + k) % 96
        a = actions[i]
        p = s["pos"][i]
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)[jnp.minimum(a, 4)]
        target = p + delta
        free = ~jnp.any(jnp.all(s["pos"] == target, axis=1))
        moved = (a >= 1) & (a <= 4) & self.walkable(s, target) & free
        n["pos"] = s["pos"].at[i].set(jnp.where(moved, target, p))
        n["ev_move"] = s["ev_move"].at[i].set(moved)
        gd = jnp.sum(jnp.abs(s["gate_pos"] - p), axis=1)
        fd = jnp.sum(jnp.abs(s["field_pos"] - p), axis=1)
        hd = jnp.sum(jnp.abs(s["store_pos"] - p), axis=1)
        bd = jnp.sum(jnp.abs(s["boat_pos"] - p), axis=1)
        g, f, h, b = jnp.argmin(gd), jnp.argmin(fd), jnp.argmin(hd), jnp.argmin(bd)
        item = s["carry"][i]
        wrench = s["tool"][i] == 1
        take = (a == 5) & (hd[h] <= 1) & (item == 0) & (s["wood"][h] > 0)
        deposit = (a == 6) & (hd[h] <= 1) & (item > 0) & jnp.where(item == 1, s["wood"][h] < 32, s["food"][h] < 32)
        build = (a == 11) & (gd[g] <= 1) & wrench & (item == 1) & (s["bridge"][g] < 3)
        upgrade = (a == 18) & (hd[h] <= 1) & wrench & (item == 1) & (s["wood"][h] >= 2) & (s["levee"][h] < 2)
        n["wood"] = s["wood"].at[h].add(-take.astype(jnp.int32) + (deposit & (item == 1)).astype(jnp.int32) - 2 * upgrade.astype(jnp.int32))
        n["food"] = s["food"].at[h].add((deposit & (item == 2)).astype(jnp.int32))
        n["score"] = s["score"].at[h].add((deposit & (item == 2)).astype(jnp.int32))
        n["levee"] = s["levee"].at[h].add(upgrade.astype(jnp.int32))
        n["bridge"] = s["bridge"].at[g].add(build.astype(jnp.int32))
        opened = (a == 9) & (gd[g] <= 1) & wrench & ~s["gate_open"][g]
        closed = (a == 10) & (gd[g] <= 1) & wrench & s["gate_open"][g]
        n["gate_open"] = s["gate_open"].at[g].set((s["gate_open"][g] | opened) & ~closed)
        n["ev_gate"] = s["ev_gate"].at[g].set(s["ev_gate"][g] | opened | closed)
        irrigate = (a == 12) & (fd[f] <= 1) & (s["tool"][i] == 2) & (s["soil"][f] < 3) & (s["water"][f] > 0)
        harvest = (a == 13) & (fd[f] <= 1) & (item == 0) & (s["growth"][f] == 6)
        n["water"] = s["water"].at[f].add(-irrigate.astype(jnp.int32))
        n["soil"] = s["soil"].at[f].set(jnp.minimum(3, s["soil"][f] + 2 * irrigate.astype(jnp.int32)))
        n["growth"] = s["growth"].at[f].set(jnp.where(harvest, 0, s["growth"][f]))
        load = (a == 16) & (bd[b] <= 1) & (item > 0) & (s["boat_load"][b] == 0)
        n["boat_load"] = s["boat_load"].at[b].set(jnp.where(load, item, s["boat_load"][b]))
        n["carry"] = s["carry"].at[i].set(jnp.where(take, 1, jnp.where(harvest, 2,
                             jnp.where(deposit | build | upgrade | load, 0, item))))
        neighbor = (jnp.sum(jnp.abs(s["pos"] - p), axis=1) == 1) & (actions == 0) & ~s["ev_trade"]
        swap_candidates = neighbor & ~s["ev_trade"][i]
        give_candidates = neighbor & (s["carry"] == 0) & ~s["ev_trade"][i]
        candidate = jnp.where(a == 7, swap_candidates, give_candidates)
        j = jnp.argmin(jnp.where(candidate, jnp.arange(96), 1000))
        swap = (a == 7) & jnp.any(swap_candidates)
        give = (a == 8) & (item > 0) & jnp.any(give_candidates)
        n["tool"] = s["tool"].at[i].set(jnp.where(swap, s["tool"][j], s["tool"][i]))
        n["tool"] = n["tool"].at[j].set(jnp.where(swap, s["tool"][i], n["tool"][j]))
        n["carry"] = n["carry"].at[i].set(jnp.where(give, 0, n["carry"][i]))
        n["carry"] = n["carry"].at[j].set(jnp.where(give, item, n["carry"][j]))
        n["ev_trade"] = s["ev_trade"].at[i].set(s["ev_trade"][i] | swap | give)
        n["ev_trade"] = n["ev_trade"].at[j].set(n["ev_trade"][j] | swap | give)
        sign = jnp.where(a == 14, 1, -1)
        tow = ((a == 14) | (a == 15)) & (bd[b] <= 1) & (s["tool"][i] == 3) & ~s["ev_tow"][b] & self.boat_allowed(s, sign)[b]
        n["boat_index"] = s["boat_index"].at[b].add(tow.astype(jnp.int32) * sign)
        n["boat_pos"] = s["path"][n["boat_index"]]
        n["ev_tow"] = s["ev_tow"].at[b].set(s["ev_tow"][b] | tow)
        dest = s["boat_dest"][b]
        cargo = n["boat_load"][b]
        room = jnp.where(cargo == 1, n["wood"][dest] < 32, n["food"][dest] < 32)
        delivered = tow & (n["boat_index"][b] == 10 * dest) & (cargo > 0) & room
        commission = delivered & (n["orders"][dest] > 0) & (cargo == 1 + (dest + 1) % 2)
        n["boat_load"] = n["boat_load"].at[b].set(jnp.where(delivered, 0, cargo))
        n["wood"] = n["wood"].at[dest].add((delivered & (cargo == 1)).astype(jnp.int32))
        n["food"] = n["food"].at[dest].add((delivered & (cargo == 2)).astype(jnp.int32))
        n["orders"] = s["orders"].at[dest].add(-commission.astype(jnp.int32))
        n["ev_delivery"] = s["ev_delivery"].at[b].set(s["ev_delivery"][b] | delivered)
        signal = (a == 17) & (gd[g] <= 1) & wrench
        recipient = jnp.minimum(g + 1, 11)
        n["report"] = s["report"].at[recipient].set(jnp.where(signal, s["water"][g], s["report"][recipient]))
        n["report_age"] = s["report_age"].at[recipient].set(jnp.where(signal, 0, s["report_age"][recipient]))
        n["ev_signal"] = s["ev_signal"].at[recipient].set(s["ev_signal"][recipient] | signal)
        n["ev_build"] = s["ev_build"].at[i].set(build | upgrade)
        n["ev_harvest"] = s["ev_harvest"].at[i].set(harvest)
        r = s["reward_components"].at[i, 0].add(-0.02 * (build.astype(jnp.float32) + upgrade + irrigate))
        r = r.at[i, 1].add(0.1 * harvest)
        r = r.at[:, 2].add(0.05 * (deposit & (item == 2)) * (s["crew"] == h))
        r = r.at[:, 3].add(0.01 * build * (jnp.sum(jnp.abs(s["pos"] - s["gate_pos"][g]), axis=1) <= 5)
                            + 0.03 * upgrade * (s["crew"] == h))
        r = r.at[:, 4].add(commission * (0.1 * (s["crew"] == dest) + 0.05 * (s["crew"] == s["boat_owner"][b])))
        n["reward_components"] = r
        return n, actions

    def advance(self, payload):
        state, actions = payload
        s = dict(state)
        valid = (actions >= 0) & (actions < 19)
        clipped = jnp.clip(actions, 0, 18)
        chosen = jnp.take_along_axis(self.legal_actions(state), clipped[:, None], axis=1)[:, 0]
        actions = jnp.where(valid & chosen, clipped, 0).astype(jnp.int32)
        s["t"] = state["t"] + jnp.asarray(1, dtype=jnp.int32)
        s["report_age"] = jnp.minimum(768, state["report_age"] + 1)
        for name in ("ev_gate", "ev_arrival", "ev_tow", "ev_delivery", "ev_signal", "ev_flood",
                     "ev_harvest", "ev_build", "ev_trade", "ev_move", "reward_components"):
            s[name] = jnp.zeros_like(state[name])
        s, actions = jax.lax.fori_loop(0, 96, self.actor_turn, (s, actions))
        arrivals = s["transit"][:, 0]
        water = s["water"] + arrivals
        water = water.at[0].add(2 * (s["t"] % 12 == 0).astype(jnp.int32))
        release = (s["gate_open"] & (water > 0)).astype(jnp.int32)
        packet = jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), release[:11]))
        s["transit"] = jnp.concatenate((s["transit"][:, 1:], packet[:, None]), axis=1)
        s["water"] = jnp.clip(water - release - (s["t"] % 16 == 0).astype(jnp.int32), 0, 8)
        s["ev_arrival"] = arrivals
        s["flood"] = s["water"] >= (7 + s["levee"])
        damage = s["flood"] & (s["t"] % 8 == 0)
        s["ev_flood"] = damage
        growth = (s["soil"] > 0) & ~s["flood"] & (s["t"] % 4 == 0)
        s["growth"] = jnp.clip(s["growth"] + growth.astype(jnp.int32) - damage.astype(jnp.int32), 0, 6)
        s["soil"] = jnp.maximum(0, s["soil"] - (s["t"] % 8 == 0).astype(jnp.int32))
        s["bridge"] = jnp.maximum(0, s["bridge"] - damage.astype(jnp.int32))
        s["reward_components"] = s["reward_components"].at[:, 5].set(
            -0.04 * damage[s["crew"]] - 0.01 * jnp.sum(damage) / 12.0 + 0.002 * jnp.sum(s["ev_delivery"]))
        s["terminated"] = jnp.asarray(False)
        s["truncated"] = s["t"] >= 768
        s["done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        return s, jnp.sum(s["reward_components"], axis=1).astype(jnp.float32)

    def absorb(self, payload):
        s, actions = payload
        return s, jnp.zeros((96,), dtype=jnp.float32)

    def raw_step(self, key, state, actions):
        s, reward = jax.lax.cond(state["done"], self.absorb, self.advance, (state, actions))
        done = jnp.full((96,), s["done"], dtype=jnp.bool_)
        return self.observe(s), s, reward, done, {"done_all": s["done"],
                "terminated": s["terminated"], "truncated": s["truncated"]}
