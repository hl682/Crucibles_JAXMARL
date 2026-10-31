"""Orchard Pollination Commons. Original authored rules; not executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 48
    observation_size = 315
    num_actions = 23
    max_steps = 768

    def directions(self):
        return jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)

    def context(self, s, i):
        p = s["pos"][i]
        q = jnp.clip(p + self.directions()[s["face"][i]], jnp.array([0, 0], jnp.int32), jnp.array([47, 23], jnp.int32))
        plant = jnp.argmin(jnp.sum(jnp.abs(s["plant_pos"] - q), axis=1))
        at_plant = jnp.all(s["plant_pos"][plant] == q)
        gate = jnp.argmin(jnp.sum(jnp.abs(s["gate_pos"] - q), axis=1))
        at_gate = jnp.all(s["gate_pos"][gate] == q)
        water = jnp.argmin(jnp.sum(jnp.abs(s["water_pos"] - p), axis=1))
        store = jnp.argmin(jnp.sum(jnp.abs(s["store_pos"] - p), axis=1))
        hive = jnp.argmin(jnp.where(s["hive_carrier"] < 0, jnp.sum(jnp.abs(s["hive_pos"] - q), axis=1), 999))
        at_hive = (s["hive_carrier"][hive] < 0) & jnp.all(s["hive_pos"][hive] == q)
        j = jnp.argmin(jnp.sum(jnp.abs(s["pos"] - q), axis=1))
        near = (j != i) & jnp.all(s["pos"][j] == q)
        return q, plant, at_plant, gate, at_gate, water, store, hive, at_hive, j, near

    def free(self, s, xy):
        inside = jnp.all((xy >= jnp.array([0, 0], jnp.int32)) & (xy <= jnp.array([47, 23], jnp.int32)))
        q = jnp.clip(xy, jnp.array([0, 0], jnp.int32), jnp.array([47, 23], jnp.int32))
        g = jnp.argmin(jnp.sum(jnp.abs(s["gate_pos"] - q), axis=1))
        tile = s["tiles"][q[1], q[0]]
        actor = jnp.any(jnp.all(s["pos"] == q, axis=1))
        hive = jnp.any(jnp.all(s["hive_pos"] == q, axis=1) & (s["hive_carrier"] < 0))
        return inside & (tile != 1) & ((tile != 3) | s["gate_open"][g]) & ~actor & ~hive

    def legal_row(self, s, i):
        q, p, atp, g, atg, w, d, h, ath, j, near = self.context(s, i)
        tool = s["tool"][i]
        held = s["held_hive"][i]
        hi = jnp.clip(held, 0, 11)
        pos = s["pos"][i]
        water_near = jnp.sum(jnp.abs(s["water_pos"][w] - pos)) <= 1
        store_near = jnp.sum(jnp.abs(s["store_pos"][d] - pos)) <= 1
        empty_drop = ~jnp.any(jnp.all(s["hive_pos"] == pos, axis=1) & (s["hive_carrier"] < 0))
        drop = (held >= 0) & (pos[1] == s["hive_lane"][hi]) & (s["tiles"][pos[1], pos[0]] == 0) & empty_drop
        pickup = (held < 0) & ath
        fence_empty = ~jnp.any(jnp.all(s["pos"] == q, axis=1)) & ~jnp.any(jnp.all(s["hive_pos"] == q, axis=1))
        give = near & (((s["fruit"][i] > 0) & (s["fruit"][j] < 3)) |
                      ((s["bucket"][i] > 0) & (s["bucket"][j] < 3)) |
                      ((s["compost"][i] > 0) & (s["compost"][j] < 3)))
        moves = jax.vmap(lambda xy: self.free(s, xy))(pos[None, :] + self.directions())
        mask = jnp.concatenate([jnp.ones((1,), jnp.bool_), moves, jnp.ones((4,), jnp.bool_), jnp.stack([
            atp & (tool == 0) & (s["seeds"][i] > 0) & ((s["stage"][p] == 0) | (s["stage"][p] == 4)),
            atp & (tool == 0) & (s["compost"][i] > 0) & ((s["stage"][p] == 1) | (s["stage"][p] == 2)) & (s["variety"][p] == 0),
            atp & (tool == 0) & ((s["disease"][p] > 0) | ((s["stage"][p] == 4) & (s["compost"][i] < 3))),
            atp & (tool == 3) & (s["stage"][p] == 3) & (s["crop"][p] > 0) & (s["fruit"][i] < 3),
            store_near & (s["fruit"][i] > 0) & (s["queue"][d] < 4),
            (tool == 1) & (((s["bucket"][i] == 0) & water_near & (s["canal"][w] > 0)) | ((s["bucket"][i] > 0) & atp & (s["wet"][p] < 8))),
            (tool == 1) & water_near,
            atg,
            give,
            (tool == 2) & (pickup | drop),
            near,
            jnp.asarray(True),
            (tool == 0) & (s["compost"][i] > 0) & (s["tiles"][q[1], q[0]] == 0) & fence_empty,
            (tool == 0) & (s["compost"][i] < 3) & (s["tiles"][q[1], q[0]] == 1)
        ])])
        return jnp.where(~s["done"], mask, jnp.arange(23) == 0)

    def legal_actions(self, s):
        return jax.vmap(lambda i: self.legal_row(s, i))(jnp.arange(48))

    def refresh(self, s):
        z = dict(s)
        p = z["pos"]
        carrier = jnp.clip(z["hive_carrier"], 0, 47)
        z["hive_pos"] = jnp.where((z["hive_carrier"] >= 0)[:, None], p[carrier], z["hive_pos"])
        distances = jnp.max(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2)
        local = distances <= 3
        other = local & ~jnp.eye(48, dtype=jnp.bool_)
        nearest = jnp.argsort(jnp.where(other, distances, 999), axis=1)[:, :4]
        detail = jnp.sum(jax.nn.one_hot(nearest, 48, dtype=jnp.int32), axis=1) > 0
        z["agent_vis"] = local & (detail | jnp.eye(48, dtype=jnp.bool_))
        z["map_vis"] = (jnp.abs(jnp.arange(48)[None, None, :] - p[:, 0, None, None]) <= 3) & (jnp.abs(jnp.arange(24)[None, :, None] - p[:, 1, None, None]) <= 3)
        z["hive_carried"] = z["hive_carrier"] >= 0
        z["bee_active"] = z["bee_health"] > 0
        for name in ("plant", "gate", "water", "store", "hive", "bee"):
            z[name + "_vis"] = (jnp.max(jnp.abs(p[:, None, :] - z[name + "_pos"][None, :, :]), axis=2) <= 3) & z[name + "_active"][None, :]
        for name, count, capacity in (("plant", 2, 24), ("hive", 1, 12), ("bee", 4, 12)):
            distance = jnp.sum(jnp.abs(p[:, None, :] - z[name + "_pos"][None, :, :]), axis=2)
            indices = jnp.argsort(jnp.where(z[name + "_vis"], distance, 999), axis=1)[:, :count]
            selected = jnp.sum(jax.nn.one_hot(indices, capacity, dtype=jnp.int32), axis=1) > 0
            z[name + "_vis"] = z[name + "_vis"] & selected
        return z

    def reset(self, key):
        g = jnp.arange(12, dtype=jnp.int32)
        base = jnp.stack([(g % 4) * 12, (g // 4) * 8], axis=1)
        pos = (base[:, None, :] + jnp.array([[5, 3], [7, 4], [2, 4], [8, 5]], jnp.int32)[None, :, :]).reshape((48, 2))
        plants = (base[:, None, :] + jnp.array([[5, 4], [8, 4]], jnp.int32)[None, :, :]).reshape((24, 2))
        gate, water, store, hive = (base + jnp.array(v, jnp.int32) for v in ([6, 4], [7, 3], [3, 6], [3, 4]))
        x = jnp.arange(48)[None, :]
        y = jnp.arange(24)[:, None]
        tiles = jnp.where((x % 12 == 6) & (y % 8 != 0), 1, 0).astype(jnp.int32)
        for pp, value in ((plants, 2), (gate, 3), (water, 4), (store, 5)):
            tiles = tiles.at[pp[:, 1], pp[:, 0]].set(value)
        s = {
            "t": jnp.asarray(0, jnp.int32), "done": jnp.asarray(False), "terminated": jnp.asarray(False), "truncated": jnp.asarray(False),
            "tiles": tiles, "map_vis": jnp.zeros((48, 24, 48), jnp.bool_),
            "pos": pos, "active": jnp.ones((48,), jnp.bool_), "agent_vis": jnp.zeros((48, 48), jnp.bool_),
            "team": jnp.arange(48, dtype=jnp.int32) // 4, "tool": jnp.tile(jnp.arange(4, dtype=jnp.int32), 12),
            "face": jnp.tile(jnp.array([2, 3, 1, 0], jnp.int32), 12),
            "fruit": jnp.zeros((48,), jnp.int32), "bucket": jnp.zeros((48,), jnp.int32),
            "compost": jnp.tile(jnp.array([1, 0, 0, 0], jnp.int32), 12), "seeds": jnp.tile(jnp.array([2, 0, 0, 0], jnp.int32), 12),
            "held_hive": jnp.full((48,), -1, jnp.int32), "message": jnp.zeros((48,), jnp.int32), "heard": jnp.zeros((48,), jnp.int32),
            "plant_pos": plants, "plant_active": jnp.ones((24,), jnp.bool_), "plant_vis": jnp.zeros((48, 24), jnp.bool_),
            "stage": jnp.full((24,), 2, jnp.int32), "age": jnp.full((24,), 8, jnp.int32),
            "wet": jnp.full((24,), 3, jnp.int32), "pollen": jnp.zeros((24,), jnp.int32),
            "disease": jnp.zeros((24,), jnp.int32), "crop": jnp.zeros((24,), jnp.int32), "variety": jnp.zeros((24,), jnp.int32),
            "gate_pos": gate, "gate_active": jnp.ones((12,), jnp.bool_), "gate_vis": jnp.zeros((48, 12), jnp.bool_), "gate_open": jnp.ones((12,), jnp.bool_),
            "water_pos": water, "water_active": jnp.ones((12,), jnp.bool_), "water_vis": jnp.zeros((48, 12), jnp.bool_),
            "canal": jnp.full((12,), 6, jnp.int32), "sluice": jnp.ones((12,), jnp.bool_), "draw": jnp.zeros((12,), jnp.int32),
            "store_pos": store, "store_active": jnp.ones((12,), jnp.bool_), "store_vis": jnp.zeros((48, 12), jnp.bool_),
            "queue": jnp.zeros((12,), jnp.int32), "delivered": jnp.zeros((12,), jnp.int32),
            "hive_pos": hive, "hive_active": jnp.ones((12,), jnp.bool_), "hive_vis": jnp.zeros((48, 12), jnp.bool_),
            "hive_lane": hive[:, 1], "hive_carrier": jnp.full((12,), -1, jnp.int32), "hive_carried": jnp.zeros((12,), jnp.bool_),
            "bee_pos": hive + jnp.array([1, 0], jnp.int32), "bee_active": jnp.ones((12,), jnp.bool_), "bee_vis": jnp.zeros((48, 12), jnp.bool_),
            "bee_dir": jnp.ones((12,), jnp.int32), "bee_spore": jnp.zeros((12,), jnp.int32),
            "bee_health": jnp.full((12,), 4, jnp.int32), "bee_hurt": jnp.zeros((12,), jnp.bool_),
            "harvested": jnp.zeros((48,), jnp.bool_), "gated": jnp.zeros((48,), jnp.bool_),
            "tended": jnp.zeros((48,), jnp.bool_), "moved_hive": jnp.zeros((48,), jnp.bool_),
            "pollinated": jnp.zeros((24,), jnp.bool_), "ripened": jnp.zeros((24,), jnp.bool_), "shipped": jnp.zeros((12,), jnp.bool_),
            "reward_parts": jnp.zeros((48, 5), jnp.float32)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        p = s["pos"]
        ids = jnp.arange(48)
        own = jnp.stack([p[:, 0] / 47.0, p[:, 1] / 23.0, s["face"] / 3.0, s["tool"] / 3.0,
             s["team"] / 11.0, s["fruit"] / 3.0, s["bucket"] / 3.0, s["compost"] / 3.0, s["seeds"] / 2.0,
             (s["held_hive"] + 1) / 12.0, s["heard"] / 3.0, jnp.full((48,), s["t"] / 768.0),
             jnp.full((48,), ((s["t"] // 64) % 4) / 3.0), s["harvested"] * 1.0,
             s["tended"] * 1.0, s["moved_hive"] * 1.0], axis=1)
        k = jnp.arange(49)
        q = p[:, None, :] + jnp.stack([k % 7 - 3, k // 7 - 3], axis=1)[None, :, :]
        inside = jnp.all((q >= jnp.array([0, 0], jnp.int32)) & (q <= jnp.array([47, 23], jnp.int32)), axis=2)
        c = jnp.clip(q, jnp.array([0, 0], jnp.int32), jnp.array([47, 23], jnp.int32))
        open_gate = jnp.any(jnp.all(q[:, :, None, :] == s["gate_pos"][None, None, :, :], axis=3) & s["gate_open"][None, None, :], axis=2)
        terrain = jnp.where(inside, s["tiles"][c[:, :, 1], c[:, :, 0]] / 5.0 + 0.1 * open_gate, -1.0)
        pm = jnp.all(q[:, :, None, :] == s["plant_pos"][None, None, :, :], axis=3)
        stage = jnp.max(jnp.where(pm, s["stage"][None, None, :] / 4.0, 0.0), axis=2)
        actors = jnp.any(jnp.all(q[:, :, None, :] == p[None, None, :, :], axis=3), axis=2)
        hives = jnp.any(jnp.all(q[:, :, None, :] == s["hive_pos"][None, None, :, :], axis=3), axis=2)
        bees = jnp.sum(jnp.all(q[:, :, None, :] == s["bee_pos"][None, None, :, :], axis=3) & s["bee_active"][None, None, :], axis=2) / 12.0
        patch = jnp.stack([terrain, stage, actors * 0.25 + hives * 0.5, bees], axis=2).reshape((48, 196))
        dist = jnp.max(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2)
        nv = s["agent_vis"] & (ids[:, None] != ids[None, :])
        ni = jnp.argsort(jnp.where(nv, dist, 999), axis=1)[:, :4]
        valid = jnp.take_along_axis(nv, ni, axis=1)
        rel = (p[ni] - p[:, None, :]) / 3.0
        nb = jnp.stack([valid * 1.0, rel[:, :, 0], rel[:, :, 1], s["tool"][ni] / 3.0,
                       s["fruit"][ni] / 3.0, s["bucket"][ni] / 3.0, s["message"][ni] / 3.0, s["team"][ni] / 11.0], axis=2)
        nb = jnp.where(valid[:, :, None], nb, 0.0).reshape((48, 32))
        pd = jnp.max(jnp.abs(p[:, None, :] - s["plant_pos"][None, :, :]), axis=2)
        pi = jnp.argsort(jnp.where(s["plant_vis"], pd, 999), axis=1)[:, :2]
        pv = jnp.take_along_axis(s["plant_vis"], pi, axis=1)
        pr = (s["plant_pos"][pi] - p[:, None, :]) / 3.0
        plant = jnp.stack([pv * 1.0, pr[:, :, 0], pr[:, :, 1], s["stage"][pi] / 4.0, s["age"][pi] / 32.0,
                          s["wet"][pi] / 8.0, s["pollen"][pi] / 3.0, s["disease"][pi] / 3.0,
                          s["crop"][pi] / 2.0, s["variety"][pi] * 1.0], axis=2)
        plant = jnp.where(pv[:, :, None], plant, 0.0).reshape((48, 20))
        hi = jnp.argmin(jnp.where(s["hive_vis"], jnp.sum(jnp.abs(p[:, None, :] - s["hive_pos"][None, :, :]), axis=2), 999), axis=1)
        hv = s["hive_vis"][ids, hi]
        hr = (s["hive_pos"][hi] - p) / 3.0
        hive = jnp.stack([hv * 1.0, hr[:, 0], hr[:, 1], (s["hive_carrier"][hi] >= 0) * 1.0, hi / 11.0, (s["hive_carrier"][hi] == ids) * 1.0], axis=1)
        wi = jnp.argmin(jnp.sum(jnp.abs(p[:, None, :] - s["water_pos"][None, :, :]), axis=2), axis=1)
        wv = s["water_vis"][ids, wi]
        wr = (s["water_pos"][wi] - p) / 3.0
        water = jnp.stack([wv * 1.0, wr[:, 0], wr[:, 1], s["canal"][wi] / 8.0, s["sluice"][wi] * 1.0, s["draw"][wi] / 2.0, wi / 11.0], axis=1)
        di = jnp.argmin(jnp.sum(jnp.abs(p[:, None, :] - s["store_pos"][None, :, :]), axis=2), axis=1)
        dv = s["store_vis"][ids, di]
        dr = (s["store_pos"][di] - p) / 3.0
        store = jnp.stack([dv * 1.0, dr[:, 0], dr[:, 1], s["queue"][di] / 4.0, s["delivered"][di] / 2.0, di / 11.0], axis=1)
        bd = jnp.max(jnp.abs(p[:, None, :] - s["bee_pos"][None, :, :]), axis=2)
        bi = jnp.argsort(jnp.where(s["bee_vis"], bd, 999), axis=1)[:, :4]
        bv = jnp.take_along_axis(s["bee_vis"], bi, axis=1)
        br = (s["bee_pos"][bi] - p[:, None, :]) / 3.0
        bee = jnp.stack([bv * 1.0, br[:, :, 0], br[:, :, 1], s["bee_dir"][bi] * 1.0,
                        s["bee_spore"][bi] / 4.0, bi / 11.0, (s["hive_carrier"][bi] < 0) * 1.0, s["bee_health"][bi] / 4.0], axis=2)
        bee = jnp.where(bv[:, :, None], bee, 0.0).reshape((48, 32))
        return jnp.concatenate([own, patch, nb, plant, jnp.where(hv[:, None], hive, 0.0),
              jnp.where(wv[:, None], water, 0.0), jnp.where(dv[:, None], store, 0.0), bee], axis=1).astype(jnp.float32)

    def act(self, i, s, a):
        i = jnp.asarray(i, jnp.int32)
        z = dict(s)
        mask = self.legal_row(z, i)
        action = jnp.where((a >= 0) & (a < 23) & mask[jnp.clip(a, 0, 22)], a, 0)
        q, p, atp, g, atg, w, d, h, ath, j, near = self.context(z, i)
        move = (action >= 1) & (action <= 4)
        direction = jnp.clip(action - 1, 0, 3)
        z["face"] = z["face"].at[i].set(jnp.where(move, direction, jnp.where((action >= 5) & (action <= 8), action - 5, z["face"][i])))
        z["pos"] = z["pos"].at[i].set(z["pos"][i] + jnp.where(move, self.directions()[direction], jnp.zeros((2,), jnp.int32)))
        plant, graft, tend, harvest, deliver = (action == v for v in (9, 10, 11, 12, 13))
        clear = tend & (z["stage"][p] == 4)
        z["seeds"] = z["seeds"].at[i].add(-plant.astype(jnp.int32))
        z["compost"] = z["compost"].at[i].add(clear.astype(jnp.int32) - graft.astype(jnp.int32))
        z["stage"] = z["stage"].at[p].set(jnp.where(plant, 1, jnp.where(clear, 0, z["stage"][p])))
        z["age"] = z["age"].at[p].set(jnp.where(plant | clear, 0, z["age"][p]))
        z["variety"] = z["variety"].at[p].set(jnp.where(plant, 0, jnp.where(graft, 1, z["variety"][p])))
        z["pollen"] = z["pollen"].at[p].set(jnp.where(plant | clear, 0, z["pollen"][p]))
        z["disease"] = z["disease"].at[p].set(jnp.where(plant, 0, jnp.maximum(0, z["disease"][p] - tend.astype(jnp.int32))))
        z["crop"] = z["crop"].at[p].add(-harvest.astype(jnp.int32))
        z["stage"] = z["stage"].at[p].set(jnp.where(harvest & (z["crop"][p] == 0), 4, z["stage"][p]))
        z["fruit"] = z["fruit"].at[i].add(harvest.astype(jnp.int32) - deliver.astype(jnp.int32))
        z["queue"] = z["queue"].at[d].add(deliver.astype(jnp.int32))
        fill = (action == 14) & (z["bucket"][i] == 0)
        irrigate = (action == 14) & ~fill
        amount = jnp.minimum(3, z["canal"][w])
        use = jnp.minimum(z["bucket"][i], 8 - z["wet"][p])
        z["canal"] = z["canal"].at[w].add(-fill.astype(jnp.int32) * amount)
        z["bucket"] = z["bucket"].at[i].add(fill.astype(jnp.int32) * amount - irrigate.astype(jnp.int32) * use)
        z["wet"] = z["wet"].at[p].add(irrigate.astype(jnp.int32) * use)
        z["sluice"] = z["sluice"].at[w].set(z["sluice"][w] ^ (action == 15))
        z["gate_open"] = z["gate_open"].at[g].set(z["gate_open"][g] ^ (action == 16))
        gf = (action == 17) & (z["fruit"][i] > 0) & (z["fruit"][j] < 3)
        gw = (action == 17) & ~gf & (z["bucket"][i] > 0) & (z["bucket"][j] < 3)
        gc = (action == 17) & ~gf & ~gw & (z["compost"][i] > 0) & (z["compost"][j] < 3)
        z["fruit"] = z["fruit"].at[i].add(-gf.astype(jnp.int32)).at[j].add(gf.astype(jnp.int32))
        z["bucket"] = z["bucket"].at[i].add(-gw.astype(jnp.int32)).at[j].add(gw.astype(jnp.int32))
        z["compost"] = z["compost"].at[i].add(-gc.astype(jnp.int32)).at[j].add(gc.astype(jnp.int32))
        old_held = z["held_hive"][i]
        pickup = (action == 18) & (old_held < 0)
        drop = (action == 18) & (old_held >= 0)
        hd = jnp.where(pickup, h, jnp.clip(old_held, 0, 11))
        z["held_hive"] = z["held_hive"].at[i].set(jnp.where(pickup, h, jnp.where(drop, -1, old_held)))
        z["hive_carrier"] = z["hive_carrier"].at[hd].set(jnp.where(pickup, i, jnp.where(drop, -1, z["hive_carrier"][hd])))
        z["hive_pos"] = z["hive_pos"].at[hd].set(jnp.where(drop | pickup, z["pos"][i], z["hive_pos"][hd]))
        ti, tj = z["tool"][i], z["tool"][j]
        z["tool"] = z["tool"].at[i].set(jnp.where(action == 19, tj, ti)).at[j].set(jnp.where(action == 19, ti, tj))
        msg = 1 + (atp & (z["stage"][p] == 2)).astype(jnp.int32) + (atp & (z["disease"][p] > 0)).astype(jnp.int32)
        z["message"] = z["message"].at[i].set(jnp.where(action == 20, msg, z["message"][i]))
        fence, remove = action == 21, action == 22
        z["tiles"] = z["tiles"].at[q[1], q[0]].set(jnp.where(fence, 1, jnp.where(remove, 0, z["tiles"][q[1], q[0]])))
        z["compost"] = z["compost"].at[i].add(remove.astype(jnp.int32) - fence.astype(jnp.int32))
        injury = (fence | ((action == 16) & s["gate_open"][g])) & jnp.all(z["bee_pos"] == q, axis=1) & z["bee_active"]
        z["bee_health"] = jnp.maximum(0, z["bee_health"] - injury.astype(jnp.int32))
        z["bee_active"] = z["bee_health"] > 0
        z["bee_hurt"] = z["bee_hurt"] | injury
        z["harvested"] = z["harvested"].at[i].set(harvest)
        z["gated"] = z["gated"].at[i].set(action == 16)
        z["tended"] = z["tended"].at[i].set(tend)
        z["moved_hive"] = z["moved_hive"].at[i].set(action == 18)
        rp = z["reward_parts"]
        rp = rp.at[i, 0].add(-0.01 * move - 0.02 * (plant | graft | tend | irrigate | fence | remove))
        rp = rp.at[i, 1].add(0.1 * harvest + 0.05 * deliver)
        rp = rp.at[i, 4].add(-0.02 * fence - 0.005 * fill * amount - 0.1 * jnp.sum(injury))
        z["reward_parts"] = rp
        carrier = jnp.clip(z["hive_carrier"], 0, 47)
        z["hive_pos"] = jnp.where((z["hive_carrier"] >= 0)[:, None], z["pos"][carrier], z["hive_pos"])
        return z

    def advance(self, s, actions):
        z = dict(s)
        for name in ("harvested", "gated", "tended", "moved_hive"):
            z[name] = jnp.zeros((48,), jnp.bool_)
        z["message"] = jnp.zeros((48,), jnp.int32)
        z["bee_hurt"] = jnp.zeros((12,), jnp.bool_)
        z["reward_parts"] = jnp.zeros((48, 5), jnp.float32)
        z = jax.lax.fori_loop(0, 48, lambda k, w: self.act((k + s["t"]) % 48, w, actions[(k + s["t"]) % 48]), z)
        tick = s["t"] + 1
        season = (tick // 64) % 4
        draw = jnp.where(z["sluice"], jnp.minimum(2, z["canal"]), 0)
        remaining = z["canal"] - draw
        incoming = jnp.roll(remaining, 1)
        left = jnp.arange(12) % 4 == 0
        rain = jnp.where((tick % 4 == 0) & (season != 3), 3, 0)
        z["canal"] = jnp.minimum(8, jnp.where(left, rain, incoming))
        z["draw"] = draw
        z["wet"] = jnp.clip(z["wet"] + jnp.repeat(draw, 2) - (tick % 4 == 0).astype(jnp.int32), 0, 8)
        flying = (tick % 2 == 0) & (season != 3) & (z["hive_carrier"] < 0) & z["bee_active"]
        bx = z["bee_pos"][:, 0]
        hx = z["hive_pos"][:, 0]
        direction = jnp.where(jnp.abs(bx - hx) > 5, jnp.sign(hx - bx), z["bee_dir"]).astype(jnp.int32)
        proposed = z["bee_pos"] + jnp.stack([direction, jnp.zeros((12,), jnp.int32)], axis=1)
        c = jnp.clip(proposed, jnp.array([0, 0], jnp.int32), jnp.array([47, 23], jnp.int32))
        tile = z["tiles"][c[:, 1], c[:, 0]]
        gate_match = jnp.all(c[:, None, :] == z["gate_pos"][None, :, :], axis=2)
        gate_closed = jnp.any(gate_match & ~z["gate_open"][None, :], axis=1)
        inward = jnp.abs(bx - hx) > 5
        allowed = jnp.all(c == proposed, axis=1) & (tile != 1) & ~gate_closed & ((jnp.abs(c[:, 0] - hx) <= 5) | inward)
        z["bee_pos"] = jnp.where((flying & allowed)[:, None], c, z["bee_pos"])
        z["bee_dir"] = jnp.where(flying, jnp.where(allowed, direction, -direction), z["bee_dir"])
        near = jnp.sum(jnp.abs(z["plant_pos"][:, None, :] - z["bee_pos"][None, :, :]), axis=2) <= 1
        visits = near & flying[None, :] & (z["stage"][:, None] == 2)
        old_pollen = z["pollen"]
        z["pollen"] = jnp.minimum(3, old_pollen + jnp.sum(visits, axis=1).astype(jnp.int32))
        z["pollinated"] = z["pollen"] > old_pollen
        exposed = jnp.any(visits & (z["bee_spore"][None, :] > 0), axis=1)
        damp = (z["wet"] >= 7) & (tick % 8 == 0)
        z["disease"] = jnp.minimum(3, z["disease"] + (exposed | damp).astype(jnp.int32))
        contaminated = jnp.any(visits & (z["disease"][:, None] >= 2), axis=0)
        z["bee_spore"] = jnp.where(contaminated, 4, jnp.maximum(0, z["bee_spore"] - flying.astype(jnp.int32)))
        growing = (z["wet"] > 0) & (z["disease"] < 3) & (season != 3) & ((z["stage"] == 1) | (z["stage"] == 2))
        z["age"] = jnp.minimum(32, z["age"] + growing.astype(jnp.int32))
        z["stage"] = jnp.where((z["stage"] == 1) & (z["age"] >= 6 - 2 * z["variety"]), 2, z["stage"])
        z["ripened"] = (z["stage"] == 2) & (z["age"] >= 14 - 2 * z["variety"]) & (z["pollen"] >= 2) & (z["disease"] < 3)
        z["stage"] = jnp.where(z["ripened"], 3, z["stage"])
        z["crop"] = jnp.where(z["ripened"], 2, z["crop"])
        g = jnp.arange(12)
        slot = (tick // 4) % 4
        priority = (jnp.arange(4) + slot) % 4
        eligible = ((z["queue"] > 0) & (z["delivered"] < 2)).reshape((3, 4))
        ranked = eligible[:, priority]
        chosen = priority[jnp.argmax(ranked, axis=1)]
        any_queue = jnp.any(ranked, axis=1)
        z["shipped"] = (tick % 4 == 0) & any_queue[g // 4] & (g % 4 == chosen[g // 4])
        z["queue"] = z["queue"] - z["shipped"].astype(jnp.int32)
        z["delivered"] = z["delivered"] + z["shipped"].astype(jnp.int32)
        z["reward_parts"] = z["reward_parts"].at[:, 2].add(0.25 * z["shipped"][z["team"]])
        local = jnp.sum(jnp.abs(z["pos"][:, None, :] - z["plant_pos"][None, :, :]), axis=2) <= 8
        ecological = jnp.sum(local * z["ripened"][None, :], axis=1)
        harmed_near = jnp.sum(jnp.abs(z["pos"][:, None, :] - z["bee_pos"][None, :, :]), axis=2) <= 8
        ecological_loss = jnp.sum(harmed_near & z["bee_hurt"][None, :], axis=1)
        z["reward_parts"] = z["reward_parts"].at[:, 3].add(0.005 * ecological - 0.005 * ecological_loss)
        hearing = jnp.sum(jnp.abs(z["pos"][:, None, :] - z["pos"][None, :, :]), axis=2) <= 6
        z["heard"] = jnp.max(jnp.where(hearing, z["message"][None, :], 0), axis=1)
        z["t"] = s["t"] + jnp.asarray(1, jnp.int32)
        z["terminated"] = jnp.all(z["delivered"] >= 2)
        z["truncated"] = (z["t"] >= 768) & ~z["terminated"]
        z["done"] = z["terminated"] | z["truncated"]
        return self.refresh(z)

    def raw_step(self, key, state, actions):
        ns = jax.lax.cond(state["done"], lambda _: state, lambda _: self.advance(state, actions), operand=None)
        rewards = jnp.where(state["done"], jnp.zeros((48,), jnp.float32), jnp.sum(ns["reward_parts"], axis=1))
        return self.observe(ns), ns, rewards, jnp.full((48,), ns["done"], jnp.bool_), {"done_all": ns["done"], "terminated": ns["terminated"], "truncated": ns["truncated"]}

