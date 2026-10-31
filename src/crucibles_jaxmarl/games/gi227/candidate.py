"""Forest Frontier Commons: original, unexecuted native spatial game source."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 128
    observation_size = 321
    num_actions = 26
    max_steps = 896

    def sight(self, pos, target, active, count):
        distance = jnp.sum(jnp.abs(pos[:, None, :] - target[None, :, :]), axis=2)
        ids = jnp.arange(target.shape[0], dtype=jnp.int32)
        score = jnp.where(active[None, :] & (distance <= 5), distance * (target.shape[0] + 1) + ids,
                          100000 + ids)
        chosen = jnp.argsort(score, axis=1)[:, :count]
        valid = jnp.take_along_axis(score, chosen, axis=1) < 100000
        return jnp.zeros(distance.shape, dtype=jnp.bool_).at[jnp.arange(128)[:, None], chosen].set(valid)

    def refresh(self, state):
        s = dict(state)
        s["agent_visible"] = self.sight(s["pos"], s["pos"], s["active"], 8)
        s["tree_visible"] = self.sight(s["pos"], s["tree_pos"], s["tree_active"], 12)
        s["field_visible"] = self.sight(s["pos"], s["field_pos"], s["village_active"], 3)
        s["well_visible"] = self.sight(s["pos"], s["well_pos"], s["village_active"], 3)
        s["store_visible"] = self.sight(s["pos"], s["store_pos"], s["village_active"], 3)
        s["bridge_visible"] = self.sight(s["pos"], s["bridge_pos"], s["bridge_active"], 3)
        s["cart_visible"] = self.sight(s["pos"], s["cart_pos"], s["village_active"], 3)
        return s

    def reset(self, key):
        d = jnp.arange(16, dtype=jnp.int32)
        ids = jnp.arange(128, dtype=jnp.int32)
        centers = jnp.stack((5 + 10 * (d % 4), 5 + 10 * (d // 4)), axis=1)
        offsets = jnp.array([[2, 0], [0, 2], [0, 0], [-2, 0], [3, 1], [1, 2], [0, -1], [-2, 1]], dtype=jnp.int32)
        tree_offsets = jnp.array([[3, 0], [3, 2], [-3, 0], [-2, -3]], dtype=jnp.int32)
        trees = (centers[:, None, :] + tree_offsets).reshape((64, 2))
        bridges = jnp.array([[20, 5], [20, 15], [20, 25], [20, 35],
                             [5, 20], [15, 20], [25, 20], [35, 20]], dtype=jnp.int32)
        x, y = jnp.arange(40, dtype=jnp.int32)[None, :], jnp.arange(40, dtype=jnp.int32)[:, None]
        tile = jnp.where((x % 10 == 5) | (y % 10 == 5), 1, 0)
        tile = jnp.where((x == 20) | (y == 20), 2, tile).astype(jnp.int32)
        tile = tile.at[bridges[:, 1], bridges[:, 0]].set(3)
        s = {"t": jnp.asarray(0, dtype=jnp.int32), "done": jnp.asarray(False),
             "terminated": jnp.asarray(False), "truncated": jnp.asarray(False),
             "map": tile, "pos": (centers[:, None, :] + offsets).reshape((128, 2)),
             "active": jnp.ones((128,), dtype=jnp.bool_), "crew": ids // 8, "profession": ids % 4,
             "tool": jnp.array([1, 2, 1, 3], dtype=jnp.int32)[ids % 4],
             "wood": jnp.where(ids % 8 == 2, 2, 0).astype(jnp.int32),
             "food": jnp.where((ids % 4 == 1) | (ids % 4 == 3), 1, 0).astype(jnp.int32),
             "water": jnp.where(ids % 4 == 3, 1, 0).astype(jnp.int32),
             "health": jnp.full((128,), 3, dtype=jnp.int32), "stamina": jnp.full((128,), 6, dtype=jnp.int32),
             "smoke": jnp.zeros((128,), dtype=jnp.int32), "tree_pos": trees,
             "tree_active": jnp.ones((64,), dtype=jnp.bool_), "tree_owner": jnp.arange(64, dtype=jnp.int32) // 4,
             "height": jnp.full((64,), 3, dtype=jnp.int32), "regrow": jnp.zeros((64,), dtype=jnp.int32),
             "fire": jnp.zeros((64,), dtype=jnp.bool_), "firebreak": jnp.zeros((64,), dtype=jnp.bool_),
             "store_pos": centers, "well_pos": centers + jnp.array([-2, 0], dtype=jnp.int32),
             "field_pos": centers + jnp.array([0, 2], dtype=jnp.int32),
             "village_active": jnp.ones((16,), dtype=jnp.bool_), "well_water": jnp.full((16,), 4, dtype=jnp.int32),
             "growth": jnp.full((16,), 2, dtype=jnp.int32), "moisture": jnp.zeros((16,), dtype=jnp.int32),
             "stock_wood": jnp.full((16,), 6, dtype=jnp.int32), "stock_food": jnp.full((16,), 6, dtype=jnp.int32),
             "level": jnp.zeros((16,), dtype=jnp.int32), "orders": jnp.full((16,), 4, dtype=jnp.int32),
             "report": jnp.zeros((16,), dtype=jnp.int32), "report_age": jnp.full((16,), 896, dtype=jnp.int32),
             "bridge_pos": bridges, "bridge_active": jnp.ones((8,), dtype=jnp.bool_),
             "bridge_health": jnp.array([0, 2, 0, 2, 2, 0, 2, 0], dtype=jnp.int32),
             "cart_pos": centers, "cart_home": d, "cart_target": 4 * (d // 4) + (d + 1) % 4,
             "cart_phase": jnp.zeros((16,), dtype=jnp.int32), "cart_cargo": jnp.zeros((16,), dtype=jnp.int32),
             "cart_stalled": jnp.zeros((16,), dtype=jnp.bool_),
             "ev_move": jnp.zeros((128,), dtype=jnp.bool_), "ev_chop": jnp.zeros((64,), dtype=jnp.bool_),
             "ev_harvest": jnp.zeros((128,), dtype=jnp.bool_), "ev_build": jnp.zeros((128,), dtype=jnp.bool_),
             "ev_trade": jnp.zeros((128,), dtype=jnp.bool_), "ev_aid": jnp.zeros((128,), dtype=jnp.bool_),
             "ev_extinguish": jnp.zeros((64,), dtype=jnp.bool_), "ev_regrow": jnp.zeros((64,), dtype=jnp.bool_),
             "ev_burn": jnp.zeros((64,), dtype=jnp.bool_), "ev_delivery": jnp.zeros((16,), dtype=jnp.bool_),
             "ev_cart_move": jnp.zeros((16,), dtype=jnp.bool_), "ev_signal": jnp.zeros((16,), dtype=jnp.bool_),
             "reward_components": jnp.zeros((128, 6), dtype=jnp.float32)}
        s = self.refresh(s)
        return self.observe(s), s

    def block(self, s, position, visible, count, values):
        distance = jnp.sum(jnp.abs(s["pos"][:, None, :] - position[None, :, :]), axis=2)
        ids = jnp.arange(position.shape[0], dtype=jnp.int32)
        order = jnp.argsort(jnp.where(visible, distance * (position.shape[0] + 1) + ids,
                                     100000 + ids), axis=1)[:, :count]
        seen = jnp.take_along_axis(visible, order, axis=1)
        rows = jnp.concatenate((seen[:, :, None], values[order] * seen[:, :, None]), axis=2)
        return rows.reshape((128, -1)).astype(jnp.float32)

    def observe(self, s):
        ids, d, tr, br = jnp.arange(128), jnp.arange(16), jnp.arange(64), jnp.arange(8)
        own = jnp.stack((jnp.full((128,), s["t"] / 896.0), s["pos"][:, 0] / 39.0, s["pos"][:, 1] / 39.0,
                         s["crew"] / 15.0, s["profession"] / 3.0, s["tool"] / 3.0, s["wood"] / 3.0,
                         s["food"] / 3.0, s["water"] / 2.0, s["health"] / 3.0, s["stamina"] / 6.0,
                         s["smoke"] / 64.0), axis=1)
        agent = jnp.stack((ids / 127.0, s["pos"][:, 0] / 39.0, s["pos"][:, 1] / 39.0, s["crew"] / 15.0,
                           s["profession"] / 3.0, s["tool"] / 3.0, s["wood"] / 3.0, s["food"] / 3.0,
                           s["water"] / 2.0, s["health"] / 3.0, s["stamina"] / 6.0), axis=1)
        tree = jnp.stack((tr / 63.0, s["tree_pos"][:, 0] / 39.0, s["tree_pos"][:, 1] / 39.0,
                          s["height"] / 3.0, s["fire"], s["firebreak"], s["regrow"] / 24.0, s["tree_owner"] / 15.0), axis=1)
        field = jnp.stack((d / 15.0, s["field_pos"][:, 0] / 39.0, s["field_pos"][:, 1] / 39.0,
                           s["growth"] / 6.0, s["moisture"] / 3.0), axis=1)
        well = jnp.stack((d / 15.0, s["well_pos"][:, 0] / 39.0, s["well_pos"][:, 1] / 39.0, s["well_water"] / 8.0), axis=1)
        store = jnp.stack((d / 15.0, s["store_pos"][:, 0] / 39.0, s["store_pos"][:, 1] / 39.0,
                           s["stock_wood"] / 32.0, s["stock_food"] / 32.0, s["level"] / 3.0,
                           s["orders"] / 4.0, s["report"] / 4.0, s["report_age"] / 896.0), axis=1)
        bridge = jnp.stack((br / 7.0, s["bridge_pos"][:, 0] / 39.0, s["bridge_pos"][:, 1] / 39.0, s["bridge_health"] / 3.0), axis=1)
        cart = jnp.stack((d / 15.0, s["cart_pos"][:, 0] / 39.0, s["cart_pos"][:, 1] / 39.0,
                          s["cart_cargo"] / 2.0, s["cart_phase"] / 2.0, s["cart_home"] / 15.0,
                          s["cart_target"] / 15.0, s["cart_stalled"]), axis=1)
        return jnp.concatenate((own, self.block(s, s["pos"], s["agent_visible"], 8, agent),
            self.block(s, s["tree_pos"], s["tree_visible"], 12, tree),
            self.block(s, s["field_pos"], s["field_visible"], 3, field),
            self.block(s, s["well_pos"], s["well_visible"], 3, well),
            self.block(s, s["store_pos"], s["store_visible"], 3, store),
            self.block(s, s["bridge_pos"], s["bridge_visible"], 3, bridge),
            self.block(s, s["cart_pos"], s["cart_visible"], 3, cart)), axis=1).astype(jnp.float32)

    def nearest(self, positions, objects):
        distance = jnp.sum(jnp.abs(positions[:, None, :] - objects[None, :, :]), axis=2)
        return jnp.argmin(distance, axis=1), jnp.min(distance, axis=1)

    def walkable(self, s, positions):
        x, y = jnp.clip(positions[..., 0], 0, 39), jnp.clip(positions[..., 1], 0, 39)
        crossing = jnp.any(jnp.all(positions[..., None, :] == s["bridge_pos"], axis=-1)
                           & (s["bridge_health"] > 0), axis=-1)
        trunk = jnp.any(jnp.all(positions[..., None, :] == s["tree_pos"], axis=-1) & (s["height"] >= 3), axis=-1)
        tile = s["map"][y, x]
        return (positions[..., 0] == x) & (positions[..., 1] == y) & (((tile != 2) & (tile != 3)) | crossing) & ~trunk

    def legal_actions(self, s):
        moves = s["pos"][:, None, :] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        occupied = jnp.any(jnp.all(moves[:, :, None, :] == s["pos"][None, None, :, :], axis=3), axis=2)
        moving = self.walkable(s, moves) & ~occupied & (s["stamina"][:, None] > 0)
        tr, td = self.nearest(s["pos"], s["tree_pos"])
        f, fd = self.nearest(s["pos"], s["field_pos"])
        w, wd = self.nearest(s["pos"], s["well_pos"])
        h, hd = self.nearest(s["pos"], s["store_pos"])
        b, bd = self.nearest(s["pos"], s["bridge_pos"])
        c, cd = self.nearest(s["pos"], s["cart_pos"])
        adjacent = jnp.sum(jnp.abs(s["pos"][:, None, :] - s["pos"][None, :, :]), axis=2) == 1
        axe, hoe, bucket = s["tool"] == 1, s["tool"] == 2, s["tool"] == 3
        load = (cd <= 1) & (s["cart_phase"][c] == 0) & (s["cart_cargo"][c] == 0)
        extras = jnp.stack(((td <= 1) & axe & (s["wood"] < 3) & (s["height"][tr] > 0) & ~s["fire"][tr] & ~s["firebreak"][tr],
            (fd <= 1) & (s["growth"][f] == 6) & (s["food"] <= 1),
            (fd <= 1) & bucket & (s["water"] > 0) & (s["moisture"][f] < 3),
            (wd <= 1) & bucket & (s["water"] < 2) & (s["well_water"][w] > 0),
            (bd <= 1) & axe & (s["wood"] > 0) & (s["bridge_health"][b] < 3),
            load & (s["wood"] > 0), load & (s["food"] > 0),
            (hd <= 1) & (s["wood"] > 0) & (s["stock_wood"][h] < 32),
            (hd <= 1) & (s["food"] > 0) & (s["stock_food"][h] < 32),
            (fd <= 1) & hoe & (s["moisture"][f] > 0) & (s["growth"][f] > 0) & (s["growth"][f] < 6),
            (s["food"] > 0) & ((s["health"] < 3) | (s["stamina"] <= 4)), jnp.any(adjacent, axis=1),
            jnp.any(adjacent & (s["health"][None, :] < 3), axis=1) & ((s["food"] > 0) | (s["water"] > 0)),
            (td <= 1) & axe & (s["wood"] > 0) & ~s["firebreak"][tr],
            (hd <= 1) & axe & (s["wood"] >= 2) & (s["food"] >= 1) & (s["level"][h] < 3),
            (fd <= 1) & hoe & (s["food"] > 0) & (s["growth"][f] == 0),
            (td <= 1) & (s["food"] > 0) & (s["height"][tr] == 0) & ~s["fire"][tr],
            (td <= 1) & bucket & (s["water"] > 0) & s["fire"][tr],
            (hd <= 1) & (s["wood"] < 3) & (s["stock_wood"][h] > 0),
            (hd <= 1) & (s["food"] < 3) & (s["stock_food"][h] > 0), (hd <= 1)), axis=1)
        live = jnp.concatenate((jnp.ones((128, 1), dtype=jnp.bool_), moving, extras), axis=1)
        can = ~s["done"] & (s["health"] > 0)
        return jnp.where(can[:, None], live, jnp.arange(26)[None, :] == 0)

    def actor_turn(self, k, carry):
        s, actions = carry
        n = dict(s)
        i = (s["t"] - 1 + k) % 128
        a, p = actions[i], s["pos"][i]
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)[jnp.minimum(a, 4)]
        target = p + delta
        free = ~jnp.any(jnp.all(s["pos"] == target, axis=1))
        moved = (a >= 1) & (a <= 4) & (s["health"][i] > 0) & (s["stamina"][i] > 0) & self.walkable(s, target) & free
        n["pos"] = s["pos"].at[i].set(jnp.where(moved, target, p))
        tile = s["map"][jnp.clip(target[1], 0, 39), jnp.clip(target[0], 0, 39)]
        cost = moved & (tile == 0)
        n["stamina"] = s["stamina"].at[i].set(jnp.clip(s["stamina"][i] - cost.astype(jnp.int32) + (a == 0).astype(jnp.int32), 0, 6))
        n["ev_move"] = s["ev_move"].at[i].set(moved)
        td = jnp.sum(jnp.abs(s["tree_pos"] - p), axis=1)
        fd = jnp.sum(jnp.abs(s["field_pos"] - p), axis=1)
        wd = jnp.sum(jnp.abs(s["well_pos"] - p), axis=1)
        hd = jnp.sum(jnp.abs(s["store_pos"] - p), axis=1)
        bd = jnp.sum(jnp.abs(s["bridge_pos"] - p), axis=1)
        cd = jnp.sum(jnp.abs(s["cart_pos"] - p), axis=1)
        tr, f, w = jnp.argmin(td), jnp.argmin(fd), jnp.argmin(wd)
        h, b, c = jnp.argmin(hd), jnp.argmin(bd), jnp.argmin(cd)
        axe, hoe, bucket = s["tool"][i] == 1, s["tool"][i] == 2, s["tool"][i] == 3
        chop = (a == 5) & (td[tr] <= 1) & axe & (s["wood"][i] < 3) & (s["height"][tr] > 0) & ~s["fire"][tr] & ~s["firebreak"][tr] & ~s["ev_chop"][tr]
        harvest = (a == 6) & (fd[f] <= 1) & (s["growth"][f] == 6) & (s["food"][i] <= 1)
        water = (a == 7) & (fd[f] <= 1) & bucket & (s["water"][i] > 0) & (s["moisture"][f] < 3)
        fill = (a == 8) & (wd[w] <= 1) & bucket & (s["water"][i] < 2) & (s["well_water"][w] > 0)
        build = (a == 9) & (bd[b] <= 1) & axe & (s["wood"][i] > 0) & (s["bridge_health"][b] < 3)
        loadable = (cd[c] <= 1) & (s["cart_phase"][c] == 0) & (s["cart_cargo"][c] == 0)
        loadw = (a == 10) & loadable & (s["wood"][i] > 0)
        loadf = (a == 11) & loadable & (s["food"][i] > 0)
        depositw = (a == 12) & (hd[h] <= 1) & (s["wood"][i] > 0) & (s["stock_wood"][h] < 32)
        depositf = (a == 13) & (hd[h] <= 1) & (s["food"][i] > 0) & (s["stock_food"][h] < 32)
        tend = (a == 14) & (fd[f] <= 1) & hoe & (s["moisture"][f] > 0) & (s["growth"][f] > 0) & (s["growth"][f] < 6)
        eat = (a == 15) & (s["food"][i] > 0) & ((s["health"][i] < 3) | (s["stamina"][i] <= 4))
        cut = (a == 18) & (td[tr] <= 1) & axe & (s["wood"][i] > 0) & ~s["firebreak"][tr]
        upgrade = (a == 19) & (hd[h] <= 1) & axe & (s["wood"][i] >= 2) & (s["food"][i] >= 1) & (s["level"][h] < 3)
        seed = (a == 20) & (fd[f] <= 1) & hoe & (s["food"][i] > 0) & (s["growth"][f] == 0)
        plant = (a == 21) & (td[tr] <= 1) & (s["food"][i] > 0) & (s["height"][tr] == 0) & ~s["fire"][tr]
        extinguish = (a == 22) & (td[tr] <= 1) & bucket & (s["water"][i] > 0) & s["fire"][tr]
        takew = (a == 23) & (hd[h] <= 1) & (s["wood"][i] < 3) & (s["stock_wood"][h] > 0)
        takef = (a == 24) & (hd[h] <= 1) & (s["food"][i] < 3) & (s["stock_food"][h] > 0)
        rest = (a == 0) & (hd[h] == 0) & (s["health"][i] < 3) & (s["stock_food"][h] > 0) & (s["t"] % 16 == 0)
        n["wood"] = s["wood"].at[i].add(chop.astype(jnp.int32) + takew - build - loadw - depositw - cut - 2 * upgrade.astype(jnp.int32))
        n["food"] = s["food"].at[i].add(2 * harvest.astype(jnp.int32) + takef - loadf - depositf - upgrade - seed - plant - eat)
        n["water"] = s["water"].at[i].add(fill.astype(jnp.int32) - water - extinguish)
        n["stock_wood"] = s["stock_wood"].at[h].add(depositw.astype(jnp.int32) - takew)
        n["stock_food"] = s["stock_food"].at[h].add(depositf.astype(jnp.int32) - takef - rest)
        n["level"] = s["level"].at[h].add(upgrade.astype(jnp.int32))
        n["well_water"] = s["well_water"].at[w].add(-fill.astype(jnp.int32))
        n["health"] = s["health"].at[i].set(jnp.minimum(3, s["health"][i] + rest.astype(jnp.int32) + eat))
        n["stamina"] = n["stamina"].at[i].set(jnp.minimum(6, n["stamina"][i] + 2 * eat.astype(jnp.int32)))
        n["bridge_health"] = s["bridge_health"].at[b].add(build.astype(jnp.int32))
        n["growth"] = s["growth"].at[f].set(jnp.where(harvest, 0, jnp.where(seed, 1, s["growth"][f] + tend.astype(jnp.int32))))
        n["moisture"] = s["moisture"].at[f].set(jnp.minimum(3, s["moisture"][f] + 2 * water.astype(jnp.int32)))
        n["height"] = s["height"].at[tr].set(jnp.where(cut, 0, jnp.where(plant, 1, s["height"][tr] - chop.astype(jnp.int32))))
        n["regrow"] = s["regrow"].at[tr].set(jnp.where(chop | cut | plant, 0, s["regrow"][tr]))
        n["firebreak"] = s["firebreak"].at[tr].set((s["firebreak"][tr] | cut) & ~plant)
        n["fire"] = s["fire"].at[tr].set(s["fire"][tr] & ~(cut | extinguish))
        n["ev_chop"] = s["ev_chop"].at[tr].set(s["ev_chop"][tr] | chop)
        n["ev_extinguish"] = s["ev_extinguish"].at[tr].set(s["ev_extinguish"][tr] | extinguish | (cut & s["fire"][tr]))
        n["ev_harvest"] = s["ev_harvest"].at[i].set(harvest)
        n["ev_build"] = s["ev_build"].at[i].set(build | cut | upgrade)
        n["cart_cargo"] = s["cart_cargo"].at[c].set(jnp.where(loadw, 1, jnp.where(loadf, 2, s["cart_cargo"][c])))
        n["cart_phase"] = s["cart_phase"].at[c].set(jnp.where(loadw | loadf, 1, s["cart_phase"][c]))
        neighbor = (jnp.sum(jnp.abs(s["pos"] - p), axis=1) == 1) & (actions == 0) & ~s["ev_trade"]
        candidates = neighbor & ~s["ev_trade"][i]
        j = jnp.argmin(jnp.where(candidates, jnp.arange(128), 1000))
        swap = (a == 16) & jnp.any(candidates)
        n["tool"] = s["tool"].at[i].set(jnp.where(swap, s["tool"][j], s["tool"][i]))
        n["tool"] = n["tool"].at[j].set(jnp.where(swap, s["tool"][i], n["tool"][j]))
        n["ev_trade"] = s["ev_trade"].at[i].set(s["ev_trade"][i] | swap)
        n["ev_trade"] = n["ev_trade"].at[j].set(n["ev_trade"][j] | swap)
        hurt = (jnp.sum(jnp.abs(s["pos"] - p), axis=1) == 1) & (s["health"] < 3)
        j = jnp.argmin(jnp.where(hurt, jnp.arange(128), 1000))
        aid = (a == 17) & jnp.any(hurt) & ((s["food"][i] > 0) | (s["water"][i] > 0))
        food_aid = aid & (s["food"][i] > 0)
        n["food"] = n["food"].at[i].add(-food_aid.astype(jnp.int32))
        n["water"] = n["water"].at[i].add(-(aid & ~food_aid).astype(jnp.int32))
        n["health"] = n["health"].at[j].set(jnp.minimum(3, n["health"][j] + aid.astype(jnp.int32)))
        n["stamina"] = n["stamina"].at[j].set(jnp.minimum(6, n["stamina"][j] + 2 * aid.astype(jnp.int32)))
        n["ev_aid"] = s["ev_aid"].at[j].set(s["ev_aid"][j] | aid)
        signal = (a == 25) & (hd[h] <= 1)
        target_village = 4 * (h // 4) + (h + 1) % 4
        local_fire = jnp.minimum(4, jnp.sum(s["fire"] & (jnp.sum(jnp.abs(s["tree_pos"] - p), axis=1) <= 5)))
        n["report"] = s["report"].at[target_village].set(jnp.where(signal, local_fire, s["report"][target_village]))
        n["report_age"] = s["report_age"].at[target_village].set(jnp.where(signal, 0, s["report_age"][target_village]))
        n["ev_signal"] = s["ev_signal"].at[target_village].set(s["ev_signal"][target_village] | signal)
        expense = chop.astype(jnp.float32) + water + build + cut + upgrade + seed + plant + extinguish + aid + eat
        r = s["reward_components"].at[i, 0].add(-0.01 * expense)
        r = r.at[i, 1].add(0.16 * harvest + 0.02 * aid)
        r = r.at[j, 1].add(0.02 * aid)
        r = r.at[:, 2].add(0.06 * upgrade * (s["crew"] == h) + 0.005 * build *
                            (jnp.sum(jnp.abs(s["pos"] - s["bridge_pos"][b]), axis=1) <= 5))
        r = r.at[:, 4].add(0.02 * extinguish * (s["crew"] == s["tree_owner"][tr]))
        n["reward_components"] = r
        return n, actions

    def cart_turn(self, k, state):
        s = dict(state)
        c = (state["t"] - 1 + k) % 16
        home = state["cart_home"][c]
        goal = state["cart_target"][c]
        p = state["cart_pos"][c]
        target = p + jnp.array([1, 0], dtype=jnp.int32) * jnp.sign(state["store_pos"][goal, 0] - p[0]).astype(jnp.int32)
        dock = jnp.any(jnp.all(target == state["store_pos"], axis=1))
        cart_block = jnp.any(jnp.all(target == state["cart_pos"], axis=1) & (jnp.arange(16) != c))
        actor_block = jnp.any(jnp.all(target == state["pos"], axis=1))
        moving = (state["cart_phase"][c] > 0) & jnp.any(target != p) & self.walkable(state, target) & (dock | ~(cart_block | actor_block))
        s["cart_pos"] = state["cart_pos"].at[c].set(jnp.where(moving, target, p))
        arrived = jnp.all(s["cart_pos"][c] == state["store_pos"][goal])
        cargo = state["cart_cargo"][c]
        room = jnp.where(cargo == 1, state["stock_wood"][goal] < 32, state["stock_food"][goal] < 32)
        delivered = (state["cart_phase"][c] == 1) & arrived & (cargo > 0) & room
        returned = (state["cart_phase"][c] == 2) & arrived
        commission = delivered & (state["orders"][goal] > 0) & (cargo == 1 + goal % 2)
        s["stock_wood"] = state["stock_wood"].at[goal].add((delivered & (cargo == 1)).astype(jnp.int32))
        s["stock_food"] = state["stock_food"].at[goal].add((delivered & (cargo == 2)).astype(jnp.int32))
        s["orders"] = state["orders"].at[goal].add(-commission.astype(jnp.int32))
        s["cart_cargo"] = state["cart_cargo"].at[c].set(jnp.where(delivered, 0, cargo))
        s["cart_phase"] = state["cart_phase"].at[c].set(jnp.where(delivered, 2, jnp.where(returned, 0, state["cart_phase"][c])))
        next_village = 4 * (home // 4) + (home + 1) % 4
        s["cart_target"] = state["cart_target"].at[c].set(jnp.where(delivered, home, jnp.where(returned, next_village, goal)))
        s["cart_stalled"] = state["cart_stalled"].at[c].set((state["cart_phase"][c] > 0) & ~moving & ~delivered & ~returned)
        s["ev_delivery"] = state["ev_delivery"].at[c].set(delivered)
        s["ev_cart_move"] = state["ev_cart_move"].at[c].set(moving)
        s["reward_components"] = state["reward_components"].at[:, 3].add(
            commission * (0.08 * (state["crew"] == goal) + 0.04 * (state["crew"] == home)))
        return s

    def advance(self, payload):
        state, actions = payload
        s = dict(state)
        clipped = jnp.clip(actions, 0, 25)
        legal = jnp.take_along_axis(self.legal_actions(state), clipped[:, None], axis=1)[:, 0]
        actions = jnp.where((actions >= 0) & (actions < 26) & legal, clipped, 0).astype(jnp.int32)
        s["t"] = state["t"] + jnp.asarray(1, dtype=jnp.int32)
        s["report_age"] = jnp.minimum(896, state["report_age"] + 1)
        for name in ("ev_move", "ev_chop", "ev_harvest", "ev_build", "ev_trade", "ev_aid", "ev_extinguish",
                     "ev_regrow", "ev_burn", "ev_delivery", "ev_cart_move", "ev_signal", "reward_components"):
            s[name] = jnp.zeros_like(state[name])
        s, actions = jax.lax.fori_loop(0, 128, self.actor_turn, (s, actions))
        s = jax.lax.fori_loop(0, 16, self.cart_turn, s)
        tree_distance = jnp.sum(jnp.abs(s["tree_pos"][:, None, :] - s["tree_pos"][None, :, :]), axis=2)
        wind = s["tree_pos"][None, :, 0] >= s["tree_pos"][:, None, 0]
        spread = jnp.any(s["fire"][:, None] & (tree_distance <= 4) & wind & ~jnp.eye(64, dtype=jnp.bool_), axis=0)
        spark = (s["t"] % 64 == 0) & (jnp.arange(64) == ((s["t"] // 64 - 1) * 7) % 64) & (s["height"] >= 3)
        dry = s["well_water"][s["tree_owner"]] < 6
        fire = (s["fire"] | ((spread | spark) & dry)) & ~s["firebreak"] & (s["height"] > 0)
        burning = fire & (s["t"] % 4 == 0)
        before = s["height"]
        s["height"] = jnp.maximum(0, before - burning.astype(jnp.int32))
        s["fire"] = fire & (s["height"] > 0)
        s["ev_burn"] = burning
        fertile = jnp.sum((tree_distance <= 6) & (s["height"][None, :] >= 3), axis=1) >= 2
        growing = (s["height"] > 0) & (s["height"] < 3) & ~s["fire"] & ~s["firebreak"] & fertile & (s["well_water"][s["tree_owner"]] > 0)
        age = jnp.where(growing, jnp.minimum(24, s["regrow"] + 1), 0)
        sprout = growing & (age >= 24)
        s["height"] = s["height"] + sprout.astype(jnp.int32)
        s["regrow"] = jnp.where(sprout, 0, age).astype(jnp.int32)
        s["ev_regrow"] = sprout & (s["height"] == 3)
        well_distance = jnp.sum(jnp.abs(s["well_pos"][:, None, :] - s["tree_pos"][None, :, :]), axis=2)
        cover = jnp.sum((well_distance <= 6) & (s["height"][None, :] >= 3), axis=1)
        recharge = (cover >= 2) & (s["t"] % 10 == 0)
        # Aquifer exchange is simultaneous along each four-village row, west to east.
        levels = s["well_water"].reshape((4, 4))
        flow = ((levels[:, :3] >= levels[:, 1:] + 2) & (levels[:, :3] > 0)).astype(jnp.int32)
        incoming = jnp.concatenate((jnp.zeros((4, 1), dtype=jnp.int32), flow), axis=1)
        outgoing = jnp.concatenate((flow, jnp.zeros((4, 1), dtype=jnp.int32)), axis=1)
        s["well_water"] = jnp.clip((levels + incoming - outgoing).reshape((16,)) + recharge.astype(jnp.int32), 0, 8)
        crop_fire = jnp.any((jnp.sum(jnp.abs(s["field_pos"][:, None, :] - s["tree_pos"][None, :, :]), axis=2) <= 3) & s["fire"][None, :], axis=1)
        crop_grow = (s["growth"] > 0) & (s["moisture"] > 0) & ~crop_fire & (s["t"] % 6 == 0)
        crop_loss = crop_fire & (s["t"] % 4 == 0)
        s["growth"] = jnp.clip(s["growth"] + crop_grow.astype(jnp.int32) - crop_loss.astype(jnp.int32), 0, 6)
        s["moisture"] = jnp.maximum(0, s["moisture"] - (s["t"] % 12 == 0).astype(jnp.int32))
        s["smoke"] = jnp.sum((jnp.sum(jnp.abs(s["pos"][:, None, :] - s["tree_pos"][None, :, :]), axis=2) <= 3) & s["fire"][None, :], axis=1).astype(jnp.int32)
        injury = (s["smoke"] > 0) & (s["t"] % 8 == 0) & (s["health"] > 0)
        s["health"] = jnp.maximum(0, s["health"] - injury.astype(jnp.int32))
        s["stamina"] = jnp.maximum(0, s["stamina"] - (s["smoke"] > 0).astype(jnp.int32))
        local_new = jnp.sum((s["crew"][:, None] == s["tree_owner"][None, :]) & s["ev_regrow"][None, :], axis=1)
        local_burn = jnp.sum((s["crew"][:, None] == s["tree_owner"][None, :]) & burning[None, :], axis=1)
        s["reward_components"] = s["reward_components"].at[:, 4].add(0.002 * local_new)
        s["reward_components"] = s["reward_components"].at[:, 5].set(-0.03 * injury - 0.02 * local_burn - 0.01 * crop_loss[s["crew"]])
        s["terminated"] = jnp.asarray(False)
        s["truncated"] = s["t"] >= 896
        s["done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        return s, jnp.sum(s["reward_components"], axis=1).astype(jnp.float32)

    def absorb(self, payload):
        s, actions = payload
        return s, jnp.zeros((128,), dtype=jnp.float32)

    def raw_step(self, key, state, actions):
        s, reward = jax.lax.cond(state["done"], self.absorb, self.advance, (state, actions))
        done = jnp.full((128,), s["done"], dtype=jnp.bool_)
        return self.observe(s), s, reward, done, {"done_all": s["done"],
                "terminated": s["terminated"], "truncated": s["truncated"]}
