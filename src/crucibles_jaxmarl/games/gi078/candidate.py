"""Thawforge Caravan. Original spatial cooperative manufacturing kernel.

Authored without executing candidate code. Integer resource accounting, explicit
raw terminal absorption, and public fixed-shape observations are intentional.
"""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 209
    num_actions = 12
    max_steps = 192

    def reset(self, key):
        grid = jnp.zeros((9, 11), dtype=jnp.int32)
        grid = grid.at[0, :].set(1).at[8, :].set(1)
        grid = grid.at[:, 0].set(1).at[:, 10].set(1)
        nodes = jnp.array([[1, 1], [1, 7], [9, 1], [9, 7], [5, 1], [5, 7]], dtype=jnp.int32)
        stations = jnp.array([[4, 4], [6, 4], [5, 3], [5, 5], [8, 4]], dtype=jnp.int32)
        grid = grid.at[nodes[:, 1], nodes[:, 0]].set(3)
        grid = grid.at[stations[:, 1], stations[:, 0]].set(2)
        s = {
            "t": jnp.array(0, dtype=jnp.int32),
            "done": jnp.array(False), "terminated": jnp.array(False),
            "truncated": jnp.array(False), "success": jnp.array(False),
            "map": grid,
            "agent_pos": jnp.array([[3, 3], [7, 3], [5, 6]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_visibility": jnp.ones((3, 3), dtype=jnp.bool_),
            "agent_role": jnp.array([0, 1, 2], dtype=jnp.int32),
            "agent_warmth": jnp.full((3,), 64, dtype=jnp.int32),
            "agent_bag": jnp.zeros((3, 5), dtype=jnp.int32),
            "agent_facing": jnp.array([[0, -1], [0, -1], [0, -1]], dtype=jnp.int32),
            "node_pos": nodes, "node_active": jnp.ones((6,), dtype=jnp.bool_),
            "node_visibility": jnp.ones((3, 6), dtype=jnp.bool_),
            "node_kind": jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32),
            "node_stock": jnp.full((6,), 8, dtype=jnp.int32),
            "station_pos": stations, "station_active": jnp.ones((5,), dtype=jnp.bool_),
            "station_visibility": jnp.ones((3, 5), dtype=jnp.bool_),
            "station_stock": jnp.zeros((5, 5), dtype=jnp.int32),
            "station_progress": jnp.zeros((5,), dtype=jnp.int32),
            "station_level": jnp.zeros((5,), dtype=jnp.int32),
            "station_heat": jnp.array([0, 0, 32, 0, 0], dtype=jnp.int32),
            "ev_harvest": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_load": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_take": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_pass": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_fire": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_craft": jnp.zeros((5,), dtype=jnp.bool_),
            "ev_upgrade": jnp.zeros((5,), dtype=jnp.bool_),
            "work_effort": jnp.zeros((5,), dtype=jnp.int32),
            "manufactured": jnp.array(0, dtype=jnp.int32),
        }
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate([
            jnp.stack([s["t"] / 192.0, jnp.mod(s["t"], 12) / 12.0, s["done"].astype(jnp.float32)]),
            s["map"].reshape(-1) / 3.0,
            s["agent_pos"].reshape(-1) / 10.0,
            s["agent_warmth"] / 80.0,
            s["agent_bag"].reshape(-1) / 6.0,
            s["agent_role"] / 2.0,
            s["node_pos"].reshape(-1) / 10.0,
            s["node_kind"] / 2.0, s["node_stock"] / 8.0,
            s["station_pos"].reshape(-1) / 10.0,
            s["station_stock"].reshape(-1) / 6.0,
            s["station_progress"] / 4.0,
            s["station_level"] / 2.0,
            s["station_heat"] / 60.0,
            s["agent_active"].astype(jnp.float32),
        ]).astype(jnp.float32)
        return jnp.concatenate([jnp.broadcast_to(public[None, :], (3, 206)), jnp.eye(3, dtype=jnp.float32)], axis=1)

    def moves(self, s):
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        target = s["agent_pos"][:, None, :] + delta[None, :, :]
        target = jnp.clip(target, jnp.array([0, 0]), jnp.array([10, 8]))
        occupied = jnp.any(jnp.all(target[:, :, None, :] == s["agent_pos"][None, None, :, :], axis=-1), axis=-1)
        ok = (s["map"][target[:, :, 1], target[:, :, 0]] == 0) & ~occupied
        return target, ok

    def load_options(self, s):
        st = s["station_stock"]
        demand = jnp.zeros((5, 5), dtype=jnp.bool_)
        demand = demand.at[0, 0].set(st[0, 0] < 2).at[0, 2].set(st[0, 2] < 1)
        demand = demand.at[1, 1].set(st[1, 1] < 2).at[1, 2].set(st[1, 2] < 1)
        demand = demand.at[2, 2].set(s["station_heat"][2] <= 48)
        demand = demand.at[3, 3].set((st[3, 3] < 1) & (s["station_level"][3] < 2))
        demand = demand.at[3, 4].set((st[3, 4] < 1) & (s["station_level"][3] < 2))
        demand = demand.at[4, 3].set(st[4, 3] < 3).at[4, 4].set(st[4, 4] < 2)
        near = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["station_pos"][None, :, :]), axis=-1) <= 1
        return near[:, :, None] & demand[None, :, :] & (s["agent_bag"][:, None, :] > 0)

    def work_options(self, s):
        st = s["station_stock"]
        ready = jnp.stack([
            (s["station_progress"][0] > 0) | ((st[0, 0] >= 2) & (st[0, 2] >= 1) & (st[0, 3] < 4)),
            (s["station_progress"][1] > 0) | ((st[1, 1] >= 2) & (st[1, 2] >= 1) & (st[1, 4] < 4)),
            jnp.array(False),
            (st[3, 3] >= 1) & (st[3, 4] >= 1) & (s["station_level"][3] < 2),
            (st[4, 3] >= 3) & (st[4, 4] >= 2),
        ])
        near = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["station_pos"][None, :, :]), axis=-1) <= 1
        return near & ready[None, :]

    def legal_actions(self, s):
        target, move = self.moves(s)
        free = jnp.sum(s["agent_bag"], axis=1) < 6
        harvest = jnp.any((jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["node_pos"][None, :, :]), axis=-1) <= 1) & (s["node_stock"][None, :] > 0), axis=1) & free
        near = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["station_pos"][None, :, :]), axis=-1) <= 1
        products = jnp.array([s["station_stock"][0, 3], s["station_stock"][1, 4], 0, 0, 0]) > 0
        take = jnp.any(near & products[None, :], axis=1) & free
        dist = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=-1)
        hand = jnp.any((dist == 1) & free[None, :], axis=1) & (jnp.sum(s["agent_bag"], axis=1) > 0)
        rest = (jnp.sum(jnp.abs(s["agent_pos"] - s["station_pos"][2]), axis=1) <= 2 + s["station_level"][3]) & (s["station_heat"][2] > 0)
        mask = jnp.concatenate([jnp.ones((3, 1), dtype=jnp.bool_), move,
            harvest[:, None], jnp.any(self.load_options(s), axis=(1, 2))[:, None], take[:, None],
            jnp.any(self.work_options(s), axis=1)[:, None], hand[:, None], rest[:, None],
            (s["agent_bag"][:, 2] > 0)[:, None]], axis=1)
        return jnp.where(s["done"], jnp.broadcast_to(jnp.arange(12) == 0, (3, 12)), mask)

    def absorb(self, key, s, actions):
        return self.observe(s), s, jnp.zeros((3,), dtype=jnp.float32), jnp.ones((3,), dtype=jnp.bool_), {
            "done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def advance(self, key, state, actions):
        s = dict(state)
        clipped = jnp.clip(actions, 0, 11)
        valid = (actions >= 0) & (actions < 12) & jnp.take_along_axis(self.legal_actions(s), clipped[:, None], axis=1)[:, 0]
        act = jnp.where(valid, clipped, 0)
        target, allowed = self.moves(s)
        chosen = target[jnp.arange(3), jnp.clip(act - 1, 0, 3)]
        moving = (act >= 1) & (act <= 4)
        priority = jnp.mod(jnp.arange(3) - s["t"], 3)
        same = jnp.all(chosen[:, None, :] == chosen[None, :, :], axis=-1)
        lose = jnp.any(same & moving[None, :] & (priority[None, :] < priority[:, None]), axis=1)
        moved = moving & ~lose
        oldpos = s["agent_pos"]
        s["agent_pos"] = jnp.where(moved[:, None], chosen, oldpos)
        s["agent_facing"] = jnp.where(moved[:, None], chosen - oldpos, s["agent_facing"])
        for field in ["ev_harvest", "ev_load", "ev_take", "ev_pass", "ev_fire"]:
            s[field] = jnp.zeros((3,), dtype=jnp.bool_)
        s["ev_craft"] = jnp.zeros((5,), dtype=jnp.bool_)
        s["ev_upgrade"] = jnp.zeros((5,), dtype=jnp.bool_)
        fire_heat = jnp.zeros((3,), dtype=jnp.int32)
        installed = jnp.array(0, dtype=jnp.int32)
        for offset in range(3):
            i = jnp.mod(s["t"] + offset, 3)
            free = 6 - jnp.sum(s["agent_bag"][i])
            nd = jnp.sum(jnp.abs(s["node_pos"] - s["agent_pos"][i]), axis=1)
            eligible = (nd <= 1) & (s["node_stock"] > 0)
            ni = jnp.argmax(eligible.astype(jnp.int32))
            kind = s["node_kind"][ni]
            amount = jnp.minimum(jnp.minimum(1 + (s["agent_role"][i] == kind).astype(jnp.int32), free), s["node_stock"][ni])
            amount = jnp.where((act[i] == 5) & jnp.any(eligible), amount, 0)
            s["node_stock"] = s["node_stock"].at[ni].add(-amount)
            s["agent_bag"] = s["agent_bag"].at[i, kind].add(amount)
            s["ev_harvest"] = s["ev_harvest"].at[i].set(amount > 0)
            options = self.load_options(s)[i].reshape(-1)
            flat = jnp.argmax(options.astype(jnp.int32))
            si, item = flat // 5, jnp.mod(flat, 5)
            load = (act[i] == 6) & jnp.any(options)
            fuel = load & (si == 2)
            stored = load & (si != 2)
            s["agent_bag"] = s["agent_bag"].at[i, item].add(-load.astype(jnp.int32))
            s["station_stock"] = s["station_stock"].at[si, item].add(stored.astype(jnp.int32))
            s["station_heat"] = s["station_heat"].at[2].add(12 * fuel.astype(jnp.int32))
            installed = installed + (load & (si == 4)).astype(jnp.int32)
            s["ev_load"] = s["ev_load"].at[i].set(load)
            near = jnp.sum(jnp.abs(s["station_pos"] - s["agent_pos"][i]), axis=1) <= 1
            outputs = jnp.array([s["station_stock"][0, 3], s["station_stock"][1, 4], 0, 0, 0])
            eligible = near & (outputs > 0)
            si = jnp.argmax(eligible.astype(jnp.int32))
            item = 3 + jnp.minimum(si, 1)
            take = (act[i] == 7) & jnp.any(eligible) & (jnp.sum(s["agent_bag"][i]) < 6)
            s["agent_bag"] = s["agent_bag"].at[i, item].add(take.astype(jnp.int32))
            s["station_stock"] = s["station_stock"].at[si, item].add(-take.astype(jnp.int32))
            s["ev_take"] = s["ev_take"].at[i].set(take)
            dist = jnp.sum(jnp.abs(s["agent_pos"] - s["agent_pos"][i]), axis=1)
            loads = jnp.sum(s["agent_bag"], axis=1)
            targets = (dist == 1) & (loads < 6)
            recipient = jnp.argmin(jnp.where(targets, loads, 99))
            item = jnp.argmax((s["agent_bag"][i] > 0).astype(jnp.int32))
            give = (act[i] == 9) & jnp.any(targets) & (loads[i] > 0)
            s["agent_bag"] = s["agent_bag"].at[i, item].add(-give.astype(jnp.int32))
            s["agent_bag"] = s["agent_bag"].at[recipient, item].add(give.astype(jnp.int32))
            s["ev_pass"] = s["ev_pass"].at[i].set(give)
            kindle = (act[i] == 11) & (s["agent_bag"][i, 2] > 0)
            s["agent_bag"] = s["agent_bag"].at[i, 2].add(-kindle.astype(jnp.int32))
            fire_heat = fire_heat + kindle.astype(jnp.int32) * jnp.where(jnp.arange(3) == i, 10, jnp.where(dist <= 1, 5, 0))
            s["ev_fire"] = s["ev_fire"].at[i].set(kindle)
        options = self.work_options(s)
        selected = jnp.argmax(options.astype(jnp.int32), axis=1)
        operating = (act == 8) & jnp.any(options, axis=1)
        workers = (selected[:, None] == jnp.arange(5)[None, :]) & operating[:, None]
        skill = jnp.ones((3, 5), dtype=jnp.int32)
        skill = skill.at[:, 0].add((s["agent_role"] == 2).astype(jnp.int32))
        skill = skill.at[:, 1].add((s["agent_role"] == 1).astype(jnp.int32))
        effort = jnp.sum(workers.astype(jnp.int32) * skill, axis=0)
        s["work_effort"] = effort
        for si in range(2):
            material = si
            output = si + 3
            stock = s["station_stock"]
            start = (s["station_progress"][si] == 0) & (stock[si, material] >= 2) & (stock[si, 2] >= 1) & (stock[si, output] < 4) & (effort[si] > 0)
            stock = stock.at[si, material].add(-2 * start.astype(jnp.int32))
            stock = stock.at[si, 2].add(-start.astype(jnp.int32))
            remaining = jnp.where(start, 4, s["station_progress"][si])
            next_remaining = jnp.maximum(0, remaining - effort[si])
            finished = (remaining > 0) & (next_remaining == 0)
            s["station_progress"] = s["station_progress"].at[si].set(next_remaining)
            s["station_stock"] = stock.at[si, output].add(finished.astype(jnp.int32))
            s["ev_craft"] = s["ev_craft"].at[si].set(finished)
        upgrade = (effort[3] > 0) & (s["station_stock"][3, 3] > 0) & (s["station_stock"][3, 4] > 0) & (s["station_level"][3] < 2)
        s["station_stock"] = s["station_stock"].at[3, 3].add(-upgrade.astype(jnp.int32)).at[3, 4].add(-upgrade.astype(jnp.int32))
        s["station_level"] = s["station_level"].at[3].add(upgrade.astype(jnp.int32))
        s["ev_upgrade"] = s["ev_upgrade"].at[3].set(upgrade)
        together = (jnp.sum(workers[:, 4]) >= 2) & (s["station_heat"][2] > 0)
        charge = jnp.where(together, jnp.minimum(3, s["station_progress"][4] + 1), jnp.maximum(0, s["station_progress"][4] - 1))
        s["station_progress"] = s["station_progress"].at[4].set(charge)
        level = s["station_level"][3]
        storm = jnp.mod(s["t"], 12) >= 8
        loss = 1 + storm.astype(jnp.int32) - ((level >= 1) & (jnp.mod(s["t"], 2) == 0)).astype(jnp.int32) - ((level >= 2) & storm).astype(jnp.int32)
        warm_zone = jnp.sum(jnp.abs(s["agent_pos"] - s["station_pos"][2]), axis=1) <= 2 + level
        burning = s["station_heat"][2] > 0
        warmth = s["agent_warmth"] - loss + fire_heat + (warm_zone & burning).astype(jnp.int32) * (3 + 5 * (act == 10).astype(jnp.int32))
        s["agent_warmth"] = jnp.clip(warmth, 0, 80)
        s["station_heat"] = s["station_heat"].at[2].set(jnp.maximum(0, s["station_heat"][2] - 1))
        s["agent_active"] = s["agent_warmth"] > 0
        s["node_active"] = s["node_stock"] > 0
        made = jnp.sum(s["ev_craft"].astype(jnp.int32))
        s["manufactured"] = s["manufactured"] + made
        success = (charge >= 3) & jnp.all(s["agent_active"])
        failed = jnp.any(~s["agent_active"])
        s["t"] = s["t"] + jnp.array(1, dtype=jnp.int32)
        s["success"] = success
        s["terminated"] = success | failed
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        reward = (-0.005 + 0.10 * made + 0.15 * installed + 0.20 * upgrade.astype(jnp.float32) + 3.0 * success.astype(jnp.float32) - 3.0 * failed.astype(jnp.float32)).astype(jnp.float32)
        return self.observe(s), s, jnp.full((3,), reward, dtype=jnp.float32), jnp.full((3,), s["done"], dtype=jnp.bool_), {
            "done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, key, state, actions)
