"""Meadow Rotations. Original rural ecology game; no author execution."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 12
    observation_size = 222
    max_steps = 128
    obs_keys = ("t", "done", "terminated", "truncated", "tiles", "agent_pos", "agent_active",
                "toolcarry", "milkcargo", "milkowner", "herd_pos", "herd_active", "herd_owner",
                "hunger", "thirst", "health", "milk", "gate_pos", "gate_active", "gate_open",
                "gate_wear", "plot_pos", "plot_active", "grass", "rest", "trough_pos", "trough_active",
                "water", "tool_pos", "tool_active", "tool_kind", "holder", "charge", "facility_pos",
                "facility_active", "facility_kind", "boards", "delivered", "gate_changed", "herd_moved",
                "fed", "drank", "produced", "delivered_now", "watered", "repaired", "milk_generated")

    def refresh(self, s):
        s = dict(s)
        held = s["holder"][None, :] == jnp.arange(3)[:, None]
        s["toolcarry"] = jnp.where(jnp.any(held, axis=1), jnp.argmax(held, axis=1), -1).astype(jnp.int32)
        s["tool_pos"] = jnp.where((s["holder"] >= 0)[:, None], s["agent_pos"][jnp.clip(s["holder"], 0, 2)], s["tool_pos"])
        s["gate_open"] = s["gate_open"] | (s["gate_wear"] == 0)
        s["tiles"] = s["tiles"].at[s["gate_pos"][:, 1], s["gate_pos"][:, 0]].set(jnp.where(s["gate_open"], 3, 2))
        return s

    def reset(self, key):
        tiles = jnp.zeros((7, 9), jnp.int32).at[:, 3].set(1).at[:, 6].set(1)
        s = {"t": jnp.array(0, jnp.int32), "done": jnp.array(False), "terminated": jnp.array(False),
             "truncated": jnp.array(False), "tiles": tiles,
             "agent_pos": jnp.array([[2, 2], [5, 4], [7, 4]], jnp.int32),
             "agent_active": jnp.ones(3, jnp.bool_), "agent_visible": jnp.ones((3, 3), jnp.bool_),
             "toolcarry": jnp.full(3, -1, jnp.int32), "milkcargo": jnp.zeros(3, jnp.int32), "milkowner": jnp.full(3, -1, jnp.int32),
             "herd_pos": jnp.array([[2, 2], [5, 4], [7, 2]], jnp.int32),
             "herd_active": jnp.ones(3, jnp.bool_), "herd_visible": jnp.ones((3, 3), jnp.bool_),
             "herd_owner": jnp.arange(3, dtype=jnp.int32), "hunger": jnp.full(3, 2, jnp.int32),
             "thirst": jnp.full(3, 2, jnp.int32), "health": jnp.full(3, 3, jnp.int32), "milk": jnp.zeros(3, jnp.int32),
             "gate_pos": jnp.array([[3, 2], [6, 4], [3, 5], [6, 1]], jnp.int32),
             "gate_active": jnp.ones(4, jnp.bool_), "gate_visible": jnp.ones((3, 4), jnp.bool_),
             "gate_open": jnp.array([False, False, True, True]), "gate_wear": jnp.full(4, 4, jnp.int32),
             "plot_pos": jnp.array([[1, 1], [1, 5], [4, 1], [4, 5], [7, 1], [7, 5]], jnp.int32),
             "plot_active": jnp.ones(6, jnp.bool_), "plot_visible": jnp.ones((3, 6), jnp.bool_),
             "grass": jnp.array([1, 6, 6, 2, 2, 6], jnp.int32), "rest": jnp.zeros(6, jnp.int32),
             "trough_pos": jnp.array([[2, 3], [7, 3]], jnp.int32), "trough_active": jnp.ones(2, jnp.bool_),
             "trough_visible": jnp.ones((3, 2), jnp.bool_), "water": jnp.full(2, 3, jnp.int32),
             "tool_pos": jnp.array([[2, 6], [5, 6]], jnp.int32), "tool_active": jnp.ones(2, jnp.bool_),
             "tool_visible": jnp.ones((3, 2), jnp.bool_), "tool_kind": jnp.array([0, 1], jnp.int32),
             "holder": jnp.full(2, -1, jnp.int32), "charge": jnp.array([0, 2], jnp.int32),
             "facility_pos": jnp.array([[0, 3], [4, 6]], jnp.int32), "facility_active": jnp.ones(2, jnp.bool_),
             "facility_visible": jnp.ones((3, 2), jnp.bool_), "facility_kind": jnp.array([0, 1], jnp.int32),
             "boards": jnp.array(3, jnp.int32), "delivered": jnp.zeros(3, jnp.int32),
             "gate_changed": jnp.zeros(4, jnp.bool_), "herd_moved": jnp.zeros(3, jnp.bool_),
             "fed": jnp.zeros(3, jnp.bool_), "drank": jnp.zeros(3, jnp.bool_), "produced": jnp.zeros(3, jnp.bool_),
             "delivered_now": jnp.zeros(3, jnp.int32), "watered": jnp.zeros(2, jnp.bool_),
             "repaired": jnp.zeros(4, jnp.bool_), "milk_generated": jnp.zeros(3, jnp.int32)}
        s = self.refresh(s)
        return self.observe(s), s

    def passable(self, s, p):
        bounds = jnp.all((p >= 0) & (p <= jnp.array([8, 6])), axis=-1)
        tile = s["tiles"][jnp.clip(p[..., 1], 0, 6), jnp.clip(p[..., 0], 0, 8)]
        return bounds & (tile != 1) & (tile != 2)

    def movement(self, s):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s["agent_pos"][:, None, :] + delta[None, :, :]
        occupied = jnp.any(jnp.all(p[:, :, None, :] == s["agent_pos"][None, None, :, :], axis=3)
                           & ~jnp.eye(3, dtype=jnp.bool_)[:, None, :], axis=2)
        return p, (self.passable(s, p) & ~occupied).at[:, 0].set(True)

    def legal_actions(self, s):
        p, move = self.movement(s)
        pos = s["agent_pos"]
        near_gate = jnp.sum(jnp.abs(pos[:, None, :] - s["gate_pos"][None, :, :]), axis=2) <= 1
        near_herd = jnp.sum(jnp.abs(pos[:, None, :] - s["herd_pos"][None, :, :]), axis=2) <= 1
        near_water = jnp.sum(jnp.abs(pos[:, None, :] - s["trough_pos"][None, :, :]), axis=2) <= 1
        loose = (s["holder"][None, :] == -1) & jnp.all(pos[:, None, :] == s["tool_pos"][None, :, :], axis=2)
        empty = (s["toolcarry"] < 0) & (s["milkcargo"] == 0)
        bucket = s["toolcarry"] == 0
        hammer = s["toolcarry"] == 1
        at_well = jnp.all(pos == s["facility_pos"][0], axis=1)
        at_barn = jnp.all(pos == s["facility_pos"][1], axis=1)
        mask = jnp.zeros((3, 12), jnp.bool_).at[:, :5].set(move)
        mask = mask.at[:, 5].set(jnp.any(near_gate & (s["gate_wear"][None, :] > 0), axis=1))
        mask = mask.at[:, 6].set(hammer & (s["charge"][1] > 0) & jnp.any(near_gate & (s["gate_wear"][None, :] < 6), axis=1))
        mask = mask.at[:, 7].set(empty & jnp.any(near_herd & (s["milk"][None, :] > 0), axis=1))
        mask = mask.at[:, 8].set(at_barn & (s["milkcargo"] > 0))
        mask = mask.at[:, 9].set((empty & jnp.any(loose, axis=1)) | ((s["toolcarry"] >= 0) & ~jnp.any(loose, axis=1)))
        mask = mask.at[:, 10].set((bucket & at_well & (s["charge"][0] < 3)) | (hammer & at_barn & (s["charge"][1] < 3) & (s["boards"] > 0)))
        mask = mask.at[:, 11].set(bucket & (s["charge"][0] > 0) & jnp.any(near_water & (s["water"][None, :] < 6), axis=1))
        return jnp.where(s["done"], jnp.arange(12)[None, :] == 0, mask)

    def observe(self, s):
        v = jnp.concatenate([s[k].astype(jnp.float32).reshape(-1) for k in self.obs_keys])
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(v[None, :], (3, self.observation_size - 3))), axis=1)

    def advance(self, payload):
        key, old, requested = payload
        s = dict(old)
        raw = jnp.clip(requested, 0, 11)
        actions = jnp.where((requested >= 0) & (requested < 12) & self.legal_actions(s)[jnp.arange(3), raw], raw, 0)
        candidates, available = self.movement(s)
        p = candidates[jnp.arange(3), jnp.where(actions <= 4, actions, 0)]
        rank = jnp.mod(jnp.arange(3) - s["t"], 3)
        clash = jnp.all(p[:, None, :] == p[None, :, :], axis=2)
        lose = jnp.any(clash & (rank[None, :] < rank[:, None]), axis=1)
        s["agent_pos"] = jnp.where(lose[:, None], s["agent_pos"], p)
        s = self.refresh(s)
        s["gate_changed"] = jnp.zeros(4, jnp.bool_)
        s["repaired"] = jnp.zeros(4, jnp.bool_)
        s["watered"] = jnp.zeros(2, jnp.bool_)
        s["delivered_now"] = jnp.zeros(3, jnp.int32)
        for k in range(3):
            i = jnp.mod(s["t"] + k, 3)
            near = jnp.sum(jnp.abs(s["gate_pos"] - s["agent_pos"][i]), axis=1) <= 1
            choices = near & (s["gate_wear"] > 0)
            g = jnp.argmax(choices)
            toggle = (actions[i] == 5) & jnp.any(choices) & ~s["gate_changed"][g]
            s["gate_open"] = s["gate_open"].at[g].set(jnp.where(toggle, ~s["gate_open"][g], s["gate_open"][g]))
            s["gate_changed"] = s["gate_changed"].at[g].set(s["gate_changed"][g] | toggle)
            repairs = near & (s["gate_wear"] < 6)
            g = jnp.argmax(repairs)
            repair = (actions[i] == 6) & jnp.any(repairs) & (s["charge"][1] > 0)
            s["gate_wear"] = s["gate_wear"].at[g].set(jnp.minimum(6, s["gate_wear"][g] + 3 * repair.astype(jnp.int32)))
            s["charge"] = s["charge"].at[1].add(-repair.astype(jnp.int32))
            s["repaired"] = s["repaired"].at[g].set(s["repaired"][g] | repair)
            cattle = (jnp.sum(jnp.abs(s["herd_pos"] - s["agent_pos"][i]), axis=1) <= 1) & (s["milk"] > 0)
            h = jnp.argmax(cattle)
            collect = (actions[i] == 7) & jnp.any(cattle)
            s["milk"] = s["milk"].at[h].add(-collect.astype(jnp.int32))
            s["milkcargo"] = s["milkcargo"].at[i].set(jnp.where(collect, 1, s["milkcargo"][i]))
            s["milkowner"] = s["milkowner"].at[i].set(jnp.where(collect, s["herd_owner"][h], s["milkowner"][i]))
            deliver = actions[i] == 8
            owner = jnp.clip(s["milkowner"][i], 0, 2)
            s["delivered_now"] = s["delivered_now"].at[owner].add(deliver.astype(jnp.int32))
            s["milkcargo"] = s["milkcargo"].at[i].set(jnp.where(deliver, 0, s["milkcargo"][i]))
            s["milkowner"] = s["milkowner"].at[i].set(jnp.where(deliver, -1, s["milkowner"][i]))
            has = jnp.any(s["holder"] == i)
            held = jnp.argmax(s["holder"] == i)
            loose = (s["holder"] == -1) & jnp.all(s["tool_pos"] == s["agent_pos"][i], axis=1)
            tool = jnp.argmax(loose)
            take = (actions[i] == 9) & ~has & jnp.any(loose)
            drop = (actions[i] == 9) & has & ~jnp.any(loose)
            s["holder"] = s["holder"].at[tool].set(jnp.where(take, i, s["holder"][tool]))
            s["holder"] = s["holder"].at[held].set(jnp.where(drop, -1, s["holder"][held]))
            s["tool_pos"] = s["tool_pos"].at[held].set(jnp.where(drop, s["agent_pos"][i], s["tool_pos"][held]))
            fill_bucket = (actions[i] == 10) & (s["toolcarry"][i] == 0)
            fill_hammer = (actions[i] == 10) & (s["toolcarry"][i] == 1)
            boards = jnp.where(fill_hammer, jnp.minimum(3 - s["charge"][1], s["boards"]), 0)
            s["charge"] = s["charge"].at[0].set(jnp.where(fill_bucket, 3, s["charge"][0])).at[1].add(boards)
            s["boards"] = s["boards"] - boards
            troughs = (jnp.sum(jnp.abs(s["trough_pos"] - s["agent_pos"][i]), axis=1) <= 1) & (s["water"] < 6)
            w = jnp.argmax(troughs)
            amount = jnp.where((actions[i] == 11) & jnp.any(troughs), jnp.minimum(s["charge"][0], 6 - s["water"][w]), 0)
            s["charge"] = s["charge"].at[0].add(-amount)
            s["water"] = s["water"].at[w].add(amount)
            s["watered"] = s["watered"].at[w].set(s["watered"][w] | (amount > 0))
        s = self.refresh(s)
        s["t"] = s["t"] + 1
        tick = jnp.mod(s["t"], 2) == 0
        s["hunger"] = jnp.minimum(8, s["hunger"] + tick.astype(jnp.int32))
        s["thirst"] = jnp.minimum(8, s["thirst"] + (jnp.mod(s["t"], 3) == 0).astype(jnp.int32))
        s["herd_moved"] = jnp.zeros(3, jnp.bool_)
        s["fed"] = jnp.zeros(3, jnp.bool_)
        s["drank"] = jnp.zeros(3, jnp.bool_)
        delta = jnp.array([[0, 0], [1, 0], [0, -1], [-1, 0], [0, 1]], jnp.int32)
        for k in range(3):
            h = jnp.mod(s["t"] + k, 3)
            here = s["herd_pos"][h]
            grass_score = 2 * s["grass"] - jnp.sum(jnp.abs(s["plot_pos"] - here), axis=1)
            water_score = 3 * s["water"] - jnp.sum(jnp.abs(s["trough_pos"] - here), axis=1)
            goal = jnp.where(s["thirst"][h] >= 4, s["trough_pos"][jnp.argmax(water_score)], s["plot_pos"][jnp.argmax(grass_score)])
            q = here + delta
            occupied = jnp.any(jnp.all(q[:, None, :] == s["herd_pos"][None, :, :], axis=2) & (jnp.arange(3)[None, :] != h), axis=1)
            can = (self.passable(s, q) & ~occupied).at[0].set(True)
            costs = jnp.sum(jnp.abs(q - goal), axis=1) + jnp.where(can, 0, 100)
            dest = jnp.where(tick, q[jnp.argmin(costs)], here)
            moved = jnp.any(dest != here)
            s["herd_pos"] = s["herd_pos"].at[h].set(dest)
            s["herd_moved"] = s["herd_moved"].at[h].set(moved)
            gate_entry = jnp.all(s["gate_pos"] == dest, axis=1) & moved
            s["gate_wear"] = jnp.maximum(0, s["gate_wear"] - gate_entry.astype(jnp.int32))
            plots = jnp.all(s["plot_pos"] == dest, axis=1) & (s["grass"] > 0)
            g = jnp.argmax(plots)
            eat = jnp.any(plots) & (s["hunger"][h] > 0)
            s["grass"] = s["grass"].at[g].add(-eat.astype(jnp.int32))
            s["hunger"] = s["hunger"].at[h].set(jnp.maximum(0, s["hunger"][h] - 3 * eat.astype(jnp.int32)))
            s["fed"] = s["fed"].at[h].set(eat)
            troughs = jnp.all(s["trough_pos"] == dest, axis=1) & (s["water"] > 0)
            w = jnp.argmax(troughs)
            drink = jnp.any(troughs) & (s["thirst"][h] > 0)
            s["water"] = s["water"].at[w].add(-drink.astype(jnp.int32))
            s["thirst"] = s["thirst"].at[h].set(jnp.maximum(0, s["thirst"][h] - 4 * drink.astype(jnp.int32)))
            s["drank"] = s["drank"].at[h].set(drink)
        occupied_plot = jnp.any(jnp.sum(jnp.abs(s["plot_pos"][:, None, :] - s["herd_pos"][None, :, :]), axis=2) <= 1, axis=1)
        s["rest"] = jnp.where(occupied_plot, 0, jnp.minimum(2, s["rest"] + 1))
        s["grass"] = jnp.minimum(6, s["grass"] + (s["rest"] >= 2).astype(jnp.int32))
        season_tick = jnp.mod(s["t"], 4) == 0
        starved = season_tick & ((s["hunger"] >= 8) | (s["thirst"] >= 8))
        s["health"] = jnp.maximum(0, s["health"] - starved.astype(jnp.int32))
        s["produced"] = season_tick & (s["hunger"] <= 3) & (s["thirst"] <= 3) & (s["health"] > 0) & (s["milk"] < 3)
        s["milk"] = s["milk"] + s["produced"].astype(jnp.int32)
        s["milk_generated"] = s["milk_generated"] + s["produced"].astype(jnp.int32)
        s["delivered"] = s["delivered"] + s["delivered_now"]
        failure = jnp.any(s["health"] == 0)
        success = jnp.all(s["delivered"] >= 2) & ~failure
        s["terminated"] = success | failure
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        reward = -0.01 + 0.2 * jnp.sum(s["delivered_now"]) + s["delivered_now"].astype(jnp.float32) + 2.0 * success - 2.0 * failure
        return self.observe(s), s, reward.astype(jnp.float32), jnp.full(3, s["done"], jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def absorb(self, payload):
        key, s, actions = payload
        return self.observe(s), s, jnp.zeros(3, jnp.float32), jnp.ones(3, jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, actions))
