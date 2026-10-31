"""Icehouse Service: original spatial thermal kitchen; authored without running."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 14
    observation_size = 209
    max_steps = 180
    obs_keys = ("t", "done", "terminated", "truncated", "tiles", "agent_pos", "agent_active", "carry",
                "item_pos", "item_active", "item_kind", "holder", "food", "stage", "temperature", "age", "clock",
                "box_charge", "door_pos", "door_active", "door_open", "room_cooling", "compressor_fuel",
                "station_pos", "station_active", "station_kind", "stock", "served", "waste", "chilled",
                "door_changed", "served_now", "processed_now", "washed_now")

    def arrange(self, state):
        s = dict(state)
        held = s["holder"][None, :] == jnp.arange(3)[:, None]
        s["carry"] = jnp.where(jnp.any(held, axis=1), jnp.argmax(held, axis=1), -1).astype(jnp.int32)
        s["item_pos"] = jnp.where((s["holder"] >= 0)[:, None], s["agent_pos"][jnp.clip(s["holder"], 0, 2)], s["item_pos"])
        s["tiles"] = s["tiles"].at[1, 3].set(jnp.where(s["door_open"][0], 3, 2))
        return s

    def reset(self, key):
        tiles = jnp.zeros((7, 11), jnp.int32).at[:, 5].set(1).at[3, 5].set(4).at[0:3, 3].set(1).at[3, 0:3].set(1).at[1, 3].set(2)
        s = {"t": jnp.array(0, jnp.int32), "done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
             "tiles": tiles, "agent_pos": jnp.array([[4, 5], [1, 1], [4, 1]], jnp.int32), "agent_active": jnp.ones(3, jnp.bool_),
             "agent_visible": jnp.ones((3, 3), jnp.bool_), "carry": jnp.full(3, -1, jnp.int32),
             "item_pos": jnp.array([[4, 5], [1, 1], [7, 3], [4, 2], [2, 2]], jnp.int32),
             "item_active": jnp.ones(5, jnp.bool_), "item_visible": jnp.ones((3, 5), jnp.bool_),
             "item_kind": jnp.array([0, 0, 0, 1, 1], jnp.int32), "holder": jnp.full(5, -1, jnp.int32),
             "food": jnp.array([1, 2, 0, 0, 0], jnp.int32), "stage": jnp.array([1, 1, 0, 0, 0], jnp.int32),
             "temperature": jnp.array([3, 3, 0, 0, 0], jnp.int32), "age": jnp.zeros(5, jnp.int32), "clock": jnp.zeros(5, jnp.int32),
             "box_charge": jnp.array([0, 0, 0, 8, 8], jnp.int32), "door_pos": jnp.array([[3, 1]], jnp.int32),
             "door_active": jnp.ones(1, jnp.bool_), "door_visible": jnp.ones((3, 1), jnp.bool_), "door_open": jnp.array([False]),
             "room_cooling": jnp.array([20], jnp.int32), "compressor_fuel": jnp.array([4], jnp.int32),
             "station_pos": jnp.array([[4, 1], [4, 5], [1, 1], [3, 5], [9, 2], [9, 4], [1, 2]], jnp.int32),
             "station_active": jnp.ones(7, jnp.bool_), "station_visible": jnp.ones((3, 7), jnp.bool_), "station_kind": jnp.arange(7, dtype=jnp.int32),
             "stock": jnp.array([2, 2], jnp.int32), "served": jnp.zeros(2, jnp.int32), "waste": jnp.zeros(2, jnp.int32),
             "chilled": jnp.zeros(5, jnp.bool_), "door_changed": jnp.array(False), "served_now": jnp.zeros(2, jnp.int32),
             "processed_now": jnp.zeros(5, jnp.bool_), "washed_now": jnp.zeros(5, jnp.bool_)}
        s = self.arrange(s)
        return self.observe(s), s

    def walking(self, s):
        steps = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s["agent_pos"][:, None, :] + steps
        tile = s["tiles"][jnp.clip(p[:, :, 1], 0, 6), jnp.clip(p[:, :, 0], 0, 10)]
        bounds = jnp.all((p >= 0) & (p <= jnp.array([10, 6])), axis=2)
        boxes = (s["item_kind"] == 1) & (s["holder"] == -1)
        blocked_box = jnp.any(jnp.all(p[:, :, None, :] == s["item_pos"][None, None, :, :], axis=3) & boxes[None, None, :], axis=2)
        blocked_agent = jnp.any(jnp.all(p[:, :, None, :] == s["agent_pos"][None, None, :, :], axis=3) & ~jnp.eye(3, dtype=jnp.bool_)[:, None, :], axis=2)
        return p, (bounds & (tile != 1) & (tile != 2) & ~blocked_box & ~blocked_agent).at[:, 0].set(True)

    def legal_actions(self, s):
        p, walk = self.walking(s)
        pos = s["agent_pos"]
        carried = s["carry"]
        h = jnp.clip(carried, 0, 4)
        empty = carried < 0
        tray = (carried >= 0) & (s["item_kind"][h] == 0)
        box = (carried >= 0) & (s["item_kind"][h] == 1)
        distance = jnp.sum(jnp.abs(pos[:, None, :] - s["item_pos"][None, :, :]), axis=2)
        loose = s["holder"][None, :] == -1
        tray_here = loose & (distance == 0) & (s["item_kind"][None, :] == 0)
        box_near = loose & (distance <= 1) & (s["item_kind"][None, :] == 1)
        clear = ~jnp.any(loose & (distance == 0), axis=1)
        station = jnp.all(pos[:, None, :] == s["station_pos"][None, :, :], axis=2)
        mask = jnp.zeros((3, 14), jnp.bool_).at[:, :5].set(walk)
        mask = mask.at[:, 5].set((empty & jnp.any(tray_here & (s["stage"][None, :] != 2), axis=1)) | (tray & clear))
        mask = mask.at[:, 6].set((empty & jnp.any(box_near, axis=1)) | (box & clear))
        mask = mask.at[:, 7].set(tray & station[:, 0] & (s["stage"][h] == 0) & (s["stock"][0] > 0))
        mask = mask.at[:, 8].set(tray & station[:, 0] & (s["stage"][h] == 0) & (s["stock"][1] > 0))
        food = s["food"][h]
        process_site = ((food == 1) & station[:, 1]) | ((food == 2) & station[:, 2])
        process_stage = (s["stage"][h] == 1) | ((food == 1) & (s["stage"][h] == 3))
        mask = mask.at[:, 9].set(tray & process_site & process_stage & clear)
        serving = ((food == 1) & station[:, 4] & (s["temperature"][h] >= 5)) | ((food == 2) & station[:, 5] & (s["temperature"][h] <= 1))
        mask = mask.at[:, 10].set(tray & (s["stage"][h] == 3) & serving & (s["served"][jnp.clip(food - 1, 0, 1)] < 2))
        near_door = jnp.sum(jnp.abs(pos - s["door_pos"][0]), axis=1) <= 1
        door_clear = ~jnp.any(jnp.all(pos == s["door_pos"][0], axis=1)) & ~jnp.any((s["item_kind"] == 1) & (s["holder"] == -1) & jnp.all(s["item_pos"] == s["door_pos"][0], axis=1))
        mask = mask.at[:, 11].set(near_door & (~s["door_open"][0] | door_clear))
        wash = tray & station[:, 3] & ((s["stage"][h] == 4) | (s["stage"][h] == 5))
        crank = empty & station[:, 6] & (s["compressor_fuel"][0] > 0) & (s["room_cooling"][0] < 20)
        mask = mask.at[:, 12].set(wash | crank)
        in_room = (pos[:, 0] <= 2) & (pos[:, 1] <= 2)
        mask = mask.at[:, 13].set(box & in_room & (s["box_charge"][h] < 8) & (s["room_cooling"][0] > 0))
        return jnp.where(s["done"], jnp.arange(14)[None, :] == 0, mask)

    def observe(self, s):
        flat = jnp.concatenate([s[k].astype(jnp.float32).reshape(-1) for k in self.obs_keys])
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(flat, (3, self.observation_size - 3))), axis=1)

    def transition(self, payload):
        key, state, requested = payload
        s = dict(state)
        a = jnp.clip(requested, 0, 13)
        a = jnp.where((requested >= 0) & (requested < 14) & self.legal_actions(s)[jnp.arange(3), a], a, 0)
        options, allowed = self.walking(s)
        proposed = options[jnp.arange(3), jnp.where(a <= 4, a, 0)]
        rank = jnp.mod(jnp.arange(3) - s["t"], 3)
        collision = jnp.any(jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2) & (rank[None, :] < rank[:, None]), axis=1)
        s["agent_pos"] = jnp.where(collision[:, None], s["agent_pos"], proposed)
        s = self.arrange(s)
        s["door_changed"] = jnp.array(False)
        s["served_now"] = jnp.zeros(2, jnp.int32)
        s["processed_now"] = jnp.zeros(5, jnp.bool_)
        s["washed_now"] = jnp.zeros(5, jnp.bool_)
        for k in range(3):
            i = jnp.mod(s["t"] + k, 3)
            act = jnp.where(self.legal_actions(s)[i, a[i]], a[i], 0)
            has = s["carry"][i] >= 0
            h = jnp.clip(s["carry"][i], 0, 4)
            dist = jnp.sum(jnp.abs(s["item_pos"] - s["agent_pos"][i]), axis=1)
            eligible = (s["holder"] == -1) & jnp.where(act == 5, (s["item_kind"] == 0) & (dist == 0) & (s["stage"] != 2), (s["item_kind"] == 1) & (dist <= 1))
            j = jnp.argmax(eligible)
            handle = (act == 5) | (act == 6)
            take = handle & ~has
            drop = handle & has
            s["holder"] = s["holder"].at[j].set(jnp.where(take, i, s["holder"][j]))
            s["holder"] = s["holder"].at[h].set(jnp.where(drop, -1, s["holder"][h]))
            load = (act == 7) | (act == 8)
            typ = jnp.where(act == 7, 1, 2)
            s["stock"] = s["stock"].at[typ - 1].add(-load.astype(jnp.int32))
            s["food"] = s["food"].at[h].set(jnp.where(load, typ, s["food"][h]))
            s["stage"] = s["stage"].at[h].set(jnp.where(load, 1, s["stage"][h]))
            start = act == 9
            s["stage"] = s["stage"].at[h].set(jnp.where(start, 2, s["stage"][h]))
            s["holder"] = s["holder"].at[h].set(jnp.where(start, -1, s["holder"][h]))
            s["clock"] = s["clock"].at[h].set(jnp.where(start, 0, s["clock"][h]))
            serve = act == 10
            f = jnp.clip(s["food"][h] - 1, 0, 1)
            s["served_now"] = s["served_now"].at[f].add(serve.astype(jnp.int32))
            toggle = (act == 11) & ~s["door_changed"]
            s["door_open"] = jnp.where(toggle, ~s["door_open"], s["door_open"])
            s["door_changed"] = s["door_changed"] | toggle
            wash = (act == 12) & has
            crank = (act == 12) & ~has
            s["waste"] = s["waste"].at[f].add((wash & (s["food"][h] > 0)).astype(jnp.int32))
            s["compressor_fuel"] = s["compressor_fuel"] - crank.astype(jnp.int32)
            s["room_cooling"] = jnp.minimum(20, s["room_cooling"] + 5 * crank.astype(jnp.int32))
            s["food"] = s["food"].at[h].set(jnp.where(serve | wash, 0, s["food"][h]))
            s["stage"] = s["stage"].at[h].set(jnp.where(wash, 0, jnp.where(serve, 5, s["stage"][h])))
            s["age"] = s["age"].at[h].set(jnp.where(load | wash | serve, 0, s["age"][h]))
            s["temperature"] = s["temperature"].at[h].set(jnp.where(wash, 0, s["temperature"][h]))
            s["washed_now"] = s["washed_now"].at[h].set(s["washed_now"][h] | wash)
            transfer = jnp.where(act == 13, jnp.minimum(8 - s["box_charge"][h], s["room_cooling"][0]), 0)
            s["box_charge"] = s["box_charge"].at[h].add(transfer)
            s["room_cooling"] = s["room_cooling"] - transfer
            s = self.arrange(s)
        s["t"] = s["t"] + 1
        room = (s["item_pos"][:, 0] <= 2) & (s["item_pos"][:, 1] <= 2)
        near_box = jnp.sum(jnp.abs(s["item_pos"][:, None, :] - s["item_pos"][None, :, :]), axis=2) <= 1
        boxes = (s["item_kind"] == 1) & (s["box_charge"] > 0)
        cooled = ((room & (s["room_cooling"][0] > 0)) | jnp.any(near_box & boxes[None, :], axis=1)) & (s["item_kind"] == 0)
        s["chilled"] = cooled
        hot_work = (s["stage"] == 2) & (s["food"] == 1)
        ambient = s["temperature"] + (jnp.mod(s["t"], 3) == 0).astype(jnp.int32) * jnp.sign(3 - s["temperature"]).astype(jnp.int32)
        temp = jnp.where(hot_work, jnp.minimum(8, s["temperature"] + 2), jnp.where(cooled, jnp.maximum(0, s["temperature"] - 2), ambient))
        s["temperature"] = jnp.where(s["item_kind"] == 0, temp, 0)
        cold_work = (s["stage"] == 2) & (s["food"] == 2)
        progress = hot_work | (cold_work & cooled & (s["temperature"] <= 1))
        clocks = jnp.where(cold_work & ~progress, 0, s["clock"] + progress.astype(jnp.int32))
        ready = (s["stage"] == 2) & (clocks >= jnp.where(s["food"] == 1, 4, 3))
        s["clock"] = jnp.where(ready, 0, clocks)
        s["stage"] = jnp.where(ready, 3, s["stage"])
        s["processed_now"] = ready
        ages = (s["food"] > 0) & ((s["food"] == 1) | ~cooled)
        s["age"] = jnp.minimum(30, s["age"] + ages.astype(jnp.int32))
        s["stage"] = jnp.where((s["food"] > 0) & (s["age"] >= 30), 4, s["stage"])
        cooling_load = ((s["food"] > 0) | (s["temperature"] > 1)) & (s["item_kind"] == 0)
        used_box = boxes & jnp.any(near_box & cooling_load[:, None], axis=0)
        s["box_charge"] = s["box_charge"] - used_box.astype(jnp.int32)
        warm_ware = room & (s["item_kind"] == 0) & (s["temperature"] > 3)
        room_load = s["door_open"][0].astype(jnp.int32) + jnp.sum((room & cold_work).astype(jnp.int32)) + jnp.sum(warm_ware.astype(jnp.int32))
        s["room_cooling"] = jnp.maximum(0, s["room_cooling"] - room_load)
        s["served"] = s["served"] + s["served_now"]
        usable = (s["food"][None, :] == jnp.array([1, 2])[:, None]) & (s["stage"][None, :] >= 1) & (s["stage"][None, :] <= 3)
        failure = jnp.any(s["stock"] + jnp.sum(usable.astype(jnp.int32), axis=1) < 2 - s["served"])
        success = jnp.all(s["served"] >= 2) & ~failure
        s["terminated"] = success | failure
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        reward = jnp.full(3, -0.01 + jnp.sum(s["served_now"]) + 2.0 * success - failure.astype(jnp.float32), jnp.float32)
        return self.observe(s), s, reward, jnp.full(3, s["done"], jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def absorb(self, payload):
        key, s, actions = payload
        return self.observe(s), s, jnp.zeros(3, jnp.float32), jnp.ones(3, jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.transition, (key, state, actions))
