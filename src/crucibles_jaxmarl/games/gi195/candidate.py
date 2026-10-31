"""Clockwork Banquet. Original prospective source, authored without execution."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 11
    observation_size = 150
    max_steps = 160
    obs_keys = ("t", "done", "terminated", "truncated", "tiles", "agent_pos", "agent_active",
                "carrying", "tray_pos", "tray_active", "holder", "food", "stage", "timer",
                "station_pos", "station_active", "station_kind", "stock", "direction", "course",
                "served", "waste", "tray_moved", "served_now", "loaded_now", "washed_now", "reversed")

    def refresh(self, s):
        s = dict(s)
        held = s["holder"][None, :] == jnp.arange(3)[:, None]
        s["carrying"] = jnp.where(jnp.any(held, axis=1), jnp.argmax(held, axis=1), -1).astype(jnp.int32)
        s["tray_pos"] = jnp.where((s["holder"] >= 0)[:, None], s["agent_pos"][jnp.clip(s["holder"], 0, 2)], s["tray_pos"])
        return s

    def reset(self, key):
        stations = jnp.array([[1, 1], [7, 5], [3, 1], [5, 1], [1, 5], [4, 5], [7, 1]], jnp.int32)
        tiles = jnp.zeros((7, 9), jnp.int32).at[3, 1:8].set(1)
        tiles = tiles.at[stations[:, 1], stations[:, 0]].set(jnp.arange(2, 9, dtype=jnp.int32))
        s = {"t": jnp.array(0, jnp.int32), "done": jnp.array(False), "terminated": jnp.array(False),
             "truncated": jnp.array(False), "tiles": tiles,
             "agent_pos": jnp.array([[1, 3], [7, 3], [4, 5]], jnp.int32),
             "agent_active": jnp.ones(3, jnp.bool_), "agent_visible": jnp.ones((3, 3), jnp.bool_),
             "carrying": jnp.full(3, -1, jnp.int32),
             "tray_pos": jnp.array([[1, 3], [4, 3], [7, 3]], jnp.int32),
             "tray_active": jnp.ones(3, jnp.bool_), "tray_visible": jnp.ones((3, 3), jnp.bool_),
             "holder": jnp.full(3, -1, jnp.int32), "food": jnp.zeros(3, jnp.int32),
             "stage": jnp.zeros(3, jnp.int32), "timer": jnp.zeros(3, jnp.int32),
             "station_pos": stations, "station_active": jnp.ones(7, jnp.bool_),
             "station_visible": jnp.ones((3, 7), jnp.bool_), "station_kind": jnp.arange(7, dtype=jnp.int32),
             "stock": jnp.array([3, 3], jnp.int32), "direction": jnp.array([1], jnp.int32),
             "course": jnp.array(0, jnp.int32), "served": jnp.zeros(2, jnp.int32),
             "waste": jnp.zeros(2, jnp.int32), "tray_moved": jnp.zeros(3, jnp.bool_),
             "served_now": jnp.array(False), "loaded_now": jnp.zeros(3, jnp.bool_),
             "washed_now": jnp.zeros(3, jnp.bool_), "reversed": jnp.array(False)}
        return self.observe(s), s

    def movement(self, s):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s["agent_pos"][:, None, :] + delta[None, :, :]
        bounds = jnp.all((p >= 0) & (p <= jnp.array([8, 6])), axis=2)
        occupied = jnp.any(jnp.all(p[:, :, None, :] == s["agent_pos"][None, None, :, :], axis=3)
                           & ~jnp.eye(3, dtype=jnp.bool_)[:, None, :], axis=2)
        return p, (bounds & ~occupied).at[:, 0].set(True)

    def legal_actions(self, s):
        p, move = self.movement(s)
        carry = s["carrying"]
        h = jnp.clip(carry, 0, 2)
        free = carry < 0
        here = jnp.all(s["agent_pos"][:, None, :] == s["tray_pos"][None, :, :], axis=2) & (s["holder"][None, :] == -1)
        at_station = jnp.all(s["agent_pos"][:, None, :] == s["station_pos"][None, :, :], axis=2)
        pantry = at_station[:, :2] & (s["stock"][None, :] > 0)
        food = s["food"][h]
        mask = jnp.zeros((3, 11), jnp.bool_).at[:, :5].set(move)
        toggle = (free & jnp.any(here & (s["stage"][None, :] != 2), axis=1)) | (~free & ~jnp.any(here, axis=1))
        mask = mask.at[:, 5].set(toggle)
        mask = mask.at[:, 6].set(~free & (s["stage"][h] == 0) & jnp.any(pantry, axis=1))
        stove = at_station[jnp.arange(3), jnp.clip(food + 1, 2, 3)]
        mask = mask.at[:, 7].set(~free & (s["stage"][h] == 1) & stove & ~jnp.any(here, axis=1))
        mask = mask.at[:, 8].set(at_station[:, 5])
        mask = mask.at[:, 9].set(~free & at_station[:, 4] & ((s["stage"][h] == 4) | (s["stage"][h] == 5)))
        demand = 1 + jnp.mod(s["course"], 2)
        mask = mask.at[:, 10].set(~free & at_station[:, 6] & (s["stage"][h] == 3) & (food == demand) & (s["course"] < 4))
        return jnp.where(s["done"], jnp.arange(11)[None, :] == 0, mask)

    def observe(self, s):
        v = jnp.concatenate([s[k].astype(jnp.float32).reshape(-1) for k in self.obs_keys])
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(v[None, :], (3, self.observation_size - 3))), axis=1)

    def advance(self, payload):
        key, old, actions = payload
        s = dict(old)
        raw = jnp.clip(actions, 0, 10)
        actions = jnp.where((actions >= 0) & (actions < 11) & self.legal_actions(s)[jnp.arange(3), raw], raw, 0)
        candidates, available = self.movement(s)
        p = candidates[jnp.arange(3), jnp.where(actions <= 4, actions, 0)]
        rank = jnp.mod(jnp.arange(3) - s["t"], 3)
        clash = jnp.all(p[:, None, :] == p[None, :, :], axis=2)
        lose = jnp.any(clash & (rank[None, :] < rank[:, None]), axis=1)
        s["agent_pos"] = jnp.where(lose[:, None], s["agent_pos"], p)
        s = self.refresh(s)
        s["served_now"] = jnp.array(False)
        s["loaded_now"] = jnp.zeros(3, jnp.bool_)
        s["washed_now"] = jnp.zeros(3, jnp.bool_)
        s["reversed"] = jnp.any(actions == 8)
        s["direction"] = jnp.where(s["reversed"], -s["direction"], s["direction"])
        for k in range(3):
            i = jnp.mod(s["t"] + k, 3)
            has = jnp.any(s["holder"] == i)
            h = jnp.argmax(s["holder"] == i)
            loose = (s["holder"] == -1) & jnp.all(s["tray_pos"] == s["agent_pos"][i], axis=1)
            pickable = loose & (s["stage"] != 2)
            picked = jnp.argmax(pickable)
            take = (actions[i] == 5) & ~has & jnp.any(pickable)
            drop = (actions[i] == 5) & has & ~jnp.any(loose)
            s["holder"] = s["holder"].at[picked].set(jnp.where(take, i, s["holder"][picked]))
            s["holder"] = s["holder"].at[h].set(jnp.where(drop, -1, s["holder"][h]))
            s["tray_pos"] = s["tray_pos"].at[h].set(jnp.where(drop, s["agent_pos"][i], s["tray_pos"][h]))
            pantry = jnp.argmax(jnp.all(s["station_pos"][:2] == s["agent_pos"][i], axis=1))
            load = (actions[i] == 6) & has & (s["stock"][pantry] > 0)
            s["stock"] = s["stock"].at[pantry].add(-load.astype(jnp.int32))
            s["food"] = s["food"].at[h].set(jnp.where(load, pantry + 1, s["food"][h]))
            s["stage"] = s["stage"].at[h].set(jnp.where(load, 1, s["stage"][h]))
            s["loaded_now"] = s["loaded_now"].at[h].set(s["loaded_now"][h] | load)
            cook = (actions[i] == 7) & has & ~jnp.any(loose)
            s["stage"] = s["stage"].at[h].set(jnp.where(cook, 2, s["stage"][h]))
            s["holder"] = s["holder"].at[h].set(jnp.where(cook, -1, s["holder"][h]))
            s["tray_pos"] = s["tray_pos"].at[h].set(jnp.where(cook, s["agent_pos"][i], s["tray_pos"][h]))
            wash = (actions[i] == 9) & has
            wasted = wash & (s["food"][h] > 0)
            food_index = jnp.clip(s["food"][h] - 1, 0, 1)
            s["waste"] = s["waste"].at[food_index].add(wasted.astype(jnp.int32))
            serve = (actions[i] == 10) & has & (s["food"][h] == 1 + jnp.mod(s["course"], 2)) & (s["course"] < 4)
            s["served"] = s["served"].at[food_index].add(serve.astype(jnp.int32))
            s["course"] = s["course"] + serve.astype(jnp.int32)
            s["served_now"] = s["served_now"] | serve
            s["food"] = s["food"].at[h].set(jnp.where(wash | serve, 0, s["food"][h]))
            s["stage"] = s["stage"].at[h].set(jnp.where(wash, 0, jnp.where(serve, 4, s["stage"][h])))
            s["timer"] = s["timer"].at[h].set(jnp.where(load | cook | wash | serve, 0, s["timer"][h]))
            s["washed_now"] = s["washed_now"].at[h].set(s["washed_now"][h] | wash)
        cooking = s["stage"] == 2
        ready = s["stage"] == 3
        timers = s["timer"] + (cooking | ready).astype(jnp.int32)
        cooked = cooking & (timers >= 3)
        spoiled = ready & (timers >= 12)
        s["stage"] = jnp.where(cooked, 3, jnp.where(spoiled, 5, s["stage"]))
        s["timer"] = jnp.where(cooked | spoiled, 0, timers)
        s["t"] = s["t"] + 1
        on_belt = (s["holder"] == -1) & (s["tray_pos"][:, 1] == 3) & (s["tray_pos"][:, 0] >= 1) & (s["tray_pos"][:, 0] <= 7)
        target = s["tray_pos"] + jnp.array([s["direction"][0], 0], jnp.int32)
        blocked = jnp.any(jnp.all(target[:, None, :] == s["tray_pos"][None, :, :], axis=2) & (s["holder"][None, :] == -1), axis=1)
        belt_move = on_belt & (jnp.mod(s["t"], 2) == 0) & (target[:, 0] >= 1) & (target[:, 0] <= 7) & ~blocked
        s["tray_moved"] = belt_move
        s["tray_pos"] = jnp.where(belt_move[:, None], target, s["tray_pos"])
        success = s["course"] == 4
        demand = 1 + jnp.mod(s["course"], 2)
        usable = (s["food"] == demand) & (s["stage"] >= 1) & (s["stage"] <= 3)
        failure = ~success & (s["stock"][demand - 1] == 0) & ~jnp.any(usable)
        s["terminated"] = success | failure
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        value = -0.01 + s["served_now"].astype(jnp.float32) + 2.0 * success - failure.astype(jnp.float32)
        reward = jnp.full(3, value, jnp.float32)
        return self.observe(s), s, reward, jnp.full(3, s["done"], jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def absorb(self, payload):
        key, s, actions = payload
        return self.observe(s), s, jnp.zeros(3, jnp.float32), jnp.ones(3, jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, actions))
