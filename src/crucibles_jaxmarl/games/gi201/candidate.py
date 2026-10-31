"""Canopy Commons: original orchard, shade and shared water game; unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 13
    observation_size = 443
    max_steps = 160
    obs_keys = ("t", "done", "terminated", "truncated", "sun", "base_tiles", "tiles", "shade_map",
                "agent_pos", "agent_active", "seed_carry", "tool_carry", "fruit_carry", "fruit_owner",
                "plant_pos", "plant_active", "plant_owner", "plant_holder", "bed_index", "stage", "moisture",
                "stress", "growth", "shaded", "fruit_generated", "bed_pos", "bed_active", "bed_occupant",
                "tool_pos", "tool_active", "tool_kind", "tool_holder", "orientation", "water_cart",
                "spring_pos", "spring_active", "spring_water", "market_pos", "market_active", "market_kind",
                "private", "community", "fruit_lost", "planted_now", "watered_now", "ripe_now", "wilted_now",
                "rotated_now", "private_now", "community_now")

    def update_geometry(self, state):
        s = dict(state)
        for i in range(3):
            seeds = s["plant_holder"] == i
            tools = s["tool_holder"] == i
            s["seed_carry"] = s["seed_carry"].at[i].set(jnp.where(jnp.any(seeds), jnp.argmax(seeds), -1))
            s["tool_carry"] = s["tool_carry"].at[i].set(jnp.where(jnp.any(tools), jnp.argmax(tools), -1))
        s["plant_pos"] = jnp.where((s["plant_holder"] >= 0)[:, None], s["agent_pos"][jnp.clip(s["plant_holder"], 0, 2)], s["plant_pos"])
        s["tool_pos"] = jnp.where((s["tool_holder"] >= 0)[:, None], s["agent_pos"][jnp.clip(s["tool_holder"], 0, 2)], s["tool_pos"])
        xx = jnp.arange(11)[None, :]
        yy = jnp.arange(7)[:, None]
        screens = (s["tool_kind"] == 0) & (s["tool_holder"] == -1)
        footprint = (xx[None, :, :] == (s["tool_pos"][:, 0] + s["orientation"])[:, None, None]) & (jnp.abs(yy[None, :, :] - s["tool_pos"][:, 1, None, None]) <= 2)
        s["shade_map"] = jnp.any(footprint & screens[:, None, None], axis=0)
        s["tiles"] = jnp.where(s["shade_map"] & (s["base_tiles"] < 2), s["base_tiles"] + 2, s["base_tiles"])
        s["shaded"] = s["shade_map"][s["plant_pos"][:, 1], s["plant_pos"][:, 0]] & (s["bed_index"] >= 0)
        assigned = s["bed_index"][:, None] == jnp.arange(6)[None, :]
        s["bed_occupant"] = jnp.where(jnp.any(assigned, axis=0), jnp.argmax(assigned, axis=0), -1).astype(jnp.int32)
        s["sun"] = jnp.mod(s["t"] // 4, 4)
        return s

    def reset(self, key):
        base = jnp.zeros((7, 11), jnp.int32).at[3, :].set(1).at[0:2, 4].set(4).at[5:7, 6].set(4)
        s = {"t": jnp.array(0, jnp.int32), "done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
             "sun": jnp.array(0, jnp.int32), "base_tiles": base, "tiles": base, "shade_map": jnp.zeros((7, 11), jnp.bool_),
             "agent_pos": jnp.array([[3, 3], [5, 3], [1, 5]], jnp.int32), "agent_active": jnp.ones(3, jnp.bool_),
             "agent_visible": jnp.ones((3, 3), jnp.bool_), "seed_carry": jnp.full(3, -1, jnp.int32), "tool_carry": jnp.full(3, -1, jnp.int32),
             "fruit_carry": jnp.zeros(3, jnp.int32), "fruit_owner": jnp.full(3, -1, jnp.int32),
             "plant_pos": jnp.array([[3, 2], [5, 2], [5, 4], [0, 5], [5, 0], [10, 5]], jnp.int32),
             "plant_active": jnp.ones(6, jnp.bool_), "plant_visible": jnp.ones((3, 6), jnp.bool_), "plant_owner": jnp.array([0, 1, 2, 0, 1, 2], jnp.int32),
             "plant_holder": jnp.full(6, -1, jnp.int32), "bed_index": jnp.array([0, 1, 4, -1, -1, -1], jnp.int32),
             "stage": jnp.array([1, 1, 2, 0, 0, 0], jnp.int32), "moisture": jnp.array([4, 4, 4, 0, 0, 0], jnp.int32),
             "stress": jnp.zeros(6, jnp.int32), "growth": jnp.zeros(6, jnp.int32), "shaded": jnp.zeros(6, jnp.bool_), "fruit_generated": jnp.zeros(6, jnp.int32),
             "bed_pos": jnp.array([[3, 2], [5, 2], [7, 2], [3, 4], [5, 4], [7, 4]], jnp.int32), "bed_active": jnp.ones(6, jnp.bool_),
             "bed_visible": jnp.ones((3, 6), jnp.bool_), "bed_occupant": jnp.full(6, -1, jnp.int32),
             "tool_pos": jnp.array([[4, 3], [8, 3], [1, 5]], jnp.int32), "tool_active": jnp.ones(3, jnp.bool_), "tool_visible": jnp.ones((3, 3), jnp.bool_),
             "tool_kind": jnp.array([0, 0, 1], jnp.int32), "tool_holder": jnp.full(3, -1, jnp.int32), "orientation": jnp.array([1, -1, 0], jnp.int32),
             "water_cart": jnp.zeros(3, jnp.int32), "spring_pos": jnp.array([[0, 3], [10, 3]], jnp.int32), "spring_active": jnp.ones(2, jnp.bool_),
             "spring_visible": jnp.ones((3, 2), jnp.bool_), "spring_water": jnp.full(2, 6, jnp.int32),
             "market_pos": jnp.array([[1, 6], [5, 6], [9, 6], [5, 0]], jnp.int32), "market_active": jnp.ones(4, jnp.bool_),
             "market_visible": jnp.ones((3, 4), jnp.bool_), "market_kind": jnp.array([0, 1, 2, 3], jnp.int32),
             "private": jnp.zeros(3, jnp.int32), "community": jnp.zeros(3, jnp.int32), "fruit_lost": jnp.zeros(3, jnp.int32),
             "planted_now": jnp.zeros(6, jnp.bool_), "watered_now": jnp.zeros(6, jnp.bool_), "ripe_now": jnp.zeros(6, jnp.bool_),
             "wilted_now": jnp.zeros(6, jnp.bool_), "rotated_now": jnp.zeros(3, jnp.bool_), "private_now": jnp.zeros(3, jnp.int32),
             "community_now": jnp.zeros(3, jnp.int32)}
        s = self.update_geometry(s)
        return self.observe(s), s

    def walking(self, s):
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s["agent_pos"][:, None, :] + directions
        tile = s["tiles"][jnp.clip(p[:, :, 1], 0, 6), jnp.clip(p[:, :, 0], 0, 10)]
        inside = jnp.all((p >= 0) & (p <= jnp.array([10, 6])), axis=2)
        tools = jnp.any(jnp.all(p[:, :, None, :] == s["tool_pos"][None, None, :, :], axis=3) & (s["tool_holder"][None, None, :] == -1), axis=2)
        peers = jnp.any(jnp.all(p[:, :, None, :] == s["agent_pos"][None, None, :, :], axis=3) & ~jnp.eye(3, dtype=jnp.bool_)[:, None, :], axis=2)
        return p, (inside & (tile != 4) & ~tools & ~peers).at[:, 0].set(True)

    def legal_actions(self, s):
        p, walk = self.walking(s)
        pos = s["agent_pos"]
        empty = (s["seed_carry"] < 0) & (s["tool_carry"] < 0) & (s["fruit_carry"] == 0)
        plant_dist = jnp.sum(jnp.abs(pos[:, None, :] - s["plant_pos"][None, :, :]), axis=2)
        tool_dist = jnp.sum(jnp.abs(pos[:, None, :] - s["tool_pos"][None, :, :]), axis=2)
        loose_seed = (s["stage"] == 0) & (s["plant_holder"] == -1)
        loose_tool = s["tool_holder"] == -1
        mask = jnp.zeros((3, 13), jnp.bool_).at[:, :5].set(walk)
        mask = mask.at[:, 5].set((empty & jnp.any((plant_dist <= 1) & loose_seed[None, :], axis=1)) | (s["seed_carry"] >= 0))
        mask = mask.at[:, 6].set((empty & jnp.any((tool_dist <= 1) & loose_tool[None, :], axis=1)) | ((s["tool_carry"] >= 0) & ~jnp.any((tool_dist == 0) & loose_tool[None, :], axis=1)))
        mask = mask.at[:, 7].set(jnp.any((tool_dist <= 1) & loose_tool[None, :] & (s["tool_kind"][None, :] == 0), axis=1))
        near_bed = jnp.sum(jnp.abs(pos[:, None, :] - s["bed_pos"][None, :, :]), axis=2) <= 1
        mask = mask.at[:, 8].set((s["seed_carry"] >= 0) & jnp.any(near_bed & (s["bed_occupant"][None, :] < 0), axis=1))
        held = jnp.clip(s["tool_carry"], 0, 2)
        cart = (s["tool_carry"] >= 0) & (s["tool_kind"][held] == 1)
        growing = (s["stage"] >= 1) & (s["stage"] <= 3)
        mask = mask.at[:, 9].set(cart & (s["water_cart"][held] > 0) & jnp.any((plant_dist <= 1) & growing[None, :] & (s["moisture"][None, :] < 5), axis=1))
        mask = mask.at[:, 10].set(empty & jnp.any((plant_dist <= 1) & (s["stage"][None, :] == 3), axis=1))
        owner = jnp.clip(s["fruit_owner"], 0, 2)
        private = jnp.all(pos == s["market_pos"][owner], axis=1)
        public = jnp.all(pos == s["market_pos"][3], axis=1)
        mask = mask.at[:, 11].set((s["fruit_carry"] > 0) & (private | public))
        spring = jnp.sum(jnp.abs(pos[:, None, :] - s["spring_pos"][None, :, :]), axis=2) <= 1
        mask = mask.at[:, 12].set(cart & (s["water_cart"][held] < 4) & jnp.any(spring & (s["spring_water"][None, :] > 0), axis=1))
        return jnp.where(s["done"], jnp.arange(13)[None, :] == 0, mask)

    def observe(self, s):
        flat = jnp.concatenate([s[k].astype(jnp.float32).reshape(-1) for k in self.obs_keys])
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(flat, (3, self.observation_size - 3))), axis=1)

    def transition(self, payload):
        key, state, requested = payload
        s = dict(state)
        a = jnp.clip(requested, 0, 12)
        a = jnp.where((requested >= 0) & (requested < 13) & self.legal_actions(s)[jnp.arange(3), a], a, 0)
        options, allowed = self.walking(s)
        target = options[jnp.arange(3), jnp.where(a <= 4, a, 0)]
        priority = jnp.mod(jnp.arange(3) - s["t"], 3)
        blocked = jnp.any(jnp.all(target[:, None, :] == target[None, :, :], axis=2) & (priority[None, :] < priority[:, None]), axis=1)
        s["agent_pos"] = jnp.where(blocked[:, None], s["agent_pos"], target)
        s = self.update_geometry(s)
        s["planted_now"] = jnp.zeros(6, jnp.bool_)
        s["watered_now"] = jnp.zeros(6, jnp.bool_)
        s["rotated_now"] = jnp.zeros(3, jnp.bool_)
        s["private_now"] = jnp.zeros(3, jnp.int32)
        s["community_now"] = jnp.zeros(3, jnp.int32)
        for k in range(3):
            i = jnp.mod(s["t"] + k, 3)
            act = jnp.where(self.legal_actions(s)[i, a[i]], a[i], 0)
            seed = s["seed_carry"][i]
            held_seed = jnp.clip(seed, 0, 5)
            near = jnp.sum(jnp.abs(s["plant_pos"] - s["agent_pos"][i]), axis=1) <= 1
            loose = near & (s["stage"] == 0) & (s["plant_holder"] == -1)
            j = jnp.argmax(loose)
            pick = (act == 5) & (seed < 0)
            drop = (act == 5) & (seed >= 0)
            s["plant_holder"] = s["plant_holder"].at[j].set(jnp.where(pick, i, s["plant_holder"][j]))
            s["plant_holder"] = s["plant_holder"].at[held_seed].set(jnp.where(drop, -1, s["plant_holder"][held_seed]))
            tool = s["tool_carry"][i]
            h = jnp.clip(tool, 0, 2)
            tool_near = (jnp.sum(jnp.abs(s["tool_pos"] - s["agent_pos"][i]), axis=1) <= 1) & (s["tool_holder"] == -1)
            j = jnp.argmax(tool_near)
            take_tool = (act == 6) & (tool < 0)
            drop_tool = (act == 6) & (tool >= 0)
            s["tool_holder"] = s["tool_holder"].at[j].set(jnp.where(take_tool, i, s["tool_holder"][j]))
            s["tool_holder"] = s["tool_holder"].at[h].set(jnp.where(drop_tool, -1, s["tool_holder"][h]))
            screen = jnp.argmax(tool_near & (s["tool_kind"] == 0))
            rotate = (act == 7) & ~s["rotated_now"][screen]
            s["orientation"] = s["orientation"].at[screen].set(jnp.where(rotate, -s["orientation"][screen], s["orientation"][screen]))
            s["rotated_now"] = s["rotated_now"].at[screen].set(s["rotated_now"][screen] | rotate)
            beds = (jnp.sum(jnp.abs(s["bed_pos"] - s["agent_pos"][i]), axis=1) <= 1) & (s["bed_occupant"] < 0)
            bed = jnp.argmax(beds)
            plant = act == 8
            s["plant_pos"] = s["plant_pos"].at[held_seed].set(jnp.where(plant, s["bed_pos"][bed], s["plant_pos"][held_seed]))
            s["bed_index"] = s["bed_index"].at[held_seed].set(jnp.where(plant, bed, s["bed_index"][held_seed]))
            s["plant_holder"] = s["plant_holder"].at[held_seed].set(jnp.where(plant, -1, s["plant_holder"][held_seed]))
            s["stage"] = s["stage"].at[held_seed].set(jnp.where(plant, 1, s["stage"][held_seed]))
            s["moisture"] = s["moisture"].at[held_seed].set(jnp.where(plant, 1, s["moisture"][held_seed]))
            s["planted_now"] = s["planted_now"].at[held_seed].set(s["planted_now"][held_seed] | plant)
            thirsty = near & (s["stage"] >= 1) & (s["stage"] <= 3) & (s["moisture"] < 5)
            j = jnp.argmax(thirsty)
            amount = jnp.where(act == 9, jnp.minimum(s["water_cart"][h], 5 - s["moisture"][j]), 0)
            s["moisture"] = s["moisture"].at[j].add(amount)
            s["water_cart"] = s["water_cart"].at[h].add(-amount)
            s["watered_now"] = s["watered_now"].at[j].set(s["watered_now"][j] | (amount > 0))
            ripe = near & (s["stage"] == 3)
            j = jnp.argmax(ripe)
            harvest = act == 10
            s["stage"] = s["stage"].at[j].set(jnp.where(harvest, 2, s["stage"][j]))
            s["growth"] = s["growth"].at[j].set(jnp.where(harvest, 0, s["growth"][j]))
            s["fruit_carry"] = s["fruit_carry"].at[i].set(jnp.where(harvest, 1, s["fruit_carry"][i]))
            s["fruit_owner"] = s["fruit_owner"].at[i].set(jnp.where(harvest, s["plant_owner"][j], s["fruit_owner"][i]))
            public = jnp.all(s["agent_pos"][i] == s["market_pos"][3])
            owner = jnp.clip(s["fruit_owner"][i], 0, 2)
            deliver = act == 11
            s["private_now"] = s["private_now"].at[owner].add((deliver & ~public).astype(jnp.int32))
            s["community_now"] = s["community_now"].at[owner].add((deliver & public).astype(jnp.int32))
            s["fruit_carry"] = s["fruit_carry"].at[i].set(jnp.where(deliver, 0, s["fruit_carry"][i]))
            s["fruit_owner"] = s["fruit_owner"].at[i].set(jnp.where(deliver, -1, s["fruit_owner"][i]))
            sources = (jnp.sum(jnp.abs(s["spring_pos"] - s["agent_pos"][i]), axis=1) <= 1) & (s["spring_water"] > 0)
            w = jnp.argmax(sources)
            fill = jnp.where(act == 12, jnp.minimum(4 - s["water_cart"][h], s["spring_water"][w]), 0)
            s["water_cart"] = s["water_cart"].at[h].add(fill)
            s["spring_water"] = s["spring_water"].at[w].add(-fill)
            s = self.update_geometry(s)
        s["t"] = s["t"] + 1
        s = self.update_geometry(s)
        live = (s["stage"] >= 1) & (s["stage"] <= 3)
        hot = s["sun"] == 1
        drain = (jnp.mod(s["t"], 3) == 0).astype(jnp.int32) + (hot & ~s["shaded"]).astype(jnp.int32)
        s["moisture"] = jnp.where(live, jnp.maximum(0, s["moisture"] - drain), s["moisture"])
        trouble = (s["moisture"] == 0) | (hot & ~s["shaded"] & (s["stage"] == 1))
        s["stress"] = jnp.where(live, jnp.clip(s["stress"] + jnp.where(trouble, 1, -1), 0, 3), s["stress"])
        s["wilted_now"] = live & (s["stress"] >= 3)
        fruit_die = s["wilted_now"] & (s["stage"] == 3)
        for i in range(3):
            s["fruit_lost"] = s["fruit_lost"].at[i].add(jnp.sum((fruit_die & (s["plant_owner"] == i)).astype(jnp.int32)))
        light = (s["sun"] != 3) & ~s["shaded"]
        grow = live & (s["stage"] <= 2) & light & (s["moisture"] > 0) & ~s["wilted_now"]
        progress = s["growth"] + grow.astype(jnp.int32)
        mature = (s["stage"] == 1) & (progress >= 4) & ~s["wilted_now"]
        s["ripe_now"] = (s["stage"] == 2) & (progress >= 4) & ~s["wilted_now"]
        s["stage"] = jnp.where(s["wilted_now"], 4, jnp.where(mature, 2, jnp.where(s["ripe_now"], 3, s["stage"])))
        s["growth"] = jnp.where(mature | s["ripe_now"] | s["wilted_now"], 0, progress)
        s["fruit_generated"] = s["fruit_generated"] + s["ripe_now"].astype(jnp.int32)
        s["spring_water"] = jnp.minimum(6, s["spring_water"] + (jnp.mod(s["t"], 3) == 0).astype(jnp.int32))
        s["private"] = s["private"] + s["private_now"]
        s["community"] = s["community"] + s["community_now"]
        success = jnp.sum(s["private"] + s["community"]) >= 6
        failure = jnp.all(s["stage"] == 4) & jnp.all(s["fruit_carry"] == 0) & ~success
        s["terminated"] = success | failure
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        reward = -0.01 + 0.1 * jnp.sum(s["private_now"]) + s["private_now"].astype(jnp.float32) + 0.6 * jnp.sum(s["community_now"]) + 2.0 * success - failure.astype(jnp.float32)
        return self.observe(s), s, reward.astype(jnp.float32), jnp.full(3, s["done"], jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def absorb(self, payload):
        key, s, actions = payload
        return self.observe(s), s, jnp.zeros(3, jnp.float32), jnp.ones(3, jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.transition, (key, state, actions))
