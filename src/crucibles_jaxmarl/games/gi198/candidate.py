"""Tideworks Yard: original blind source; no author-side execution."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 14
    observation_size = 254
    max_steps = 168
    obs_keys = ("t", "done", "terminated", "truncated", "tiles", "phase", "agent_pos", "agent_active",
                "cargo", "section", "soaked", "pontoon_pos", "pontoon_active", "pontoon_holder",
                "pontoon_owner", "shared", "anchored", "links", "connected", "crate_pos", "crate_active",
                "crate_holder", "crate_owner", "depot_pos", "depot_active", "delivered", "drifted",
                "anchor_changed", "linked_now", "delivered_now", "sunk_now")

    def refresh(self, state):
        s = dict(state)
        s["phase"] = jnp.mod(s["t"] // 3, 4)
        for role in range(3):
            has_c = s["crate_holder"] == role
            has_p = s["pontoon_holder"] == role
            s["cargo"] = s["cargo"].at[role].set(jnp.where(jnp.any(has_c), jnp.argmax(has_c), -1))
            s["section"] = s["section"].at[role].set(jnp.where(jnp.any(has_p), jnp.argmax(has_p), -1))
        s["crate_pos"] = jnp.where((s["crate_holder"] >= 0)[:, None], s["agent_pos"][jnp.clip(s["crate_holder"], 0, 2)], s["crate_pos"])
        s["pontoon_pos"] = jnp.where((s["pontoon_holder"] >= 0)[:, None], s["agent_pos"][jnp.clip(s["pontoon_holder"], 0, 2)], s["pontoon_pos"])
        deployed = s["pontoon_holder"] == -1
        near = jnp.sum(jnp.abs(s["pontoon_pos"][:, None, :] - s["pontoon_pos"][None, :, :]), axis=2) == 1
        s["links"] = s["links"] & near & deployed[:, None] & deployed[None, :]
        reach = s["links"] | jnp.eye(5, dtype=jnp.bool_)
        for k in range(5):
            reach = reach | (reach[:, k, None] & reach[k, None, :])
        s["connected"] = reach
        s["crate_active"] = s["crate_holder"] >= -1
        s["crate_visible"] = jnp.broadcast_to(s["crate_active"][None, :], (3, 6))
        return s

    def reset(self, key):
        tiles = jnp.zeros((7, 11), jnp.int32).at[:, 4:7].set(1).at[0, 4:7].set(2).at[6, 4:7].set(3)
        links = jnp.zeros((5, 5), jnp.bool_).at[0, 1].set(True).at[1, 0].set(True).at[1, 2].set(True).at[2, 1].set(True)
        s = {"t": jnp.array(0, jnp.int32), "done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
             "tiles": tiles, "phase": jnp.array(0, jnp.int32), "agent_pos": jnp.array([[3, 3], [5, 3], [7, 3]], jnp.int32),
             "agent_active": jnp.ones(3, jnp.bool_), "agent_visible": jnp.ones((3, 3), jnp.bool_),
             "cargo": jnp.full(3, -1, jnp.int32), "section": jnp.full(3, -1, jnp.int32), "soaked": jnp.zeros(3, jnp.bool_),
             "pontoon_pos": jnp.array([[4, 3], [5, 3], [6, 3], [3, 5], [7, 5]], jnp.int32),
             "pontoon_active": jnp.ones(5, jnp.bool_), "pontoon_visible": jnp.ones((3, 5), jnp.bool_),
             "pontoon_holder": jnp.full(5, -1, jnp.int32), "pontoon_owner": jnp.array([0, 1, 2, 0, 2], jnp.int32),
             "shared": jnp.array([True, True, True, False, False]), "anchored": jnp.array([True, False, False, False, False]),
             "links": links, "connected": jnp.eye(5, dtype=jnp.bool_),
             "crate_pos": jnp.array([[2, 2], [5, 3], [8, 4], [2, 4], [8, 2], [7, 5]], jnp.int32),
             "crate_active": jnp.ones(6, jnp.bool_), "crate_visible": jnp.ones((3, 6), jnp.bool_),
             "crate_holder": jnp.full(6, -1, jnp.int32), "crate_owner": jnp.array([0, 1, 2, 0, 1, 2], jnp.int32),
             "depot_pos": jnp.array([[9, 1], [9, 3], [1, 5]], jnp.int32), "depot_active": jnp.ones(3, jnp.bool_),
             "depot_visible": jnp.ones((3, 3), jnp.bool_), "delivered": jnp.zeros(3, jnp.int32),
             "drifted": jnp.zeros(5, jnp.bool_), "anchor_changed": jnp.zeros(5, jnp.bool_), "linked_now": jnp.zeros(5, jnp.bool_),
             "delivered_now": jnp.zeros(3, jnp.int32), "sunk_now": jnp.zeros(6, jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def support(self, s, points):
        terrain = s["tiles"][jnp.clip(points[..., 1], 0, 6), jnp.clip(points[..., 0], 0, 10)]
        land = (terrain == 0) | (terrain == 2) | ((terrain == 3) & (s["phase"] == 0))
        floating = jnp.any(jnp.all(points[..., None, :] == s["pontoon_pos"], axis=-1) & (s["pontoon_holder"] == -1), axis=-1)
        return land | floating

    def destinations(self, s):
        d = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s["agent_pos"][:, None, :] + d
        inside = jnp.all((p >= 0) & (p <= jnp.array([10, 6])), axis=2)
        hits = jnp.all(p[:, :, None, :] == s["pontoon_pos"][None, None, :, :], axis=3) & (s["pontoon_holder"][None, None, :] == -1)
        permission = s["shared"][None, :] | (s["pontoon_owner"][None, :] == jnp.arange(3)[:, None])
        forbidden = jnp.any(hits & ~permission[:, None, :], axis=2)
        occupied = jnp.any(jnp.all(p[:, :, None, :] == s["agent_pos"][None, None, :, :], axis=3) & ~jnp.eye(3, dtype=jnp.bool_)[:, None, :], axis=2)
        ok = inside & self.support(s, p) & ~forbidden & ~occupied
        return p, ok.at[:, 0].set(True)

    def legal_actions(self, s):
        p, move = self.destinations(s)
        pos = s["agent_pos"]
        empty = (s["cargo"] < 0) & (s["section"] < 0)
        nearby = (jnp.sum(jnp.abs(pos[:, None, :] - s["pontoon_pos"][None, :, :]), axis=2) <= 1) & (s["pontoon_holder"][None, :] == -1)
        passengers = jnp.any(jnp.all(s["pontoon_pos"][:, None, :] == pos[None, :, :], axis=2), axis=1)
        freight = jnp.any(jnp.all(s["pontoon_pos"][:, None, :] == s["crate_pos"][None, :, :], axis=2) & s["crate_active"][None, :], axis=1)
        liftable = ~s["anchored"] & ~jnp.any(s["links"], axis=1) & ~passengers & ~freight
        loose = jnp.any(jnp.all(pos[:, None, :] == s["crate_pos"][None, :, :], axis=2) & (s["crate_holder"][None, :] == -1), axis=1)
        mask = jnp.zeros((3, 14), jnp.bool_).at[:, :5].set(move)
        mask = mask.at[:, 5].set((empty & loose) | (s["cargo"] >= 0))
        mask = mask.at[:, 6].set(empty & jnp.any(nearby & liftable[None, :], axis=1))
        for action, dx in ((7, 1), (8, -1)):
            target = pos + jnp.array([dx, 0], jnp.int32)
            free = ~jnp.any(jnp.all(target[:, None, :] == s["pontoon_pos"][None, :, :], axis=2) & (s["pontoon_holder"][None, :] == -1), axis=1)
            free = free & ~jnp.any(jnp.all(target[:, None, :] == pos[None, :, :], axis=2), axis=1)
            water = (target[:, 0] >= 4) & (target[:, 0] <= 6) & (target[:, 1] >= 1) & (target[:, 1] <= 5)
            mask = mask.at[:, action].set((s["section"] >= 0) & free & water)
        mask = mask.at[:, 9].set(jnp.any(nearby, axis=1)).at[:, 10].set(jnp.any(nearby, axis=1)).at[:, 11].set(jnp.any(nearby & jnp.any(s["links"], axis=1)[None, :], axis=1))
        owned = nearby & (s["pontoon_owner"][None, :] == jnp.arange(3)[:, None])
        mask = mask.at[:, 12].set(jnp.any(owned, axis=1))
        owner = s["crate_owner"][jnp.clip(s["cargo"], 0, 5)]
        mask = mask.at[:, 13].set((s["cargo"] >= 0) & jnp.all(pos == s["depot_pos"][owner], axis=1))
        return jnp.where(s["done"], jnp.arange(14)[None, :] == 0, mask)

    def observe(self, s):
        flat = jnp.concatenate([s[k].astype(jnp.float32).reshape(-1) for k in self.obs_keys])
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(flat, (3, self.observation_size - 3))), axis=1)

    def transition(self, payload):
        key, state, requested = payload
        s = dict(state)
        a = jnp.clip(requested, 0, 13)
        a = jnp.where((requested >= 0) & (requested < 14) & self.legal_actions(s)[jnp.arange(3), a], a, 0)
        candidates, ok = self.destinations(s)
        proposed = candidates[jnp.arange(3), jnp.where(a <= 4, a, 0)]
        rank = jnp.mod(jnp.arange(3) - s["t"], 3)
        loses = jnp.any(jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2) & (rank[None, :] < rank[:, None]), axis=1)
        s["agent_pos"] = jnp.where(loses[:, None], s["agent_pos"], proposed)
        s = self.refresh(s)
        s["delivered_now"] = jnp.zeros(3, jnp.int32)
        s["anchor_changed"] = jnp.zeros(5, jnp.bool_)
        s["linked_now"] = jnp.zeros(5, jnp.bool_)
        for k in range(3):
            i = jnp.mod(s["t"] + k, 3)
            valid = self.legal_actions(s)[i, a[i]]
            act = jnp.where(valid, a[i], 0)
            held = s["cargo"][i]
            h = jnp.clip(held, 0, 5)
            loose = (s["crate_holder"] == -1) & jnp.all(s["crate_pos"] == s["agent_pos"][i], axis=1)
            c = jnp.argmax(loose)
            pick = (act == 5) & (held < 0)
            drop = (act == 5) & (held >= 0)
            s["crate_holder"] = s["crate_holder"].at[c].set(jnp.where(pick, i, s["crate_holder"][c])).at[h].set(jnp.where(drop, -1, jnp.where(pick & (c == h), i, s["crate_holder"][h])))
            near = (jnp.sum(jnp.abs(s["pontoon_pos"] - s["agent_pos"][i]), axis=1) <= 1) & (s["pontoon_holder"] == -1)
            occupied = jnp.any(jnp.all(s["pontoon_pos"][:, None, :] == s["agent_pos"][None, :, :], axis=2), axis=1)
            freight = jnp.any(jnp.all(s["pontoon_pos"][:, None, :] == s["crate_pos"][None, :, :], axis=2) & s["crate_active"][None, :], axis=1)
            can_lift = near & ~s["anchored"] & ~jnp.any(s["links"], axis=1) & ~occupied & ~freight
            j = jnp.argmax(can_lift)
            s["pontoon_holder"] = s["pontoon_holder"].at[j].set(jnp.where(act == 6, i, s["pontoon_holder"][j]))
            held_section = jnp.clip(s["section"][i], 0, 4)
            place = (act == 7) | (act == 8)
            target = s["agent_pos"][i] + jnp.array([jnp.where(act == 7, 1, -1), 0], jnp.int32)
            s["pontoon_pos"] = s["pontoon_pos"].at[held_section].set(jnp.where(place, target, s["pontoon_pos"][held_section]))
            s["pontoon_holder"] = s["pontoon_holder"].at[held_section].set(jnp.where(place, -1, s["pontoon_holder"][held_section]))
            eligible = near & jnp.where(act == 11, jnp.any(s["links"], axis=1), True) & jnp.where(act == 12, s["pontoon_owner"] == i, True)
            j = jnp.argmax(eligible)
            toggle = (act == 9) & ~s["anchor_changed"][j]
            s["anchored"] = s["anchored"].at[j].set(jnp.where(toggle, ~s["anchored"][j], s["anchored"][j]))
            s["anchor_changed"] = s["anchor_changed"].at[j].set(s["anchor_changed"][j] | toggle)
            adjacent = (jnp.sum(jnp.abs(s["pontoon_pos"] - s["pontoon_pos"][j]), axis=1) == 1) & (s["pontoon_holder"] == -1)
            row = jnp.where(act == 10, s["links"][j] | adjacent, jnp.where(act == 11, False, s["links"][j]))
            s["links"] = s["links"].at[j, :].set(row).at[:, j].set(row)
            s["linked_now"] = s["linked_now"].at[j].set(s["linked_now"][j] | (act == 10) | (act == 11))
            s["shared"] = s["shared"].at[j].set(jnp.where(act == 12, ~s["shared"][j], s["shared"][j]))
            delivered = act == 13
            owner = s["crate_owner"][h]
            s["crate_holder"] = s["crate_holder"].at[h].set(jnp.where(delivered, -2, s["crate_holder"][h]))
            s["delivered_now"] = s["delivered_now"].at[owner].add(delivered.astype(jnp.int32))
            s = self.refresh(s)
        s["t"] = s["t"] + 1
        s = self.refresh(s)
        pulse = jnp.mod(s["t"], 6) == 0
        dy = jnp.where(jnp.mod(s["t"], 12) == 6, 1, -1)
        shift = jnp.array([0, dy], jnp.int32)
        roots = jnp.min(jnp.where(s["connected"], jnp.arange(5)[None, :], 5), axis=1)
        s["drifted"] = jnp.zeros(5, jnp.bool_)
        for root in range(5):
            member = roots == root
            target = s["pontoon_pos"] + shift
            water = (s["pontoon_pos"][:, 0] >= 4) & (s["pontoon_pos"][:, 0] <= 6) & (s["pontoon_holder"] == -1)
            bounds = (target[:, 1] >= 1) & (target[:, 1] <= 5)
            other = jnp.any(jnp.all(target[:, None, :] == s["pontoon_pos"][None, :, :], axis=2) & ~member[None, :] & (s["pontoon_holder"][None, :] == -1), axis=1)
            riders = jnp.any(jnp.all(s["agent_pos"][:, None, :] == s["pontoon_pos"][None, :, :], axis=2) & member[None, :], axis=1)
            arriving = s["agent_pos"] + shift
            collision = jnp.any(jnp.all(arriving[:, None, :] == s["agent_pos"][None, :, :], axis=2) & riders[:, None] & ~riders[None, :])
            drift = pulse & jnp.any(member) & ~jnp.any(member & s["anchored"]) & jnp.all(~member | (water & bounds & ~other)) & ~collision
            aboard = jnp.any(jnp.all(s["crate_pos"][:, None, :] == s["pontoon_pos"][None, :, :], axis=2) & member[None, :], axis=1) & s["crate_active"]
            s["pontoon_pos"] = s["pontoon_pos"] + (member & drift)[:, None].astype(jnp.int32) * shift
            s["agent_pos"] = s["agent_pos"] + (riders & drift)[:, None].astype(jnp.int32) * shift
            s["crate_pos"] = s["crate_pos"] + (aboard & drift)[:, None].astype(jnp.int32) * shift
            s["drifted"] = s["drifted"] | (member & drift)
        s = self.refresh(s)
        s["soaked"] = ~self.support(s, s["agent_pos"])
        s["sunk_now"] = s["crate_active"] & ~self.support(s, s["crate_pos"])
        s["crate_holder"] = jnp.where(s["sunk_now"], -3, s["crate_holder"])
        s["delivered"] = s["delivered"] + s["delivered_now"]
        failure = jnp.any(s["soaked"]) | jnp.any(s["crate_holder"] == -3)
        success = (jnp.sum(s["delivered"]) == 6) & ~failure
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
        return jax.lax.cond(state["done"], self.absorb, self.transition, (key, state, actions))
