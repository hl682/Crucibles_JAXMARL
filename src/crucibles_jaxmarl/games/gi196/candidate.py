"""Beacon Marches. Original phased frontier game, not executed by its author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 12
    observation_size = 301
    max_steps = 90

    def refresh(self, s):
        s = dict(s)
        xx = jnp.arange(9)[None, :]
        yy = jnp.arange(9)[:, None]
        local = (jnp.abs(xx[None, :, :] - s["agent_pos"][:, 0, None, None])
                 + jnp.abs(yy[None, :, :] - s["agent_pos"][:, 1, None, None])) <= 2
        radius = (jnp.abs(xx[None, :, :] - s["beacon_pos"][:, 0, None, None])
                  + jnp.abs(yy[None, :, :] - s["beacon_pos"][:, 1, None, None])) <= 3
        owners = jnp.clip(s["beacon_owner"], 0, 2)
        allied = s["grants"][owners, :].T > 0
        access = ((s["beacon_owner"][None, :] == jnp.arange(3)[:, None]) | allied) & s["beacon_built"][None, :] & (s["beacon_fuel"][None, :] > 0)
        s["map_visible"] = local | jnp.any(access[:, :, None, None] & radius[None, :, :, :], axis=1)
        s["agent_visible"] = s["map_visible"][:, s["agent_pos"][:, 1], s["agent_pos"][:, 0]]
        s["home_visible"] = s["map_visible"][:, s["home_pos"][:, 1], s["home_pos"][:, 0]] | jnp.eye(3, dtype=jnp.bool_)
        s["beacon_visible"] = s["map_visible"][:, s["beacon_pos"][:, 1], s["beacon_pos"][:, 0]] | (s["beacon_owner"][None, :] == jnp.arange(3)[:, None])
        s["pass_access"] = s["grants"][s["pass_owner"]]
        s["pass_offers"] = s["offers"][s["pass_owner"]]
        s["phase"] = jnp.mod(s["t"], 3)
        return s

    def reset(self, key):
        tiles = jnp.zeros((9, 9), jnp.int32).at[:, 4].set(1)
        tiles = tiles.at[jnp.array([2, 4, 6]), 4].set(3)
        rough = jax.random.bernoulli(key, 0.35, (6,))
        tiles = tiles.at[jnp.array([2, 2, 4, 4, 6, 6]), jnp.array([2, 6, 2, 6, 2, 6])].set(jnp.where(rough, 2, 0))
        homes = jnp.array([[1, 2], [1, 6], [7, 4]], jnp.int32)
        s = {"t": jnp.array(0, jnp.int32), "phase": jnp.array(0, jnp.int32),
             "done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
             "tiles": tiles, "map_visible": jnp.zeros((3, 9, 9), jnp.bool_),
             "agent_pos": homes, "agent_active": jnp.ones(3, jnp.bool_), "agent_visible": jnp.ones((3, 3), jnp.bool_),
             "carried": jnp.full(3, 2, jnp.int32), "bank": jnp.full(3, 6, jnp.int32), "spent": jnp.zeros(3, jnp.int32),
             "home_pos": homes, "home_active": jnp.ones(3, jnp.bool_), "home_visible": jnp.eye(3, dtype=jnp.bool_),
             "outpost_pos": jnp.array([[7, 2], [7, 6], [1, 4]], jnp.int32),
             "outpost_active": jnp.ones(3, jnp.bool_), "outpost_visible": jnp.ones((3, 3), jnp.bool_),
             "defense": jnp.full(3, 3, jnp.int32), "delivered": jnp.zeros(3, jnp.int32),
             "pass_pos": jnp.array([[4, 2], [4, 4], [4, 6]], jnp.int32), "pass_owner": jnp.array([1, 2, 0], jnp.int32),
             "pass_active": jnp.ones(3, jnp.bool_), "pass_visible": jnp.ones((3, 3), jnp.bool_),
             "offers": jnp.zeros((3, 3), jnp.int32), "grants": jnp.zeros((3, 3), jnp.int32),
             "pass_access": jnp.zeros((3, 3), jnp.int32), "pass_offers": jnp.zeros((3, 3), jnp.int32),
             "beacon_pos": jnp.array([[3, 2], [5, 2], [3, 4], [5, 4], [3, 6], [5, 6]], jnp.int32),
             "beacon_active": jnp.ones(6, jnp.bool_), "beacon_visible": jnp.zeros((3, 6), jnp.bool_),
             "beacon_built": jnp.zeros(6, jnp.bool_), "beacon_owner": jnp.full(6, -1, jnp.int32),
             "beacon_fuel": jnp.zeros(6, jnp.int32), "recharged": jnp.zeros(6, jnp.bool_),
             "delivered_now": jnp.zeros(3, jnp.int32), "offer_now": jnp.zeros((3, 3), jnp.bool_),
             "accepted_now": jnp.zeros((3, 3), jnp.bool_), "blocked": jnp.zeros(3, jnp.bool_),
             "pressure": jnp.array(False)}
        s = self.refresh(s)
        return self.observe(s), s

    def movement(self, s):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s["agent_pos"][:, None, :] + delta[None, :, :]
        bounds = jnp.all((p >= 0) & (p <= 8), axis=2)
        terrain = s["tiles"][jnp.clip(p[:, :, 1], 0, 8), jnp.clip(p[:, :, 0], 0, 8)]
        pass_match = jnp.all(p[:, :, None, :] == s["pass_pos"][None, None, :, :], axis=3)
        owner = s["pass_owner"][jnp.argmax(pass_match, axis=2)]
        permit = (owner == jnp.arange(3)[:, None]) | (s["grants"][owner, jnp.arange(3)[:, None]] > 0)
        occupied = jnp.any(jnp.all(p[:, :, None, :] == s["agent_pos"][None, None, :, :], axis=3)
                           & ~jnp.eye(3, dtype=jnp.bool_)[:, None, :], axis=2)
        ok = bounds & (terrain != 1) & (terrain != 2) & (~jnp.any(pass_match, axis=2) | permit) & ~occupied & (s["phase"] == 1)
        return p, ok.at[:, 0].set(True)

    def legal_actions(self, s):
        p, movement = self.movement(s)
        ids = jnp.arange(3)
        at_home = jnp.all(s["agent_pos"] == s["home_pos"], axis=1)
        at_outpost = jnp.all(s["agent_pos"] == s["outpost_pos"], axis=1)
        node = jnp.all(s["agent_pos"][:, None, :] == s["beacon_pos"][None, :, :], axis=2)
        owner = jnp.clip(s["beacon_owner"], 0, 2)
        maintain = ~s["beacon_built"][None, :] | (s["beacon_owner"][None, :] == ids[:, None]) | (s["grants"][owner, :].T > 0)
        can_build = jnp.any(node & maintain & (s["beacon_fuel"][None, :] < 6), axis=1)
        works = s["phase"] == 2
        council = (s["phase"] == 0) & at_home & (s["carried"] > 0)
        mask = jnp.zeros((3, 12), jnp.bool_).at[:, :5].set(movement)
        mask = mask.at[:, 5].set(works & at_home & (s["carried"] < 2) & (s["bank"] > 0))
        mask = mask.at[:, 6].set(works & at_outpost & (s["carried"] > 0) & (s["delivered"] < 2))
        mask = mask.at[:, 7].set(works & can_build & (s["carried"] > 0))
        mask = mask.at[:, 8].set(council & (s["offers"][ids, jnp.mod(ids + 1, 3)] < 3))
        mask = mask.at[:, 9].set(council & (s["offers"][ids, jnp.mod(ids + 2, 3)] < 3))
        mask = mask.at[:, 10].set(council & (s["offers"][jnp.mod(ids + 1, 3), ids] > 0))
        mask = mask.at[:, 11].set(council & (s["offers"][jnp.mod(ids + 2, 3), ids] > 0))
        return jnp.where(s["done"], jnp.arange(12)[None, :] == 0, mask)

    def role_observation(self, s, i):
        av = s["agent_visible"][i]
        hv = s["home_visible"][i]
        bv = s["beacon_visible"][i]
        couriers = jnp.concatenate((jnp.where(av[:, None], s["agent_pos"], -1), jnp.where(av, s["carried"], -1)[:, None], av[:, None]), axis=1)
        homes = jnp.concatenate((jnp.where(hv[:, None], s["home_pos"], -1), jnp.where(hv, s["bank"], -1)[:, None], hv[:, None]), axis=1)
        beacons = jnp.concatenate((jnp.where(bv[:, None], s["beacon_pos"], -1), jnp.where(bv, s["beacon_owner"], -1)[:, None],
                                   jnp.where(bv, s["beacon_fuel"], -1)[:, None], jnp.where(bv, s["beacon_built"].astype(jnp.int32), -1)[:, None], bv[:, None]), axis=1)
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32)[i],
                                jnp.array([s["t"], s["phase"], s["done"], s["terminated"], s["truncated"]], jnp.float32),
                                jnp.array([s["carried"][i], s["bank"][i], s["spent"][i], s["blocked"][i]], jnp.float32),
                                couriers.reshape(-1), homes.reshape(-1), s["outpost_pos"].reshape(-1), s["defense"], s["delivered"],
                                s["pass_pos"].reshape(-1), s["pass_owner"], s["offers"].reshape(-1), s["grants"].reshape(-1),
                                beacons.reshape(-1), jnp.where(s["map_visible"][i], s["tiles"], -1).reshape(-1),
                                s["map_visible"][i].reshape(-1), s["delivered_now"], s["offer_now"].reshape(-1), s["accepted_now"].reshape(-1),
                                jnp.where(bv, s["recharged"].astype(jnp.int32), -1), s["pressure"].reshape(1))).astype(jnp.float32)

    def observe(self, s):
        return jnp.stack([self.role_observation(s, i) for i in range(3)])

    def advance(self, payload):
        key, old, requested = payload
        s = dict(old)
        ids = jnp.arange(3)
        raw = jnp.clip(requested, 0, 11)
        legal = self.legal_actions(s)
        valid = (requested >= 0) & (requested < 12) & legal[ids, raw]
        actions = jnp.where(valid, raw, 0)
        candidates, available = self.movement(s)
        p = candidates[ids, jnp.where(actions <= 4, actions, 0)]
        rank = jnp.mod(ids - s["t"], 3)
        clash = jnp.all(p[:, None, :] == p[None, :, :], axis=2)
        lose = jnp.any(clash & (rank[None, :] < rank[:, None]), axis=1)
        s["agent_pos"] = jnp.where(lose[:, None], s["agent_pos"], p)
        s["blocked"] = (s["phase"] == 1) & (requested >= 1) & (requested <= 4) & (~valid | lose)
        offer = (actions == 8) | (actions == 9)
        accept = (actions == 10) | (actions == 11)
        target = jnp.mod(ids + jnp.where(actions == 8, 1, 2), 3)
        offering_owner = jnp.mod(ids + jnp.where(actions == 10, 1, 2), 3)
        s["offer_now"] = jnp.eye(3, dtype=jnp.bool_)[target] & offer[:, None]
        s["accepted_now"] = jnp.eye(3, dtype=jnp.bool_)[offering_owner].T & accept[None, :]
        s["offers"] = jnp.where(s["accepted_now"], 0, jnp.where(s["offer_now"], 3, s["offers"]))
        s["grants"] = jnp.where(s["accepted_now"], 4, s["grants"])
        cost = (offer | accept).astype(jnp.int32)
        s["carried"] = s["carried"] - cost
        s["spent"] = s["spent"] + cost
        load = actions == 5
        amount = jnp.where(load, jnp.minimum(2 - s["carried"], s["bank"]), 0)
        s["carried"] = s["carried"] + amount
        s["bank"] = s["bank"] - amount
        s["delivered_now"] = (actions == 6).astype(jnp.int32)
        s["carried"] = s["carried"] - s["delivered_now"]
        s["delivered"] = s["delivered"] + s["delivered_now"]
        s["defense"] = jnp.minimum(6, s["defense"] + 2 * s["delivered_now"])
        s["recharged"] = jnp.zeros(6, jnp.bool_)
        for k in range(3):
            i = jnp.mod(s["t"] + k, 3)
            node = jnp.argmax(jnp.all(s["beacon_pos"] == s["agent_pos"][i], axis=1))
            build = actions[i] == 7
            claim = build & ~s["beacon_built"][node]
            s["beacon_owner"] = s["beacon_owner"].at[node].set(jnp.where(claim, i, s["beacon_owner"][node]))
            s["beacon_built"] = s["beacon_built"].at[node].set(s["beacon_built"][node] | build)
            s["beacon_fuel"] = s["beacon_fuel"].at[node].set(jnp.minimum(6, s["beacon_fuel"][node] + 3 * build.astype(jnp.int32)))
            s["carried"] = s["carried"].at[i].add(-build.astype(jnp.int32))
            s["spent"] = s["spent"].at[i].add(build.astype(jnp.int32))
            s["recharged"] = s["recharged"].at[node].set(s["recharged"][node] | build)
        works = (s["phase"] == 2).astype(jnp.int32)
        s["offers"] = jnp.maximum(0, s["offers"] - works)
        s["grants"] = jnp.maximum(0, s["grants"] - works)
        s["beacon_fuel"] = jnp.maximum(0, s["beacon_fuel"] - works)
        s["t"] = s["t"] + 1
        s["pressure"] = jnp.mod(s["t"], 15) == 0
        s["defense"] = jnp.maximum(0, s["defense"] - s["pressure"].astype(jnp.int32))
        failure = jnp.any(s["defense"] == 0)
        success = jnp.all(s["delivered"] >= 2) & ~failure
        s["terminated"] = failure | success
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        reward = -0.01 + 0.25 * jnp.sum(s["delivered_now"]) + 0.75 * s["delivered_now"].astype(jnp.float32) - 0.1 * s["pressure"] + 2.0 * success - 2.0 * failure
        return self.observe(s), s, reward.astype(jnp.float32), jnp.full(3, s["done"], jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def absorb(self, payload):
        key, s, actions = payload
        return self.observe(s), s, jnp.zeros(3, jnp.float32), jnp.ones(3, jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, actions))
