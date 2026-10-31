"""Signal Coast League: original explicit-commitment coast game; unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 12
    observation_size = 218
    max_steps = 120

    def visibility(self, state):
        s = dict(state)
        distances = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2)
        s["agent_visible"] = distances <= s["radius"][:, None]
        local_c = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["caravan_pos"][None, :, :]), axis=2) <= s["radius"][:, None]
        s["caravan_visible"] = local_c | jnp.eye(3, dtype=jnp.bool_)
        s["home_visible"] = (jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["home_pos"][None, :, :]), axis=2) <= s["radius"][:, None]) | jnp.eye(3, dtype=jnp.bool_)
        from_scout = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["hazard_pos"][None, :, :]), axis=2) <= s["radius"][:, None]
        from_caravan = jnp.sum(jnp.abs(s["caravan_pos"][:, None, :] - s["hazard_pos"][None, :, :]), axis=2) <= 1
        s["hazard_visible"] = from_scout | from_caravan
        s["warning_active"] = s["warning_ttl"] > 0
        s["warning_visible"] = (jnp.arange(9)[None, :] // 3 == jnp.arange(3)[:, None]) & s["warning_active"][None, :]
        s["phase"] = jnp.mod(s["t"], 3)
        return s

    def reset(self, key):
        tiles = jnp.zeros((7, 13), jnp.int32).at[1, 1:12].set(1).at[3, 1:12].set(1).at[5, 1:12].set(1)
        tiles = tiles.at[1:6, 1].set(1).at[1:6, 6].set(1).at[1:6, 11].set(1)
        homes = jnp.array([[1, 1], [6, 3], [11, 5]], jnp.int32)
        s = {"t": jnp.array(0, jnp.int32), "phase": jnp.array(0, jnp.int32), "done": jnp.array(False),
             "terminated": jnp.array(False), "truncated": jnp.array(False), "tiles": tiles,
             "agent_pos": jnp.array([[1, 0], [6, 2], [11, 6]], jnp.int32), "agent_active": jnp.ones(3, jnp.bool_),
             "agent_visible": jnp.eye(3, dtype=jnp.bool_), "radius": jnp.array([3, 3, 4], jnp.int32),
             "caravan_pos": homes, "caravan_target": homes, "caravan_active": jnp.ones(3, jnp.bool_),
             "caravan_visible": jnp.eye(3, dtype=jnp.bool_), "cargo": jnp.full(3, 2, jnp.int32), "lost": jnp.zeros(3, jnp.int32),
             "home_pos": homes, "home_active": jnp.ones(3, jnp.bool_), "home_visible": jnp.eye(3, dtype=jnp.bool_), "bank": jnp.full(3, 2, jnp.int32),
             "destination_pos": jnp.array([[11, 1], [1, 3], [6, 5]], jnp.int32), "destination_active": jnp.ones(3, jnp.bool_),
             "destination_visible": jnp.ones((3, 3), jnp.bool_), "delivered": jnp.zeros(3, jnp.int32),
             "tower_pos": jnp.array([[1, 0], [6, 2], [11, 6]], jnp.int32), "tower_active": jnp.ones(3, jnp.bool_),
             "tower_visible": jnp.ones((3, 3), jnp.bool_), "tower_fuel": jnp.full(3, 4, jnp.int32),
             "stop_pos": jnp.array([[1, 1], [6, 1], [11, 1], [1, 3], [6, 3], [11, 3], [1, 5], [6, 5], [11, 5]], jnp.int32),
             "stop_active": jnp.ones(9, jnp.bool_), "stop_visible": jnp.ones((3, 9), jnp.bool_),
             "hazard_pos": jnp.array([[3, 1], [8, 3], [9, 5]], jnp.int32), "hazard_active": jnp.ones(3, jnp.bool_),
             "hazard_visible": jnp.eye(3, dtype=jnp.bool_), "hazard_dir": jnp.where(jax.random.bernoulli(key, 0.5, (3,)), 1, -1).astype(jnp.int32),
             "warning_pos": jnp.zeros((9, 2), jnp.int32), "warning_active": jnp.zeros(9, jnp.bool_),
             "warning_visible": jnp.zeros((3, 9), jnp.bool_), "warning_ttl": jnp.zeros(9, jnp.int32),
             "warning_dir": jnp.zeros(9, jnp.int32), "warning_sender": jnp.full(9, -1, jnp.int32),
             "warning_stamp": jnp.full(9, -1, jnp.int32), "warning_hazard": jnp.tile(jnp.arange(3, dtype=jnp.int32), 3),
             "signal_now": jnp.zeros(3, jnp.bool_), "delivered_now": jnp.zeros(3, jnp.int32), "blocked": jnp.zeros(3, jnp.bool_),
             "hit": jnp.zeros(3, jnp.bool_)}
        s = self.visibility(s)
        return self.observe(s), s

    def walking(self, s):
        steps = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s["agent_pos"][:, None, :] + steps
        bounds = jnp.all((p >= 0) & (p <= jnp.array([12, 6])), axis=2)
        occupied = jnp.any(jnp.all(p[:, :, None, :] == s["agent_pos"][None, None, :, :], axis=3) & ~jnp.eye(3, dtype=jnp.bool_)[:, None, :], axis=2)
        return p, (bounds & ~occupied).at[:, 0].set(True)

    def legal_actions(self, s):
        p, walking = self.walking(s)
        tower = jnp.all(s["agent_pos"][:, None, :] == s["tower_pos"][None, :, :], axis=2)
        fuel = jnp.any(tower & (s["tower_fuel"][None, :] > 0), axis=1)
        stopped = jnp.all(s["caravan_pos"] == s["caravan_target"], axis=1)
        driver = jnp.sum(jnp.abs(s["agent_pos"] - s["caravan_pos"]), axis=1) <= 1
        command = stopped & driver
        cp = s["caravan_pos"]
        mask = jnp.zeros((3, 12), jnp.bool_).at[:, :5].set(walking)
        mask = mask.at[:, 5].set(fuel & jnp.any(s["hazard_visible"], axis=1))
        mask = mask.at[:, 6].set(command & (cp[:, 0] < 11))
        mask = mask.at[:, 7].set(command & (cp[:, 0] > 1))
        mask = mask.at[:, 8].set(command & (cp[:, 1] > 1))
        mask = mask.at[:, 9].set(command & (cp[:, 1] < 5))
        mask = mask.at[:, 10].set(command & jnp.all(cp == s["home_pos"], axis=1) & (s["cargo"] < 2) & (s["bank"] > 0))
        mask = mask.at[:, 11].set(command & jnp.all(cp == s["destination_pos"], axis=1) & (s["cargo"] > 0) & (s["delivered"] < 2))
        return jnp.where(s["done"], jnp.arange(12)[None, :] == 0, mask)

    def role_observation(self, s, i):
        av, cv, hv, zv = s["agent_visible"][i], s["caravan_visible"][i], s["home_visible"][i], s["hazard_visible"][i]
        scouts = jnp.concatenate((jnp.where(av[:, None], s["agent_pos"], -1), av[:, None]), axis=1)
        caravans = jnp.concatenate((jnp.where(cv[:, None], s["caravan_pos"], -1), jnp.where(cv[:, None], s["caravan_target"], -1),
                                    jnp.where(cv, s["cargo"], -1)[:, None], jnp.where(cv, s["lost"], -1)[:, None], cv[:, None]), axis=1)
        homes = jnp.concatenate((jnp.where(hv[:, None], s["home_pos"], -1), jnp.where(hv, s["bank"], -1)[:, None], hv[:, None]), axis=1)
        hazards = jnp.concatenate((jnp.where(zv[:, None], s["hazard_pos"], -1), jnp.where(zv, s["hazard_dir"], -1)[:, None], zv[:, None]), axis=1)
        indices = 3 * i + jnp.arange(3)
        live = s["warning_ttl"][indices] > 0
        warning = jnp.concatenate((jnp.where(live[:, None], s["warning_pos"][indices], -1), jnp.where(live, s["warning_dir"][indices], -1)[:, None],
                                   s["warning_ttl"][indices, None], jnp.where(live, s["warning_sender"][indices], -1)[:, None],
                                   jnp.where(live, s["warning_stamp"][indices], -1)[:, None], live[:, None]), axis=1)
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32)[i], jnp.array([s["t"], s["done"], s["terminated"], s["truncated"], s["phase"]], jnp.float32),
                                s["radius"], s["tiles"].reshape(-1), s["delivered"], s["signal_now"], s["tower_pos"].reshape(-1), s["tower_fuel"],
                                s["stop_pos"].reshape(-1), s["destination_pos"].reshape(-1), scouts.reshape(-1), caravans.reshape(-1), homes.reshape(-1),
                                hazards.reshape(-1), warning.reshape(-1), jnp.array([s["blocked"][i], s["hit"][i]], jnp.float32))).astype(jnp.float32)

    def observe(self, s):
        return jnp.stack([self.role_observation(s, i) for i in range(3)])

    def transition(self, payload):
        key, state, requested = payload
        s = dict(state)
        a = jnp.clip(requested, 0, 11)
        a = jnp.where((requested >= 0) & (requested < 12) & self.legal_actions(s)[jnp.arange(3), a], a, 0)
        options, mask = self.walking(s)
        pp = options[jnp.arange(3), jnp.where(a <= 4, a, 0)]
        rank = jnp.mod(jnp.arange(3) - s["t"], 3)
        loses = jnp.any(jnp.all(pp[:, None, :] == pp[None, :, :], axis=2) & (rank[None, :] < rank[:, None]), axis=1)
        s["agent_pos"] = jnp.where(loses[:, None], s["agent_pos"], pp)
        s["warning_ttl"] = jnp.maximum(0, s["warning_ttl"] - 1)
        s["signal_now"] = a == 5
        for k in range(3):
            i = jnp.mod(s["t"] + k, 3)
            tower = jnp.argmax(jnp.all(s["tower_pos"] == s["agent_pos"][i], axis=1))
            hazard = jnp.argmax(state["hazard_visible"][i])
            send = a[i] == 5
            recipients = 3 * jnp.arange(3) + hazard
            s["tower_fuel"] = s["tower_fuel"].at[tower].add(-send.astype(jnp.int32))
            s["warning_pos"] = s["warning_pos"].at[recipients].set(jnp.where(send, state["hazard_pos"][hazard], s["warning_pos"][recipients]))
            s["warning_dir"] = s["warning_dir"].at[recipients].set(jnp.where(send, state["hazard_dir"][hazard], s["warning_dir"][recipients]))
            s["warning_ttl"] = s["warning_ttl"].at[recipients].set(jnp.where(send, 6, s["warning_ttl"][recipients]))
            s["warning_sender"] = s["warning_sender"].at[recipients].set(jnp.where(send, i, s["warning_sender"][recipients]))
            s["warning_stamp"] = s["warning_stamp"].at[recipients].set(jnp.where(send, state["t"], s["warning_stamp"][recipients]))
        direction = jnp.stack((jnp.where(a == 6, 5, jnp.where(a == 7, -5, 0)), jnp.where(a == 8, -2, jnp.where(a == 9, 2, 0))), axis=1).astype(jnp.int32)
        command = (a >= 6) & (a <= 9)
        s["caravan_target"] = jnp.where(command[:, None], s["caravan_pos"] + direction, s["caravan_target"])
        load = jnp.where(a == 10, jnp.minimum(2 - s["cargo"], s["bank"]), 0)
        s["bank"] = s["bank"] - load
        s["cargo"] = s["cargo"] + load
        s["delivered_now"] = jnp.where(a == 11, jnp.minimum(s["cargo"], 2 - s["delivered"]), 0)
        s["cargo"] = s["cargo"] - s["delivered_now"]
        s["delivered"] = s["delivered"] + s["delivered_now"]
        step = jnp.sign(s["caravan_target"] - s["caravan_pos"]).astype(jnp.int32)
        target = s["caravan_pos"] + step
        old_occupied = jnp.any(jnp.all(target[:, None, :] == s["caravan_pos"][None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        contested = jnp.any(jnp.all(target[:, None, :] == target[None, :, :], axis=2) & (rank[None, :] < rank[:, None]), axis=1)
        moving = jnp.any(step != 0, axis=1)
        s["blocked"] = moving & (old_occupied | contested)
        s["caravan_pos"] = jnp.where(s["blocked"][:, None], s["caravan_pos"], target)
        s["t"] = s["t"] + 1
        drift = jnp.mod(s["t"], 3) == 0
        next_x = s["hazard_pos"][:, 0] + s["hazard_dir"]
        bounce = (next_x < 2) | (next_x > 10)
        hd = jnp.where(bounce, -s["hazard_dir"], s["hazard_dir"])
        s["hazard_dir"] = jnp.where(drift, hd, s["hazard_dir"])
        s["hazard_pos"] = s["hazard_pos"].at[:, 0].add(drift.astype(jnp.int32) * hd)
        safe = jnp.any(jnp.all(s["caravan_pos"][:, None, :] == s["stop_pos"][None, :, :], axis=2), axis=1)
        danger = jnp.any(jnp.sum(jnp.abs(s["caravan_pos"][:, None, :] - s["hazard_pos"][None, :, :]), axis=2) <= 1, axis=1)
        s["hit"] = ~safe & danger & (s["cargo"] > 0)
        s["cargo"] = s["cargo"] - s["hit"].astype(jnp.int32)
        s["lost"] = s["lost"] + s["hit"].astype(jnp.int32)
        failure = jnp.any(s["bank"] + s["cargo"] + s["delivered"] < 2)
        success = jnp.all(s["delivered"] == 2) & ~failure
        s["terminated"] = success | failure
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        s = self.visibility(s)
        reward = -0.01 + 0.2 * jnp.sum(s["delivered_now"]) + s["delivered_now"].astype(jnp.float32) - 0.25 * s["hit"].astype(jnp.float32) + 2.0 * success - 2.0 * failure
        return self.observe(s), s, reward.astype(jnp.float32), jnp.full(3, s["done"], jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def absorb(self, payload):
        key, s, actions = payload
        return self.observe(s), s, jnp.zeros(3, jnp.float32), jnp.ones(3, jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.transition, (key, state, actions))
