"""Beacon Convoy: private surveys, costly radio and action-dependent escorts."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 79
    num_actions = 13
    max_steps = 64

    def reset(self, key):
        unsafe = jax.random.randint(key, (), 0, 2, dtype=jnp.int32)
        terrain = jnp.ones((7, 11), jnp.int32)
        terrain = terrain.at[1, 1:10].set(0).at[5, 1:10].set(0)
        terrain = terrain.at[1:6, 1].set(0).at[1:6, 9].set(0)
        s = {
            "t": jnp.array(0, jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False), "terrain": terrain,
            "agent_pos": jnp.array([[1, 3], [1, 2], [1, 4]], jnp.int32),
            "agent_active": jnp.ones(3, jnp.bool_), "agent_visible": jnp.ones((3, 3), jnp.bool_),
            "fuel": jnp.array([28, 24, 24], jnp.int32), "hull": jnp.full(3, 2, jnp.int32),
            "delivered": jnp.zeros(3, jnp.bool_), "wrecked": jnp.zeros(3, jnp.bool_),
            "goal_pos": jnp.array([[9, 3], [9, 1], [9, 5]], jnp.int32),
            "goal_active": jnp.ones(3, jnp.bool_), "goal_visible": jnp.ones((3, 3), jnp.bool_),
            "hazard_pos": jnp.array([[5, 1], [5, 5]], jnp.int32),
            "hazard_active": jnp.ones(2, jnp.bool_),
            "hazard_visible": jnp.array([[True, True], [False, False], [False, False]], jnp.bool_),
            "danger": jnp.arange(2, dtype=jnp.int32) == unsafe,
            "depot_pos": jnp.array([[3, 1], [3, 5]], jnp.int32),
            "depot_active": jnp.ones(2, jnp.bool_), "depot_visible": jnp.ones((3, 2), jnp.bool_),
            "stock": jnp.full(2, 8, jnp.int32), "guard": jnp.zeros(3, jnp.bool_),
            "message": jnp.zeros(3, jnp.int32), "message_ttl": jnp.zeros(3, jnp.int32),
            "audited": jnp.zeros(3, jnp.bool_), "promise": jnp.zeros(3, jnp.int32),
            "promise_ttl": jnp.zeros(3, jnp.int32), "kept": jnp.zeros(3, jnp.int32),
            "broken": jnp.zeros(3, jnp.int32), "honest": jnp.zeros(3, jnp.int32),
            "false_signal": jnp.zeros(3, jnp.int32), "hit": jnp.zeros(3, jnp.bool_),
            "escorted": jnp.zeros(3, jnp.bool_), "transfer": jnp.zeros(3, jnp.int32),
            "arrived": jnp.zeros(3, jnp.bool_)
        }
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate((
            s["t"].reshape(1), s["agent_pos"].reshape(-1), s["fuel"], s["hull"], s["delivered"],
            s["wrecked"], s["goal_pos"].reshape(-1), s["stock"], s["guard"], s["message"],
            s["message_ttl"], s["audited"], s["promise"], s["promise_ttl"], s["kept"], s["broken"],
            s["honest"], s["false_signal"], s["hit"], s["escorted"], s["transfer"], s["arrived"],
            s["done"].reshape(1), s["terminated"].reshape(1), s["truncated"].reshape(1)
        )).astype(jnp.float32)
        known = s["hazard_visible"]
        risk = jnp.where(known, s["danger"][None, :].astype(jnp.int32), -1)
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), known.astype(jnp.float32),
                                risk.astype(jnp.float32), jnp.broadcast_to(public, (3, 72))), axis=1)

    def legal_actions(self, s):
        p = s["agent_pos"]
        d = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = p[:, None, :] + d[None, :, :]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] <= 10) & (q[:, :, 1] >= 0) & (q[:, :, 1] <= 6)
        tile = s["terrain"][jnp.clip(q[:, :, 1], 0, 6), jnp.clip(q[:, :, 0], 0, 10)]
        move = inside & (tile == 0) & (s["fuel"][:, None] > 0)
        near = jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2) <= 1
        peer = ~jnp.eye(3, dtype=jnp.bool_) & s["agent_active"][None, :] & (s["fuel"][None, :] <= 30)
        depot = jnp.all(p[:, None, :] == s["depot_pos"][None, :, :], axis=2)
        rest = jnp.stack((s["fuel"] > 0, s["fuel"] > 0, s["fuel"] > 0,
                          (s["fuel"] > 0) & (s["promise"] == 0),
                          (s["fuel"] > 0) & (s["promise"] == 0),
                          (s["fuel"] >= 2) & jnp.any(near & peer, axis=1),
                          jnp.all(p == s["goal_pos"], axis=1),
                          jnp.any(depot & (s["stock"] > 0)[None, :], axis=1) & (s["fuel"] <= 28)), axis=1)
        mask = jnp.concatenate((jnp.ones((3, 1), jnp.bool_), move, rest), axis=1)
        return mask & ((s["agent_active"] & ~s["done"])[:, None] | (jnp.arange(13)[None, :] == 0))

    def raw_step(self, key, s, actions):
        ac = jnp.clip(actions, 0, 12)
        valid = (actions >= 0) & (actions < 13) & self.legal_actions(s)[jnp.arange(3), ac]
        ac = jnp.where(valid, ac, 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        proposal = s["agent_pos"] + delta[ac]
        moving = (ac >= 1) & (ac <= 4)
        at_gate = jnp.all(proposal[:, None, :] == s["hazard_pos"][None, :, :], axis=2) & moving[:, None]
        lane = jnp.argmax(at_gate, axis=1)
        risky = jnp.any(at_gate & s["danger"][None, :], axis=1)
        guarding = ac == 7
        near_gate = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["hazard_pos"][None, :, :]), axis=2) <= 1
        helpers = (near_gate[:, lane].T & guarding[None, :] & ~jnp.eye(3, dtype=jnp.bool_))
        escorted = risky & jnp.any(helpers, axis=1)
        hit = risky & ~escorted
        pos = jnp.where(hit[:, None], s["agent_pos"], proposal)
        hull = jnp.maximum(0, s["hull"] - hit)
        wrecked = s["wrecked"] | (hull == 0)
        arrived = ac == 11
        delivered = s["delivered"] | arrived

        # Fuel transfers are physical, simultaneous, and target the lowest-fuel eligible neighbour.
        distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2)
        eligible = (distance <= 1) & ~jnp.eye(3, dtype=jnp.bool_) & s["agent_active"][None, :] & (s["fuel"][None, :] <= 30)
        target = jnp.argmin(jnp.where(eligible, s["fuel"][None, :] * 4 + jnp.arange(3)[None, :], 999), axis=1)
        gifts = jax.nn.one_hot(target, 3, dtype=jnp.int32) * (2 * (ac == 10))[:, None]
        transfer = jnp.sum(gifts, axis=0) - 2 * (ac == 10)
        fuel = jnp.maximum(0, s["fuel"] - moving - ((ac >= 5) & (ac <= 9)) - 2 * hit + transfer)
        stock = s["stock"]
        # Shared depot contention uses a public rotating priority, never a strategy oracle.
        for offset in range(3):
            i = (s["t"] + offset) % 3
            di = jnp.argmin(jnp.sum(jnp.abs(s["agent_pos"][i] - s["depot_pos"]), axis=1))
            amount = jnp.where(ac[i] == 12, jnp.minimum(4, stock[di]), 0)
            stock = stock.at[di].add(-amount)
            fuel = fuel.at[i].add(amount)
        fuel = jnp.minimum(32, fuel)

        # Radio states are claims; knowledge is separate and no legal mask reveals danger.
        signal = (ac == 5) | (ac == 6)
        message_ttl = jnp.where(signal, 8, jnp.maximum(0, s["message_ttl"] - 1))
        message = jnp.where(signal, ac - 4, jnp.where(message_ttl > 0, s["message"], 0))
        reached_lane = jnp.any(at_gate, axis=0)
        checked = (~signal & ~s["audited"] & (s["message_ttl"] > 0)
                   & reached_lane[jnp.maximum(s["message"] - 1, 0)])
        truth = ~s["danger"][jnp.maximum(s["message"] - 1, 0)]
        audited = jnp.where(signal, False, s["audited"] | checked)
        honest = s["honest"] + (checked & truth)
        false_signal = s["false_signal"] + (checked & ~truth)
        pledge = (ac == 8) | (ac == 9)
        fulfilled = jnp.any(helpers & risky[:, None] & ((lane[:, None] + 1) == s["promise"][None, :]), axis=0)
        ttl = jnp.maximum(0, s["promise_ttl"] - 1)
        broken_now = (s["promise"] > 0) & (ttl == 0) & ~fulfilled
        promise = jnp.where(pledge, ac - 7, jnp.where(fulfilled | broken_now, 0, s["promise"]))
        ttl = jnp.where(pledge, 10, jnp.where(promise > 0, ttl, 0))
        known = s["hazard_visible"] | (jnp.sum(jnp.abs(pos[:, None, :] - s["hazard_pos"][None, :, :]), axis=2) <= 1)
        t = s["t"] + 1
        term = jnp.all(delivered | wrecked)
        trunc = (t >= self.max_steps) & ~term
        rewards = (4.0 * arrived + 0.25 * jnp.sum(arrived) - hit - 2.0 * (wrecked & ~s["wrecked"])
                   - 0.125 * signal - 0.0625 * ((ac == 7) | (ac == 10)) - 0.015625 * s["agent_active"]).astype(jnp.float32)
        ns = {**s, "t": t, "done": term | trunc, "terminated": term, "truncated": trunc,
              "agent_pos": pos, "agent_active": ~(delivered | wrecked), "fuel": fuel, "hull": hull,
              "delivered": delivered, "wrecked": wrecked, "stock": stock, "guard": guarding,
              "hazard_visible": known, "message": message, "message_ttl": message_ttl, "audited": audited,
              "promise": promise, "promise_ttl": ttl, "kept": s["kept"] + fulfilled,
              "broken": s["broken"] + broken_now, "honest": honest, "false_signal": false_signal,
              "hit": hit, "escorted": escorted, "transfer": transfer, "arrived": arrived}
        ns = jax.tree.map(lambda old, new: jnp.where(s["done"], old, new), s, ns)
        rewards = jnp.where(s["done"], jnp.zeros(3, jnp.float32), rewards)
        info = {"done_all": ns["done"], "terminated": ns["terminated"], "truncated": ns["truncated"]}
        return self.observe(ns), ns, rewards, jnp.full(3, ns["done"], jnp.bool_), info
