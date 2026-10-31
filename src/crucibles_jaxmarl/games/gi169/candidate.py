"""Mutual Cover: original unexecuted author kernel."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 79
    num_actions = 10
    max_steps = 64

    def refresh(self, s):
        s = dict(s)
        s["agent_active"] = ~(s["delivered"] | s["failed"])
        s["agent_vis"] = jnp.broadcast_to(s["agent_active"][None, :], (3, 3))
        s["cargo_pos"] = s["pos"]
        s["cargo_active"] = s["agent_active"]
        s["cargo_vis"] = s["agent_vis"]
        s["hazard_phase"] = (s["t"] + jnp.arange(3, dtype=jnp.int32)) % 4
        s["hazard_active"] = s["hazard_phase"] < 2
        s["hazard_vis"] = jnp.ones((3, 3), jnp.bool_)
        return s

    def reset(self, key):
        s = {
            "t": jnp.array(0, jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "map": jnp.zeros((7, 9), jnp.int32),
            "pos": jnp.array([[1, 1], [1, 3], [1, 5]], jnp.int32),
            "health": jnp.full(3, 3, jnp.int32), "cash": jnp.full(3, 6, jnp.int32),
            "escrow": jnp.zeros(3, jnp.int32), "partner": jnp.full(3, -1, jnp.int32),
            "pact_ttl": jnp.zeros(3, jnp.int32), "shield": jnp.zeros(3, jnp.int32),
            "delivered": jnp.zeros(3, jnp.bool_), "failed": jnp.zeros(3, jnp.bool_),
            "offer_to": jnp.full(3, -1, jnp.int32), "offer_ttl": jnp.zeros(3, jnp.int32),
            "escort_to": jnp.full(3, -1, jnp.int32),
            "agent_active": jnp.ones(3, jnp.bool_), "agent_vis": jnp.ones((3, 3), jnp.bool_),
            "cargo_pos": jnp.array([[1, 1], [1, 3], [1, 5]], jnp.int32),
            "cargo_active": jnp.ones(3, jnp.bool_), "cargo_vis": jnp.ones((3, 3), jnp.bool_),
            "goal_pos": jnp.array([[7, 1], [7, 3], [7, 5]], jnp.int32),
            "goal_active": jnp.ones(3, jnp.bool_), "goal_vis": jnp.ones((3, 3), jnp.bool_),
            "hazard_pos": jnp.array([[4, 1], [4, 3], [4, 5]], jnp.int32),
            "hazard_active": jnp.zeros(3, jnp.bool_), "hazard_vis": jnp.ones((3, 3), jnp.bool_),
            "hazard_phase": jnp.zeros(3, jnp.int32), "damage": jnp.zeros(3, jnp.int32),
            "delivery_event": jnp.zeros(3, jnp.bool_), "settlement_event": jnp.zeros(3, jnp.bool_),
            "repair_event": jnp.zeros(3, jnp.bool_), "offer_event": jnp.zeros(3, jnp.bool_),
            "accept_event": jnp.zeros(3, jnp.bool_)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def nearby(self, s, i):
        d = jnp.sum(jnp.abs(s["pos"] - s["pos"][i]), axis=1)
        eligible = (jnp.arange(3) != i) & s["agent_active"]
        target = jnp.argmin(jnp.where(eligible, d, 99))
        return target, jnp.where(eligible[target], d[target], 99)

    def repair_target(self, s, i):
        d = jnp.sum(jnp.abs(s["pos"] - s["pos"][i]), axis=1)
        injured = s["agent_active"] & (s["health"] < 3) & (d <= 1)
        # Nearby injured peer first; self is the fallback.
        score = jnp.where(injured, d + 3 * (jnp.arange(3) == i), 99)
        r = jnp.argmin(score)
        return r, jnp.any(injured)

    def legal_actions(self, s):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        mask = jnp.zeros((3, 10), jnp.bool_).at[:, 0].set(True)
        for i in range(3):
            live = ~s["done"] & s["agent_active"][i]
            for a in range(1, 5):
                p = s["pos"][i] + delta[a]
                mask = mask.at[i, a].set(live & jnp.all((p >= 0) & (p <= jnp.array([8, 6]))))
            q, d = self.nearby(s, i)
            r, hurt = self.repair_target(s, i)
            mask = mask.at[i, 5].set(live & (s["cash"][i] >= 2) & (s["shield"][i] == 0))
            mask = mask.at[i, 6].set(live & (s["cash"][i] >= 1) & hurt)
            mask = mask.at[i, 7].set(live & (d <= 2) & (s["partner"][i] < 0) & (s["partner"][q] < 0) & (s["cash"][i] >= 2))
            dist = jnp.sum(jnp.abs(s["pos"] - s["pos"][i]), axis=1)
            incoming = (s["offer_to"] == i) & (s["offer_ttl"] > 0) & s["agent_active"] & (s["partner"] < 0) & (s["cash"] >= 2) & (dist <= 2)
            mask = mask.at[i, 8].set(live & (s["partner"][i] < 0) & (s["cash"][i] >= 2) & jnp.any(incoming))
            mask = mask.at[i, 9].set(live & (d <= 1) & (s["cash"][i] >= 1))
        return mask

    def observe(self, s):
        public = jnp.concatenate((s["pos"].reshape(-1), s["health"], s["cash"], s["escrow"], s["partner"], s["pact_ttl"], s["shield"], s["delivered"].astype(jnp.int32), s["failed"].astype(jnp.int32), s["offer_to"], s["offer_ttl"], s["escort_to"], s["goal_pos"].reshape(-1), s["hazard_pos"].reshape(-1), s["hazard_active"].astype(jnp.int32), s["hazard_phase"], s["t"].reshape(1), s["damage"], s["delivery_event"].astype(jnp.int32), s["settlement_event"].astype(jnp.int32), s["repair_event"].astype(jnp.int32), s["offer_event"].astype(jnp.int32), s["accept_event"].astype(jnp.int32)))
        return jnp.concatenate((jnp.eye(3, dtype=jnp.int32), jnp.broadcast_to(public[None, :], (3, 76))), axis=1).astype(jnp.float32)

    def raw_step(self, key, s, actions):
        ac = jnp.clip(actions, 0, 9)
        a = jnp.where((actions >= 0) & (actions < 10) & self.legal_actions(s)[jnp.arange(3), ac], ac, 0)
        n = dict(s)
        n["t"] = s["t"] + jnp.array(1, jnp.int32)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        n["pos"] = s["pos"] + delta[a]
        n["shield"] = jnp.maximum(s["shield"] - 1, 0)
        n["offer_ttl"] = jnp.maximum(s["offer_ttl"] - 1, 0)
        n["offer_to"] = jnp.where(n["offer_ttl"] > 0, s["offer_to"], -1)
        n["pact_ttl"] = jnp.maximum(s["pact_ttl"] - 1, 0)
        n["escort_to"] = jnp.full(3, -1, jnp.int32)
        n["repair_event"] = jnp.zeros(3, jnp.bool_)
        n["offer_event"] = jnp.zeros(3, jnp.bool_)
        n["accept_event"] = jnp.zeros(3, jnp.bool_)
        # COMMITMENT: acceptance locks two actual coins per member, no unsecured promise.
        for k in range(3):
            i = (s["t"] + k) % 3
            dist = jnp.sum(jnp.abs(n["pos"] - n["pos"][i]), axis=1)
            incoming = (s["offer_to"] == i) & (s["offer_ttl"] > 0) & n["agent_active"] & (n["partner"] < 0) & (n["cash"] >= 2) & (dist <= 2)
            j = jnp.argmin(jnp.where(incoming, jnp.arange(3), 99))
            accept = (a[i] == 8) & (n["partner"][i] < 0) & (n["cash"][i] >= 2) & jnp.any(incoming)
            n["cash"] = n["cash"].at[i].add(-2 * accept.astype(jnp.int32)).at[j].add(-2 * accept.astype(jnp.int32))
            n["escrow"] = n["escrow"].at[i].set(jnp.where(accept, 2, n["escrow"][i])).at[j].set(jnp.where(accept, 2, n["escrow"][j]))
            n["partner"] = n["partner"].at[i].set(jnp.where(accept, j, n["partner"][i])).at[j].set(jnp.where(accept, i, n["partner"][j]))
            n["pact_ttl"] = n["pact_ttl"].at[i].set(jnp.where(accept, 6, n["pact_ttl"][i])).at[j].set(jnp.where(accept, 6, n["pact_ttl"][j]))
            n["offer_ttl"] = n["offer_ttl"].at[j].set(jnp.where(accept, 0, n["offer_ttl"][j]))
            n["offer_to"] = n["offer_to"].at[j].set(jnp.where(accept, -1, n["offer_to"][j]))
            n["accept_event"] = n["accept_event"].at[i].set(accept)
        # CHOICES: offers, protection, repair and current-tick escort all use positions.
        for k in range(3):
            i = (s["t"] + k) % 3
            q, d = self.nearby(n, i)
            offer = (a[i] == 7) & (d <= 2) & (n["partner"][i] < 0) & (n["partner"][q] < 0) & (n["cash"][i] >= 2)
            n["offer_to"] = n["offer_to"].at[i].set(jnp.where(offer, q, n["offer_to"][i]))
            n["offer_ttl"] = n["offer_ttl"].at[i].set(jnp.where(offer, 3, n["offer_ttl"][i]))
            n["offer_event"] = n["offer_event"].at[i].set(offer)
            insure = (a[i] == 5) & (n["cash"][i] >= 2)
            n["cash"] = n["cash"].at[i].add(-2 * insure.astype(jnp.int32))
            n["shield"] = n["shield"].at[i].set(jnp.where(insure, 3, n["shield"][i]))
            r, hurt = self.repair_target(n, i)
            repair = (a[i] == 6) & (n["cash"][i] >= 1) & hurt
            n["cash"] = n["cash"].at[i].add(-repair.astype(jnp.int32))
            n["health"] = n["health"].at[r].set(jnp.where(repair, jnp.minimum(n["health"][r] + 1, 3), n["health"][r]))
            n["repair_event"] = n["repair_event"].at[r].set(n["repair_event"][r] | repair)
            escort = (a[i] == 9) & (d <= 1) & (n["cash"][i] >= 1)
            n["cash"] = n["cash"].at[i].add(-escort.astype(jnp.int32))
            n["escort_to"] = n["escort_to"].at[i].set(jnp.where(escort, q, -1))
        # HAZARD: public periodic cells; loss depends on actual position and protection.
        n = self.refresh(n)
        on = jnp.all(n["pos"][:, None, :] == n["hazard_pos"][None, :, :], axis=2)
        danger = jnp.any(on & n["hazard_active"][None, :], axis=1) & n["agent_active"]
        escort_count = jnp.sum(n["escort_to"][:, None] == jnp.arange(3)[None, :], axis=0)
        n["damage"] = jnp.where(danger, jnp.maximum(2 - (n["shield"] > 0).astype(jnp.int32) - escort_count, 0), 0)
        n["health"] = jnp.maximum(n["health"] - n["damage"], 0)
        new_failure = n["agent_active"] & (n["health"] == 0)
        n["failed"] = s["failed"] | new_failure
        n["delivery_event"] = n["agent_active"] & ~new_failure & jnp.all(n["pos"] == n["goal_pos"], axis=1)
        n["delivered"] = s["delivered"] | n["delivery_event"]
        # SETTLEMENT: one casualty receives all four; two receive two each; expiry/delivery refunds.
        n["settlement_event"] = jnp.zeros(3, jnp.bool_)
        for i in range(3):
            for j in range(i + 1, 3):
                linked = (n["partner"][i] == j) & (n["partner"][j] == i)
                hi, hj = n["damage"][i] > 0, n["damage"][j] > 0
                finish = n["delivered"][i] | n["delivered"][j] | n["failed"][i] | n["failed"][j]
                settle = linked & (hi | hj | finish | (n["pact_ttl"][i] == 0))
                pay_i = jnp.where(hi & ~hj, 4, jnp.where(hj & ~hi, 0, 2))
                pay_j = 4 - pay_i
                n["cash"] = n["cash"].at[i].add(jnp.where(settle, pay_i, 0)).at[j].add(jnp.where(settle, pay_j, 0))
                n["escrow"] = n["escrow"].at[i].set(jnp.where(settle, 0, n["escrow"][i])).at[j].set(jnp.where(settle, 0, n["escrow"][j]))
                n["partner"] = n["partner"].at[i].set(jnp.where(settle, -1, n["partner"][i])).at[j].set(jnp.where(settle, -1, n["partner"][j]))
                n["pact_ttl"] = n["pact_ttl"].at[i].set(jnp.where(settle, 0, n["pact_ttl"][i])).at[j].set(jnp.where(settle, 0, n["pact_ttl"][j]))
                n["settlement_event"] = n["settlement_event"].at[i].set(n["settlement_event"][i] | settle).at[j].set(n["settlement_event"][j] | settle)
        reward = n["cash"] + n["escrow"] - s["cash"] - s["escrow"] + 10 * n["delivery_event"].astype(jnp.int32) - 6 * new_failure.astype(jnp.int32) - 2 * n["damage"]
        n["terminated"] = jnp.all(n["delivered"] | n["failed"])
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["done"] = n["terminated"] | n["truncated"]
        n = self.refresh(n)
        n = {k: jnp.where(s["done"], s[k], n[k]) for k in s}
        reward = jnp.where(s["done"], 0, reward).astype(jnp.float32)
        return self.observe(n), n, reward, jnp.full(3, n["done"], jnp.bool_), {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
