"""Workshop Exchange: escrowed handoffs, complementary tools and timed assembly."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 94
    num_actions = 14
    max_steps = 96

    def reset(self, key):
        stations = jnp.array([[2, 2], [6, 2], [6, 6], [2, 6]], jnp.int32)
        piles = jnp.array([[4, 1], [7, 4], [4, 7], [1, 4]], jnp.int32)
        terrain = jnp.zeros((9, 9), jnp.int32).at[4, 4].set(2)
        terrain = terrain.at[stations[:, 1], stations[:, 0]].set(1)
        terrain = terrain.at[piles[:, 1], piles[:, 0]].set(3)
        s = {
            "t": jnp.array(0, jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False), "terrain": terrain,
            "agent_pos": stations, "agent_active": jnp.ones(4, jnp.bool_),
            "agent_visible": jnp.ones((4, 4), jnp.bool_), "raw": jnp.full(4, 2, jnp.int32),
            "inventory": jnp.zeros((4, 4), jnp.int32), "tool": jnp.arange(4, dtype=jnp.int32),
            "station_pos": stations, "station_active": jnp.ones(4, jnp.bool_),
            "station_visible": jnp.ones((4, 4), jnp.bool_), "job_timer": jnp.zeros(4, jnp.int32),
            "shelf": jnp.zeros(4, jnp.int32), "pile_pos": piles,
            "pile_active": jnp.ones(4, jnp.bool_), "pile_visible": jnp.ones((4, 4), jnp.bool_),
            "stock": jnp.full(4, 4, jnp.int32), "offer_live": jnp.zeros(4, jnp.bool_),
            "offer_need": jnp.full(4, -1, jnp.int32), "offer_ttl": jnp.zeros(4, jnp.int32),
            "assembly_pos": jnp.array([[4, 4]], jnp.int32), "assembly_active": jnp.ones(1, jnp.bool_),
            "assembly_visible": jnp.ones((4, 1), jnp.bool_), "assembly_parts": jnp.zeros((1, 4), jnp.int32),
            "contributors": jnp.full(4, -1, jnp.int32), "machine_timer": jnp.zeros(1, jnp.int32),
            "products": jnp.zeros(1, jnp.int32), "trade_from": jnp.full(4, -1, jnp.int32),
            "expired": jnp.zeros(4, jnp.bool_), "sold": jnp.zeros(4, jnp.bool_),
            "deposited": jnp.zeros(4, jnp.bool_), "job_done": jnp.zeros(4, jnp.bool_),
            "gathered": jnp.zeros(4, jnp.bool_)
        }
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate((
            s["t"].reshape(1), s["agent_pos"].reshape(-1), s["raw"], s["inventory"].reshape(-1),
            s["job_timer"], s["shelf"], s["stock"], s["offer_live"], s["offer_need"], s["offer_ttl"],
            s["assembly_parts"].reshape(-1), s["contributors"], s["machine_timer"], s["products"],
            s["trade_from"], s["expired"], s["sold"], s["deposited"], s["job_done"], s["gathered"],
            s["done"].reshape(1), s["terminated"].reshape(1), s["truncated"].reshape(1)
        )).astype(jnp.float32)
        return jnp.concatenate((jnp.eye(4, dtype=jnp.float32), jnp.broadcast_to(public, (4, 90))), axis=1)

    def offers(self, s):
        d = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2)
        ask = jnp.clip(s["offer_need"], 0, 3)
        payment = jnp.where((s["offer_need"] == 4)[None, :], s["raw"][:, None], s["inventory"][:, ask])
        return (d <= 1) & ~jnp.eye(4, dtype=jnp.bool_) & s["offer_live"][None, :] & (payment > 0)

    def legal_actions(self, s):
        p = s["agent_pos"]
        at_station = jnp.all(p == s["station_pos"], axis=1)
        near_pile = jnp.sum(jnp.abs(p[:, None, :] - s["pile_pos"][None, :, :]), axis=2) <= 1
        own_parts = s["inventory"][jnp.arange(4), jnp.arange(4)] > 0
        deposit = jnp.any((s["inventory"] > 0) & (s["assembly_parts"] == 0), axis=1)
        mask = jnp.stack((
            jnp.ones(4, jnp.bool_), p[:, 1] > 0, p[:, 0] < 8, p[:, 1] < 8, p[:, 0] > 0,
            jnp.any(near_pile & (s["stock"] > 0)[None, :], axis=1) & (s["raw"] < 6),
            at_station & (s["raw"] > 0) & (s["job_timer"] == 0) & (s["shelf"] == 0),
            at_station & (s["shelf"] > 0), own_parts & ~s["offer_live"], own_parts & ~s["offer_live"],
            jnp.any(self.offers(s), axis=1),
            jnp.all(p == s["assembly_pos"][0], axis=1) & (s["machine_timer"][0] == 0) & deposit,
            at_station & jnp.any(s["inventory"] > 0, axis=1), s["offer_live"]
        ), axis=1)
        return mask & (~s["done"] | (jnp.arange(14)[None, :] == 0))

    def raw_step(self, key, s, actions):
        ac = jnp.clip(actions, 0, 13)
        valid = (actions >= 0) & (actions < 14) & self.legal_actions(s)[jnp.arange(4), ac]
        ac = jnp.where(valid, ac, 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        pos = s["agent_pos"] + delta[ac]
        start = ac == 6
        collect = ac == 7
        job_done = s["job_timer"] == 1
        job_timer = jnp.where(start, 3 + (jnp.arange(4, dtype=jnp.int32) % 2), jnp.maximum(0, s["job_timer"] - 1))
        shelf = s["shelf"] + job_done - collect
        raw = s["raw"] - start
        inv = s["inventory"] + jnp.eye(4, dtype=jnp.int32) * collect[:, None]
        stock = s["stock"]
        gathered = jnp.zeros(4, jnp.bool_)
        pile_distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["pile_pos"][None, :, :]), axis=2)
        chosen_pile = jnp.argmin(jnp.where((pile_distance <= 1) & (s["stock"] > 0)[None, :], pile_distance, 999), axis=1)
        for offset in range(4):
            i = (s["t"] + offset) % 4
            pi = chosen_pile[i]
            take = (ac[i] == 5) & (stock[pi] > 0)
            stock = stock.at[pi].add(-take.astype(jnp.int32))
            raw = raw.at[i].add(take.astype(jnp.int32))
            gathered = gathered.at[i].set(take)

        # Every buyer names one pre-step adjacent offer. Clearing never asks a later policy.
        available = self.offers(s) & (ac != 13)[None, :]
        distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2)
        chosen = jnp.argmin(jnp.where(available, distance * 4 + jnp.arange(4)[None, :], 999), axis=1)
        live = s["offer_live"]
        trade_from = jnp.full(4, -1, jnp.int32)
        for offset in range(4):
            buyer = (s["t"] + offset) % 4
            seller = chosen[buyer]
            settle = (ac[buyer] == 10) & available[buyer, seller] & live[seller]
            need = s["offer_need"][seller]
            part = jnp.clip(need, 0, 3)
            component_payment = settle & (need < 4)
            raw_payment = settle & (need == 4)
            inv = inv.at[buyer, part].add(-component_payment.astype(jnp.int32))
            inv = inv.at[seller, part].add(component_payment.astype(jnp.int32))
            inv = inv.at[buyer, seller].add(settle.astype(jnp.int32))
            raw = raw.at[buyer].add(-raw_payment.astype(jnp.int32))
            raw = raw.at[seller].add(raw_payment.astype(jnp.int32))
            live = live.at[seller].set(live[seller] & ~settle)
            trade_from = trade_from.at[buyer].set(jnp.where(settle, seller, -1))
        ttl = jnp.maximum(0, s["offer_ttl"] - 1)
        expired = live & (ttl == 0)
        returned = live & (expired | (ac == 13))
        inv = inv + jnp.eye(4, dtype=jnp.int32) * returned[:, None]
        live = live & ~returned
        post = (ac == 8) | (ac == 9)
        inv = inv - jnp.eye(4, dtype=jnp.int32) * post[:, None]
        live = live | post
        need = jnp.where(post, jnp.where(ac == 8, (jnp.arange(4, dtype=jnp.int32) + 1) % 4, 4), s["offer_need"])
        need = jnp.where(live, need, -1)
        ttl = jnp.where(post, 8, jnp.where(live, ttl, 0))

        # Deposits are real parts; a full assembly takes three further raw steps.
        finish = s["machine_timer"][0] == 1
        parts = jnp.where(finish, jnp.zeros((1, 4), jnp.int32), s["assembly_parts"])
        contributors = jnp.where(finish, jnp.full(4, -1, jnp.int32), s["contributors"])
        deposited = jnp.zeros(4, jnp.bool_)
        chosen_part = jnp.argmax((s["inventory"] > 0) & (s["assembly_parts"] == 0), axis=1)
        for offset in range(4):
            i = (s["t"] + offset) % 4
            kind = chosen_part[i]
            put = (ac[i] == 11) & (parts[0, kind] == 0)
            inv = inv.at[i, kind].add(-put.astype(jnp.int32))
            parts = parts.at[0, kind].set(jnp.where(put, 1, parts[0, kind]))
            contributors = contributors.at[kind].set(jnp.where(put, i, contributors[kind]))
            deposited = deposited.at[i].set(put)
        begin = (s["machine_timer"][0] == 0) & jnp.all(parts == 1)
        timer = jnp.where(begin, jnp.array([3], jnp.int32), jnp.maximum(0, s["machine_timer"] - 1))
        products = s["products"] + finish
        sold = ac == 12
        sale_kind = jnp.argmax(s["inventory"] > 0, axis=1)
        inv = inv.at[jnp.arange(4), sale_kind].add(-sold.astype(jnp.int32))
        contribution = jnp.sum(jax.nn.one_hot(jnp.maximum(s["contributors"], 0), 4, dtype=jnp.int32)
                               * (s["contributors"] >= 0)[:, None], axis=0)
        rewards = (finish * (1.0 + contribution) + 1.5 * sold - 0.125 * start
                   - 0.0625 * (post | (ac == 13)) - 0.015625).astype(jnp.float32)
        t = s["t"] + 1
        term = products[0] >= 2
        trunc = (t >= self.max_steps) & ~term
        ns = {**s, "t": t, "done": term | trunc, "terminated": term, "truncated": trunc,
              "agent_pos": pos, "raw": raw, "inventory": inv, "job_timer": job_timer, "shelf": shelf,
              "stock": stock, "offer_live": live, "offer_need": need, "offer_ttl": ttl,
              "assembly_parts": parts, "contributors": contributors, "machine_timer": timer,
              "products": products, "trade_from": trade_from, "expired": expired,
              "sold": sold, "deposited": deposited, "job_done": job_done, "gathered": gathered}
        ns = jax.tree.map(lambda old, new: jnp.where(s["done"], old, new), s, ns)
        rewards = jnp.where(s["done"], jnp.zeros(4, jnp.float32), rewards)
        info = {"done_all": ns["done"], "terminated": ns["terminated"], "truncated": ns["truncated"]}
        return self.observe(ns), ns, rewards, jnp.full(4, ns["done"], jnp.bool_), info
