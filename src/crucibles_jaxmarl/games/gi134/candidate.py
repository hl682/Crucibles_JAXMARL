"""Tide Ledger: original three-role auction and salvage world; not executed."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 64
    num_actions = 11
    max_steps = 64

    def refresh(self, s):
        n = dict(s)
        held = s["carrier"][None, :] == jnp.arange(3)[:, None]
        n["carrying"] = jnp.where(jnp.any(held, axis=1), jnp.argmax(held, axis=1), -1).astype(jnp.int32)
        n["private_pos"] = s["agent_pos"]
        n["object_active"] = ~s["delivered"]
        return n

    def reset(self, key):
        wall = jnp.zeros((7, 9), dtype=jnp.int32)
        wall = wall.at[:, 4].set(1)
        wall = wall.at[1, 4].set(0).at[5, 4].set(0)
        budgets = jax.random.randint(jax.random.fold_in(key, 1), (3,), 4, 8, dtype=jnp.int32)
        s = {
            "t": jnp.array(0, jnp.int32),
            "done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
            "tilemap": wall,
            "agent_pos": jnp.array([[1, 1], [1, 3], [1, 5]], jnp.int32),
            "agent_active": jnp.ones((3,), jnp.bool_),
            "agent_visibility": jnp.ones((3, 3), jnp.bool_),
            "carrying": jnp.full((3,), -1, jnp.int32),
            "bids": jnp.zeros((3, 3), jnp.int32),
            "budget": budgets,
            "initial_tokens": jnp.sum(budgets, dtype=jnp.int32),
            "treasury": jnp.array(0, jnp.int32),
            "values": jax.random.randint(jax.random.fold_in(key, 0), (3, 3), 2, 9, dtype=jnp.int32),
            "private_pos": jnp.array([[1, 1], [1, 3], [1, 5]], jnp.int32),
            "private_active": jnp.ones((3,), jnp.bool_),
            "private_visibility": jnp.eye(3, dtype=jnp.bool_),
            "object_pos": jnp.array([[7, 1], [7, 3], [7, 5]], jnp.int32),
            "object_active": jnp.ones((3,), jnp.bool_),
            "object_visibility": jnp.ones((3, 3), jnp.bool_),
            "owner": jnp.full((3,), -1, jnp.int32),
            "carrier": jnp.full((3,), -1, jnp.int32),
            "delivered": jnp.zeros((3,), jnp.bool_),
            "deadline": jnp.array([28, 32, 36], jnp.int32),
            "depot_pos": jnp.array([[0, 3]], jnp.int32),
            "depot_active": jnp.ones((1,), jnp.bool_),
            "depot_visibility": jnp.ones((3, 1), jnp.bool_),
            "auction_closed": jnp.array(False),
            "picked": jnp.zeros((3,), jnp.bool_),
            "released": jnp.zeros((3,), jnp.bool_),
            "delivered_now": jnp.zeros((3,), jnp.bool_),
        }
        return self.observe(s), s

    def legal_actions(self, s):
        ids = jnp.arange(3)
        physical = s["t"] >= 6
        lot = jnp.minimum(s["t"] // 2, 2)
        mask = jnp.zeros((3, 11), jnp.bool_).at[:, 0].set(True)
        mask = mask.at[:, 1:5].set(jnp.broadcast_to(physical & ~s["done"], (3, 4)))
        same = jnp.all(s["agent_pos"][:, None, :] == s["object_pos"][None, :, :], axis=2)
        can_pick = jnp.any(same & (s["carrier"][None, :] < 0) & ~s["delivered"][None, :] &
                           ((s["owner"][None, :] < 0) | (s["owner"][None, :] == ids[:, None])), axis=1)
        at_depot = jnp.all(s["agent_pos"] == s["depot_pos"][0], axis=1)
        interact = jnp.where(s["carrying"] >= 0, at_depot, can_pick)
        mask = mask.at[:, 5].set(physical & interact & ~s["done"])
        levels = jnp.arange(1, 4)
        bid_ok = (levels[None, :] > s["bids"][:, lot, None]) & (levels[None, :] <= s["budget"][:, None])
        mask = mask.at[:, 6:9].set(bid_ok & ~physical & ~s["done"])
        can_release = jnp.any((s["owner"][None, :] == ids[:, None]) & (s["carrier"][None, :] < 0) &
                              ~s["delivered"][None, :], axis=1)
        mask = mask.at[:, 9].set(physical & can_release & ~s["done"])
        return mask.at[:, 10].set(physical & (s["carrying"] >= 0) & ~s["done"])

    def raw_step(self, key, s, actions):
        ids = jnp.arange(3)
        clipped = jnp.clip(actions, 0, 10)
        valid = (actions >= 0) & (actions < 11) & self.legal_actions(s)[ids, clipped]
        a = jnp.where(valid, clipped, 0)
        n = dict(s)
        nt = s["t"] + jnp.array(1, jnp.int32)
        n["t"] = nt
        lot = jnp.minimum(s["t"] // 2, 2)
        auction = s["t"] < 6
        new_bid = jnp.where((a >= 6) & (a <= 8) & auction, a - 5, s["bids"][:, lot])
        bids = s["bids"].at[:, lot].set(new_bid)
        closing = auction & ((s["t"] % 2) == 1)
        best = jnp.max(bids[:, lot])
        rank = (ids - lot) % 3
        winner = jnp.argmin(jnp.where(bids[:, lot] == best, rank, 9))
        award = closing & (best > 0)
        payment = jnp.where(award & (ids == winner), best, 0).astype(jnp.int32)
        n["bids"] = bids
        n["budget"] = s["budget"] - payment
        n["treasury"] = s["treasury"] + jnp.sum(payment, dtype=jnp.int32)
        owner = s["owner"].at[lot].set(jnp.where(award, winner, s["owner"][lot]))
        n["auction_closed"] = closing
        reward = -payment.astype(jnp.float32)
        # Movement attempts use old occupancy: no swaps, no following into vacated cells.
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        moving = (a >= 1) & (a <= 4)
        proposed = s["agent_pos"] + delta[jnp.where(moving, a, 0)]
        inside = jnp.all((proposed >= jnp.array([0, 0])) & (proposed <= jnp.array([8, 6])), axis=1)
        safe = jnp.clip(proposed, jnp.array([0, 0]), jnp.array([8, 6]))
        floor = s["tilemap"][safe[:, 1], safe[:, 0]] == 0
        occupied = jnp.any(jnp.all(proposed[:, None, :] == s["agent_pos"][None, :, :], axis=2) &
                           ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2), axis=1) > 1
        p = jnp.where((moving & inside & floor & ~occupied & ~contested)[:, None], proposed, s["agent_pos"])
        n["agent_pos"] = p
        # Claims expire before pickup; a carried claim cannot be confiscated.
        expiry = (nt >= s["deadline"]) & (s["carrier"] < 0) & ~s["delivered"] & (owner >= 0)
        owner = jnp.where(expiry, -1, owner)
        released = expiry
        for i in range(3):
            eligible = (owner == i) & (s["carrier"] < 0) & ~s["delivered"]
            q = jnp.argmax(eligible)
            release = (a[i] == 9) & jnp.any(eligible)
            owner = owner.at[q].set(jnp.where(release, -1, owner[q]))
            released = released.at[q].set(released[q] | release)
        carrier = s["carrier"]
        obj = s["object_pos"]
        delivered = s["delivered"]
        picked = jnp.zeros((3,), jnp.bool_)
        delivered_now = jnp.zeros((3,), jnp.bool_)
        for i in range(3):
            held = carrier == i
            h = jnp.argmax(held)
            carrying = jnp.any(held)
            at_depot = jnp.all(p[i] == s["depot_pos"][0])
            deliver = (a[i] == 5) & carrying & at_depot
            drop = (a[i] == 10) & carrying
            delivered = delivered.at[h].set(delivered[h] | deliver)
            delivered_now = delivered_now.at[h].set(delivered_now[h] | deliver)
            carrier = carrier.at[h].set(jnp.where(deliver | drop, -1, carrier[h]))
            obj = obj.at[h].set(jnp.where((deliver | drop), p[i], obj[h]))
            reward = reward + deliver.astype(jnp.float32)
            reward = reward.at[i].add(jnp.where(deliver, s["values"][i, h].astype(jnp.float32), 0.0))
            available = jnp.all(obj == p[i], axis=1) & (carrier < 0) & ~delivered & ((owner < 0) | (owner == i))
            q = jnp.argmax(available)
            pickup = (a[i] == 5) & ~carrying & jnp.any(available)
            carrier = carrier.at[q].set(jnp.where(pickup, i, carrier[q]))
            picked = picked.at[q].set(picked[q] | pickup)
        obj = jnp.where((carrier >= 0)[:, None], p[jnp.maximum(carrier, 0)], obj)
        n["object_pos"], n["owner"], n["carrier"] = obj, owner, carrier
        n["delivered"], n["picked"], n["released"], n["delivered_now"] = delivered, picked, released, delivered_now
        natural = jnp.all(delivered)
        reward = reward + 2.0 * natural.astype(jnp.float32)
        n["terminated"] = natural
        n["truncated"] = (nt >= self.max_steps) & ~natural
        n["done"] = n["terminated"] | n["truncated"]
        n = self.refresh(n)
        n = jax.tree.map(lambda old, new: jnp.where(s["done"], old, new), s, n)
        reward = jnp.where(s["done"], jnp.zeros((3,), jnp.float32), reward).astype(jnp.float32)
        dones = jnp.full((3,), n["done"], jnp.bool_)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, reward, dones, info

    def observe(self, s):
        common = jnp.concatenate((
            jnp.array([s["t"] / 64.0, (s["t"] < 6).astype(jnp.float32),
                       jnp.minimum(s["t"] // 2, 2) / 2.0, s["treasury"] / 9.0]),
            s["agent_pos"].reshape(-1) / 8.0, (s["carrying"] + 1) / 3.0,
            s["bids"].reshape(-1) / 3.0, s["object_pos"].reshape(-1) / 8.0,
            (s["owner"] + 1) / 3.0, (s["carrier"] + 1) / 3.0,
            s["delivered"].astype(jnp.float32), s["deadline"] / 64.0,
        ))
        private = jnp.concatenate((s["budget"][:, None] / 7.0, s["values"] / 8.0,
                                   jnp.eye(3, dtype=jnp.float32)), axis=1)
        events = jnp.concatenate((s["released"], s["picked"], s["delivered_now"], s["auction_closed"][None])).astype(jnp.float32)
        return jnp.concatenate((jnp.broadcast_to(common, (3, 40)), private,
                                jnp.broadcast_to(events, (3, 10)), jnp.zeros((3, 7))), axis=1).astype(jnp.float32)
