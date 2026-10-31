"""Three Quays: physical cargo, revocable quotes and time-locked conditional exchange."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 104
    num_actions = 10
    max_steps = 120

    def reset(self, key):
        tile = jnp.zeros((9, 9), dtype=jnp.int32)
        tile = tile.at[3, jnp.array([1, 2, 6, 7])].set(1)
        tile = tile.at[5, jnp.array([1, 2, 3, 5, 6, 7])].set(1)
        s = {
            "t": jnp.array(0, dtype=jnp.int32),
            "done": jnp.array(False, dtype=jnp.bool_),
            "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
            "map": tile,
            "agent_pos": jnp.array([[1, 1], [7, 1], [4, 7]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_vis": jnp.ones((3, 3), dtype=jnp.bool_),
            "home_pos": jnp.array([[1, 1], [7, 1], [4, 7]], dtype=jnp.int32),
            "home_active": jnp.ones((3,), dtype=jnp.bool_),
            "home_vis": jnp.ones((3, 3), dtype=jnp.bool_),
            "pad_pos": jnp.array([[3, 4], [5, 4], [4, 5]], dtype=jnp.int32),
            "pad_active": jnp.ones((3,), dtype=jnp.bool_),
            "pad_vis": jnp.ones((3, 3), dtype=jnp.bool_),
            "quote": jnp.array([1, 2, 0], dtype=jnp.int32),
            "demand": jnp.array([2, 0, 1], dtype=jnp.int32),
            "delivered": jnp.zeros((3,), dtype=jnp.int32),
            "goods_pos": jnp.array([[1, 1], [1, 1], [7, 1], [7, 1], [4, 7], [4, 7]], dtype=jnp.int32),
            "goods_active": jnp.ones((6,), dtype=jnp.bool_),
            "goods_vis": jnp.ones((3, 6), dtype=jnp.bool_),
            "goods_kind": jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32),
            "goods_owner": jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32),
            "goods_carrier": jnp.full((6,), -1, dtype=jnp.int32),
            "goods_escrow": jnp.full((6,), -1, dtype=jnp.int32),
            "goods_timer": jnp.zeros((6,), dtype=jnp.int32),
            "goods_target": jnp.full((6,), -1, dtype=jnp.int32),
            "offer_event": jnp.zeros((3,), dtype=jnp.bool_),
            "exchange_event": jnp.zeros((3,), dtype=jnp.bool_),
            "withdraw_event": jnp.zeros((3,), dtype=jnp.bool_),
            "gift_event": jnp.zeros((3,), dtype=jnp.bool_),
            "delivery_event": jnp.zeros((3,), dtype=jnp.bool_),
            "pickup_event": jnp.zeros((3,), dtype=jnp.bool_),
        }
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate((
            jnp.stack((s["t"], s["done"].astype(jnp.int32),
                       s["terminated"].astype(jnp.int32), s["truncated"].astype(jnp.int32))),
            s["agent_pos"].reshape(-1), s["home_pos"].reshape(-1), s["pad_pos"].reshape(-1),
            s["quote"], s["demand"], s["delivered"], s["goods_pos"].reshape(-1),
            s["goods_active"].astype(jnp.int32), s["goods_kind"], s["goods_owner"],
            s["goods_carrier"], s["goods_escrow"], s["goods_timer"], s["goods_target"],
            s["offer_event"].astype(jnp.int32), s["exchange_event"].astype(jnp.int32),
            s["withdraw_event"].astype(jnp.int32), s["gift_event"].astype(jnp.int32),
            s["delivery_event"].astype(jnp.int32), s["pickup_event"].astype(jnp.int32),
        ))
        return jnp.concatenate((jnp.arange(3, dtype=jnp.int32)[:, None],
                                jnp.broadcast_to(public, (3, 103))), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        return jnp.broadcast_to((~s["done"]) | (jnp.arange(10) == 0), (3, 10))

    def advance(self, s, actions):
        n = dict(s)
        clipped = jnp.clip(actions, 0, 9)
        valid = (actions >= 0) & (actions < 10) & self.legal_actions(s)[jnp.arange(3), clipped]
        a = jnp.where(valid, clipped, 0)
        n["goods_timer"] = jnp.maximum(s["goods_timer"] - 1, 0)
        n["offer_event"] = jnp.zeros((3,), dtype=jnp.bool_)
        n["exchange_event"] = jnp.zeros((3,), dtype=jnp.bool_)
        n["withdraw_event"] = jnp.zeros((3,), dtype=jnp.bool_)
        n["gift_event"] = jnp.zeros((3,), dtype=jnp.bool_)
        n["delivery_event"] = jnp.zeros((3,), dtype=jnp.bool_)
        n["pickup_event"] = jnp.zeros((3,), dtype=jnp.bool_)
        moves = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)

        def move(k, pos):
            i = (k + s["t"]) % 3
            target = pos[i] + moves[a[i]]
            inside = jnp.all((target >= 0) & (target <= 8))
            safe = jnp.clip(target, 0, 8)
            occupied = jnp.any(jnp.all(pos == target, axis=1) & (jnp.arange(3) != i))
            ok = inside & ~occupied & (s["map"][safe[1], safe[0]] == 0)
            return pos.at[i].set(jnp.where(ok, target, pos[i]))

        n["agent_pos"] = jax.lax.fori_loop(0, 3, move, s["agent_pos"])
        n["goods_pos"] = jnp.where((n["goods_carrier"] >= 0)[:, None],
                                   n["agent_pos"][jnp.maximum(n["goods_carrier"], 0)], n["goods_pos"])

        # A merchant cannot move a good they do not own, unless it was publicly gifted.
        def handle(k, q):
            i = (k + s["t"]) % 3
            held = (q["goods_carrier"] == i) & q["goods_active"]
            has = jnp.any(held)
            good = jnp.argmax(held)
            at_home = jnp.all(q["agent_pos"][i] == q["home_pos"][i])
            at_pad = jnp.all(q["agent_pos"][i] == q["pad_pos"][i])
            owned_offer = (q["goods_escrow"] == i) & q["goods_active"]
            has_offer = jnp.any(owned_offer)
            offered_good = jnp.argmax(owned_offer)
            fulfil = (a[i] == 5) & has & at_home & (q["goods_kind"][good] == q["demand"][i]) & (q["delivered"][i] < 2)
            drop = (a[i] == 5) & has & ~fulfil
            q["goods_active"] = q["goods_active"].at[good].set(q["goods_active"][good] & ~fulfil)
            q["goods_carrier"] = q["goods_carrier"].at[good].set(jnp.where(fulfil | drop, -1, q["goods_carrier"][good]))
            q["goods_owner"] = q["goods_owner"].at[good].set(jnp.where(fulfil, -2, q["goods_owner"][good]))
            q["delivered"] = q["delivered"].at[i].add(fulfil.astype(jnp.int32))
            q["delivery_event"] = q["delivery_event"].at[i].set(fulfil)
            available = (q["goods_active"] & (q["goods_carrier"] < 0) & (q["goods_escrow"] < 0)
                         & ((q["goods_owner"] == i) | (q["goods_owner"] == -1))
                         & jnp.all(q["goods_pos"] == q["agent_pos"][i], axis=1))
            lift = (a[i] == 5) & ~has & jnp.any(available)
            loose = jnp.argmax(available)
            q["goods_carrier"] = q["goods_carrier"].at[loose].set(jnp.where(lift, i, q["goods_carrier"][loose]))
            q["goods_owner"] = q["goods_owner"].at[loose].set(jnp.where(lift, i, q["goods_owner"][loose]))
            q["pickup_event"] = q["pickup_event"].at[i].set(lift)
            commit = (a[i] == 6) & has & at_pad & ~has_offer
            q["goods_carrier"] = q["goods_carrier"].at[good].set(jnp.where(commit, -1, q["goods_carrier"][good]))
            q["goods_escrow"] = q["goods_escrow"].at[good].set(jnp.where(commit, i, q["goods_escrow"][good]))
            q["goods_timer"] = q["goods_timer"].at[good].set(jnp.where(commit, 3, q["goods_timer"][good]))
            q["goods_target"] = q["goods_target"].at[good].set(jnp.where(commit, q["quote"][i], q["goods_target"][good]))
            q["offer_event"] = q["offer_event"].at[i].set(commit)
            change_quote = (a[i] == 7) & ~has_offer
            other_quote = jnp.where(q["quote"][i] == (i + 1) % 3, (i + 2) % 3, (i + 1) % 3)
            q["quote"] = q["quote"].at[i].set(jnp.where(change_quote, other_quote, q["quote"][i]))
            withdraw = (a[i] == 8) & at_pad & has_offer & (q["goods_timer"][offered_good] == 0)
            q["goods_escrow"] = q["goods_escrow"].at[offered_good].set(jnp.where(withdraw, -1, q["goods_escrow"][offered_good]))
            q["goods_target"] = q["goods_target"].at[offered_good].set(jnp.where(withdraw, -1, q["goods_target"][offered_good]))
            q["withdraw_event"] = q["withdraw_event"].at[i].set(withdraw)
            gift = (a[i] == 9) & has
            q["goods_carrier"] = q["goods_carrier"].at[good].set(jnp.where(gift, -1, q["goods_carrier"][good]))
            q["goods_owner"] = q["goods_owner"].at[good].set(jnp.where(gift, -1, q["goods_owner"][good]))
            q["gift_event"] = q["gift_event"].at[i].set(gift)
            return q

        n = jax.lax.fori_loop(0, 3, handle, n)

        # Mature offers settle only in an entire compatible two- or three-party cycle.
        offers = n["goods_escrow"][None, :] == jnp.arange(3)[:, None]
        present = jnp.any(offers, axis=1)
        ids = jnp.argmax(offers, axis=1)
        recipient = n["goods_target"][ids]
        kind = n["goods_kind"][ids]
        ready = present & (n["goods_timer"][ids] == 0)
        pairs = jnp.array([[0, 1], [0, 2], [1, 2]], dtype=jnp.int32)

        def match_pair(k, settled):
            i, j = pairs[k, 0], pairs[k, 1]
            ok = (ready[i] & ready[j] & ~settled[i] & ~settled[j]
                  & (recipient[i] == j) & (recipient[j] == i)
                  & (kind[i] == n["demand"][j]) & (kind[j] == n["demand"][i]))
            settled = settled.at[i].set(settled[i] | ok)
            return settled.at[j].set(settled[j] | ok)

        settled = jax.lax.fori_loop(0, 3, match_pair, jnp.zeros((3,), dtype=jnp.bool_))
        target_safe = jnp.clip(recipient, 0, 2)
        target_counts = jnp.sum((recipient[:, None] == jnp.arange(3)[None, :]).astype(jnp.int32), axis=0, dtype=jnp.int32)
        ring = (jnp.all(ready) & ~jnp.any(settled) & jnp.all(target_counts == 1)
                & jnp.all(recipient != jnp.arange(3)) & jnp.all(kind == n["demand"][target_safe]))
        settled = settled | ring
        seller = jnp.maximum(n["goods_escrow"], 0)
        transferred = (n["goods_escrow"] >= 0) & settled[seller]
        target = jnp.clip(n["goods_target"], 0, 2)
        n["exchange_event"] = jnp.any(transferred[:, None] & (target[:, None] == jnp.arange(3)[None, :]), axis=0)
        n["goods_owner"] = jnp.where(transferred, n["goods_target"], n["goods_owner"])
        n["goods_escrow"] = jnp.where(transferred, -1, n["goods_escrow"])
        n["goods_target"] = jnp.where(transferred, -1, n["goods_target"])
        n["goods_timer"] = jnp.where(transferred, 0, n["goods_timer"])
        n["goods_pos"] = jnp.where((n["goods_carrier"] >= 0)[:, None],
                                   n["agent_pos"][jnp.maximum(n["goods_carrier"], 0)], n["goods_pos"])
        success = jnp.all(n["delivered"] == 2)
        n["t"] = s["t"] + 1
        n["terminated"] = success
        n["truncated"] = (n["t"] >= self.max_steps) & ~success
        n["done"] = n["terminated"] | n["truncated"]
        deliveries = jnp.sum(n["delivery_event"].astype(jnp.int32), dtype=jnp.int32)
        rewards = (-jnp.ones((3,), dtype=jnp.int32) + 4 * n["delivery_event"].astype(jnp.int32)
                   + deliveries + n["exchange_event"].astype(jnp.int32) + 3 * success.astype(jnp.int32))
        return n, rewards.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        proposed, rewards = self.advance(state, actions)
        next_state = jax.tree.map(lambda old, new: jnp.where(state["done"], old, new), state, proposed)
        rewards = jnp.where(state["done"], jnp.zeros((3,), dtype=jnp.float32), rewards)
        dones = jnp.full((3,), next_state["done"], dtype=jnp.bool_)
        info = {"done_all": next_state["done"], "terminated": next_state["terminated"],
                "truncated": next_state["truncated"]}
        return self.observe(next_state), next_state, rewards, dones, info
