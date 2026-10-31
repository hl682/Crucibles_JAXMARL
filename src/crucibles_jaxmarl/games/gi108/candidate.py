"""Original spatial escrow barter market with private delivery orders."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 149
    num_actions = 15
    max_steps = 96

    def reset(self, key):
        agent_pos = jnp.array([[1, 3], [7, 3], [4, 5]], dtype=jnp.int32)
        destinations = jnp.array([[0, 1], [8, 1], [4, 6]], dtype=jnp.int32)
        order_kind = (jnp.arange(3, dtype=jnp.int32) + 1 + jax.random.randint(key, (3,), 0, 2, dtype=jnp.int32)) % 3
        owner = jnp.array([0, 0, 1, 1, 2, 2, -1, -1, -1], dtype=jnp.int32)
        goods = jnp.concatenate((jnp.repeat(agent_pos, 2, axis=0),
                                 jnp.array([[3, 3], [4, 3], [5, 3]], dtype=jnp.int32)), axis=0)
        terrain = jnp.zeros((7, 9), dtype=jnp.int32).at[3, 3:6].set(1)
        terrain = terrain.at[destinations[:, 1], destinations[:, 0]].set(2)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "episode_done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
            "terrain": terrain,
            "agent_pos": agent_pos, "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "role": jnp.arange(3, dtype=jnp.int32),
            "offer_good": jnp.full((3,), -1, dtype=jnp.int32),
            "ask_kind": jnp.full((3,), -1, dtype=jnp.int32),
            "offer_timer": jnp.zeros((3,), dtype=jnp.int32),
            "delivery_count": jnp.zeros((3,), dtype=jnp.int32),
            "good_pos": goods, "good_owner": owner,
            "good_kind": jnp.array([0, 0, 1, 1, 2, 2, 0, 1, 2], dtype=jnp.int32),
            "good_status": jnp.where(owner >= 0, 1, 0).astype(jnp.int32),
            "good_locked": jnp.zeros((9,), dtype=jnp.bool_),
            "good_destination": jnp.full((9,), -1, dtype=jnp.int32),
            "good_active": jnp.ones((9,), dtype=jnp.bool_),
            "good_visible": jnp.ones((3, 9), dtype=jnp.bool_),
            "destination_pos": destinations, "destination_active": jnp.ones((3,), dtype=jnp.bool_),
            "destination_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "order_pos": destinations, "order_active": jnp.ones((3,), dtype=jnp.bool_),
            "order_visible": jnp.eye(3, dtype=jnp.bool_),
            "order_kind": order_kind, "order_remaining": jnp.full((3,), 2, dtype=jnp.int32),
            "trade_count": jnp.array(0, dtype=jnp.int32),
            "pickup_count": jnp.array(0, dtype=jnp.int32),
            "expired_count": jnp.array(0, dtype=jnp.int32),
            "delivery_event": jnp.zeros((3,), dtype=jnp.int32)
        }
        return self.observe(state), state

    def observe(self, state):
        private = jnp.concatenate((jnp.eye(3, dtype=jnp.float32),
                                   jax.nn.one_hot(state["order_kind"], 3, dtype=jnp.float32),
                                   state["order_remaining"][:, None] / 2.0), axis=1)
        public = jnp.concatenate((
            state["agent_pos"].reshape(-1) / 8.0, state["good_pos"].reshape(-1) / 8.0,
            state["good_kind"] / 2.0, (state["good_owner"] + 2) / 4.0,
            state["good_status"] / 2.0, state["good_locked"].astype(jnp.float32),
            (state["offer_good"] + 1) / 9.0, (state["ask_kind"] + 1) / 3.0,
            state["offer_timer"] / 6.0, state["delivery_count"] / 2.0,
            state["destination_pos"].reshape(-1) / 8.0,
            jnp.reshape(state["t"] / 96.0, (1,)), state["terrain"].reshape(-1) / 2.0
        )).astype(jnp.float32)
        return jnp.concatenate((private, jnp.broadcast_to(public, (3, 142))), axis=1).astype(jnp.float32)

    def legal_actions(self, state):
        return jnp.where(state["episode_done"], jnp.arange(15)[None, :] == 0,
                         jnp.ones((3, 15), dtype=jnp.bool_))

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions, 0, 14)
        a = jnp.where((actions >= 0) & (actions < 15) &
                      self.legal_actions(state)[jnp.arange(3), safe], safe, 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)
        pos = jnp.clip(state["agent_pos"] + delta[a], jnp.array([0, 0]), jnp.array([8, 6]))
        owner = state["good_owner"]
        gp = state["good_pos"]
        destination = state["good_destination"]
        timer = jnp.maximum(state["offer_timer"] - 1, 0)
        offer = jnp.where(timer > 0, state["offer_good"], -1)
        ask = jnp.where((a >= 8) & (a <= 10), a - 8, state["ask_kind"])
        for i in range(3):
            posting = (a[i] >= 5) & (a[i] <= 7)
            eligible = (owner == i) & (state["good_kind"] == a[i] - 5)
            chosen = jnp.argmax(eligible)
            new_offer = jnp.where(jnp.any(eligible), chosen, -1)
            offer = offer.at[i].set(jnp.where(posting, new_offer, offer[i]))
            timer = timer.at[i].set(jnp.where(posting, jnp.where(new_offer >= 0, 6, 0), timer[i]))
        withdraw = a == 12
        offer = jnp.where(withdraw, -1, offer)
        timer = jnp.where(withdraw, 0, timer)
        ask = jnp.where(withdraw, -1, ask)
        matched = jnp.zeros((3,), dtype=jnp.bool_)
        trades = jnp.array(0, dtype=jnp.int32)
        distance = jnp.sum(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=2)
        for i in range(3):
            for j in range(3):
                gi = jnp.clip(offer[i], 0, 8)
                gj = jnp.clip(offer[j], 0, 8)
                reciprocal = (ask[i] == state["good_kind"][gj]) & (ask[j] == state["good_kind"][gi])
                valid = (offer[i] >= 0) & (offer[j] >= 0) & (owner[gi] == i) & (owner[gj] == j)
                trade = (i < j) & (a[i] == 11) & (a[j] == 11) & (distance[i, j] <= 1)
                trade = trade & valid & reciprocal & ~matched[i] & ~matched[j]
                owner = jnp.where((jnp.arange(9) == gi) & trade, j, owner)
                owner = jnp.where((jnp.arange(9) == gj) & trade, i, owner)
                matched = matched.at[i].set(matched[i] | trade)
                matched = matched.at[j].set(matched[j] | trade)
                offer = offer.at[i].set(jnp.where(trade, -1, offer[i]))
                offer = offer.at[j].set(jnp.where(trade, -1, offer[j]))
                timer = timer.at[i].set(jnp.where(trade, 0, timer[i]))
                timer = timer.at[j].set(jnp.where(trade, 0, timer[j]))
                trades = trades + trade.astype(jnp.int32)
        pickups = jnp.array(0, dtype=jnp.int32)
        for turn in range(3):
            i = (state["t"] + turn) % 3
            available = (owner == -1) & jnp.all(gp == pos[i], axis=1)
            room = jnp.sum(owner == i) < 2
            take = (a[i] == 13) & room & jnp.any(available)
            chosen = jnp.argmax(available)
            owner = jnp.where((jnp.arange(9) == chosen) & take, i, owner)
            pickups = pickups + take.astype(jnp.int32)
        locked = jnp.any((jnp.arange(9)[None, :] == offer[:, None]) & (timer > 0)[:, None], axis=0)
        remaining = state["order_remaining"]
        counts = state["delivery_count"]
        delivery = jnp.zeros((3,), dtype=jnp.int32)
        for i in range(3):
            eligible = (owner == i) & (state["good_kind"] == state["order_kind"][i]) & ~locked
            at_destination = jnp.all(pos[i] == state["destination_pos"][i])
            deliver = (a[i] == 14) & at_destination & (remaining[i] > 0) & jnp.any(eligible)
            chosen = jnp.argmax(eligible)
            consumed = (jnp.arange(9) == chosen) & deliver
            owner = jnp.where(consumed, -2, owner)
            destination = jnp.where(consumed, i, destination)
            gp = jnp.where(consumed[:, None], state["destination_pos"][i], gp)
            remaining = remaining.at[i].add(-deliver.astype(jnp.int32))
            counts = counts.at[i].add(deliver.astype(jnp.int32))
            delivery = delivery.at[i].set(deliver.astype(jnp.int32))
        gp = jnp.where((owner >= 0)[:, None], pos[jnp.clip(owner, 0, 2)], gp)
        status = jnp.where(owner >= 0, 1, jnp.where(owner == -2, 2, 0)).astype(jnp.int32)
        t = state["t"] + jnp.array(1, dtype=jnp.int32)
        terminated = jnp.all(remaining == 0)
        truncated = (t >= self.max_steps) & ~terminated
        nxt = dict(state)
        nxt["t"] = t
        nxt["agent_pos"] = pos
        nxt["offer_good"] = offer
        nxt["ask_kind"] = ask
        nxt["offer_timer"] = timer
        nxt["good_pos"] = gp
        nxt["good_owner"] = owner
        nxt["good_status"] = status
        nxt["good_locked"] = locked
        nxt["good_destination"] = destination
        nxt["order_remaining"] = remaining
        nxt["delivery_count"] = counts
        nxt["trade_count"] = trades
        nxt["pickup_count"] = pickups
        nxt["expired_count"] = jnp.sum((state["offer_timer"] == 1) & (state["offer_good"] >= 0)).astype(jnp.int32)
        nxt["delivery_event"] = delivery
        nxt["terminated"] = terminated
        nxt["truncated"] = truncated
        nxt["episode_done"] = terminated | truncated
        reward = (delivery + 0.1 * (jnp.sum(delivery) - delivery) - 0.002).astype(jnp.float32)
        nxt = jax.tree.map(lambda new, old: jnp.where(state["episode_done"], old, new), nxt, state)
        reward = jnp.where(state["episode_done"], jnp.zeros((3,), dtype=jnp.float32), reward).astype(jnp.float32)
        dones = jnp.full((3,), nxt["episode_done"], dtype=jnp.bool_)
        info = {"done_all": nxt["episode_done"], "terminated": nxt["terminated"], "truncated": nxt["truncated"]}
        return self.observe(nxt), nxt, reward, dones, info
