"""Crossbar Couriers: one-slot buffers, expiring packets and reserved cells."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 13
    observation_size = 83
    max_steps = 64

    def refresh(self, s):
        s = dict(s)
        s["packet_active"] = s["packet_status"] < 2
        s["agent_load"] = jnp.any((s["packet_carrier"][None, :] == jnp.arange(3)[:, None]) &
                                 (s["packet_status"][None, :] == 1), axis=1).astype(jnp.int32)
        s["agent_visible"] = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2) <= 3
        distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["packet_pos"][None, :, :]), axis=2)
        s["packet_visible"] = ((distance <= 2) | (s["packet_carrier"][None, :] == jnp.arange(3)[:, None])) & s["packet_active"][None, :]
        s["reservation_active"] = s["reservation_ttl"] > 0
        s["reservation_visible"] = jnp.broadcast_to(s["reservation_active"], (3, 3))
        return s

    def reset(self, key):
        right_or_bottom = jax.random.bernoulli(key, 0.5, (4,)).astype(jnp.int32)
        wall = (jnp.arange(9)[None, :] == 4) & ~((jnp.arange(7)[:, None] == 1) | (jnp.arange(7)[:, None] == 3) | (jnp.arange(7)[:, None] == 5))
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "episode_done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "agent_pos": jnp.array([[2, 1], [2, 5], [6, 3]], dtype=jnp.int32),
            "agent_active": jnp.ones(3, dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "agent_load": jnp.zeros(3, dtype=jnp.int32),
            "depot_pos": jnp.array([[1, 1], [1, 5], [7, 1], [7, 5]], dtype=jnp.int32),
            "depot_active": jnp.ones(4, dtype=jnp.bool_),
            "depot_visible": jnp.ones((3, 4), dtype=jnp.bool_),
            "packet_pos": jnp.array([[1, 1], [1, 5], [7, 1], [7, 5]], dtype=jnp.int32),
            "packet_status": jnp.zeros(4, dtype=jnp.int32),
            "packet_carrier": jnp.full(4, -1, dtype=jnp.int32),
            "packet_destination": jnp.array([2, 2, 0, 0], dtype=jnp.int32) + right_or_bottom,
            "packet_sponsor": jnp.array([0, 1, 2, 2], dtype=jnp.int32),
            "packet_ttl": jnp.array([36, 44, 36, 44], dtype=jnp.int32),
            "packet_active": jnp.ones(4, dtype=jnp.bool_),
            "packet_visible": jnp.zeros((3, 4), dtype=jnp.bool_),
            "reservation_pos": jnp.zeros((3, 2), dtype=jnp.int32),
            "reservation_ttl": jnp.zeros(3, dtype=jnp.int32),
            "reservation_active": jnp.zeros(3, dtype=jnp.bool_),
            "reservation_visible": jnp.zeros((3, 3), dtype=jnp.bool_),
            "reservation_owner": jnp.arange(3, dtype=jnp.int32),
            "delivery_event": jnp.zeros(4, dtype=jnp.bool_),
            "expiry_event": jnp.zeros(4, dtype=jnp.bool_),
            "handoff_event": jnp.zeros(4, dtype=jnp.bool_),
            "pickup_event": jnp.zeros(4, dtype=jnp.bool_),
            "blocked_event": jnp.zeros(3, dtype=jnp.bool_),
            "delivered_total": jnp.array(0, dtype=jnp.int32),
            "expired_total": jnp.array(0, dtype=jnp.int32),
            "map": wall.astype(jnp.int32),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        # Zero unknown values only after collecting each entity's attributes; presence bits disambiguate zero.
        agents = jnp.concatenate([s["agent_pos"].astype(jnp.float32) / 8.0,
                                  s["agent_load"][:, None].astype(jnp.float32)], axis=1)
        av = s["agent_visible"]
        local_agents = jnp.concatenate([av[:, :, None].astype(jnp.float32),
                                        jnp.where(av[:, :, None], agents[None, :, :], 0.0)], axis=2).reshape((3, 12))
        packets = jnp.concatenate([s["packet_pos"].astype(jnp.float32) / 8.0,
                                   s["packet_destination"][:, None].astype(jnp.float32) / 4.0,
                                   s["packet_ttl"][:, None].astype(jnp.float32) / 64.0,
                                   s["packet_status"][:, None].astype(jnp.float32) / 4.0,
                                   s["packet_carrier"][:, None].astype(jnp.float32) / 4.0,
                                   s["packet_sponsor"][:, None].astype(jnp.float32) / 4.0], axis=1)
        pv = s["packet_visible"]
        local_packets = jnp.concatenate([pv[:, :, None].astype(jnp.float32),
                                         jnp.where(pv[:, :, None], packets[None, :, :], 0.0)], axis=2).reshape((3, 32))
        pub = jnp.concatenate([s["depot_pos"].ravel().astype(jnp.float32) / 8.0,
                               s["reservation_pos"].ravel().astype(jnp.float32) / 8.0,
                               s["reservation_ttl"].astype(jnp.float32) / 4.0,
                               s["reservation_active"].astype(jnp.float32),
                               s["delivery_event"].astype(jnp.float32),
                               s["expiry_event"].astype(jnp.float32),
                               jnp.array([s["t"] / 64.0, s["delivered_total"] / 4.0,
                                          s["expired_total"] / 4.0, s["episode_done"]], dtype=jnp.float32)])
        own = jnp.concatenate([jnp.eye(3, dtype=jnp.float32), s["agent_pos"].astype(jnp.float32) / 8.0,
                               s["agent_load"][:, None].astype(jnp.float32),
                               s["blocked_event"][:, None].astype(jnp.float32)], axis=1)
        return jnp.concatenate([own, local_agents, local_packets, jnp.broadcast_to(pub, (3, pub.shape[0]))], axis=1)

    def legal_actions(self, s):
        return jnp.broadcast_to((jnp.arange(13) == 0) | ~s["episode_done"], (3, 13))

    def raw_step(self, key, state, actions):
        s = dict(state)
        a = jnp.where((actions >= 0) & (actions < 13), actions, 0)
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        # Broadcast reservations persist for three decision frames. New requests compete by rotating priority.
        rt = jnp.maximum(0, s["reservation_ttl"] - 1)
        rp = s["reservation_pos"]
        for k in range(3):
            i = (s["t"] + k) % 3
            target = s["agent_pos"][i] + dirs[jnp.clip(a[i] - 8, 0, 3)]
            bounds = jnp.all((target >= jnp.array([0, 0])) & (target <= jnp.array([8, 6])))
            q = jnp.clip(target, jnp.array([0, 0]), jnp.array([8, 6]))
            conflict = jnp.any((rt > 0) & (jnp.arange(3) != i) & jnp.all(rp == q, axis=1))
            reserve = (a[i] >= 8) & (a[i] <= 11) & bounds & (s["map"][q[1], q[0]] == 0) & ~conflict
            rp = rp.at[i].set(jnp.where(reserve, q, rp[i]))
            rt = rt.at[i].set(jnp.where(reserve, 3, rt[i]))
        moving = (a >= 1) & (a <= 4)
        proposal = jnp.clip(s["agent_pos"] + jnp.where(moving[:, None], dirs[jnp.clip(a - 1, 0, 3)], 0), jnp.array([0, 0]), jnp.array([8, 6]))
        wall = s["map"][proposal[:, 1], proposal[:, 0]] != 0
        reserved = jnp.any(jnp.all(proposal[:, None, :] == rp[None, :, :], axis=2) &
                           (rt > 0)[None, :] & (jnp.arange(3)[:, None] != jnp.arange(3)[None, :]), axis=1)
        # An occupied old cell remains unavailable for this whole tick: no swaps or follow-through races.
        occupied = jnp.any(jnp.all(proposal[:, None, :] == s["agent_pos"][None, :, :], axis=2) &
                           (jnp.arange(3)[:, None] != jnp.arange(3)[None, :]), axis=1)
        possible = moving & ~wall & ~reserved & ~occupied
        priority = (jnp.arange(3) - s["t"]) % 3
        duplicate = jnp.all(proposal[:, None, :] == proposal[None, :, :], axis=2)
        beaten = jnp.any(duplicate & possible[None, :] & (priority[None, :] < priority[:, None]), axis=1)
        moved = possible & ~beaten
        pos = jnp.where(moved[:, None], proposal, s["agent_pos"])
        blocked = moving & ~jnp.any(pos != s["agent_pos"], axis=1)
        status, carrier, pp = s["packet_status"], s["packet_carrier"], s["packet_pos"]
        ttl = s["packet_ttl"]
        pickup, handoff = jnp.zeros(4, dtype=jnp.bool_), jnp.zeros(4, dtype=jnp.bool_)
        # Drop/pick requests occur before offers. Only one packet may occupy each courier buffer.
        for k in range(3):
            i = (s["t"] + k) % 3
            held = (status == 1) & (carrier == i)
            drop = held & (a[i] == 12)
            status = jnp.where(drop, 0, status)
            carrier = jnp.where(drop, -1, carrier)
            pp = jnp.where(drop[:, None], pos[i], pp)
            held = (status == 1) & (carrier == i)
            nearby = (status == 0) & jnp.all(pp == pos[i], axis=1)
            p = jnp.argmax(nearby)
            take = (a[i] == 5) & ~jnp.any(held) & jnp.any(nearby)
            status = status.at[p].set(jnp.where(take, 1, status[p]))
            carrier = carrier.at[p].set(jnp.where(take, i, carrier[p]))
            pickup = pickup.at[p].set(pickup[p] | take)
        for k in range(3):
            i = (s["t"] + k) % 3
            held = (status == 1) & (carrier == i)
            loaded = jnp.any((status[None, :] == 1) & (carrier[None, :] == jnp.arange(3)[:, None]), axis=1)
            accept = (a == 7) & ~loaded & (jnp.arange(3) != i) & (jnp.sum(jnp.abs(pos - pos[i]), axis=1) == 1)
            receiver = jnp.argmin(jnp.where(accept, priority, 9))
            p = jnp.argmax(held)
            give = (a[i] == 6) & jnp.any(held) & jnp.any(accept)
            carrier = carrier.at[p].set(jnp.where(give, receiver, carrier[p]))
            handoff = handoff.at[p].set(handoff[p] | give)
        pp = jnp.where((status == 1)[:, None], pos[jnp.clip(carrier, 0, 2)], pp)
        # Arrival has priority over the same tick's expiry; final receipts are public radio messages.
        destination = s["depot_pos"][s["packet_destination"]]
        delivered = (status == 1) & jnp.all(pp == destination, axis=1)
        handler = carrier
        ttl = jnp.where(status < 2, jnp.maximum(0, ttl - 1), ttl)
        expired = (status < 2) & ~delivered & (ttl == 0)
        status = jnp.where(delivered, 2, jnp.where(expired, 3, status))
        carrier = jnp.where(delivered | expired, -1, carrier)
        s["agent_pos"], s["reservation_pos"], s["reservation_ttl"] = pos, rp, rt
        s["packet_pos"], s["packet_status"], s["packet_carrier"], s["packet_ttl"] = pp, status, carrier, ttl
        s["delivery_event"], s["expiry_event"] = delivered, expired
        s["handoff_event"], s["pickup_event"], s["blocked_event"] = handoff, pickup, blocked
        s["delivered_total"] = s["delivered_total"] + jnp.sum(delivered, dtype=jnp.int32)
        s["expired_total"] = s["expired_total"] + jnp.sum(expired, dtype=jnp.int32)
        s["t"] = state["t"] + jnp.array(1, dtype=jnp.int32)
        s["terminated"] = jnp.all(status >= 2)
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        delivered_f, expired_f = delivered.astype(jnp.float32), expired.astype(jnp.float32)
        own = (jnp.arange(3)[:, None] == s["packet_sponsor"][None, :]).astype(jnp.float32)
        handled = (jnp.arange(3)[:, None] == handler[None, :]).astype(jnp.float32)
        reward = 0.25 * jnp.sum(delivered_f - expired_f) + jnp.sum((own + handled) * delivered_f[None, :], axis=1) - 0.5 * jnp.sum(own * expired_f[None, :], axis=1)
        s = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, s)
        reward = jnp.where(state["episode_done"], jnp.zeros(3, dtype=jnp.float32), reward)
        info = {"done_all": s["episode_done"], "terminated": s["terminated"], "truncated": s["truncated"]}
        return self.observe(s), s, reward.astype(jnp.float32), jnp.full((3,), s["episode_done"], dtype=jnp.bool_), info
