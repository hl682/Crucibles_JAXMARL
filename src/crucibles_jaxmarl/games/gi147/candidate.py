"""Harbor barter and complementary cargo. Original author kernel; unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 146
    num_actions = 14
    max_steps = 90

    def reset(self, key):
        # PRIVATE_DEMANDS: each type is required by exactly two roles; permutation is hidden.
        missing = jax.random.permutation(key, jnp.arange(3, dtype=jnp.int32))
        demand = (jnp.arange(3)[None, :] != missing[:, None]).astype(jnp.int32)
        xy = jnp.stack((jnp.arange(63) % 9, jnp.arange(63) // 9), axis=1)
        x, y = xy[:, 0], xy[:, 1]
        wall = (x == 0) | (x == 8) | (y == 0) | (y == 6) | ((x == 4) & (y <= 3))
        tiles = jnp.where(wall, 1, 0).reshape(7, 9).astype(jnp.int32)
        tiles = tiles.at[1, 3].set(3).at[1, 5].set(3).at[2, 3].set(2).at[2, 5].set(2)
        pos = jnp.array([[2, 4], [3, 4], [5, 4]], jnp.int32)
        owner = jnp.array([0, 0, 1, 1, 2, 2], jnp.int32)
        s = {
            "t": jnp.array(0, jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "tiles": tiles, "agent_pos": pos, "agent_active": jnp.ones(3, jnp.bool_),
            "agent_vis": jnp.zeros((3, 3), jnp.bool_),
            "demand": demand, "delivered_counts": jnp.zeros((3, 3), jnp.int32),
            "cooldown": jnp.zeros(3, jnp.int32), "offer": jnp.full(3, -1, jnp.int32),
            "offer_ttl": jnp.zeros(3, jnp.int32), "inventory_count": jnp.full(3, 2, jnp.int32),
            "finished": jnp.zeros(3, jnp.bool_), "private_credit": jnp.zeros(3, jnp.int32),
            "cargo_pos": pos[owner], "cargo_active": jnp.ones(6, jnp.bool_),
            "cargo_vis": jnp.zeros((3, 6), jnp.bool_),
            "cargo_type": jnp.array([0, 0, 1, 1, 2, 2], jnp.int32),
            "cargo_owner": owner, "cargo_delivered_by": jnp.full(6, -1, jnp.int32),
            "dock_pos": jnp.array([[3, 1], [5, 1]], jnp.int32),
            "dock_active": jnp.ones(2, jnp.bool_), "dock_vis": jnp.ones((3, 2), jnp.bool_),
            "dock_owner": jnp.full(2, -1, jnp.int32), "dock_timer": jnp.zeros(2, jnp.int32),
            "trade_event": jnp.zeros(3, jnp.bool_), "delivery_event": jnp.zeros(3, jnp.bool_),
            "reservation_event": jnp.zeros(2, jnp.bool_),
            "cargo_total": jnp.array(6, jnp.int32), "type_totals": jnp.full(3, 2, jnp.int32),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def refresh(self, state):
        s = dict(state)
        p = s["agent_pos"]
        s["agent_vis"] = jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2) <= 3
        s["cargo_vis"] = jnp.sum(jnp.abs(p[:, None, :] - s["cargo_pos"][None, :, :]), axis=2) <= 3
        s["inventory_count"] = jnp.sum((s["cargo_owner"][None, :] == jnp.arange(3)[:, None]).astype(jnp.int32), axis=1)
        s["private_credit"] = jnp.sum(s["delivered_counts"], axis=1)
        s["finished"] = jnp.all(s["delivered_counts"] >= s["demand"], axis=1)
        s["agent_active"] = ~s["finished"]
        s["cargo_total"] = jnp.sum(s["cargo_active"].astype(jnp.int32))
        s["type_totals"] = jnp.sum((s["cargo_type"][None, :] == jnp.arange(3)[:, None]).astype(jnp.int32), axis=1)
        return s

    def records(self, position, attributes, visibility):
        values = jnp.concatenate((position, attributes), axis=1)
        hidden = jnp.where(visibility[:, :, None], values[None, :, :], -1)
        return jnp.concatenate((visibility[:, :, None].astype(jnp.int32), hidden), axis=2).reshape(3, -1)

    def observe(self, state):
        s = state
        public = jnp.stack((s["t"], s["done"].astype(jnp.int32), s["cargo_total"]))
        own = jnp.concatenate((jnp.arange(3)[:, None], s["agent_pos"], s["cooldown"][:, None],
                               s["demand"], s["delivered_counts"]), axis=1)
        agents = self.records(s["agent_pos"], jnp.stack((s["cooldown"], s["offer"], s["offer_ttl"], s["inventory_count"], s["finished"]), axis=1), s["agent_vis"])
        cargo = self.records(s["cargo_pos"], jnp.stack((s["cargo_type"], s["cargo_owner"], s["cargo_delivered_by"]), axis=1), s["cargo_vis"])
        docks = self.records(s["dock_pos"], jnp.stack((s["dock_owner"], s["dock_timer"]), axis=1), s["dock_vis"])
        return jnp.concatenate((jnp.broadcast_to(public, (3, 3)), own,
                                jnp.broadcast_to(s["tiles"].reshape(1, 63), (3, 63)),
                                agents, cargo, docks), axis=1).astype(jnp.float32)

    def legal_actions(self, state):
        s = state
        p = s["agent_pos"]
        directions = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]], jnp.int32)
        target = p[:, None, :] + directions[None, :, :]
        x, y = jnp.clip(target[:, :, 0], 0, 8), jnp.clip(target[:, :, 1], 0, 6)
        inside = (target[:, :, 0] >= 0) & (target[:, :, 0] <= 8) & (target[:, :, 1] >= 0) & (target[:, :, 1] <= 6)
        moves = inside & (s["tiles"][y, x] != 1) & (s["cooldown"][:, None] == 0)
        inventory = s["cargo_owner"][None, :] == jnp.arange(3)[:, None]
        offers = jnp.any(inventory[:, :, None] & (s["cargo_type"][None, :, None] == jnp.arange(3)[None, None, :]), axis=1)
        neighbor = jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2) == 1
        ids = jnp.arange(3)
        accept_next = neighbor[ids, (ids + 1) % 3] & s["agent_active"][(ids + 1) % 3]
        accept_prev = neighbor[ids, (ids + 2) % 3] & s["agent_active"][(ids + 2) % 3]
        near_dock = jnp.any(jnp.sum(jnp.abs(p[:, None, :] - s["dock_pos"][None, :, :]), axis=2) <= 1, axis=1)
        at_dock = jnp.any(jnp.all(p[:, None, :] == s["dock_pos"][None, :, :], axis=2), axis=1)
        need = s["demand"] > s["delivered_counts"]
        deliverable = jnp.any(inventory & need[jnp.arange(3)[:, None], s["cargo_type"][None, :]], axis=1)
        owned_lease = jnp.any(s["dock_owner"][None, :] == ids[:, None], axis=1)
        extra = jnp.concatenate((offers, jnp.ones((3, 1), jnp.bool_),
                                 jnp.stack((accept_next, accept_prev, near_dock, at_dock & deliverable, owned_lease), axis=1)), axis=1)
        mask = jnp.concatenate((jnp.ones((3, 1), jnp.bool_), moves, extra), axis=1)
        return jnp.where((s["done"] | s["finished"])[:, None], jnp.arange(14)[None, :] == 0, mask)

    def raw_step(self, key, state, actions):
        actions = jnp.asarray(actions, jnp.int32)
        clipped = jnp.clip(actions, 0, 13)
        valid = (actions == clipped) & self.legal_actions(state)[jnp.arange(3), clipped]
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, jnp.where(valid, clipped, 0)))

    def absorb(self, bundle):
        key, s, a = bundle
        return self.observe(s), s, jnp.zeros(3, jnp.float32), jnp.ones(3, jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def advance(self, bundle):
        key, old, a = bundle
        s = dict(old)
        ids = jnp.arange(3)
        rank = (ids - old["t"]) % 3
        p = old["agent_pos"]
        directions = jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], jnp.int32)
        target = p + directions[jnp.minimum(a, 4)] * (a <= 4)[:, None]
        at_gate = jnp.all(target[:, None, :] == old["dock_pos"][None, :, :], axis=2)
        lease_denied = jnp.any(at_gate & (old["dock_owner"][None, :] >= 0) & (old["dock_owner"][None, :] != ids[:, None]), axis=1)
        moving = (a >= 1) & (a <= 4) & ~lease_denied
        target = jnp.where(moving[:, None], target, p)
        occupied = jnp.any(jnp.all(target[:, None, :] == p[None, :, :], axis=2) & old["agent_active"][None, :] & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        same = jnp.all(target[:, None, :] == target[None, :, :], axis=2)
        contested = jnp.sum((same & old["agent_active"][None, :]).astype(jnp.int32), axis=1) > 1
        moved = moving & ~occupied & ~contested
        p = jnp.where(moved[:, None], target, p)
        s["agent_pos"] = p
        # ROUTE_COST: entering a shallow tile requires one stationary transition afterwards.
        s["cooldown"] = jnp.where(moved & (old["tiles"][p[:, 1], p[:, 0]] == 2), 1, jnp.maximum(old["cooldown"] - 1, 0))
        # CONSENSUAL_BARTER: both actors name each other; both old offers must be live.
        partner = jnp.where(a == 9, (ids + 1) % 3, jnp.where(a == 10, (ids + 2) % 3, -1))
        owner = old["cargo_owner"]
        has_offer = jnp.any((owner[None, :] == ids[:, None]) & (old["cargo_type"][None, :] == old["offer"][:, None]), axis=1)
        traded = jnp.zeros(3, jnp.bool_)
        for i in range(3):
            for j in range(3):
                mutual = (i < j) & (partner[i] == j) & (partner[j] == i)
                close = jnp.sum(jnp.abs(p[i] - p[j])) == 1
                trade = mutual & close & has_offer[i] & has_offer[j] & (old["offer_ttl"][i] > 0) & (old["offer_ttl"][j] > 0)
                ci = jnp.argmax((old["cargo_owner"] == i) & (old["cargo_type"] == old["offer"][i]))
                cj = jnp.argmax((old["cargo_owner"] == j) & (old["cargo_type"] == old["offer"][j]))
                owner = owner.at[ci].set(jnp.where(trade, j, owner[ci]))
                owner = owner.at[cj].set(jnp.where(trade, i, owner[cj]))
                traded = traded.at[i].set(traded[i] | trade).at[j].set(traded[j] | trade)
        fresh_offer = (a >= 5) & (a <= 7)
        ttl = jnp.where(fresh_offer, 4, jnp.maximum(old["offer_ttl"] - 1, 0))
        clear = (a == 8) | traded | (ttl == 0)
        s["offer_ttl"] = jnp.where(clear, 0, ttl)
        s["offer"] = jnp.where(clear, -1, jnp.where(fresh_offer, a - 5, old["offer"]))
        s["trade_event"] = traded
        # LEASES: old lease controls movement; release/expiry then new reservations apply.
        timers = jnp.maximum(old["dock_timer"] - 1, 0)
        lease = jnp.where(timers > 0, old["dock_owner"], -1)
        release = (old["dock_owner"] >= 0) & (a[jnp.maximum(old["dock_owner"], 0)] == 13)
        lease = jnp.where(release, -1, lease)
        timers = jnp.where(release, 0, timers)
        near = jnp.sum(jnp.abs(p[:, None, :] - old["dock_pos"][None, :, :]), axis=2) <= 1
        selected = jnp.argmax(near, axis=1)
        reservations = jnp.zeros(2, jnp.bool_)
        for d in range(2):
            wants = (a == 11) & near[:, d] & (selected == d)
            # An occupied dock can only be booked by its present occupant.
            occupant = jnp.all(p == old["dock_pos"][d], axis=1) & old["agent_active"]
            wants = wants & (~jnp.any(occupant) | occupant)
            winner = jnp.argmin(jnp.where(wants, rank, 9))
            grant = (lease[d] < 0) & jnp.any(wants)
            lease = lease.at[d].set(jnp.where(grant, winner, lease[d]))
            timers = timers.at[d].set(jnp.where(grant, 3, timers[d]))
            reservations = reservations.at[d].set(grant)
        # DELIVERY: lowest-id demanded item is consumed into a permanent dock ledger.
        delivered = jnp.zeros(3, jnp.bool_)
        cargo_pos = old["cargo_pos"]
        delivered_by = old["cargo_delivered_by"]
        counts = old["delivered_counts"]
        for i in range(3):
            dock_here = jnp.all(p[i] == old["dock_pos"], axis=1)
            access = jnp.any(dock_here & ((old["dock_owner"] < 0) | (old["dock_owner"] == i)))
            need = old["demand"][i] > counts[i]
            eligible = (owner == i) & need[old["cargo_type"]]
            item = jnp.argmax(eligible)
            deliver = (a[i] == 12) & access & jnp.any(eligible)
            kind = old["cargo_type"][item]
            counts = counts.at[i, kind].add(deliver.astype(jnp.int32))
            owner = owner.at[item].set(jnp.where(deliver, -1, owner[item]))
            delivered_by = delivered_by.at[item].set(jnp.where(deliver, i, delivered_by[item]))
            cargo_pos = cargo_pos.at[item].set(jnp.where(deliver, p[i], cargo_pos[item]))
            delivered = delivered.at[i].set(deliver)
        cargo_pos = jnp.where((owner >= 0)[:, None], p[jnp.maximum(owner, 0)], cargo_pos)
        s["cargo_owner"], s["cargo_pos"], s["cargo_delivered_by"] = owner, cargo_pos, delivered_by
        s["delivered_counts"], s["delivery_event"] = counts, delivered
        s["dock_owner"], s["dock_timer"], s["reservation_event"] = lease, timers, reservations
        s = self.refresh(s)
        # Completed boats depart physically and retain only NOOP until joint completion.
        abandoned = (s["dock_owner"] >= 0) & s["finished"][jnp.maximum(s["dock_owner"], 0)]
        s["dock_owner"] = jnp.where(abandoned, -1, s["dock_owner"])
        s["dock_timer"] = jnp.where(abandoned, 0, s["dock_timer"])
        s["t"] = old["t"] + 1
        s["terminated"] = jnp.all(s["finished"])
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        newly_finished = s["finished"] & ~old["finished"]
        reward = (6 * delivered.astype(jnp.int32) + 3 * newly_finished.astype(jnp.int32) - old["agent_active"].astype(jnp.int32)).astype(jnp.float32)
        dones = s["finished"] | s["done"]
        info = {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}
        return self.observe(s), s, reward, dones, info

