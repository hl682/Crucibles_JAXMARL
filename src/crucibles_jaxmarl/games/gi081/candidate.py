"""Dockside Promises: bonded right-of-way and physical robot deliveries."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 254
    num_actions = 11
    max_steps = 128

    def bridges(self, s):
        s = dict(s)
        phase = jnp.mod(s["t"], 12)
        occupied = jnp.any(jnp.all(s["agent_pos"][:, None, :] == s["bridge_pos"][None, :, :], axis=-1), axis=0)
        inside = (s["agent_pos"][:, 1] == 4) & (s["agent_pos"][:, 0] >= 5) & (s["agent_pos"][:, 0] <= 7)
        opened = jnp.array([phase < 4, s["lease_owner"] >= 0, (phase >= 6) & (phase < 10)]) | occupied
        opened = opened.at[1].set(opened[1] | jnp.any(inside))
        s["bridge_open"] = opened
        s["bridge_owner"] = jnp.array([-1, s["lease_owner"], -1], dtype=jnp.int32)
        s["map"] = s["map"].at[s["bridge_pos"][:, 1], s["bridge_pos"][:, 0]].set(jnp.where(opened, 0, 2))
        return s

    def reset(self, key):
        grid = jnp.zeros((9, 13), dtype=jnp.int32)
        grid = grid.at[0, :].set(1).at[8, :].set(1)
        grid = grid.at[:, 0].set(1).at[:, 12].set(1).at[1:8, 6].set(1)
        starts = jnp.array([[2, 3], [10, 3], [2, 5], [10, 5]], dtype=jnp.int32)
        destinations = jnp.array([[10, 5], [2, 5], [10, 3], [2, 3]], dtype=jnp.int32)
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False), "success": jnp.array(False),
            "map": grid, "agent_pos": starts,
            "agent_active": jnp.ones((4,), dtype=jnp.bool_),
            "agent_visibility": jnp.ones((4, 4), dtype=jnp.bool_),
            "agent_battery": jnp.full((4,), 24, dtype=jnp.int32),
            "agent_credits": jnp.full((4,), 4, dtype=jnp.int32),
            "agent_cargo": -jnp.ones((4,), dtype=jnp.int32),
            "agent_facing": jnp.array([[1, 0], [-1, 0], [1, 0], [-1, 0]], dtype=jnp.int32),
            "request_ttl": jnp.zeros((4,), dtype=jnp.int32),
            "request_dir": jnp.zeros((4,), dtype=jnp.int32),
            "request_age": jnp.zeros((4,), dtype=jnp.int32),
            "agent_bond": jnp.zeros((4,), dtype=jnp.int32),
            "yield_ttl": jnp.zeros((4,), dtype=jnp.int32),
            "yield_rewarded": jnp.zeros((4,), dtype=jnp.bool_),
            "agent_completed": jnp.zeros((4,), dtype=jnp.int32),
            "lease_owner": jnp.array(-1, dtype=jnp.int32),
            "lease_ttl": jnp.array(0, dtype=jnp.int32),
            "lease_dir": jnp.array(0, dtype=jnp.int32), "lease_entered": jnp.array(False),
            "parcel_pos": jnp.concatenate([starts, starts], axis=0),
            "parcel_active": jnp.array([True, True, True, True, False, False, False, False]),
            "parcel_visibility": jnp.ones((4, 8), dtype=jnp.bool_),
            "parcel_dest": jnp.concatenate([destinations, destinations], axis=0),
            "parcel_owner": jnp.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=jnp.int32),
            "parcel_status": jnp.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=jnp.int32),
            "parcel_release": jnp.array([0, 0, 0, 0, 36, 36, 36, 36], dtype=jnp.int32),
            "parcel_deadline": jnp.array([55, 55, 55, 55, 115, 115, 115, 115], dtype=jnp.int32),
            "charger_pos": starts, "charger_active": jnp.ones((4,), dtype=jnp.bool_),
            "charger_visibility": jnp.ones((4, 4), dtype=jnp.bool_),
            "charger_busy": jnp.zeros((4,), dtype=jnp.bool_),
            "bridge_pos": jnp.array([[6, 2], [6, 4], [6, 6]], dtype=jnp.int32),
            "bridge_active": jnp.ones((3,), dtype=jnp.bool_),
            "bridge_visibility": jnp.ones((4, 3), dtype=jnp.bool_),
            "bridge_open": jnp.zeros((3,), dtype=jnp.bool_),
            "bridge_owner": -jnp.ones((3,), dtype=jnp.int32),
            "ev_request": jnp.zeros((4,), dtype=jnp.bool_),
            "ev_yield": jnp.zeros((4,), dtype=jnp.bool_),
            "ev_cross": jnp.zeros((4,), dtype=jnp.bool_),
            "ev_yield_bonus": jnp.zeros((4,), dtype=jnp.bool_),
            "ev_transfer": jnp.zeros((4,), dtype=jnp.bool_),
            "ev_swap": jnp.zeros((4,), dtype=jnp.bool_),
            "ev_pickup": jnp.zeros((8,), dtype=jnp.bool_),
            "ev_delivery": jnp.zeros((8,), dtype=jnp.bool_),
            "ev_expired": jnp.zeros((8,), dtype=jnp.bool_),
        }
        s = self.bridges(s)
        return self.observe(s), s

    def observe(self, s):
        common = jnp.concatenate([
            jnp.array([s["t"] / 128.0, jnp.mod(s["t"], 12) / 12.0, (s["lease_owner"] + 1) / 4.0,
                       s["lease_ttl"] / 8.0, s["lease_dir"].astype(jnp.float32), s["lease_entered"].astype(jnp.float32)]),
            s["map"].reshape(-1) / 2.0, s["agent_pos"].reshape(-1) / 12.0,
            s["agent_battery"] / 24.0, s["agent_credits"] / 8.0,
            (s["agent_cargo"] + 1) / 8.0, s["request_ttl"] / 10.0,
            s["request_dir"].astype(jnp.float32), s["request_age"] / 10.0,
            s["yield_ttl"] / 6.0, s["yield_rewarded"].astype(jnp.float32),
            s["agent_completed"] / 2.0, s["agent_facing"].reshape(-1).astype(jnp.float32),
            s["parcel_pos"].reshape(-1) / 12.0, s["parcel_dest"].reshape(-1) / 12.0,
            s["parcel_owner"] / 3.0, s["parcel_status"] / 4.0,
            s["parcel_release"] / 128.0, s["parcel_deadline"] / 128.0,
            s["charger_pos"].reshape(-1) / 12.0, s["bridge_open"].astype(jnp.float32),
        ]).astype(jnp.float32)
        return jnp.concatenate([jnp.broadcast_to(common[None, :], (4, 250)), jnp.eye(4, dtype=jnp.float32)], axis=1)

    def inside(self, pos):
        return (pos[..., 1] == 4) & (pos[..., 0] >= 5) & (pos[..., 0] <= 7)

    def move_options(self, s):
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        target = jnp.clip(s["agent_pos"][:, None, :] + dirs[None, :, :], jnp.array([0, 0]), jnp.array([12, 8]))
        occupied = jnp.any(jnp.all(target[:, :, None, :] == s["agent_pos"][None, None, :, :], axis=-1), axis=-1)
        corridor = self.inside(target)
        entitlement = ((jnp.arange(4) == s["lease_owner"]) | self.inside(s["agent_pos"])) & (s["yield_ttl"] == 0)
        energy = s["agent_battery"] >= 1 + (s["agent_cargo"] >= 0).astype(jnp.int32)
        ok = (s["map"][target[:, :, 1], target[:, :, 0]] == 0) & ~occupied & (~corridor | entitlement[:, None]) & energy[:, None]
        return target, ok

    def legal_actions(self, s):
        target, move = self.move_options(s)
        own = s["parcel_owner"][None, :] == jnp.arange(4)[:, None]
        near = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["parcel_pos"][None, :, :]), axis=-1) <= 1
        pick = jnp.any(own & near & (s["parcel_status"][None, :] == 1), axis=1)
        ci = jnp.maximum(0, s["agent_cargo"])
        deliver = jnp.all(s["agent_pos"] == s["parcel_dest"][ci], axis=1)
        interact = jnp.where(s["agent_cargo"] >= 0, deliver, pick)
        entrance = jnp.stack([jnp.where(s["agent_pos"][:, 0] < 6, 4, 8), jnp.full((4,), 4, dtype=jnp.int32)], axis=1)
        request = (jnp.sum(jnp.abs(s["agent_pos"] - entrance), axis=1) <= 3) & ~self.inside(s["agent_pos"]) & (s["agent_credits"] > 0) & (s["request_ttl"] == 0) & (s["lease_owner"] != jnp.arange(4)) & (s["yield_ttl"] == 0)
        yielding = ~self.inside(s["agent_pos"]) & (s["yield_ttl"] == 0)
        charge = jnp.any(jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["charger_pos"][None, :, :]), axis=-1) <= 1, axis=1) & (s["agent_battery"] < 24)
        adjacent = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=-1) == 1
        transfer = (s["agent_battery"] >= 4) & jnp.any(adjacent & (s["agent_battery"][None, :] <= 22), axis=1)
        swap = (s["agent_battery"] >= 2) & ~self.inside(s["agent_pos"]) & jnp.any(adjacent & (s["agent_battery"][None, :] >= 2) & ~self.inside(s["agent_pos"])[None, :], axis=1)
        mask = jnp.concatenate([jnp.ones((4, 1), dtype=jnp.bool_), move, interact[:, None], request[:, None], yielding[:, None], charge[:, None], transfer[:, None], swap[:, None]], axis=1)
        return jnp.where(s["done"], jnp.broadcast_to(jnp.arange(11) == 0, (4, 11)), mask)

    def absorb(self, key, s, actions):
        return self.observe(s), s, jnp.zeros((4,), dtype=jnp.float32), jnp.ones((4,), dtype=jnp.bool_), {
            "done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def advance(self, key, state, actions):
        s = dict(state)
        a = jnp.clip(actions, 0, 10)
        valid = (actions >= 0) & (actions < 11) & jnp.take_along_axis(self.legal_actions(s), a[:, None], axis=1)[:, 0]
        act = jnp.where(valid, a, 0)
        target, ok = self.move_options(s)
        chosen = target[jnp.arange(4), jnp.clip(act - 1, 0, 3)]
        moving = (act >= 1) & (act <= 4)
        priority = jnp.mod(jnp.arange(4) - s["t"], 4)
        collision = jnp.all(chosen[:, None, :] == chosen[None, :, :], axis=-1)
        lose = jnp.any(collision & moving[None, :] & (priority[None, :] < priority[:, None]), axis=1)
        moved = moving & ~lose
        oldpos = s["agent_pos"]
        s["agent_pos"] = jnp.where(moved[:, None], chosen, oldpos)
        s["agent_facing"] = jnp.where(moved[:, None], chosen - oldpos, s["agent_facing"])
        s["agent_battery"] = s["agent_battery"] - moved.astype(jnp.int32) * (1 + (s["agent_cargo"] >= 0).astype(jnp.int32))
        for name in ["ev_request", "ev_yield", "ev_cross", "ev_yield_bonus", "ev_transfer", "ev_swap"]:
            s[name] = jnp.zeros((4,), dtype=jnp.bool_)
        for name in ["ev_pickup", "ev_delivery", "ev_expired"]:
            s[name] = jnp.zeros((8,), dtype=jnp.bool_)
        s["charger_busy"] = jnp.zeros((4,), dtype=jnp.bool_)
        s["request_ttl"] = jnp.maximum(0, s["request_ttl"] - 1)
        s["request_age"] = jnp.where(s["request_ttl"] > 0, jnp.minimum(10, s["request_age"] + 1), 0)
        s["agent_bond"] = jnp.where((s["request_ttl"] == 0) & (jnp.arange(4) != s["lease_owner"]), 0, s["agent_bond"])
        s["yield_ttl"] = jnp.maximum(0, s["yield_ttl"] - 1)
        s["lease_ttl"] = jnp.maximum(0, s["lease_ttl"] - 1)
        paired = jnp.zeros((4,), dtype=jnp.bool_)
        for i in range(4):
            near = jnp.sum(jnp.abs(s["agent_pos"] - s["agent_pos"][i]), axis=1) == 1
            possible = near & (jnp.arange(4) > i) & (act == 10) & ~paired & ~self.inside(s["agent_pos"]) & (s["agent_battery"] >= 2)
            j = jnp.argmax(possible.astype(jnp.int32))
            swap = (act[i] == 10) & ~paired[i] & ~self.inside(s["agent_pos"][i]) & (s["agent_battery"][i] >= 2) & jnp.any(possible)
            pi, pj = s["agent_pos"][i], s["agent_pos"][j]
            s["agent_pos"] = s["agent_pos"].at[i].set(jnp.where(swap, pj, pi)).at[j].set(jnp.where(swap, pi, pj))
            s["agent_facing"] = s["agent_facing"].at[i].set(jnp.where(swap, pj - pi, s["agent_facing"][i])).at[j].set(jnp.where(swap, pi - pj, s["agent_facing"][j]))
            s["agent_battery"] = s["agent_battery"].at[i].add(-2 * swap.astype(jnp.int32)).at[j].add(-2 * swap.astype(jnp.int32))
            paired = paired.at[i].set(paired[i] | swap).at[j].set(paired[j] | swap)
        s["ev_swap"] = paired
        for offset in range(4):
            i = jnp.mod(s["t"] + offset, 4)
            want = act[i] == 6
            s["request_ttl"] = s["request_ttl"].at[i].set(jnp.where(want, 10, s["request_ttl"][i]))
            s["request_age"] = s["request_age"].at[i].set(jnp.where(want, 0, s["request_age"][i]))
            s["request_dir"] = s["request_dir"].at[i].set(jnp.where(want, jnp.where(s["agent_pos"][i, 0] < 6, 1, -1), s["request_dir"][i]))
            s["agent_credits"] = s["agent_credits"].at[i].add(-want.astype(jnp.int32))
            s["agent_bond"] = s["agent_bond"].at[i].set(jnp.where(want, 1, s["agent_bond"][i]))
            s["ev_request"] = s["ev_request"].at[i].set(want)
            pledge = act[i] == 7
            s["yield_ttl"] = s["yield_ttl"].at[i].set(jnp.where(pledge, 6, s["yield_ttl"][i]))
            s["yield_rewarded"] = s["yield_rewarded"].at[i].set(jnp.where(pledge, False, s["yield_rewarded"][i]))
            s["request_ttl"] = s["request_ttl"].at[i].set(jnp.where(pledge, 0, s["request_ttl"][i]))
            s["agent_bond"] = s["agent_bond"].at[i].set(jnp.where(pledge, 0, s["agent_bond"][i]))
            s["lease_owner"] = jnp.where(pledge & (s["lease_owner"] == i), -1, s["lease_owner"])
            s["ev_yield"] = s["ev_yield"].at[i].set(pledge)
            charges = (jnp.sum(jnp.abs(s["charger_pos"] - s["agent_pos"][i]), axis=1) <= 1) & ~s["charger_busy"]
            ch = jnp.argmax(charges.astype(jnp.int32))
            charge = (act[i] == 8) & jnp.any(charges)
            s["agent_battery"] = s["agent_battery"].at[i].set(jnp.minimum(24, s["agent_battery"][i] + 6 * charge.astype(jnp.int32)))
            s["charger_busy"] = s["charger_busy"].at[ch].set(s["charger_busy"][ch] | charge)
            near = jnp.sum(jnp.abs(s["agent_pos"] - s["agent_pos"][i]), axis=1) == 1
            targets = near & (s["agent_battery"] <= 22)
            other = jnp.argmin(jnp.where(targets, s["agent_battery"], 99))
            transfer = (act[i] == 9) & (s["agent_battery"][i] >= 4) & jnp.any(targets)
            s["agent_battery"] = s["agent_battery"].at[i].add(-2 * transfer.astype(jnp.int32)).at[other].add(2 * transfer.astype(jnp.int32))
            s["ev_transfer"] = s["ev_transfer"].at[i].set(transfer)
            cargo = s["agent_cargo"][i]
            ci = jnp.maximum(0, cargo)
            delivery = (act[i] == 5) & (cargo >= 0) & jnp.all(s["agent_pos"][i] == s["parcel_dest"][ci])
            candidates = (s["parcel_owner"] == i) & (s["parcel_status"] == 1) & (jnp.sum(jnp.abs(s["parcel_pos"] - s["agent_pos"][i]), axis=1) <= 1)
            pick_id = jnp.argmax(candidates.astype(jnp.int32))
            pickup = (act[i] == 5) & (cargo < 0) & jnp.any(candidates)
            s["parcel_status"] = s["parcel_status"].at[ci].set(jnp.where(delivery, 3, s["parcel_status"][ci]))
            s["parcel_status"] = s["parcel_status"].at[pick_id].set(jnp.where(pickup, 2, s["parcel_status"][pick_id]))
            s["agent_cargo"] = s["agent_cargo"].at[i].set(jnp.where(delivery, -1, jnp.where(pickup, pick_id, cargo)))
            s["agent_completed"] = s["agent_completed"].at[i].add(delivery.astype(jnp.int32))
            s["agent_credits"] = s["agent_credits"].at[i].set(jnp.minimum(8, s["agent_credits"][i] + delivery.astype(jnp.int32)))
            s["ev_delivery"] = s["ev_delivery"].at[ci].set(s["ev_delivery"][ci] | delivery)
            s["ev_pickup"] = s["ev_pickup"].at[pick_id].set(s["ev_pickup"][pick_id] | pickup)
        owner = s["lease_owner"]
        oi = jnp.maximum(0, owner)
        entered = s["lease_entered"] | ((owner >= 0) & self.inside(s["agent_pos"][oi]))
        opposite = jnp.where(s["lease_dir"] > 0, s["agent_pos"][oi, 0] >= 8, s["agent_pos"][oi, 0] <= 4)
        crossed = (owner >= 0) & entered & opposite
        s["ev_cross"] = s["ev_cross"].at[oi].set(crossed)
        s["agent_credits"] = s["agent_credits"].at[oi].set(jnp.minimum(8, s["agent_credits"][oi] + crossed.astype(jnp.int32) * s["agent_bond"][oi]))
        opposite_yielder = jnp.where(s["lease_dir"] > 0, s["agent_pos"][:, 0] >= 8, s["agent_pos"][:, 0] <= 4)
        bonus = crossed & (s["yield_ttl"] > 0) & ~s["yield_rewarded"] & opposite_yielder & (jnp.arange(4) != owner)
        s["ev_yield_bonus"] = bonus
        s["agent_credits"] = jnp.minimum(8, s["agent_credits"] + bonus.astype(jnp.int32))
        s["yield_rewarded"] = s["yield_rewarded"] | bonus
        finished = crossed | ((owner >= 0) & (s["lease_ttl"] == 0))
        s["agent_bond"] = s["agent_bond"].at[oi].set(jnp.where(finished, 0, s["agent_bond"][oi]))
        s["lease_owner"] = jnp.where(finished, -1, owner)
        s["lease_entered"] = jnp.where(finished, False, entered)
        supporters = (s["yield_ttl"][None, :] > 0) & jnp.where(s["request_dir"][:, None] > 0, s["agent_pos"][None, :, 0] >= 8, s["agent_pos"][None, :, 0] <= 4)
        eligible = (s["request_ttl"] > 0) & ((s["request_age"] >= 4) | jnp.any(supporters, axis=1)) & (s["yield_ttl"] == 0)
        rank = 100 * s["request_age"] - priority
        winner = jnp.argmax(jnp.where(eligible, rank, -9999))
        grant = (s["lease_owner"] < 0) & ~jnp.any(self.inside(s["agent_pos"])) & jnp.any(eligible)
        s["lease_owner"] = jnp.where(grant, winner, s["lease_owner"])
        s["lease_ttl"] = jnp.where(grant, 8, jnp.where(s["lease_owner"] < 0, 0, s["lease_ttl"]))
        s["lease_dir"] = jnp.where(grant, s["request_dir"][winner], s["lease_dir"])
        s["lease_entered"] = jnp.where(grant, False, s["lease_entered"])
        s["request_ttl"] = s["request_ttl"].at[winner].set(jnp.where(grant, 0, s["request_ttl"][winner]))
        cargo_match = s["agent_cargo"][:, None] == jnp.arange(8)[None, :]
        carriers = jnp.argmax(cargo_match.astype(jnp.int32), axis=0)
        s["parcel_pos"] = jnp.where((s["parcel_status"] == 2)[:, None], s["agent_pos"][carriers], s["parcel_pos"])
        s["t"] = s["t"] + jnp.array(1, dtype=jnp.int32)
        s["parcel_status"] = jnp.where((s["parcel_status"] == 0) & (s["t"] >= s["parcel_release"]), 1, s["parcel_status"])
        expired = ((s["parcel_status"] == 1) | (s["parcel_status"] == 2)) & (s["t"] > s["parcel_deadline"])
        s["ev_expired"] = expired
        s["parcel_status"] = jnp.where(expired, 4, s["parcel_status"])
        ci = jnp.maximum(0, s["agent_cargo"])
        s["agent_cargo"] = jnp.where((s["agent_cargo"] >= 0) & expired[ci], -1, s["agent_cargo"])
        s["parcel_active"] = (s["parcel_status"] == 1) | (s["parcel_status"] == 2)
        s["success"] = jnp.all(s["parcel_status"] == 3)
        s["terminated"] = jnp.all(s["parcel_status"] >= 3)
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        timeliness = jnp.maximum(0.0, (s["parcel_deadline"] - s["t"]) / s["parcel_deadline"])
        owned = s["parcel_owner"][None, :] == jnp.arange(4)[:, None]
        rewards = jnp.sum(owned * (s["ev_delivery"].astype(jnp.float32) * (1.0 + 0.5 * timeliness) - 0.5 * expired.astype(jnp.float32))[None, :], axis=1)
        rewards = rewards + 0.1 * jnp.sum(s["ev_delivery"].astype(jnp.int32)) - 0.1 * jnp.sum(expired.astype(jnp.int32)) - 0.002 * (s["agent_cargo"] >= 0).astype(jnp.float32)
        s = self.bridges(s)
        return self.observe(s), s, rewards.astype(jnp.float32), jnp.full((4,), s["done"], dtype=jnp.bool_), {
            "done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, key, state, actions)
