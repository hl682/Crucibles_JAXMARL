import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 224
    num_actions = 11
    max_steps = 128

    def reset(self, key):
        # EVIDENCE: original_river_and_actual_carts
        terrain = jnp.zeros((7, 9), dtype=jnp.int32).at[:, 3:6].set(1)
        bridge_pos = jnp.stack((3 + jnp.mod(jnp.arange(21), 3), jnp.arange(21) // 3), axis=1).astype(jnp.int32)
        start = jnp.array([[1, 1], [1, 3], [1, 5]], dtype=jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "map": terrain,
            "agent_pos": start,
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "wood": jnp.full((3,), 4, dtype=jnp.int32),
            "facing": jnp.full((3,), 2, dtype=jnp.int32),
            "credits": jnp.zeros((3,), dtype=jnp.int32),
            "carried": jnp.ones((3,), dtype=jnp.bool_),
            "bridge_pos": bridge_pos,
            "bridge_active": jnp.zeros((21,), dtype=jnp.bool_),
            "bridge_visible": jnp.ones((3, 21), dtype=jnp.bool_),
            "durability": jnp.zeros((21,), dtype=jnp.int32),
            "bridge_owner": jnp.full((21,), -1, dtype=jnp.int32),
            "pending": jnp.zeros((21,), dtype=jnp.bool_),
            "cargo_pos": start,
            "cargo_active": jnp.ones((3,), dtype=jnp.bool_),
            "cargo_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "cargo_owner": jnp.arange(3, dtype=jnp.int32),
            "delivered": jnp.zeros((3,), dtype=jnp.bool_),
            "depot_pos": start,
            "depot_active": jnp.ones((3,), dtype=jnp.bool_),
            "depot_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "supply": jnp.full((3,), 4, dtype=jnp.int32),
            "goal_pos": jnp.array([[8, 1], [8, 3], [8, 5]], dtype=jnp.int32),
            "goal_active": jnp.ones((3,), dtype=jnp.bool_),
            "goal_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "goal_owner": jnp.arange(3, dtype=jnp.int32),
            "events": jnp.zeros((7,), dtype=jnp.bool_),
            "episode_done": jnp.array(False),
            "terminated": jnp.array(False),
            "truncated": jnp.array(False),
        }
        return self.observe(state), state

    def legal_actions(self, state):
        live = jnp.ones((3, 11), dtype=jnp.bool_)
        resting = jnp.broadcast_to(jnp.arange(11)[None, :] == 0, (3, 11))
        return jnp.where(state["episode_done"], resting, live)

    def observe(self, state):
        # EVIDENCE: public_infrastructure_observation
        public = jnp.concatenate((
            state["agent_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["wood"].astype(jnp.float32) / 8.0,
            state["facing"].astype(jnp.float32) / 4.0,
            state["credits"].astype(jnp.float32) / 256.0,
            state["carried"].astype(jnp.float32),
            state["map"].reshape(-1).astype(jnp.float32),
            state["bridge_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["durability"].astype(jnp.float32) / 4.0,
            (state["bridge_owner"] + 1).astype(jnp.float32) / 4.0,
            state["pending"].astype(jnp.float32),
            state["cargo_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["delivered"].astype(jnp.float32),
            state["depot_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["supply"].astype(jnp.float32) / 4.0,
            state["goal_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["t"][None].astype(jnp.float32) / 128.0,
            state["events"].astype(jnp.float32),
            jnp.stack((state["episode_done"], state["terminated"], state["truncated"])).astype(jnp.float32),
        ))
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(public, (3, 221))), axis=1)

    def raw_step(self, key, state, actions):
        a = jnp.where((actions >= 0) & (actions < 11), actions, 0).astype(jnp.int32)
        ids = jnp.arange(3, dtype=jnp.int32)
        ranks = jnp.mod(ids - state["t"], 3)
        old = state["agent_pos"]
        direction = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        moving = (a >= 1) & (a <= 4)
        facing = jnp.where(moving, a, state["facing"])
        target = jnp.clip(old + direction[jnp.where(moving, a, 0)], jnp.array([0, 0]), jnp.array([8, 6]))
        # EVIDENCE: constructed_passability_and_congestion
        target_bridges = jnp.all(target[:, None, :] == state["bridge_pos"][None, :, :], axis=2)
        supported = jnp.any(target_bridges & (state["durability"] > 0)[None, :], axis=1)
        water_target = state["map"][target[:, 1], target[:, 0]] == 1
        occupied = jnp.any(jnp.all(target[:, None, :] == old[None, :, :], axis=2) & (ids[:, None] != ids[None, :]), axis=1)
        competing = jnp.all(target[:, None, :] == target[None, :, :], axis=2)
        loses = jnp.any(competing & (ranks[None, :] < ranks[:, None]), axis=1)
        moved = moving & (~water_target | supported) & ~occupied & ~loses & jnp.any(target != old, axis=1)
        pos = jnp.where(moved[:, None], target, old)

        # EVIDENCE: ownership_tolls_and_cart_wear
        entered = moved[:, None] & jnp.all(pos[:, None, :] == state["bridge_pos"][None, :, :], axis=2)
        bridge_index = jnp.argmax(entered, axis=1).astype(jnp.int32)
        owner_at_entry = state["bridge_owner"][bridge_index]
        toll = jnp.any(entered, axis=1) & (owner_at_entry >= 0) & (owner_at_entry != ids)
        received = jnp.sum((owner_at_entry[:, None] == ids[None, :]) & toll[:, None], axis=0).astype(jnp.int32)
        credit_delta = received - toll.astype(jnp.int32)
        wear = jnp.sum(entered.astype(jnp.int32) * (1 + state["carried"].astype(jnp.int32))[:, None], axis=0)
        durability = jnp.maximum(state["durability"] - wear, 0)
        bridge_owner = state["bridge_owner"]
        bridge_occupied = jnp.any(jnp.all(pos[:, None, :] == state["bridge_pos"][None, :, :], axis=2), axis=0)
        front_raw = pos + direction[facing]
        front = jnp.clip(front_raw, jnp.array([0, 0]), jnp.array([8, 6]))
        front_matches = jnp.all(front[:, None, :] == state["bridge_pos"][None, :, :], axis=2)
        front_index = jnp.argmax(front_matches, axis=1).astype(jnp.int32)
        has_front = jnp.any(front_matches, axis=1) & jnp.all(front_raw == front, axis=1)
        wood = state["wood"]
        supply = state["supply"]
        events = jnp.zeros((7,), dtype=jnp.bool_)

        # EVIDENCE: lasting_construction_maintenance_and_access
        for k in range(3):
            i = jnp.mod(k + state["t"], 3)
            b = front_index[i]
            existing = (durability[b] > 0) | bridge_occupied[b]
            build = (a[i] == 5) & has_front[i] & ~existing & (wood[i] >= 2)
            repair = (a[i] == 5) & has_front[i] & existing & (durability[b] < 4) & (wood[i] >= 1)
            durability = durability.at[b].set(jnp.where(build, 4, jnp.where(repair, jnp.minimum(4, durability[b] + 2), durability[b])))
            bridge_owner = bridge_owner.at[b].set(jnp.where(build, i, bridge_owner[b]))
            wood = wood.at[i].add(-2 * build.astype(jnp.int32) - repair.astype(jnp.int32))
            near_depot = jnp.sum(jnp.abs(state["depot_pos"] - pos[i]), axis=1) <= 1
            available = near_depot & (supply > 0)
            depot = jnp.argmax(available).astype(jnp.int32)
            refill = (a[i] == 8) & (wood[i] < 6) & jnp.any(available)
            wood = wood.at[i].add(refill.astype(jnp.int32))
            supply = supply.at[depot].add(-refill.astype(jnp.int32))
            recipients = (ids != i) & (jnp.sum(jnp.abs(pos - pos[i]), axis=1) <= 1) & (wood < 6)
            recipient = jnp.argmin(jnp.where(recipients, ranks, 99))
            give = (a[i] == 7) & (wood[i] > 0) & jnp.any(recipients)
            wood = wood.at[i].add(-give.astype(jnp.int32))
            wood = wood.at[recipient].add(give.astype(jnp.int32))
            salvage = (a[i] == 9) & has_front[i] & (durability[b] > 0) & ~bridge_occupied[b] & (bridge_owner[b] == i)
            durability = durability.at[b].set(jnp.where(salvage, 0, durability[b]))
            bridge_owner = bridge_owner.at[b].set(jnp.where(salvage, -1, bridge_owner[b]))
            wood = wood.at[i].set(jnp.minimum(6, wood[i] + salvage.astype(jnp.int32)))
            waive = (a[i] == 10) & has_front[i] & (durability[b] > 0) & (bridge_owner[b] == i)
            bridge_owner = bridge_owner.at[b].set(jnp.where(waive, -1, bridge_owner[b]))
            events = events.at[0].set(events[0] | build)
            events = events.at[1].set(events[1] | repair)
            events = events.at[2].set(events[2] | give)
            events = events.at[3].set(events[3] | salvage)
            events = events.at[4].set(events[4] | waive)
        pending = bridge_occupied & (durability == 0)
        bridge_active = (durability > 0) | pending
        bridge_owner = jnp.where(bridge_active, bridge_owner, -1)

        # EVIDENCE: deliveries_mix_shared_and_private_payoffs
        pickup = (a == 6) & ~state["carried"] & ~state["delivered"] & jnp.all(pos == state["cargo_pos"], axis=1)
        drop = (a == 6) & state["carried"]
        carried = (state["carried"] & ~drop) | pickup
        cargo_pos = jnp.where((state["carried"] | carried)[:, None], pos, state["cargo_pos"])
        new_delivery = carried & ~state["delivered"] & jnp.all(pos == state["goal_pos"], axis=1)
        delivered = state["delivered"] | new_delivery
        carried = carried & ~new_delivery
        success = jnp.all(delivered)
        own = new_delivery.astype(jnp.float32)
        rewards = own + (jnp.sum(own) - own) / 4.0 + credit_delta.astype(jnp.float32) / 32.0
        events = events.at[5].set(jnp.any(toll))
        events = events.at[6].set(jnp.any(new_delivery))

        # EVIDENCE: natural_priority_absorbing_terminal
        n = dict(state)
        n["t"] = state["t"] + 1
        n["agent_pos"] = pos
        n["wood"] = wood
        n["facing"] = facing
        n["credits"] = state["credits"] + credit_delta
        n["carried"] = carried
        n["durability"] = durability
        n["bridge_owner"] = bridge_owner
        n["pending"] = pending
        n["bridge_active"] = bridge_active
        n["cargo_pos"] = cargo_pos
        n["delivered"] = delivered
        n["cargo_active"] = ~delivered
        n["supply"] = supply
        n["events"] = events
        n["terminated"] = success
        n["truncated"] = (n["t"] >= self.max_steps) & ~success
        n["episode_done"] = n["terminated"] | n["truncated"]
        n = jax.tree.map(lambda new, previous: jnp.where(state["episode_done"], previous, new), n, state)
        rewards = jnp.where(state["episode_done"], jnp.zeros((3,), dtype=jnp.float32), rewards)
        done = n["episode_done"]
        info = {"done_all": done, "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, rewards, jnp.full((3,), done, dtype=jnp.bool_), info