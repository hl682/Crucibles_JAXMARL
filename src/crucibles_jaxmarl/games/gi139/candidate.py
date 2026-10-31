"""Three Bloom Exchange: original pollen handoffs and delayed shared harvests.

One packet slot per courier; a different handler must deliver each pollen packet.
"""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 120
    num_actions = 12
    max_steps = 96

    def refresh(self, state):
        n = dict(state)
        n["plant_flowering"] = (state["plant_stage"] == 0) & (jnp.mod(state["t"] // 4, 2) == state["plant_species"])
        agent_distance = jnp.sum(jnp.abs(state["agent_pos"][:, None, :] - state["agent_pos"][None, :, :]), axis=2)
        plant_distance = jnp.sum(jnp.abs(state["agent_pos"][:, None, :] - state["plant_pos"][None, :, :]), axis=2)
        n["agent_visibility"] = agent_distance <= 3
        n["plant_visibility"] = plant_distance <= 3
        owner = state["fruit_owner"]
        carried = (owner >= 0) & (owner < 3)
        n["fruit_pos"] = jnp.where(carried[:, None], state["agent_pos"][jnp.clip(owner, 0, 2)], state["fruit_pos"])
        n["fruit_active"] = (owner == -1) | carried
        fruit_distance = jnp.sum(jnp.abs(state["agent_pos"][:, None, :] - n["fruit_pos"][None, :, :]), axis=2)
        n["fruit_visibility"] = (fruit_distance <= 3) & n["fruit_active"][None, :]
        return n

    def reset(self, key):
        plants = jnp.array([[1, 1], [7, 1], [1, 5], [7, 5], [4, 1], [4, 5]], dtype=jnp.int32)
        species = jnp.array([0, 1, 1, 0, 0, 1], dtype=jnp.int32)
        state = {
            "t": jnp.asarray(0, dtype=jnp.int32),
            "episode_done": jnp.asarray(False), "terminated": jnp.asarray(False), "truncated": jnp.asarray(False),
            "agent_pos": jnp.array([[1, 2], [4, 2], [7, 2]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_visibility": jnp.ones((3, 3), dtype=jnp.bool_),
            "preference": jax.random.randint(key, (3,), 0, 2, dtype=jnp.int32),
            "pollen_kind": jnp.full((3,), -1, dtype=jnp.int32),
            "pollen_collector": jnp.full((3,), -1, dtype=jnp.int32),
            "fruit_load": jnp.full((3,), -1, dtype=jnp.int32),
            "plant_pos": plants, "plant_active": jnp.ones((6,), dtype=jnp.bool_),
            "plant_visibility": jnp.ones((3, 6), dtype=jnp.bool_),
            "plant_species": species, "plant_stock": jnp.full((6,), 2, dtype=jnp.int32),
            "plant_stage": jnp.zeros((6,), dtype=jnp.int32),
            "plant_growth": jnp.zeros((6,), dtype=jnp.int32),
            "plant_flowering": jnp.zeros((6,), dtype=jnp.bool_),
            "fruit_pos": plants, "fruit_active": jnp.zeros((6,), dtype=jnp.bool_),
            "fruit_visibility": jnp.zeros((3, 6), dtype=jnp.bool_),
            "fruit_species": species, "fruit_owner": jnp.full((6,), -2, dtype=jnp.int32),
            "depot_pos": jnp.array([[4, 3]], dtype=jnp.int32),
            "depot_active": jnp.ones((1,), dtype=jnp.bool_),
            "depot_visibility": jnp.ones((3, 1), dtype=jnp.bool_),
            "delivered": jnp.zeros((1,), dtype=jnp.int32),
            "pollen_used": jnp.asarray(0, dtype=jnp.int32),
            "pollen_discarded": jnp.asarray(0, dtype=jnp.int32),
            "gather_event": jnp.asarray(0, dtype=jnp.int32),
            "handoff_event": jnp.asarray(0, dtype=jnp.int32),
            "pollination_event": jnp.asarray(0, dtype=jnp.int32),
            "ripening_event": jnp.asarray(0, dtype=jnp.int32),
            "harvest_event": jnp.asarray(0, dtype=jnp.int32),
            "delivery_event": jnp.asarray(0, dtype=jnp.int32),
            "collision_event": jnp.asarray(0, dtype=jnp.int32),
        }
        state = self.refresh(state)
        return self.observe(state), state

    def legal_actions(self, state):
        # Commands are attempts: missing nearby resources cause a deterministic no-op.
        # This mask reveals no unseen plant, inventory or preference information.
        return jnp.broadcast_to((~state["episode_done"]) | (jnp.arange(12) == 0), (3, 12))

    def advance(self, state, actions):
        n = dict(state)
        a = jnp.where((actions >= 0) & (actions < 12), actions, 0)
        ids = jnp.arange(3, dtype=jnp.int32)
        pids = jnp.arange(6, dtype=jnp.int32)
        rank = jnp.mod(ids - state["t"], 3)
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        p = state["agent_pos"]
        moving = (a >= 1) & (a <= 4)
        q = p + directions[jnp.where(moving, a, 0)]
        inside = (q[:, 0] >= 0) & (q[:, 0] <= 8) & (q[:, 1] >= 0) & (q[:, 1] <= 6)
        occupied = jnp.any(jnp.all(q[:, None, :] == p[None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        q = jnp.where((moving & inside & ~occupied)[:, None], q, p)
        conflict = jnp.any(jnp.all(q[:, None, :] == q[None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        n["agent_pos"] = jnp.where(conflict[:, None], p, q)
        n["collision_event"] = jnp.sum((moving & (occupied | conflict)).astype(jnp.int32))
        n["gather_event"] = jnp.asarray(0, dtype=jnp.int32)
        n["handoff_event"] = jnp.asarray(0, dtype=jnp.int32)
        n["pollination_event"] = jnp.asarray(0, dtype=jnp.int32)
        n["harvest_event"] = jnp.asarray(0, dtype=jnp.int32)
        n["delivery_event"] = jnp.asarray(0, dtype=jnp.int32)

        # Only growth already in progress advances. A fresh graft takes three later steps.
        growing = state["plant_stage"] == 1
        n["plant_growth"] = jnp.where(growing, jnp.maximum(state["plant_growth"] - 1, 0), state["plant_growth"])
        ripened = growing & (n["plant_growth"] == 0)
        n["plant_stage"] = jnp.where(ripened, 2, state["plant_stage"])
        n["fruit_owner"] = jnp.where(ripened, -1, state["fruit_owner"])
        n["ripening_event"] = jnp.sum(ripened.astype(jnp.int32))

        # Explicit GIVE/RECEIVE rendezvous; collector identity travels with the packet.
        for offset in range(3):
            i = jnp.mod(state["t"] + offset, 3)
            adjacent = jnp.sum(jnp.abs(n["agent_pos"] - n["agent_pos"][i]), axis=1) == 1
            receivers = adjacent & (a == 7) & (n["pollen_kind"] < 0) & (n["fruit_load"] < 0) & (ids != i)
            receiver = jnp.argmin(jnp.where(receivers, rank, 9)).astype(jnp.int32)
            give = (a[i] == 6) & (n["pollen_kind"][i] >= 0) & jnp.any(receivers)
            old_kind = n["pollen_kind"][i]
            old_collector = n["pollen_collector"][i]
            n["pollen_kind"] = jnp.where(give & (ids == receiver), old_kind, n["pollen_kind"])
            n["pollen_collector"] = jnp.where(give & (ids == receiver), old_collector, n["pollen_collector"])
            n["pollen_kind"] = jnp.where(give & (ids == i), -1, n["pollen_kind"])
            n["pollen_collector"] = jnp.where(give & (ids == i), -1, n["pollen_collector"])
            n["handoff_event"] = n["handoff_event"] + give.astype(jnp.int32)

        delivered_by = jnp.zeros((3,), dtype=jnp.bool_)
        preferred_by = jnp.zeros((3,), dtype=jnp.bool_)
        # Plant conflicts resolve by rotating actor order, then lowest eligible plant id.
        for offset in range(3):
            i = jnp.mod(state["t"] + offset, 3)
            near = jnp.sum(jnp.abs(n["plant_pos"] - n["agent_pos"][i]), axis=1) <= 1
            empty = (n["pollen_kind"][i] < 0) & (n["fruit_load"][i] < 0)
            gatherable = near & state["plant_flowering"] & (n["plant_stage"] == 0) & (n["plant_stock"] > 0)
            gather_target = jnp.argmin(jnp.where(gatherable, pids, 99)).astype(jnp.int32)
            gather = (a[i] == 5) & empty & jnp.any(gatherable)
            n["plant_stock"] = n["plant_stock"] - (gather & (pids == gather_target)).astype(jnp.int32)
            n["pollen_kind"] = jnp.where(gather & (ids == i), n["plant_species"][gather_target], n["pollen_kind"])
            n["pollen_collector"] = jnp.where(gather & (ids == i), i, n["pollen_collector"])
            n["gather_event"] = n["gather_event"] + gather.astype(jnp.int32)

            compatible = near & state["plant_flowering"] & (n["plant_stage"] == 0) & (n["plant_species"] != n["pollen_kind"][i])
            graft_target = jnp.argmin(jnp.where(compatible, pids, 99)).astype(jnp.int32)
            graft = (a[i] == 8) & (n["pollen_kind"][i] >= 0) & (n["pollen_collector"][i] != i) & jnp.any(compatible)
            n["plant_stage"] = jnp.where(graft & (pids == graft_target), 1, n["plant_stage"])
            n["plant_growth"] = jnp.where(graft & (pids == graft_target), 3, n["plant_growth"])
            n["pollen_kind"] = jnp.where(graft & (ids == i), -1, n["pollen_kind"])
            n["pollen_collector"] = jnp.where(graft & (ids == i), -1, n["pollen_collector"])
            n["pollen_used"] = n["pollen_used"] + graft.astype(jnp.int32)
            n["pollination_event"] = n["pollination_event"] + graft.astype(jnp.int32)

            empty = (n["pollen_kind"][i] < 0) & (n["fruit_load"][i] < 0)
            harvestable = near & (n["plant_stage"] == 2)
            harvest_target = jnp.argmin(jnp.where(harvestable, pids, 99)).astype(jnp.int32)
            harvest = (a[i] == 9) & empty & jnp.any(harvestable)
            n["plant_stage"] = jnp.where(harvest & (pids == harvest_target), 3, n["plant_stage"])
            n["fruit_owner"] = jnp.where(harvest & (pids == harvest_target), i, n["fruit_owner"])
            n["fruit_load"] = jnp.where(harvest & (ids == i), harvest_target, n["fruit_load"])
            n["harvest_event"] = n["harvest_event"] + harvest.astype(jnp.int32)

            load = n["fruit_load"][i]
            at_depot = jnp.sum(jnp.abs(n["agent_pos"][i] - n["depot_pos"][0])) <= 1
            deliver = (a[i] == 10) & (load >= 0) & at_depot
            preferred = deliver & (n["fruit_species"][jnp.maximum(load, 0)] == n["preference"][i])
            n["fruit_owner"] = jnp.where(deliver & (pids == load), 3, n["fruit_owner"])
            n["fruit_pos"] = jnp.where((deliver & (pids == load))[:, None], n["depot_pos"][0], n["fruit_pos"])
            n["fruit_load"] = jnp.where(deliver & (ids == i), -1, n["fruit_load"])
            n["delivered"] = n["delivered"] + deliver.astype(jnp.int32)
            n["delivery_event"] = n["delivery_event"] + deliver.astype(jnp.int32)
            delivered_by = delivered_by | (deliver & (ids == i))
            preferred_by = preferred_by | (preferred & (ids == i))

            discard = (a[i] == 11) & (n["pollen_kind"][i] >= 0)
            n["pollen_kind"] = jnp.where(discard & (ids == i), -1, n["pollen_kind"])
            n["pollen_collector"] = jnp.where(discard & (ids == i), -1, n["pollen_collector"])
            n["pollen_discarded"] = n["pollen_discarded"] + discard.astype(jnp.int32)

        n["t"] = state["t"] + 1
        n["terminated"] = n["delivered"][0] == 6
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["episode_done"] = n["terminated"] | n["truncated"]
        n = self.refresh(n)
        reward = (0.5 * jnp.sum(delivered_by.astype(jnp.float32)) + 0.5 * preferred_by.astype(jnp.float32)).astype(jnp.float32)
        return n, reward

    def observe(self, state):
        own = jnp.stack((jnp.arange(3, dtype=jnp.float32) / 4.0, state["preference"].astype(jnp.float32),
                         (state["pollen_kind"] + 1) / 2.0, (state["pollen_collector"] + 1) / 4.0,
                         (state["fruit_load"] + 1) / 8.0), axis=1)
        clock = jnp.array([state["t"] / 128.0, jnp.mod(state["t"] // 4, 2), state["delivered"][0] / 8.0,
                           state["pollen_discarded"] / 16.0, state["pollen_used"] / 16.0], dtype=jnp.float32)
        agent_data = jnp.concatenate((state["agent_pos"] / 16.0, (state["pollen_kind"][:, None] + 1) / 2.0,
                                      (state["pollen_collector"][:, None] + 1) / 4.0,
                                      (state["fruit_load"][:, None] + 1) / 8.0), axis=1)
        av = state["agent_visibility"]
        local_agents = jnp.concatenate((av[:, :, None].astype(jnp.float32),
                                        jnp.where(av[:, :, None], agent_data[None, :, :], 0.0)), axis=2).reshape((3, 18))
        plant_data = jnp.concatenate((state["plant_pos"] / 16.0, state["plant_species"][:, None].astype(jnp.float32),
                                      state["plant_flowering"][:, None].astype(jnp.float32), state["plant_stock"][:, None] / 2.0,
                                      state["plant_stage"][:, None] / 4.0, state["plant_growth"][:, None] / 4.0), axis=1)
        pv = state["plant_visibility"]
        local_plants = jnp.concatenate((pv[:, :, None].astype(jnp.float32),
                                        jnp.where(pv[:, :, None], plant_data[None, :, :], 0.0)), axis=2).reshape((3, 48))
        fruit_data = jnp.concatenate((state["fruit_pos"] / 16.0, state["fruit_species"][:, None].astype(jnp.float32),
                                      (state["fruit_owner"][:, None] + 2) / 8.0), axis=1)
        fv = state["fruit_visibility"]
        local_fruit = jnp.concatenate((fv[:, :, None].astype(jnp.float32),
                                       jnp.where(fv[:, :, None], fruit_data[None, :, :], 0.0)), axis=2).reshape((3, 30))
        bells = jnp.array([state["gather_event"], state["handoff_event"], state["pollination_event"],
                           state["ripening_event"], state["harvest_event"], state["delivery_event"],
                           state["collision_event"]], dtype=jnp.float32) / 8.0
        return jnp.concatenate((own, jnp.broadcast_to(clock, (3, 5)), local_agents, local_plants, local_fruit,
                                jnp.broadcast_to(state["depot_pos"][0] / 16.0, (3, 2)),
                                jnp.broadcast_to(bells, (3, 7)), jnp.zeros((3, 5), dtype=jnp.float32)), axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        next_state, rewards = jax.lax.cond(state["episode_done"],
            lambda s, a: (s, jnp.zeros((3,), dtype=jnp.float32)), self.advance, state, actions)
        dones = jnp.full((3,), next_state["episode_done"], dtype=jnp.bool_)
        info = {"done_all": next_state["episode_done"], "terminated": next_state["terminated"], "truncated": next_state["truncated"]}
        return self.observe(next_state), next_state, rewards, dones, info
