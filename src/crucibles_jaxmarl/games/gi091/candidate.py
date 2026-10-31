import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 293
    num_actions = 10
    max_steps = 96

    def reset(self, key):
        # EVIDENCE: original_world
        cells = jnp.stack((jnp.mod(jnp.arange(63), 9), jnp.arange(63) // 9), axis=1).astype(jnp.int32)
        terrain = jnp.zeros((7, 9), dtype=jnp.int32)
        terrain = terrain.at[jnp.array([0, 1, 3, 5, 6]), 4].set(1)
        terrain = terrain.at[3, 0].set(2)
        terrain = terrain.at[jnp.array([1, 5, 3]), jnp.array([1, 1, 3])].set(3)
        fire = jnp.zeros((63,), dtype=jnp.int32).at[jnp.array([26, 44])].set(3)
        fuel = ((terrain.reshape(-1) != 1) & (terrain.reshape(-1) != 2) & (fire == 0)).astype(jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "map": terrain,
            "cell_pos": cells,
            "fire": fire,
            "fuel": fuel,
            "barrier": jnp.zeros((63,), dtype=jnp.int32),
            "fire_active": fire > 0,
            "barrier_active": jnp.zeros((63,), dtype=jnp.bool_),
            "cell_visible": jnp.ones((3, 63), dtype=jnp.bool_),
            "agent_pos": jnp.array([[1, 2], [1, 3], [1, 4]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "water": jnp.full((3,), 3, dtype=jnp.int32),
            "facing": jnp.full((3,), 2, dtype=jnp.int32),
            "stun": jnp.zeros((3,), dtype=jnp.int32),
            "asset_pos": jnp.array([[1, 1], [1, 5], [3, 3]], dtype=jnp.int32),
            "asset_active": jnp.ones((3,), dtype=jnp.bool_),
            "asset_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "health": jnp.full((3,), 3, dtype=jnp.int32),
            "depot_pos": jnp.array([[0, 3]], dtype=jnp.int32),
            "depot_active": jnp.ones((1,), dtype=jnp.bool_),
            "depot_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "stock": jnp.array([18], dtype=jnp.int32),
            "events": jnp.zeros((7,), dtype=jnp.bool_),
            "episode_done": jnp.array(False),
            "terminated": jnp.array(False),
            "truncated": jnp.array(False),
        }
        return self.observe(state), state

    def legal_actions(self, state):
        live = jnp.ones((3, 10), dtype=jnp.bool_)
        resting = jnp.broadcast_to(jnp.arange(10)[None, :] == 0, (3, 10))
        return jnp.where(state["episode_done"], resting, live)

    def observe(self, state):
        # EVIDENCE: public_observation
        public = jnp.concatenate((
            state["agent_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["water"].astype(jnp.float32) / 4.0,
            state["facing"].astype(jnp.float32) / 4.0,
            state["stun"].astype(jnp.float32) / 4.0,
            state["map"].reshape(-1).astype(jnp.float32) / 4.0,
            state["fire"].astype(jnp.float32) / 4.0,
            state["fuel"].astype(jnp.float32),
            state["barrier"].astype(jnp.float32) / 4.0,
            state["asset_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["health"].astype(jnp.float32) / 4.0,
            state["depot_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["stock"].astype(jnp.float32) / 32.0,
            state["t"][None].astype(jnp.float32) / 128.0,
            state["events"].astype(jnp.float32),
            jnp.stack((state["episode_done"], state["terminated"], state["truncated"])).astype(jnp.float32),
        ))
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(public, (3, 290))), axis=1)

    def raw_step(self, key, state, actions):
        a = jnp.where((actions >= 0) & (actions < 10) & (state["stun"] == 0), actions, 0).astype(jnp.int32)
        old = state["agent_pos"]
        ids = jnp.arange(3, dtype=jnp.int32)
        direction = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        is_move = (a >= 1) & (a <= 4)
        facing = jnp.where(is_move, a, state["facing"])
        proposed = old + direction[jnp.where(is_move, a, 0)]
        target = jnp.clip(proposed, jnp.array([0, 0]), jnp.array([8, 6]))
        index = target[:, 1] * 9 + target[:, 0]
        # EVIDENCE: barriers_change_partner_routes
        blocked = (state["map"][target[:, 1], target[:, 0]] == 1) | (state["barrier"][index] > 0)
        occupied = jnp.any(jnp.all(target[:, None, :] == old[None, :, :], axis=2) & (ids[:, None] != ids[None, :]), axis=1)
        duplicate = jnp.sum(jnp.all(target[:, None, :] == target[None, :, :], axis=2), axis=1) > 1
        pos = jnp.where((is_move & ~blocked & ~occupied & ~duplicate)[:, None], target, old)
        front_raw = pos + direction[facing]
        front = jnp.clip(front_raw, jnp.array([0, 0]), jnp.array([8, 6]))
        inside = jnp.all(front_raw == front, axis=1)
        front_index = front[:, 1] * 9 + front[:, 0]
        current_index = pos[:, 1] * 9 + pos[:, 0]
        cell_ids = jnp.arange(63, dtype=jnp.int32)
        fire = state["fire"]
        barrier = state["barrier"]
        water = state["water"]
        stock = state["stock"][0]
        events = jnp.zeros((7,), dtype=jnp.bool_)

        # EVIDENCE: scarce_shared_resources_and_transfers
        for k in range(3):
            i = jnp.mod(k + state["t"], 3)
            spray = (a[i] == 5) & (water[i] > 0)
            sprayed_cells = (cell_ids == current_index[i]) | ((cell_ids == front_index[i]) & inside[i])
            fire = jnp.where(spray & sprayed_cells, 0, fire)
            water = water.at[i].add(-spray.astype(jnp.int32))
            front_free = ~jnp.any(jnp.all(pos == front[i], axis=1))
            build = (a[i] == 6) & (water[i] >= 2) & inside[i] & front_free & (state["map"][front[i, 1], front[i, 0]] == 0) & (fire[front_index[i]] == 0) & (barrier[front_index[i]] == 0)
            barrier = barrier.at[front_index[i]].set(jnp.where(build, 2, barrier[front_index[i]]))
            water = water.at[i].add(-2 * build.astype(jnp.int32))
            refill = (a[i] == 7) & (water[i] < 3) & (stock > 0) & (jnp.sum(jnp.abs(pos[i] - state["depot_pos"][0])) <= 1)
            water = water.at[i].add(refill.astype(jnp.int32))
            stock = stock - refill.astype(jnp.int32)
            recipients = (ids != i) & (jnp.sum(jnp.abs(pos - pos[i]), axis=1) <= 1) & (water < 3)
            recipient = jnp.argmin(jnp.where(recipients, jnp.mod(ids - state["t"], 3), 99))
            give = (a[i] == 8) & (water[i] > 0) & jnp.any(recipients)
            water = water.at[i].add(-give.astype(jnp.int32))
            water = water.at[recipient].add(give.astype(jnp.int32))
            dismantle = (a[i] == 9) & inside[i] & (barrier[front_index[i]] > 0)
            barrier = barrier.at[front_index[i]].set(jnp.where(dismantle, 0, barrier[front_index[i]]))
            water = water.at[i].set(jnp.minimum(3, water[i] + dismantle.astype(jnp.int32)))
            events = events.at[0].set(events[0] | spray)
            events = events.at[1].set(events[1] | build)
            events = events.at[2].set(events[2] | give)
            events = events.at[3].set(events[3] | dismantle)

        # EVIDENCE: persistent_spreading_hazard
        adjacent = jnp.sum(jnp.abs(state["cell_pos"][:, None, :] - state["cell_pos"][None, :, :]), axis=2) == 1
        exposed = jnp.any((fire > 0)[:, None] & adjacent, axis=0)
        spread_tick = jnp.mod(state["t"] + 1, 3) == 0
        ignition = spread_tick & exposed & (state["fuel"] > 0) & (barrier == 0)
        next_fire = jnp.where(ignition, 3, jnp.maximum(fire - 1, 0)).astype(jnp.int32)
        fuel = jnp.where(ignition, 0, state["fuel"])
        barrier = jnp.maximum(barrier - (spread_tick & exposed & (barrier > 0)).astype(jnp.int32), 0)
        asset_index = state["asset_pos"][:, 1] * 9 + state["asset_pos"][:, 0]
        damage = (next_fire[asset_index] > 0) & (state["health"] > 0)
        health = state["health"] - damage.astype(jnp.int32)
        in_fire = next_fire[current_index] > 0
        water = jnp.maximum(0, water - in_fire.astype(jnp.int32))
        stun = jnp.where(in_fire, 2, jnp.maximum(state["stun"] - 1, 0))
        failure = jnp.all(health == 0)
        success = jnp.all(next_fire == 0) & ~failure
        terminated = failure | success
        events = events.at[4].set(jnp.any(damage))
        events = events.at[5].set(success)
        events = events.at[6].set(failure)
        reward = success.astype(jnp.float32) - failure.astype(jnp.float32) - jnp.sum(damage.astype(jnp.float32)) / 16.0

        # EVIDENCE: natural_priority_absorbing_terminal
        n = dict(state)
        n["t"] = state["t"] + 1
        n["agent_pos"] = pos
        n["water"] = water
        n["facing"] = facing
        n["stun"] = stun
        n["stock"] = stock[None]
        n["fire"] = next_fire
        n["fuel"] = fuel
        n["barrier"] = barrier
        n["fire_active"] = next_fire > 0
        n["barrier_active"] = barrier > 0
        n["health"] = health
        n["asset_active"] = health > 0
        n["events"] = events
        n["terminated"] = terminated
        n["truncated"] = (n["t"] >= self.max_steps) & ~terminated
        n["episode_done"] = n["terminated"] | n["truncated"]
        n = jax.tree.map(lambda new, previous: jnp.where(state["episode_done"], previous, new), n, state)
        rewards = jnp.full((3,), jnp.where(state["episode_done"], 0.0, reward), dtype=jnp.float32)
        done = n["episode_done"]
        info = {"done_all": done, "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, rewards, jnp.full((3,), done, dtype=jnp.bool_), info