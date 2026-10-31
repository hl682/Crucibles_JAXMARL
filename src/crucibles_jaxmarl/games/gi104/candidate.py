"""Original Orchard Watershed Commons kernel; author hypotheses are unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 164
    num_actions = 12
    max_steps = 128

    def distances(self, a, b):
        return jnp.sum(jnp.abs(a[:, None, :] - b[None, :, :]), axis=-1)

    def reset(self, key):
        # No stochastic draw: water, fruit and root propagation follow local rules.
        grid = jnp.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 3, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 3, 0, 0, 2, 0, 0, 3, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]], jnp.int32)
        s = {
            "t": jnp.array(0, jnp.int32), "episode_done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "success": jnp.array(False), "collapse": jnp.array(False), "map": grid,
            "agent_pos": jnp.array([[1, 4], [4, 1], [7, 4]], jnp.int32),
            "agent_active": jnp.ones(3, jnp.bool_), "agent_vis": jnp.ones((3, 3), jnp.bool_),
            "agent_fruit": jnp.zeros(3, jnp.int32), "agent_water": jnp.ones(3, jnp.int32),
            "agent_bank": jnp.zeros(3, jnp.int32),
            "patch_pos": jnp.array([[2, 3], [4, 2], [6, 3], [6, 5], [4, 6], [2, 5]], jnp.int32),
            "patch_active": jnp.ones(6, jnp.bool_), "patch_vis": jnp.ones((3, 6), jnp.bool_),
            "patch_stock": jnp.full(6, 3, jnp.int32), "patch_roots": jnp.full(6, 3, jnp.int32),
            "patch_soil": jnp.full(6, 2, jnp.int32),
            "home_pos": jnp.array([[1, 4], [4, 1], [7, 4]], jnp.int32),
            "home_active": jnp.ones(3, jnp.bool_), "home_vis": jnp.ones((3, 3), jnp.bool_),
            "pond_pos": jnp.array([[4, 4]], jnp.int32), "pond_active": jnp.ones(1, jnp.bool_),
            "pond_vis": jnp.ones((3, 1), jnp.bool_),
            "event_harvest": jnp.zeros(3, jnp.bool_), "event_plant": jnp.zeros(3, jnp.bool_),
            "event_water": jnp.zeros(3, jnp.bool_), "event_gift": jnp.zeros(3, jnp.bool_),
            "event_deposit": jnp.zeros(3, jnp.bool_), "event_growth": jnp.zeros(6, jnp.bool_)}
        return self.observe(s), s

    def legal_actions(self, s):
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s["agent_pos"][:, None, :] + delta[None, :, :]
        tiles = s["map"][jnp.clip(p[:, :, 1], 0, 8), jnp.clip(p[:, :, 0], 0, 8)]
        moves = (tiles != 1) & jnp.all((p >= 0) & (p <= 8), axis=-1)
        at_home = jnp.all(s["agent_pos"] == s["home_pos"], axis=1)
        one = jnp.ones((3, 1), jnp.bool_)
        mask = jnp.concatenate((one, moves, (s["agent_fruit"] < 4)[:, None],
            (s["agent_fruit"] > 0)[:, None], (s["agent_water"] > 0)[:, None], one,
            (at_home & (s["agent_fruit"] > 0) & (s["agent_bank"] < 6))[:, None],
            (s["agent_fruit"] > 0)[:, None], (s["agent_fruit"] < 4)[:, None]), axis=1)
        return mask & (~s["episode_done"] | (jnp.arange(12)[None, :] == 0))

    def move(self, s, a):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s["agent_pos"] + delta[jnp.where(a <= 4, a, 0)]
        same = jnp.all(p[:, None, :] == p[None, :, :], axis=-1)
        occupied = jnp.all(p[:, None, :] == s["agent_pos"][None, :, :], axis=-1)
        blocked = jnp.any((same | occupied) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        return jnp.where(blocked[:, None], s["agent_pos"], p)

    def advance(self, s, a):
        n = dict(s)
        n["t"] = s["t"] + jnp.array(1, jnp.int32)
        n["agent_pos"] = self.move(s, a)
        n["event_harvest"] = jnp.zeros(3, jnp.bool_)
        n["event_plant"] = jnp.zeros(3, jnp.bool_)
        n["event_water"] = jnp.zeros(3, jnp.bool_)
        n["event_gift"] = jnp.zeros(3, jnp.bool_)
        distances = self.distances(n["agent_pos"], n["patch_pos"])
        target = jnp.argmin(distances, axis=1)
        near = jnp.min(distances, axis=1) <= 1
        # Shared patch contests rotate first mover by t mod 3; each action has one target.
        for offset in range(3):
            i = jnp.mod(s["t"] + offset, 3)
            p = target[i]
            harvest = ((a[i] == 5) & near[i] & (n["patch_stock"][p] > 0)
                       & (n["patch_roots"][p] > 0) & (n["agent_fruit"][i] < 4))
            last_fruit = harvest & (n["patch_stock"][p] == 1)
            n["patch_stock"] = n["patch_stock"].at[p].add(-harvest.astype(jnp.int32))
            n["patch_roots"] = n["patch_roots"].at[p].add(-last_fruit.astype(jnp.int32))
            n["agent_fruit"] = n["agent_fruit"].at[i].add(harvest.astype(jnp.int32))
            n["event_harvest"] = n["event_harvest"].at[i].set(harvest)
            plant = ((a[i] == 6) & near[i] & (n["agent_fruit"][i] > 0)
                     & ((n["patch_roots"][p] < 4) | (n["patch_stock"][p] < 5)
                        | (n["patch_soil"][p] < 3)))
            n["agent_fruit"] = n["agent_fruit"].at[i].add(-plant.astype(jnp.int32))
            n["patch_roots"] = n["patch_roots"].at[p].set(jnp.minimum(
                4, n["patch_roots"][p] + plant.astype(jnp.int32)))
            n["patch_stock"] = n["patch_stock"].at[p].set(jnp.minimum(
                5, n["patch_stock"][p] + plant.astype(jnp.int32)))
            n["patch_soil"] = n["patch_soil"].at[p].set(jnp.minimum(
                3, n["patch_soil"][p] + plant.astype(jnp.int32)))
            n["event_plant"] = n["event_plant"].at[i].set(plant)
            irrigate = (a[i] == 7) & near[i] & (n["agent_water"][i] > 0) & (n["patch_roots"][p] > 0)
            n["agent_water"] = n["agent_water"].at[i].add(-irrigate.astype(jnp.int32))
            n["patch_soil"] = n["patch_soil"].at[p].set(jnp.where(irrigate, 3, n["patch_soil"][p]))
            n["event_water"] = n["event_water"].at[i].set(irrigate)

        at_pond = self.distances(n["agent_pos"], n["pond_pos"])[:, 0] <= 1
        n["agent_water"] = jnp.where((a == 8) & at_pond, 2, n["agent_water"])
        for offset in range(3):
            i = jnp.mod(s["t"] + offset, 3)
            d = jnp.sum(jnp.abs(n["agent_pos"] - n["agent_pos"][i]), axis=1)
            ready = (a == 11) & (d <= 1) & (n["agent_fruit"] < 4) & (jnp.arange(3) != i)
            rank = jnp.mod(jnp.arange(3) - s["t"], 3)
            receiver = jnp.argmin(jnp.where(ready, rank, 9))
            gift = (a[i] == 10) & (n["agent_fruit"][i] > 0) & jnp.any(ready)
            n["agent_fruit"] = n["agent_fruit"].at[i].add(-gift.astype(jnp.int32))
            n["agent_fruit"] = n["agent_fruit"].at[receiver].add(gift.astype(jnp.int32))
            n["event_gift"] = n["event_gift"].at[i].set(gift)

        at_home = jnp.all(n["agent_pos"] == n["home_pos"], axis=1)
        delivered = jnp.where((a == 9) & at_home,
            jnp.minimum(n["agent_fruit"], 6 - n["agent_bank"]), 0)
        n["agent_bank"] = n["agent_bank"] + delivered
        n["agent_fruit"] = n["agent_fruit"] - delivered
        n["event_deposit"] = delivered > 0

        # A four-tick ecological pulse uses one common post-action snapshot.
        # Moisture reaches neighbouring patches now, but aids their fruit on a later pulse.
        separation = self.distances(n["patch_pos"], n["patch_pos"])
        adjacent = (separation > 0) & (separation <= 3)
        donor = (n["patch_roots"] >= 2) & (n["patch_stock"] >= 2)
        seed_support = jnp.any(adjacent & donor[None, :], axis=1)
        wet_neighbour = jnp.any(adjacent & (n["patch_soil"][None, :] >= 3), axis=1)
        pulse = jnp.mod(n["t"], 4) == 0
        growth = (pulse & (n["patch_roots"] > 0) & (n["patch_soil"] > 0)
                  & (n["patch_stock"] < 5) & ((n["patch_stock"] >= 2) | seed_support))
        n["patch_stock"] = n["patch_stock"] + growth.astype(jnp.int32)
        new_soil = jnp.maximum(n["patch_soil"] - 1, wet_neighbour.astype(jnp.int32))
        n["patch_soil"] = jnp.where(pulse, new_soil, n["patch_soil"])
        n["event_growth"] = growth
        n["success"] = jnp.all(n["agent_bank"] >= 6)
        n["collapse"] = ~n["success"] & jnp.all(n["patch_roots"] == 0) & jnp.all(n["agent_fruit"] == 0)
        n["terminated"] = n["success"] | n["collapse"]
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["episode_done"] = n["terminated"] | n["truncated"]
        reward = (delivered + 2 * n["success"].astype(jnp.int32)
                  - 2 * n["collapse"].astype(jnp.int32)).astype(jnp.float32)
        return n, reward

    def observe(self, s):
        clock = jnp.array([s["t"], s["episode_done"], s["terminated"], s["truncated"],
                           s["success"], s["collapse"]], jnp.int32)
        agents = jnp.concatenate((s["agent_pos"], s["agent_fruit"][:, None],
                                  s["agent_water"][:, None], s["agent_bank"][:, None]), axis=1).reshape(-1)
        patches = jnp.concatenate((s["patch_pos"], s["patch_stock"][:, None],
                                    s["patch_roots"][:, None], s["patch_soil"][:, None]), axis=1).reshape(-1)
        shared = jnp.concatenate((clock, agents, patches, s["home_pos"].reshape(-1),
            s["pond_pos"].reshape(-1), s["map"].reshape(-1), s["event_harvest"],
            s["event_plant"], s["event_water"], s["event_gift"], s["event_deposit"],
            s["event_growth"])).astype(jnp.int32)
        obs = jnp.concatenate((jnp.eye(3, dtype=jnp.int32), jnp.broadcast_to(shared, (3, 161))), axis=1)
        return obs.astype(jnp.float32) / 256.0

    def raw_step(self, key, state, actions):
        clipped = jnp.clip(actions, 0, self.num_actions - 1)
        allowed = self.legal_actions(state)[jnp.arange(3), clipped]
        valid = (actions >= 0) & (actions < self.num_actions) & allowed
        a = jnp.where(valid, clipped, 0).astype(jnp.int32)
        proposed, reward = self.advance(state, a)
        n = jax.tree.map(lambda new, old: jnp.where(state["episode_done"], old, new), proposed, state)
        reward = jnp.where(state["episode_done"], jnp.zeros(3, jnp.float32), reward)
        dones = jnp.full((3,), n["episode_done"], jnp.bool_)
        info = {"done_all": n["episode_done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, reward, dones, info
