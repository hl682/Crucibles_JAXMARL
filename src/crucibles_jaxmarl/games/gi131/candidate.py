"""Firebreak Crews: finite water, fuel-front propagation and survivor relays."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 514
    num_actions = 10
    max_steps = 120

    def refresh(self, s):
        s = dict(s)
        carried = (s["survivor_carrier"] >= 0) & (s["survivor_carrier"] < 3)
        owners = jnp.clip(s["survivor_carrier"], 0, 2)
        s["survivor_pos"] = jnp.where(carried[:, None], s["agent_pos"][owners], s["survivor_pos"])
        s["survivor_active"] = s["survivor_carrier"] != 4
        s["agent_vis"] = jnp.ones((3, 3), dtype=jnp.bool_)
        s["survivor_vis"] = jnp.ones((3, 6), dtype=jnp.bool_)
        s["cell_vis"] = jnp.ones((3, 63), dtype=jnp.bool_)
        s["tank_vis"] = jnp.ones((3, 1), dtype=jnp.bool_)
        s["safe_vis"] = jnp.ones((3, 1), dtype=jnp.bool_)
        s["fire_level"] = s["fire"].reshape(63)
        s["fuel_level"] = s["fuel"].reshape(63)
        s["smoke_level"] = s["smoke"].reshape(63)
        return s

    def reset(self, key):
        tile = jnp.zeros((7, 9), dtype=jnp.int32)
        tile = tile.at[0, :].set(1).at[:, 0].set(1).at[1, 1].set(1)
        tile = tile.at[2, 4].set(2).at[3, 4].set(2).at[4, 4].set(2)
        fuel = jnp.where(tile == 0, 3, 0).astype(jnp.int32)
        fire = jnp.zeros((7, 9), dtype=jnp.int32).at[3, 7].set(2).at[5, 6].set(2)
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "tile": tile, "fuel": fuel, "fire": fire,
            "smoke": jnp.where(fire > 0, 3, 0).astype(jnp.int32),
            "cell_pos": jnp.stack((jnp.tile(jnp.arange(9), 7), jnp.repeat(jnp.arange(7), 9)), axis=1).astype(jnp.int32),
            "cell_active": jnp.ones((63,), dtype=jnp.bool_),
            "agent_pos": jnp.array([[0, 1], [1, 0], [1, 1]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "facing": jnp.array([1, 2, 1], dtype=jnp.int32),
            "agent_health": jnp.full((3,), 3, dtype=jnp.int32),
            "water": jnp.full((3,), 3, dtype=jnp.int32),
            "tank_pos": jnp.array([[0, 2]], dtype=jnp.int32),
            "tank_active": jnp.ones((1,), dtype=jnp.bool_),
            "tank_stock": jnp.array([6], dtype=jnp.int32),
            "spent_water": jnp.array(0, dtype=jnp.int32),
            "safe_pos": jnp.array([[0, 0]], dtype=jnp.int32),
            "safe_active": jnp.ones((1,), dtype=jnp.bool_),
            "survivor_pos": jnp.array([[7, 1], [7, 5], [5, 3], [4, 5], [3, 2], [2, 5]], dtype=jnp.int32),
            "survivor_carrier": jnp.full((6,), -1, dtype=jnp.int32),
            "survivor_health": jnp.full((6,), 3, dtype=jnp.int32),
            "last_rescued": jnp.zeros((6,), dtype=jnp.bool_),
            "last_lost": jnp.zeros((6,), dtype=jnp.bool_),
            "last_injury": jnp.zeros((3,), dtype=jnp.bool_),
            "last_spread": jnp.zeros((63,), dtype=jnp.bool_),
            "last_cut": jnp.zeros((63,), dtype=jnp.bool_),
            "last_spray": jnp.zeros((63,), dtype=jnp.int32),
            "last_refill": jnp.zeros((3,), dtype=jnp.bool_),
            "last_share": jnp.zeros((3,), dtype=jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate(tuple(s[k].reshape(-1).astype(jnp.float32) for k in (
            "t", "done", "terminated", "truncated", "tile", "fire_level", "fuel_level",
            "smoke_level", "agent_pos", "facing", "agent_health", "water", "tank_pos",
            "tank_stock", "spent_water", "safe_pos", "survivor_pos", "survivor_carrier",
            "survivor_health", "last_rescued", "last_lost", "last_injury", "last_spread",
            "last_cut", "last_spray", "last_refill", "last_share")))
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(public, (3, public.shape[0]))), axis=1)

    def legal_actions(self, s):
        live = ~s["done"] & (s["agent_health"] > 0)
        return (jnp.ones((3, 10), dtype=jnp.bool_) & live[:, None]).at[:, 0].set(True)

    def neighbors(self, grid):
        p = jnp.pad(grid, ((1, 1), (1, 1)))
        return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]

    def move(self, s, a):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        target = s["agent_pos"] + delta[jnp.where(a <= 4, a, 0)]
        inside = jnp.all((target >= 0) & (target <= jnp.array([8, 6])), axis=1)
        q = jnp.clip(target, jnp.array([0, 0]), jnp.array([8, 6]))
        passable = (s["tile"][q[:, 1], q[:, 0]] != 2) & (s["fire"][q[:, 1], q[:, 0]] == 0) & (s["smoke"][q[:, 1], q[:, 0]] < 3)
        occupied = jnp.any(jnp.all(q[:, None, :] == s["agent_pos"][None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        conflict = jnp.sum(jnp.all(q[:, None, :] == q[None, :, :], axis=2), axis=1) > 1
        return jnp.where((inside & passable & ~occupied & ~conflict)[:, None], q, s["agent_pos"])

    def advance(self, s, a):
        n = dict(s)
        n["t"] = s["t"] + 1
        n["agent_pos"] = self.move(s, a)
        n["facing"] = jnp.where((a >= 1) & (a <= 4), a - 1, s["facing"])
        directions = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        target = n["agent_pos"] + directions[n["facing"]]
        inside = jnp.all((target >= 0) & (target <= jnp.array([8, 6])), axis=1)
        q = jnp.clip(target, jnp.array([0, 0]), jnp.array([8, 6]))
        # WATER_AND_BREAKS: simultaneous spray spends each actor's unit, then clears resolve.
        spraying = (a == 6) & (s["water"] > 0) & inside
        sprays = jnp.zeros((7, 9), dtype=jnp.int32).at[q[:, 1], q[:, 0]].add(spraying.astype(jnp.int32))
        fire = jnp.where(sprays > 0, 0, s["fire"])
        cut_vote = (a == 5) & inside & (fire[q[:, 1], q[:, 0]] == 0) & (s["tile"][q[:, 1], q[:, 0]] == 0)
        cuts = jnp.zeros((7, 9), dtype=jnp.int32).at[q[:, 1], q[:, 0]].add(cut_vote.astype(jnp.int32)) > 0
        fuel = jnp.where(cuts, 0, s["fuel"])
        water = s["water"] - spraying.astype(jnp.int32)
        tank = s["tank_stock"][0]
        carrier = s["survivor_carrier"]
        carried = (carrier >= 0) & (carrier < 3)
        survivor_pos = jnp.where(carried[:, None], n["agent_pos"][jnp.clip(carrier, 0, 2)], s["survivor_pos"])
        refilled = jnp.zeros((3,), dtype=jnp.bool_)
        shared = jnp.zeros((3,), dtype=jnp.bool_)
        # ROTATING_INTERACTIONS: unique ownership, finite tank and directed water handoffs.
        for offset in range(3):
            i = (s["t"] + offset) % 3
            holding = jnp.any(carrier == i)
            drop = (a[i] == 7) & holding
            carrier = jnp.where((carrier == i) & drop, -1, carrier)
            nearby = jnp.sum(jnp.abs(survivor_pos - n["agent_pos"][i]), axis=1) <= 1
            eligible = nearby & (carrier == -1) & (s["survivor_health"] > 0)
            sid = jnp.argmax(eligible.astype(jnp.int32))
            pickup = (a[i] == 7) & ~holding & jnp.any(eligible)
            carrier = carrier.at[sid].set(jnp.where(pickup, i, carrier[sid]))
            survivor_pos = survivor_pos.at[sid].set(jnp.where(pickup, n["agent_pos"][i], survivor_pos[sid]))
            take = (a[i] == 8) & (jnp.sum(jnp.abs(n["agent_pos"][i] - s["tank_pos"][0])) <= 1) & (water[i] < 6) & (tank > 0)
            water = water.at[i].add(take.astype(jnp.int32))
            tank = tank - take.astype(jnp.int32)
            refilled = refilled.at[i].set(take)
            ally = jnp.all(n["agent_pos"] == target[i], axis=1) & (jnp.arange(3) != i) & (s["agent_health"] > 0) & (water < 6)
            recipient = jnp.argmax(ally.astype(jnp.int32))
            give = (a[i] == 9) & (water[i] > 0) & jnp.any(ally)
            water = water.at[i].add(-give.astype(jnp.int32)).at[recipient].add(give.astype(jnp.int32))
            shared = shared.at[i].set(give)
        # RESCUE: any carried survivor entering the four-cell safe corner is resolved once.
        carried = (carrier >= 0) & (carrier < 3)
        survivor_pos = jnp.where(carried[:, None], n["agent_pos"][jnp.clip(carrier, 0, 2)], survivor_pos)
        rescued = carried & (survivor_pos[:, 0] <= 1) & (survivor_pos[:, 1] <= 1)
        rescuer = carrier
        carrier = jnp.where(rescued, 3, carrier)
        survivor_pos = jnp.where(rescued[:, None], s["safe_pos"][0], survivor_pos)
        # SPREAD_ORDER: deterministic orthogonal fronts every third turn, without wrapping.
        burning = fire > 0
        fuel = jnp.maximum(fuel - burning.astype(jnp.int32), 0)
        ignition = ((n["t"] % 3 == 0) & (self.neighbors(burning.astype(jnp.int32)) > 0)
                    & (fuel > 0) & ~burning & (s["tile"] == 0))
        fire = jnp.where((burning & (fuel > 0)) | ignition, 2, 0).astype(jnp.int32)
        smoke = jnp.maximum(jnp.maximum(s["smoke"] - 1 - 2 * sprays, 0),
                            2 * (self.neighbors((fire > 0).astype(jnp.int32)) > 0).astype(jnp.int32))
        smoke = jnp.where(fire > 0, 3, smoke).astype(jnp.int32)
        # CONSEQUENCES: protection changes partners' exposure; disabled carriers release cargo.
        p = n["agent_pos"]
        injury = ((fire[p[:, 1], p[:, 0]] > 0) | (smoke[p[:, 1], p[:, 0]] >= 3)) & (s["agent_health"] > 0)
        health = s["agent_health"] - injury.astype(jnp.int32)
        disabled_carrier = (carrier >= 0) & (carrier < 3) & (health[jnp.clip(carrier, 0, 2)] == 0)
        carrier = jnp.where(disabled_carrier, -1, carrier)
        exposed = ((fire[survivor_pos[:, 1], survivor_pos[:, 0]] > 0) | (smoke[survivor_pos[:, 1], survivor_pos[:, 0]] >= 3)) & (carrier < 3) & (s["survivor_health"] > 0)
        survivor_health = s["survivor_health"] - exposed.astype(jnp.int32)
        lost = (survivor_health == 0) & (carrier < 3)
        carrier = jnp.where(lost, 4, carrier)
        n["fire"], n["fuel"], n["smoke"] = fire, fuel, smoke
        n["water"], n["tank_stock"] = water, jnp.array([tank], dtype=jnp.int32)
        n["spent_water"] = s["spent_water"] + jnp.sum(spraying.astype(jnp.int32))
        n["agent_health"] = health
        n["survivor_carrier"], n["survivor_pos"], n["survivor_health"] = carrier, survivor_pos, survivor_health
        n["last_rescued"], n["last_lost"], n["last_injury"] = rescued, lost, injury
        n["last_spread"], n["last_cut"], n["last_spray"] = ignition.reshape(63), cuts.reshape(63), sprays.reshape(63)
        n["last_refill"], n["last_share"] = refilled, shared
        role = jnp.arange(3)
        rescue_credit = jnp.sum(rescued[:, None] & (rescuer[:, None] == role[None, :]), axis=0)
        priorities = (jnp.arange(6)[:, None] % 3) == role[None, :]
        priority_gain = jnp.sum(priorities * rescued[:, None], axis=0)
        priority_loss = jnp.sum(priorities * lost[:, None], axis=0)
        rewards = 0.4 * jnp.sum(rescued) - 0.4 * jnp.sum(lost) + 0.8 * rescue_credit + 0.6 * priority_gain - 0.6 * priority_loss - 0.1 * injury
        n["terminated"] = jnp.all(carrier >= 3) | jnp.all(health == 0)
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["done"] = n["terminated"] | n["truncated"]
        return self.refresh(n), rewards.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions, 0, 9)
        valid = self.legal_actions(state)[jnp.arange(3), safe] & (safe == actions)
        n, rewards = self.advance(state, jnp.where(valid, safe, 0))
        # ABSORBING_TERMINAL: no auto-reset and no clock advance after completion.
        n = {k: jnp.where(state["done"], state[k], n[k]) for k in state}
        rewards = jnp.where(state["done"], jnp.zeros((3,), dtype=jnp.float32), rewards)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, rewards, jnp.full((3,), n["done"], dtype=jnp.bool_), info
