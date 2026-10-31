"""Original hose-network rescue game; authored without execution."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 242
    num_actions = 11
    max_steps = 96

    def reset(self, key):
        cells = jnp.stack((jnp.arange(63) % 9, jnp.arange(63) // 9), axis=1).astype(jnp.int32)
        brush = cells[:, 0] >= 3
        fire = jnp.zeros((63,), dtype=jnp.int32).at[25].set(1).at[43].set(1)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "episode_done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
            "agent_pos": jnp.array([[1, 3], [2, 3], [3, 3]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "role": jnp.arange(3, dtype=jnp.int32),
            "water": jnp.array([4, 3, 2], dtype=jnp.int32),
            "coupled": jnp.zeros((3,), dtype=jnp.bool_),
            "connected": jnp.zeros((3,), dtype=jnp.bool_),
            "terrain": brush.astype(jnp.int32).reshape(7, 9),
            "cell_pos": cells, "fire": fire,
            "fuel": jnp.where(brush, 4, 0).astype(jnp.int32),
            "fire_active": fire > 0, "fire_visible": jnp.ones((3, 63), dtype=jnp.bool_),
            "victim_pos": jnp.array([[4, 1], [5, 1], [6, 2], [4, 5], [5, 5], [6, 4]], dtype=jnp.int32),
            "victim_owner": jnp.full((6,), -1, dtype=jnp.int32),
            "victim_status": jnp.zeros((6,), dtype=jnp.int32),
            "victim_distress": jnp.zeros((6,), dtype=jnp.int32),
            "victim_active": jnp.ones((6,), dtype=jnp.bool_),
            "victim_visible": jnp.ones((3, 6), dtype=jnp.bool_),
            "pump_pos": jnp.array([[1, 3]], dtype=jnp.int32),
            "pump_active": jnp.ones((1,), dtype=jnp.bool_),
            "pump_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "reservoir": jnp.array([36], dtype=jnp.int32),
            "pump_on": jnp.zeros((1,), dtype=jnp.bool_),
            "shelter_pos": jnp.array([[0, 3]], dtype=jnp.int32),
            "shelter_active": jnp.ones((1,), dtype=jnp.bool_),
            "shelter_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "rescued": jnp.array(0, dtype=jnp.int32), "lost": jnp.array(0, dtype=jnp.int32),
            "sprays": jnp.array(0, dtype=jnp.int32), "cuts": jnp.array(0, dtype=jnp.int32),
            "handoffs": jnp.array(0, dtype=jnp.int32)
        }
        return self.observe(state), state

    def observe(self, state):
        public = jnp.concatenate((
            state["agent_pos"].reshape(-1) / 8.0, state["water"] / 4.0,
            state["coupled"].astype(jnp.float32), state["connected"].astype(jnp.float32),
            state["fire"] / 3.0, state["fuel"] / 4.0, state["terrain"].reshape(-1).astype(jnp.float32),
            state["victim_pos"].reshape(-1) / 8.0, state["victim_status"] / 3.0,
            (state["victim_owner"] + 1) / 3.0, state["victim_distress"] / 3.0,
            jnp.stack((state["reservoir"][0] / 36.0, state["pump_on"][0].astype(jnp.float32),
                       state["rescued"] / 6.0, state["lost"] / 6.0, state["t"] / 96.0))
        )).astype(jnp.float32)
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(public, (3, 239))), axis=1)

    def legal_actions(self, state):
        return jnp.where(state["episode_done"], jnp.arange(11)[None, :] == 0,
                         jnp.ones((3, 11), dtype=jnp.bool_))

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions, 0, 10)
        a = jnp.where((actions >= 0) & (actions < 11) &
                      self.legal_actions(state)[jnp.arange(3), safe], safe, 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)
        carry = jnp.any(state["victim_owner"][None, :] == jnp.arange(3)[:, None], axis=1)
        can_walk = ~carry | (jnp.arange(3) == 2) | (state["t"] % 2 == 0)
        proposed = jnp.clip(state["agent_pos"] + delta[a], jnp.array([0, 0]), jnp.array([8, 6]))
        heat = state["fire"][proposed[:, 1] * 9 + proposed[:, 0]]
        pos = jnp.where((can_walk & (heat < 2))[:, None], proposed, state["agent_pos"])
        stationary = jnp.all(pos == state["agent_pos"], axis=1)
        coupled = stationary & (state["coupled"] | (a == 5))
        safe_hose = state["fire"][pos[:, 1] * 9 + pos[:, 0]] < 2
        pump_distance = jnp.sum(jnp.abs(pos - state["pump_pos"][0]), axis=1)
        crank = (a == 8) & (pump_distance <= 1)
        pump_on = jnp.any(crank) & (state["reservoir"][0] > 0)
        connected = coupled & safe_hose & (pump_distance <= 2) & pump_on
        distance = jnp.sum(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=2)
        reach = jnp.array([3, 4, 3], dtype=jnp.int32)
        edges = (distance <= jnp.maximum(reach[:, None], reach[None, :])) & coupled[:, None] & coupled[None, :]
        edges = edges & safe_hose[:, None] & safe_hose[None, :]
        for hop in range(3):
            connected = connected | (coupled & jnp.any(edges & connected[None, :], axis=1))
        water = state["water"]
        reserve = state["reservoir"][0]
        capacity = jnp.array([4, 4, 3], dtype=jnp.int32)
        pump_rate = 1 + crank[0].astype(jnp.int32)
        for turn in range(3):
            i = (state["t"] + turn) % 3
            amount = jnp.minimum(jnp.minimum(capacity[i] - water[i], pump_rate), reserve)
            amount = jnp.where(connected[i], amount, 0)
            water = water.at[i].add(amount)
            reserve = reserve - amount
        nearby = jnp.sum(jnp.abs(pos[:, None, :] - state["cell_pos"][None, :, :]), axis=2)
        targetable = (nearby <= 1) & (state["fire"][None, :] > 0)
        target = jnp.argmax(jnp.where(targetable, state["fire"][None, :], -1), axis=1)
        use = (a == 6) & (water > 0) & jnp.any(targetable, axis=1)
        power = jnp.array([2, 2, 1], dtype=jnp.int32)
        spray = jnp.sum((jnp.arange(63)[None, :] == target[:, None]) * (use * power)[:, None], axis=0)
        fire = jnp.maximum(state["fire"] - spray, 0)
        water = water - use.astype(jnp.int32)
        cut_radius = jnp.array([0, 1, 0], dtype=jnp.int32)
        cut = jnp.any((a == 7)[:, None] & (nearby <= cut_radius[:, None]), axis=0) & (fire == 0)
        fuel = jnp.where(cut, 0, state["fuel"])
        terrain = jnp.where(cut, 0, state["terrain"].reshape(-1)).reshape(7, 9)
        vpos = state["victim_pos"]
        owner = state["victim_owner"]
        status = state["victim_status"]
        personal = jnp.zeros((3,), dtype=jnp.float32)
        for i in range(3):
            carrying = owner == i
            at_shelter = jnp.all(pos[i] == state["shelter_pos"][0])
            delivered = carrying & (a[i] == 9) & at_shelter
            status = jnp.where(delivered, 2, status)
            owner = jnp.where(delivered, -1, owner)
            personal = personal.at[i].add(jnp.sum(delivered) * 0.2)
            available = (status == 0) & jnp.all(vpos == pos[i], axis=1)
            take = (a[i] == 9) & ~jnp.any(carrying) & jnp.any(available)
            chosen = jnp.argmax(available)
            pickup = (jnp.arange(6) == chosen) & take
            owner = jnp.where(pickup, i, owner)
            status = jnp.where(pickup, 1, status)
        exchanged = jnp.zeros((3,), dtype=jnp.bool_)
        handoffs = jnp.array(0, dtype=jnp.int32)
        for i in range(3):
            occupied = jnp.any(owner[None, :] == jnp.arange(3)[:, None], axis=1)
            eligible = (a == 10) & ~occupied & ~exchanged & (distance[i] <= 1) & (jnp.arange(3) != i)
            transfer = (a[i] == 10) & occupied[i] & ~exchanged[i] & jnp.any(eligible)
            receiver = jnp.argmax(eligible)
            owner = jnp.where((owner == i) & transfer, receiver, owner)
            exchanged = exchanged.at[i].set(exchanged[i] | transfer)
            exchanged = exchanged.at[receiver].set(exchanged[receiver] | transfer)
            handoffs = handoffs + transfer.astype(jnp.int32)
        vpos = jnp.where((status == 1)[:, None], pos[jnp.clip(owner, 0, 2)], vpos)
        t = state["t"] + jnp.array(1, dtype=jnp.int32)
        adjacent = jnp.sum(jnp.abs(state["cell_pos"][:, None, :] - state["cell_pos"][None, :, :]), axis=2) == 1
        exposed = jnp.any(adjacent & (fire[None, :] > 0), axis=1)
        ignition = (t % 4 == 0) & exposed & (fuel > 0) & (fire == 0)
        fire = jnp.where(ignition, 1, jnp.where((fire > 0) & (fuel > 0), jnp.minimum(fire + 1, 3), 0))
        fuel = jnp.maximum(fuel - (fire > 0).astype(jnp.int32), 0)
        fire = jnp.where(fuel > 0, fire, 0)
        victim_heat = fire[vpos[:, 1] * 9 + vpos[:, 0]]
        distress = jnp.where((status == 0) & (victim_heat >= 2),
                            jnp.minimum(state["victim_distress"] + 1, 3), state["victim_distress"])
        killed = (status == 0) & (distress >= 3)
        status = jnp.where(killed, 3, status)
        rescued = jnp.sum(status == 2).astype(jnp.int32)
        lost = jnp.sum(status == 3).astype(jnp.int32)
        terminated = jnp.all(status >= 2)
        truncated = (t >= self.max_steps) & ~terminated
        nxt = dict(state)
        nxt["t"] = t
        nxt["agent_pos"] = pos
        nxt["water"] = water
        nxt["coupled"] = coupled
        nxt["connected"] = connected
        nxt["reservoir"] = jnp.reshape(reserve, (1,))
        nxt["pump_on"] = jnp.reshape(pump_on, (1,))
        nxt["terrain"] = terrain
        nxt["fire"] = fire
        nxt["fuel"] = fuel
        nxt["fire_active"] = fire > 0
        nxt["victim_pos"] = vpos
        nxt["victim_owner"] = owner
        nxt["victim_status"] = status
        nxt["victim_distress"] = distress
        nxt["rescued"] = rescued
        nxt["lost"] = lost
        nxt["sprays"] = jnp.sum(use).astype(jnp.int32)
        nxt["cuts"] = jnp.sum(cut & (state["fuel"] > 0)).astype(jnp.int32)
        nxt["handoffs"] = handoffs
        nxt["terminated"] = terminated
        nxt["truncated"] = truncated
        nxt["episode_done"] = terminated | truncated
        reward = jnp.full((3,), (rescued - state["rescued"]) - (lost - state["lost"]) - 0.005, dtype=jnp.float32) + personal
        nxt = jax.tree.map(lambda new, old: jnp.where(state["episode_done"], old, new), nxt, state)
        reward = jnp.where(state["episode_done"], jnp.zeros((3,), dtype=jnp.float32), reward).astype(jnp.float32)
        dones = jnp.full((3,), nxt["episode_done"], dtype=jnp.bool_)
        info = {"done_all": nxt["episode_done"], "terminated": nxt["terminated"], "truncated": nxt["truncated"]}
        return self.observe(nxt), nxt, reward, dones, info
