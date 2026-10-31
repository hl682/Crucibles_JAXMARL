import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 154
    num_actions = 10
    max_steps = 128

    def visibility(self, pos, code_pos):
        owner = jnp.arange(3)[:, None] == jnp.arange(2)[None, :]
        near = jnp.sum(jnp.abs(pos[:, None, :] - code_pos[None, :, :]), axis=2) <= 1
        return owner & near

    def reset(self, key):
        # EVIDENCE: separated_roles_and_private_clues
        terrain = jnp.ones((7, 13), dtype=jnp.int32)
        terrain = terrain.at[:, 0:4].set(0)
        terrain = terrain.at[:, 5:8].set(0)
        terrain = terrain.at[:, 10].set(0)
        terrain = terrain.at[6, 9:13].set(0)
        terrain = terrain.at[3, 11].set(0)
        pos = jnp.array([[1, 1], [6, 5], [10, 6]], dtype=jnp.int32)
        code_pos = jnp.array([[1, 1], [6, 5]], dtype=jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "map": terrain,
            "agent_pos": pos,
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "energy": jnp.full((3,), 4, dtype=jnp.int32),
            "signal_cooldown": jnp.zeros((3,), dtype=jnp.int32),
            "panel_pos": jnp.array([[2, 4], [6, 2]], dtype=jnp.int32),
            "panel_active": jnp.ones((2,), dtype=jnp.bool_),
            "panel_visible": jnp.ones((3, 2), dtype=jnp.bool_),
            "panel_value": jnp.full((2,), -1, dtype=jnp.int32),
            "gate_pos": jnp.array([[10, 4], [10, 2]], dtype=jnp.int32),
            "gate_active": jnp.ones((2,), dtype=jnp.bool_),
            "gate_visible": jnp.ones((3, 2), dtype=jnp.bool_),
            "gate_timer": jnp.zeros((2,), dtype=jnp.int32),
            "gate_open": jnp.zeros((2,), dtype=jnp.bool_),
            "jam": jnp.zeros((2,), dtype=jnp.int32),
            "code_pos": code_pos,
            "code_active": jnp.ones((2,), dtype=jnp.bool_),
            "code_visible": self.visibility(pos, code_pos),
            "code_value": jax.random.randint(key, (2,), 0, 2, dtype=jnp.int32),
            "signal_pos": pos,
            "signal_active": jnp.zeros((3,), dtype=jnp.bool_),
            "signal_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "signal_value": jnp.zeros((3,), dtype=jnp.int32),
            "signal_ttl": jnp.zeros((3,), dtype=jnp.int32),
            "cargo_pos": jnp.array([[10, 6], [11, 6]], dtype=jnp.int32),
            "cargo_active": jnp.ones((2,), dtype=jnp.bool_),
            "cargo_visible": jnp.ones((3, 2), dtype=jnp.bool_),
            "cargo_owner": jnp.full((2,), -1, dtype=jnp.int32),
            "cargo_delivered": jnp.zeros((2,), dtype=jnp.bool_),
            "goal_pos": jnp.array([[10, 0]], dtype=jnp.int32),
            "goal_active": jnp.ones((1,), dtype=jnp.bool_),
            "goal_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "events": jnp.zeros((6,), dtype=jnp.bool_),
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
        # EVIDENCE: role_observation_and_signal_lifetime
        private_codes = jnp.where(state["code_visible"], state["code_value"][None, :], -1).astype(jnp.float32)
        public = jnp.concatenate((
            state["agent_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["energy"].astype(jnp.float32) / 4.0,
            state["panel_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["panel_value"].astype(jnp.float32),
            state["gate_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["gate_timer"].astype(jnp.float32) / 8.0,
            state["jam"].astype(jnp.float32) / 4.0,
            state["code_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            jnp.where(state["signal_ttl"] > 0, state["signal_value"], -1).astype(jnp.float32),
            state["signal_ttl"].astype(jnp.float32) / 4.0,
            state["signal_cooldown"].astype(jnp.float32) / 4.0,
            state["cargo_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            (state["cargo_owner"] + 1).astype(jnp.float32) / 4.0,
            state["cargo_delivered"].astype(jnp.float32),
            state["goal_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["map"].reshape(-1).astype(jnp.float32),
            state["t"][None].astype(jnp.float32) / 128.0,
            state["events"].astype(jnp.float32),
            jnp.stack((state["episode_done"], state["terminated"], state["truncated"])).astype(jnp.float32),
        ))
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), private_codes, state["code_visible"].astype(jnp.float32), jnp.broadcast_to(public, (3, 147))), axis=1)

    def raw_step(self, key, state, actions):
        a = jnp.where((actions >= 0) & (actions < 10), actions, 0).astype(jnp.int32)
        old = state["agent_pos"]
        ids = jnp.arange(3, dtype=jnp.int32)
        direction = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        moving = (a >= 1) & (a <= 4)
        target = jnp.clip(old + direction[jnp.where(moving, a, 0)], jnp.array([0, 0]), jnp.array([12, 6]))
        at_gate = jnp.all(target[:, None, :] == state["gate_pos"][None, :, :], axis=2)
        gate_blocked = jnp.any(at_gate & ~state["gate_open"][None, :], axis=1)
        wall = state["map"][target[:, 1], target[:, 0]] == 1
        occupied = jnp.any(jnp.all(target[:, None, :] == old[None, :, :], axis=2) & (ids[:, None] != ids[None, :]), axis=1)
        duplicate = jnp.sum(jnp.all(target[:, None, :] == target[None, :, :], axis=2), axis=1) > 1
        pos = jnp.where((moving & ~gate_blocked & ~wall & ~occupied & ~duplicate)[:, None], target, old)

        # EVIDENCE: limited_causally_effective_messages
        signal_ttl = jnp.maximum(state["signal_ttl"] - 1, 0)
        cooldown = jnp.maximum(state["signal_cooldown"] - 1, 0)
        send = ((a == 7) | (a == 8)) & (state["signal_cooldown"] == 0)
        signal_value = jnp.where(send, a - 7, state["signal_value"])
        signal_ttl = jnp.where(send, 4, signal_ttl)
        cooldown = jnp.where(send, 3, cooldown)

        # EVIDENCE: cross_wired_remote_gates
        gate_timer = jnp.maximum(state["gate_timer"] - 1, 0)
        jam = jnp.maximum(state["jam"] - 1, 0)
        energy = state["energy"]
        at_sensor = jnp.concatenate((jnp.all(pos[:2] == state["code_pos"], axis=1), jnp.array([False])))
        energy = jnp.minimum(4, energy + ((a == 0) & at_sensor).astype(jnp.int32))
        panel_value = state["panel_value"]
        opened = jnp.array(False)
        fault = jnp.array(False)
        for i in range(2):
            at_panel = jnp.all(pos[i] == state["panel_pos"][i])
            activate = ((a[i] == 5) | (a[i] == 6)) & at_panel & (energy[i] > 0) & (state["jam"][i] == 0)
            value = a[i] - 5
            correct = activate & (value == state["code_value"][1 - i])
            wrong = activate & ~correct
            gate_timer = gate_timer.at[i].set(jnp.where(correct, 6, jnp.where(wrong, 0, gate_timer[i])))
            jam = jam.at[i].set(jnp.where(wrong, 2, jam[i]))
            panel_value = panel_value.at[i].set(jnp.where(activate, value, panel_value[i]))
            energy = energy.at[i].add(-activate.astype(jnp.int32))
            opened = opened | correct
            fault = fault | wrong

        # EVIDENCE: physical_cargo_and_round_change
        owners = state["cargo_owner"]
        carried = owners == 2
        had_cargo = jnp.any(carried)
        matching = jnp.all(state["cargo_pos"] == pos[2], axis=1) & ~state["cargo_delivered"] & (owners < 0)
        pick_id = jnp.argmax(matching).astype(jnp.int32)
        pick = (a[2] == 9) & ~had_cargo & jnp.any(matching)
        release = (a[2] == 9) & had_cargo
        at_goal = jnp.all(pos[2] == state["goal_pos"][0])
        delivery_mask = carried & release & at_goal
        delivered_now = jnp.any(delivery_mask)
        delivered = state["cargo_delivered"] | delivery_mask
        owners = jnp.where(carried & release, -1, owners)
        owners = owners.at[pick_id].set(jnp.where(pick, 2, owners[pick_id]))
        cargo_pos = jnp.where((carried | (owners == 2))[:, None], pos[2], state["cargo_pos"])
        new_codes = jax.random.randint(key, (2,), 0, 2, dtype=jnp.int32)
        code_value = jnp.where(delivered_now, new_codes, state["code_value"])
        gate_timer = jnp.where(delivered_now, jnp.zeros((2,), dtype=jnp.int32), gate_timer)
        signal_ttl = jnp.where(delivered_now, jnp.zeros((3,), dtype=jnp.int32), signal_ttl)
        success = jnp.all(delivered)
        events = jnp.stack((jnp.any(send), opened, fault, pick | release, delivered_now, delivered_now))
        reward = delivered_now.astype(jnp.float32) / 2.0 + success.astype(jnp.float32) / 2.0

        # EVIDENCE: natural_priority_absorbing_terminal
        n = dict(state)
        n["t"] = state["t"] + 1
        n["agent_pos"] = pos
        n["energy"] = energy
        n["signal_cooldown"] = cooldown
        n["panel_value"] = panel_value
        n["gate_timer"] = gate_timer
        n["gate_open"] = gate_timer > 0
        n["jam"] = jam
        n["code_value"] = code_value
        n["code_visible"] = self.visibility(pos, state["code_pos"])
        n["signal_pos"] = pos
        n["signal_value"] = signal_value
        n["signal_ttl"] = signal_ttl
        n["signal_active"] = signal_ttl > 0
        n["cargo_pos"] = cargo_pos
        n["cargo_owner"] = owners
        n["cargo_delivered"] = delivered
        n["cargo_active"] = ~delivered
        n["events"] = events
        n["terminated"] = success
        n["truncated"] = (n["t"] >= self.max_steps) & ~success
        n["episode_done"] = n["terminated"] | n["truncated"]
        n = jax.tree.map(lambda new, previous: jnp.where(state["episode_done"], previous, new), n, state)
        rewards = jnp.full((3,), jnp.where(state["episode_done"], 0.0, reward), dtype=jnp.float32)
        done = n["episode_done"]
        info = {"done_all": done, "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, rewards, jnp.full((3,), done, dtype=jnp.bool_), info