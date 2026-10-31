import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        keys = jax.random.split(key, 3)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "switch": jax.random.randint(keys[0], (), 0, 2, dtype=jnp.int32),
            "target": jax.random.randint(keys[1], (2,), 0, 2, dtype=jnp.int32),
            "priority": jax.random.randint(keys[2], (), 0, 2, dtype=jnp.int32),
            "packet": jnp.full((2,), -1, dtype=jnp.int32),
            "age": jnp.zeros((2,), dtype=jnp.int32),
            "lock": jnp.array(0, dtype=jnp.int32),
            "energy": jnp.full((2,), 0.8, dtype=jnp.float32),
            "blocked": jnp.zeros((2,), dtype=jnp.bool_),
            "last_result": jnp.zeros((2,), dtype=jnp.float32),
            "traffic": jnp.array(0, dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        roles = jnp.arange(2, dtype=jnp.int32)
        observations = jnp.zeros((2, 16), dtype=jnp.float32)
        observations = observations.at[:, 0].set(state["t"] / 32.0)
        observations = observations.at[:, 1].set(state["energy"])
        observations = observations.at[:, 2].set(jnp.roll(state["energy"], 1))
        observations = observations.at[:, 3].set(state["target"])
        observations = observations.at[:, 4].set((state["packet"] + 1) / 2.0)
        observations = observations.at[:, 5].set((jnp.roll(state["packet"], 1) + 1) / 2.0)
        observations = observations.at[:, 6].set(state["age"] / 2.0)
        observations = observations.at[:, 7].set(jnp.roll(state["age"], 1) / 2.0)
        observations = observations.at[:, 8].set(state["switch"])
        observations = observations.at[:, 9].set(state["lock"] / 2.0)
        observations = observations.at[:, 10].set(roles == state["priority"])
        observations = observations.at[:, 11].set(state["blocked"])
        observations = observations.at[:, 12].set(jnp.roll(state["blocked"], 1))
        observations = observations.at[:, 13].set(state["last_result"])
        observations = observations.at[:, 14].set(jnp.roll(state["last_result"], 1))
        observations = observations.at[:, 15].set(state["traffic"] / 64.0)
        return observations

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        roles = jnp.arange(2, dtype=jnp.int32)
        fresh_target = jax.random.randint(key, (2,), 0, 2, dtype=jnp.int32)
        toggling = (
            (actions == 3) & (state["energy"] >= 0.08) & (state["lock"] == 0)
        )
        parity = jnp.mod(jnp.sum(toggling.astype(jnp.int32)), 2)
        switch = jnp.mod(state["switch"] + parity, 2).astype(jnp.int32)
        clamping = (actions == 4) & (state["energy"] >= 0.25)
        lock = jnp.where(
            jnp.any(clamping), 2, jnp.maximum(state["lock"] - 1, 0)
        ).astype(jnp.int32)
        active = state["packet"] >= 0
        due = active & (state["age"] >= 2)
        destination = jnp.mod(state["packet"] + switch, 2)
        collision = jnp.all(due) & (destination[0] == destination[1])
        served = due & ((~collision) | (roles == state["priority"]))
        blocked = due & (~served)
        correct = served & (destination == state["target"])
        wrong = served & (~correct)
        launching = (
            ((actions == 1) | (actions == 2))
            & (~active) & (state["energy"] >= 0.15)
        )
        new_packet = jnp.where(launching, actions - 1, -1).astype(jnp.int32)
        remaining = active & (~served)
        packet = jnp.where(remaining, state["packet"], new_packet).astype(jnp.int32)
        age = jnp.where(
            remaining, jnp.minimum(2, state["age"] + 1), jnp.where(launching, 1, 0)
        ).astype(jnp.int32)
        energy = jnp.clip(
            state["energy"] + 0.04
            + 0.06 * (actions == 0).astype(jnp.float32)
            + 0.1 * correct.astype(jnp.float32)
            - 0.15 * launching.astype(jnp.float32)
            - 0.08 * toggling.astype(jnp.float32)
            - 0.25 * clamping.astype(jnp.float32),
            0.0, 1.0,
        )
        rewards = jnp.clip(
            0.8 * correct.astype(jnp.float32) - 0.6 * wrong.astype(jnp.float32)
            - 0.04 * blocked.astype(jnp.float32)
            - 0.03 * launching.astype(jnp.float32)
            - 0.02 * toggling.astype(jnp.float32)
            - 0.05 * clamping.astype(jnp.float32),
            -1.0, 1.0,
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "switch": switch,
            "target": jnp.where(served, fresh_target, state["target"]).astype(jnp.int32),
            "priority": jnp.where(
                collision, 1 - state["priority"], state["priority"]
            ).astype(jnp.int32),
            "packet": packet,
            "age": age,
            "lock": lock,
            "energy": energy.astype(jnp.float32),
            "blocked": blocked,
            "last_result": jnp.where(
                served, jnp.where(correct, 1.0, -1.0), state["last_result"]
            ).astype(jnp.float32),
            "traffic": state["traffic"] + jnp.sum(served.astype(jnp.int32)),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
