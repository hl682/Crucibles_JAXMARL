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
            "energy": jax.random.uniform(keys[0], (2,), minval=1.2, maxval=2.8, dtype=jnp.float32),
            "queue": jnp.full((2, 3), -1, dtype=jnp.int32),
            "gate": jax.random.randint(keys[1], (2,), 0, 2, dtype=jnp.int32),
            "preference": jax.random.randint(keys[2], (2,), 0, 2, dtype=jnp.int32),
            "last_in": jnp.zeros((2,), dtype=jnp.float32),
            "last_out": jnp.zeros((2,), dtype=jnp.float32),
            "last_waste": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        incoming_queue = jnp.roll(state["queue"], 1, axis=0)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["energy"] / 3.0)
        obs = obs.at[:, 2].set(state["gate"].astype(jnp.float32))
        obs = obs.at[:, 3].set(state["preference"].astype(jnp.float32))
        obs = obs.at[:, 4].set(jnp.roll(state["gate"], 1).astype(jnp.float32))
        obs = obs.at[:, 5].set(jnp.roll(state["energy"], 1) / 3.0)
        obs = obs.at[:, 6:9].set((state["queue"].astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 9].set((incoming_queue[:, 0].astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 10].set((incoming_queue[:, 1] >= 0).astype(jnp.float32))
        obs = obs.at[:, 11].set(state["last_in"])
        obs = obs.at[:, 12].set(state["last_out"])
        obs = obs.at[:, 13].set(state["last_waste"])
        obs = obs.at[:, 14].set((state["energy"] >= 1.0).astype(jnp.float32))
        obs = obs.at[:, 15].set(jnp.arange(2, dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        launch_requested = (actions == 1) | (actions == 2)
        launched = launch_requested & (state["energy"] >= 1.0)
        package = jnp.where(launched, actions - 1, -1).astype(jnp.int32)
        incoming = jnp.roll(state["queue"][:, 0], 1)
        present = incoming >= 0
        matched = present & (incoming == state["gate"])
        preferred = incoming == state["preference"]
        incoming_value = matched.astype(jnp.float32) * (0.6 + 0.4 * preferred.astype(jnp.float32))
        outgoing_value = jnp.roll(incoming_value, 1)
        wasted = (present & ~matched).astype(jnp.float32)
        gate = jnp.where(actions == 3, 0, jnp.where(actions == 4, 1, state["gate"])).astype(jnp.int32)
        queue = jnp.concatenate((state["queue"][:, 1:], package[:, None]), axis=1)
        energy = jnp.clip(
            state["energy"] - launched.astype(jnp.float32) + 0.2
            + 0.5 * matched.astype(jnp.float32),
            0.0,
            3.0,
        )
        rewards = (
            0.5 * incoming_value + 0.4 * outgoing_value
            - 0.05 * launched.astype(jnp.float32) - 0.05 * wasted
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "energy": energy.astype(jnp.float32),
            "queue": queue,
            "gate": gate,
            "preference": state["preference"],
            "last_in": incoming_value.astype(jnp.float32),
            "last_out": outgoing_value.astype(jnp.float32),
            "last_waste": wasted,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info