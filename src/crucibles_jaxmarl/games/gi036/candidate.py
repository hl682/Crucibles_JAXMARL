import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        energy_key, order_key = jax.random.split(key, 2)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "stock": jnp.zeros((2,), dtype=jnp.float32),
            "energy": jax.random.uniform(energy_key, (2,), minval=0.8, maxval=1.1, dtype=jnp.float32),
            "stage": jnp.zeros((2,), dtype=jnp.int32),
            "target": jax.random.randint(order_key, (2,), 0, 2, dtype=jnp.int32),
            "catalyst": jnp.zeros((2, 2), dtype=jnp.float32),
            "last_success": jnp.zeros((2,), dtype=jnp.float32),
            "completed": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        other = jnp.array([1, 0], dtype=jnp.int32)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["stock"] / 4.0)
        obs = obs.at[:, 2].set(state["energy"] / 1.5)
        obs = obs.at[:, 3].set(state["stage"].astype(jnp.float32))
        obs = obs.at[:, 4].set(state["target"].astype(jnp.float32))
        obs = obs.at[:, 5].set(state["catalyst"][:, 0] / 2.0)
        obs = obs.at[:, 6].set(state["catalyst"][:, 1] / 2.0)
        obs = obs.at[:, 7].set(state["stage"][other].astype(jnp.float32))
        obs = obs.at[:, 8].set(state["target"][other].astype(jnp.float32))
        obs = obs.at[:, 9].set(state["energy"][other] / 1.5)
        obs = obs.at[:, 10].set(state["catalyst"][other, 0] / 2.0)
        obs = obs.at[:, 11].set(state["catalyst"][other, 1] / 2.0)
        obs = obs.at[:, 12].set(state["last_success"])
        obs = obs.at[:, 13].set(state["last_success"][other])
        obs = obs.at[:, 14].set(state["completed"] / 16.0)
        obs = obs.at[:, 15].set(state["completed"][other] / 16.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        target_mask = jax.nn.one_hot(state["target"], 2, dtype=jnp.float32)
        available = jnp.sum(state["catalyst"] * target_mask, axis=1)
        harvest = (actions == 1) & (state["energy"] >= 0.2)
        send = ((actions == 2) | (actions == 4)) & (state["energy"] >= 0.25)
        success = (actions == 3) & (state["stock"] >= 1.0) & (state["energy"] >= 0.35) & (available >= 1.0)
        completed = success & (state["stage"] == 1)
        failed_process = (actions == 3) & ~success
        harvest_f = harvest.astype(jnp.float32)
        send_f = send.astype(jnp.float32)
        success_f = success.astype(jnp.float32)
        completed_f = completed.astype(jnp.float32)
        emitted_color = (actions == 4).astype(jnp.int32)
        emitted = jax.nn.one_hot(emitted_color, 2, dtype=jnp.float32) * send_f[:, None]
        retained = jnp.maximum(state["catalyst"] - target_mask * success_f[:, None], 0.0) * 0.65
        next_catalyst = jnp.clip(retained + jnp.roll(emitted, 1, axis=0), 0.0, 2.0)
        next_stock = jnp.clip(state["stock"] + harvest_f - success_f, 0.0, 4.0)
        next_energy = jnp.clip(state["energy"] + 0.04 + 0.3 * (actions == 0).astype(jnp.float32) - 0.2 * harvest_f - 0.25 * send_f - 0.35 * success_f - 0.05 * failed_process.astype(jnp.float32), 0.0, 1.5)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "stock": next_stock,
            "energy": next_energy,
            "stage": jnp.where(success, 1 - state["stage"], state["stage"]).astype(jnp.int32),
            "target": jnp.where(completed, 1 - state["target"], state["target"]).astype(jnp.int32),
            "catalyst": next_catalyst,
            "last_success": success_f,
            "completed": state["completed"] + completed_f,
        }
        rewards = jnp.clip(0.12 * success_f + 0.6 * completed_f + 0.2 * jnp.roll(completed_f, 1) - 0.025 * send_f - 0.03 * failed_process.astype(jnp.float32), -1.0, 1.0).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info