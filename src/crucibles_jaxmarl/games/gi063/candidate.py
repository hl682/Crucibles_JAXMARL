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
            "seeds": jnp.ones((2,), dtype=jnp.float32),
            "immature": jnp.zeros((2,), dtype=jnp.float32),
            "flower": jnp.zeros((2,), dtype=jnp.float32),
            "fruit": jnp.zeros((2,), dtype=jnp.float32),
            "moisture": jax.random.uniform(keys[0], (2,), minval=0.65, maxval=0.85, dtype=jnp.float32),
            "pest": jax.random.uniform(keys[1], (2,), minval=0.05, maxval=0.15, dtype=jnp.float32),
            "pollen": jnp.zeros((2,), dtype=jnp.float32),
            "fertility": jax.random.uniform(keys[2], (2,), minval=0.8, maxval=1.2, dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        seeds = state["seeds"]
        immature = state["immature"]
        flower = state["flower"]
        fruit = state["fruit"]
        moisture = state["moisture"]
        pest = state["pest"]
        pollen = state["pollen"]
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(seeds)
        obs = obs.at[:, 2].set(seeds[::-1])
        obs = obs.at[:, 3].set(immature)
        obs = obs.at[:, 4].set(immature[::-1])
        obs = obs.at[:, 5].set(flower)
        obs = obs.at[:, 6].set(flower[::-1])
        obs = obs.at[:, 7].set(fruit)
        obs = obs.at[:, 8].set(fruit[::-1])
        obs = obs.at[:, 9].set(moisture)
        obs = obs.at[:, 10].set(moisture[::-1])
        obs = obs.at[:, 11].set(pest)
        obs = obs.at[:, 12].set(pest[::-1])
        obs = obs.at[:, 13].set(pollen)
        obs = obs.at[:, 14].set(pollen[::-1])
        obs = obs.at[:, 15].set((state["fertility"] - 0.8) / 0.4)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        harvest = jnp.where(actions == 1, jnp.minimum(state["fruit"], 0.5), 0.0)
        planted = jnp.where(actions == 2, jnp.minimum(state["seeds"], 0.3), 0.0)
        watering = (actions == 3).astype(jnp.float32)
        clearing = (actions == 4).astype(jnp.float32)
        rainfall = jax.random.uniform(key, (), minval=0.02, maxval=0.06, dtype=jnp.float32)
        maturation = 0.4 * state["immature"] * state["moisture"] * state["fertility"]
        fertilized = jnp.minimum(state["flower"], 0.85 * state["pollen"]) * state["moisture"] * (1.0 - 0.55 * state["pest"])
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "seeds": jnp.clip(state["seeds"] - planted + 0.65 * harvest, 0.0, 1.0).astype(jnp.float32),
            "immature": jnp.clip(0.98 * state["immature"] - maturation + planted, 0.0, 1.0).astype(jnp.float32),
            "flower": jnp.clip(0.94 * state["flower"] + maturation - fertilized, 0.0, 1.0).astype(jnp.float32),
            "fruit": jnp.clip(0.96 * (state["fruit"] - harvest) + fertilized * (1.0 - 0.35 * state["pest"]), 0.0, 1.0).astype(jnp.float32),
            "moisture": jnp.clip(0.9 * state["moisture"] + rainfall + 0.3 * watering[::-1] - 0.05 * state["immature"], 0.0, 1.0).astype(jnp.float32),
            "pest": jnp.clip(0.92 * state["pest"] + 0.04 * state["fruit"] + 0.03 * state["pest"][::-1] - 0.18 * clearing, 0.0, 1.0).astype(jnp.float32),
            "pollen": (0.9 * state["flower"][::-1] * (1.0 - 0.4 * state["pest"][::-1])).astype(jnp.float32),
            "fertility": state["fertility"],
        }
        rewards = jnp.clip(
            0.95 * harvest - 0.06 * planted - 0.04 * watering - 0.03 * clearing,
            -1.0, 1.0,
        ).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
