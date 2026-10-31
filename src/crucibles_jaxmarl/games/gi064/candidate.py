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
            "biomass": jax.random.uniform(keys[0], (2,), minval=0.25, maxval=0.55, dtype=jnp.float32),
            "affinity": jax.random.uniform(keys[1], (2,), minval=0.75, maxval=1.25, dtype=jnp.float32),
            "catalyst": jnp.full((2,), 0.15, dtype=jnp.float32),
            "nutrient": jax.random.uniform(keys[2], (), minval=0.45, maxval=0.75, dtype=jnp.float32),
            "toxin": jnp.array(0.12, dtype=jnp.float32),
            "last_harvest": jnp.zeros((2,), dtype=jnp.float32),
            "last_uptake": jnp.zeros((2,), dtype=jnp.float32),
            "last_action": jnp.zeros((2,), dtype=jnp.int32),
            "total_harvest": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["biomass"])
        obs = obs.at[:, 1].set(jnp.roll(state["biomass"], 1))
        obs = obs.at[:, 2].set(state["catalyst"])
        obs = obs.at[:, 3].set(jnp.roll(state["catalyst"], 1))
        obs = obs.at[:, 4].set(state["nutrient"])
        obs = obs.at[:, 5].set(state["toxin"])
        obs = obs.at[:, 6].set(state["affinity"] / 1.25)
        obs = obs.at[:, 7].set(state["last_harvest"] / 0.38)
        obs = obs.at[:, 8].set(jnp.roll(state["last_harvest"], 1) / 0.38)
        obs = obs.at[:, 9].set(state["last_uptake"] / 0.15)
        obs = obs.at[:, 10].set(jnp.roll(state["last_uptake"], 1) / 0.15)
        obs = obs.at[:, 11].set(state["last_action"].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 12].set(jnp.roll(state["last_action"], 1).astype(jnp.float32) / 4.0)
        obs = obs.at[:, 13].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 14].set(state["total_harvest"] / 16.0)
        obs = obs.at[:, 15].set(jnp.roll(state["total_harvest"], 1) / 16.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        grow = (actions == 1).astype(jnp.float32)
        take = (actions == 2).astype(jnp.float32)
        clean = (actions == 3).astype(jnp.float32)
        release = (actions == 4).astype(jnp.float32)
        harvest = 0.38 * state["biomass"] * take
        donation = 0.22 * state["biomass"] * release
        demand = 0.12 * grow * state["affinity"] * (0.25 + 0.75 * jnp.roll(state["catalyst"], 1))
        ration = jnp.minimum(1.0, state["nutrient"] / jnp.maximum(jnp.sum(demand), 0.0001))
        uptake = demand * ration
        new_biomass = jnp.clip(
            0.985 * state["biomass"] - harvest - donation
            + 0.9 * uptake * (1.0 - 0.65 * state["toxin"]), 0.0, 1.0
        )
        new_catalyst = jnp.clip(0.58 * state["catalyst"] + 1.6 * donation, 0.0, 1.0)
        new_nutrient = jnp.clip(state["nutrient"] - jnp.sum(uptake) + 0.08 + 0.06 * jnp.sum(clean), 0.0, 1.0)
        new_toxin = jnp.clip(
            0.85 * state["toxin"] + 0.28 * jnp.sum(harvest)
            + 0.10 * jnp.sum(grow) - 0.12 * jnp.sum(clean), 0.0, 1.0
        )
        rewards = jnp.tanh(
            2.0 * harvest - 0.05 * grow - 0.04 * clean
            - 0.01 * release - 0.10 * new_toxin
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "biomass": new_biomass.astype(jnp.float32),
            "affinity": state["affinity"],
            "catalyst": new_catalyst.astype(jnp.float32),
            "nutrient": new_nutrient.astype(jnp.float32),
            "toxin": new_toxin.astype(jnp.float32),
            "last_harvest": harvest.astype(jnp.float32),
            "last_uptake": uptake.astype(jnp.float32),
            "last_action": actions.astype(jnp.int32),
            "total_harvest": (state["total_harvest"] + harvest).astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
