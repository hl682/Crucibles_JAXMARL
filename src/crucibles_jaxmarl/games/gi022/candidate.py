import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        keys = jax.random.split(key, 4)
        biomass = jax.random.uniform(keys[0], (2,), minval=0.3, maxval=0.65, dtype=jnp.float32)
        nutrient = jax.random.uniform(keys[1], (2,), minval=0.35, maxval=0.7, dtype=jnp.float32)
        toxin = jax.random.uniform(keys[2], (2,), minval=0.05, maxval=0.2, dtype=jnp.float32)
        affinity = jax.random.uniform(keys[3], (2,), minval=0.35, maxval=0.75, dtype=jnp.float32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "biomass": biomass,
            "nutrient": nutrient,
            "toxin": toxin,
            "valve": jnp.ones((2,), dtype=jnp.bool_),
            "feed_near": jnp.zeros((2,), dtype=jnp.float32),
            "feed_far": jnp.zeros((2,), dtype=jnp.float32),
            "waste": jnp.zeros((2,), dtype=jnp.float32),
            "affinity": affinity,
            "remote": jnp.roll(jnp.stack((biomass, nutrient, toxin), axis=1), 1, axis=0),
            "last_reward": jnp.zeros((2,), dtype=jnp.float32),
            "last_harvest": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(jnp.array([-1.0, 1.0], dtype=jnp.float32))
        obs = obs.at[:, 2].set(state["biomass"])
        obs = obs.at[:, 3].set(state["remote"][:, 0])
        obs = obs.at[:, 4].set(state["nutrient"])
        obs = obs.at[:, 5].set(state["remote"][:, 1])
        obs = obs.at[:, 6].set(state["toxin"])
        obs = obs.at[:, 7].set(state["remote"][:, 2])
        obs = obs.at[:, 8].set(state["valve"].astype(jnp.float32))
        obs = obs.at[:, 9].set(jnp.roll(state["valve"], 1).astype(jnp.float32))
        obs = obs.at[:, 10].set(state["feed_near"])
        obs = obs.at[:, 11].set(state["feed_far"])
        obs = obs.at[:, 12].set(state["waste"])
        obs = obs.at[:, 13].set(state["affinity"])
        obs = obs.at[:, 14].set(state["last_reward"])
        obs = obs.at[:, 15].set(state["last_harvest"])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        harvested = (actions == 1).astype(jnp.float32) * 0.45 * state["biomass"]
        fed = (actions == 2).astype(jnp.float32)
        switched = actions == 3
        filtered = (actions == 4).astype(jnp.float32) * 0.55 * state["toxin"]
        valve = jnp.where(switched, ~state["valve"], state["valve"])
        connected = jnp.all(valve).astype(jnp.float32)
        nutrient_base = jnp.clip(state["nutrient"] + state["feed_near"] - 0.015, 0.0, 1.0)
        toxin_base = jnp.clip(0.94 * state["toxin"] + state["waste"] - filtered, 0.0, 1.0)
        nutrient_mixed = jnp.clip(
            nutrient_base + 0.16 * connected * (jnp.roll(nutrient_base, 1) - nutrient_base),
            0.0, 1.0,
        )
        toxin = jnp.clip(
            toxin_base + 0.12 * connected * (jnp.roll(toxin_base, 1) - toxin_base)
            + 0.08 * state["feed_near"],
            0.0, 1.0,
        )
        remaining = jnp.maximum(state["biomass"] - harvested, 0.0)
        efficiency = 0.3 + 0.7 * jnp.exp(-4.0 * jnp.square(nutrient_mixed - state["affinity"]))
        growth = (
            0.025 * (1.0 - remaining)
            + 0.18 * remaining * (1.0 - remaining) * efficiency * (1.0 - toxin)
        )
        biomass = jnp.clip(remaining + growth - 0.03 * toxin * remaining, 0.0, 1.0)
        nutrient = jnp.clip(nutrient_mixed - 0.16 * growth, 0.0, 1.0)
        waste = 0.6 * jnp.roll(filtered, 1)
        rewards = jnp.tanh(
            2.5 * harvested * (1.0 - state["toxin"])
            - 0.08 * fed
            - 0.04 * switched.astype(jnp.float32)
            - 0.07 * (actions == 4).astype(jnp.float32)
            - 0.08 * toxin
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "biomass": biomass.astype(jnp.float32),
            "nutrient": nutrient.astype(jnp.float32),
            "toxin": toxin.astype(jnp.float32),
            "valve": valve,
            "feed_near": state["feed_far"],
            "feed_far": (0.22 * fed).astype(jnp.float32),
            "waste": waste.astype(jnp.float32),
            "affinity": state["affinity"],
            "remote": jnp.roll(jnp.stack((state["biomass"], state["nutrient"], state["toxin"]), axis=1), 1, axis=0),
            "last_reward": rewards,
            "last_harvest": harvested.astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
