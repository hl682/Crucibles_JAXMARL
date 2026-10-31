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
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "position": jax.random.randint(keys[0], (2,), 0, 8, dtype=jnp.int32),
            "energy": jnp.full((2,), 2.0, dtype=jnp.float32),
            "nutrient": jax.random.uniform(keys[1], (2, 8), minval=0.1, maxval=0.4, dtype=jnp.float32),
            "sprout": jnp.zeros((2, 8), dtype=jnp.float32),
            "ripe": jnp.zeros((8,), dtype=jnp.float32),
            "soil": jax.random.uniform(keys[2], (8,), minval=0.5, maxval=1.0, dtype=jnp.float32),
            "taste": jax.random.uniform(keys[3], (2, 8), minval=0.4, maxval=1.0, dtype=jnp.float32),
            "last_harvest": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        roles = jnp.arange(2, dtype=jnp.int32)
        other = jnp.array([1, 0], dtype=jnp.int32)
        position = state["position"]
        left = jnp.mod(position - 1, 8)
        right = jnp.mod(position + 1, 8)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(position.astype(jnp.float32) / 7.0)
        obs = obs.at[:, 2].set(position[other].astype(jnp.float32) / 7.0)
        obs = obs.at[:, 3].set(state["energy"] / 3.0)
        obs = obs.at[:, 4].set(state["nutrient"][roles, position] / 2.0)
        obs = obs.at[:, 5].set(state["nutrient"][other, position] / 2.0)
        obs = obs.at[:, 6].set(jnp.sum(state["sprout"], axis=0)[position] / 6.0)
        obs = obs.at[:, 7].set(state["ripe"][position] / 3.0)
        obs = obs.at[:, 8].set(state["soil"][position])
        obs = obs.at[:, 9].set(state["taste"][roles, position])
        obs = obs.at[:, 10].set(state["ripe"][left] / 3.0)
        obs = obs.at[:, 11].set(state["ripe"][right] / 3.0)
        obs = obs.at[:, 12].set(state["soil"][left])
        obs = obs.at[:, 13].set(state["soil"][right])
        obs = obs.at[:, 14].set(state["last_harvest"])
        obs = obs.at[:, 15].set(roles.astype(jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        roles = jnp.arange(2, dtype=jnp.int32)
        location = jax.nn.one_hot(state["position"], 8, dtype=jnp.float32)
        plant = ((actions == 3) & (state["energy"] >= 0.5)).astype(jnp.float32)
        deposit = 0.7 * location * plant[:, None]
        enriched = jnp.clip(state["nutrient"] + deposit, 0.0, 2.0)
        growth = 0.35 * jnp.minimum(enriched[0], enriched[1]) * state["soil"]
        requests = location * (actions == 4).astype(jnp.float32)[:, None]
        total_requests = jnp.sum(requests, axis=0)
        removed = jnp.minimum(state["ripe"], 0.8 * total_requests)
        share = removed / jnp.maximum(total_requests, 1.0)
        harvest = jnp.sum(requests * share[None, :], axis=1)
        taste = state["taste"][roles, state["position"]]
        first_stage = 0.55 * state["sprout"][0] + growth
        second_stage = 0.6 * state["sprout"][1] + 0.45 * state["sprout"][0]
        movement = (actions == 2).astype(jnp.int32) - (actions == 1).astype(jnp.int32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "position": jnp.mod(state["position"] + movement, 8).astype(jnp.int32),
            "energy": jnp.clip(state["energy"] - 0.5 * plant + 0.04 + 0.2 * (actions == 0).astype(jnp.float32), 0.0, 3.0),
            "nutrient": jnp.clip(0.98 * (enriched - growth[None, :]), 0.0, 2.0),
            "sprout": jnp.clip(jnp.stack((first_stage, second_stage)), 0.0, 3.0),
            "ripe": jnp.clip(0.96 * (state["ripe"] - removed + 0.4 * state["sprout"][1]), 0.0, 3.0),
            "soil": jnp.clip(state["soil"] + 0.035 - 0.15 * removed - 0.025 * jnp.sum(deposit, axis=0), 0.1, 1.0),
            "taste": state["taste"],
            "last_harvest": harvest,
        }
        utility = 1.1 * taste * harvest - 0.025 * plant - 0.01 * jnp.abs(movement).astype(jnp.float32)
        rewards = jnp.tanh(utility).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
