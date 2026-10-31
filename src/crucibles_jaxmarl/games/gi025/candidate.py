"""Coupled reaction chambers with a shared coolant pool and delayed exhaust."""

import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        draws = jax.random.uniform(key, (4, 2), dtype=jnp.float32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "charge": 0.25 + 0.20 * draws[0],
            "product": 0.05 + 0.10 * draws[1],
            "temperature": 0.25 + 0.30 * draws[2],
            "catalyst": 0.10 + 0.20 * draws[3],
            "coolant": jnp.array(0.80, dtype=jnp.float32),
            "plume": jnp.zeros((2,), dtype=jnp.float32),
            "last_reaction": jnp.zeros((2,), dtype=jnp.float32),
            "last_shock": jnp.zeros((2,), dtype=jnp.bool_),
            "last_reward": jnp.zeros((2,), dtype=jnp.float32),
            "last_action": jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"] / 32.0)
        obs = obs.at[:, 1].set(state["charge"])
        obs = obs.at[:, 2].set(state["product"])
        obs = obs.at[:, 3].set(state["temperature"] / 1.2)
        obs = obs.at[:, 4].set(jnp.roll(state["temperature"], 1, axis=0) / 1.2)
        obs = obs.at[:, 5].set(state["catalyst"])
        obs = obs.at[:, 6].set(state["coolant"])
        obs = obs.at[:, 7].set(state["plume"] / 0.084)
        obs = obs.at[:, 8].set(jnp.roll(state["plume"], 1, axis=0) / 0.084)
        obs = obs.at[:, 9].set(state["last_reaction"] / 0.25)
        obs = obs.at[:, 10].set(jnp.roll(state["last_reaction"], 1, axis=0) / 0.25)
        obs = obs.at[:, 11].set(state["last_shock"].astype(jnp.float32))
        obs = obs.at[:, 12].set(jnp.roll(state["last_shock"], 1, axis=0).astype(jnp.float32))
        obs = obs.at[:, 13].set(state["last_action"] / 4.0)
        obs = obs.at[:, 14].set(jnp.roll(state["last_action"], 1, axis=0) / 4.0)
        obs = obs.at[:, 15].set(state["last_reward"])
        return obs.astype(jnp.float32)

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        feed = actions == 1
        catalyse = actions == 2
        cool = actions == 3
        harvest_action = actions == 4
        thermal_window = jnp.clip(
            1.0 - 1.7 * jnp.abs(state["temperature"] - 0.5), 0.0, 1.0
        )
        reaction = (
            jnp.minimum(state["charge"], 0.05 + 0.20 * state["catalyst"])
            * thermal_window
        )
        coolant_request = 0.24 * cool.astype(jnp.float32)
        request_total = jnp.sum(coolant_request)
        allocation = jnp.minimum(
            1.0, state["coolant"] / jnp.maximum(request_total, 0.001)
        )
        coolant_used = coolant_request * allocation
        harvest = jnp.where(
            harvest_action, jnp.minimum(state["product"], 0.6), 0.0
        )
        predicted_temperature = (
            state["temperature"]
            + 0.38 * reaction
            + 0.08 * feed.astype(jnp.float32)
            + 0.10
            * (jnp.roll(state["temperature"], 1, axis=0) - state["temperature"])
            - 0.75 * coolant_used
            + jnp.roll(state["plume"], 1, axis=0)
            - 0.035
        )
        shock = predicted_temperature > 0.98
        temperature = jnp.where(
            shock, 0.58, jnp.clip(predicted_temperature, 0.05, 1.2)
        )
        charge = jnp.clip(
            state["charge"] - reaction + 0.24 * feed.astype(jnp.float32),
            0.0,
            1.0,
        )
        product = jnp.clip(state["product"] - harvest + reaction, 0.0, 1.0)
        catalyst = jnp.clip(
            0.96 * state["catalyst"] + 0.30 * catalyse.astype(jnp.float32),
            0.0,
            1.0,
        )
        charge = jnp.where(shock, 0.35 * charge, charge)
        product = jnp.where(shock, 0.20 * product, product)
        catalyst = jnp.where(shock, 0.50 * catalyst, catalyst)
        coolant = jnp.clip(
            state["coolant"] - jnp.sum(coolant_used) + 0.08, 0.0, 1.0
        )
        costs = (
            0.030 * feed.astype(jnp.float32)
            + 0.020 * catalyse.astype(jnp.float32)
            + 0.010 * cool.astype(jnp.float32)
        )
        rewards = jnp.clip(
            0.65 * harvest
            - costs
            - 0.30 * shock.astype(jnp.float32)
            - 0.06 * jnp.roll(shock, 1, axis=0).astype(jnp.float32),
            -1.0,
            1.0,
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "charge": charge.astype(jnp.float32),
            "product": product.astype(jnp.float32),
            "temperature": temperature.astype(jnp.float32),
            "catalyst": catalyst.astype(jnp.float32),
            "coolant": coolant.astype(jnp.float32),
            "plume": (0.35 * coolant_used).astype(jnp.float32),
            "last_reaction": reaction.astype(jnp.float32),
            "last_shock": shock,
            "last_reward": rewards,
            "last_action": actions.astype(jnp.int32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
