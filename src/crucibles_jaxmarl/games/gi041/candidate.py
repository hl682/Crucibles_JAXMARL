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
            "reservoir": jax.random.uniform(keys[0], (2,), minval=2.5, maxval=3.5, dtype=jnp.float32),
            "sediment": jax.random.uniform(keys[1], (2,), minval=0.05, maxval=0.2, dtype=jnp.float32),
            "cistern": jnp.full((2,), 0.8, dtype=jnp.float32),
            "transit_water": jnp.zeros((2,), dtype=jnp.float32),
            "transit_silt": jnp.zeros((2,), dtype=jnp.float32),
            "rain": jax.random.uniform(keys[2], (2,), minval=0.12, maxval=0.2, dtype=jnp.float32),
            "demand": jax.random.uniform(keys[3], (2,), minval=0.65, maxval=1.1, dtype=jnp.float32),
            "last_utility": jnp.zeros((2,), dtype=jnp.float32),
            "last_out": jnp.zeros((2,), dtype=jnp.float32),
            "last_loss": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["reservoir"] / 6.0)
        obs = obs.at[:, 1].set(state["sediment"] / 2.5)
        obs = obs.at[:, 2].set(state["cistern"] / 2.0)
        obs = obs.at[:, 3].set(state["transit_water"] / 8.0)
        obs = obs.at[:, 4].set(state["transit_silt"] / 2.5)
        obs = obs.at[:, 5].set(state["rain"] / 0.25)
        obs = obs.at[:, 6].set(state["demand"] / 1.1)
        obs = obs.at[:, 7].set(state["last_utility"])
        obs = obs.at[:, 8].set(state["last_out"] / 8.0)
        obs = obs.at[:, 9].set(state["last_loss"] / 8.0)
        obs = obs.at[:, 10].set(jnp.roll(state["last_out"], 1) / 8.0)
        obs = obs.at[:, 11].set(jnp.roll(state["last_utility"], 1))
        obs = obs.at[:, 12].set(jnp.sum(state["reservoir"] + state["cistern"]) / 16.0)
        obs = obs.at[:, 13].set(jnp.mod(state["t"], 8) / 8.0)
        obs = obs.at[:, 14].set(state["t"] / 32.0)
        obs = obs.at[:, 15].set(jnp.array([-1.0, 1.0], dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        keys = jax.random.split(key, 2)
        available = state["reservoir"] + state["transit_water"] + state["rain"]
        deposit = jnp.clip(
            state["sediment"] + state["transit_silt"] + 0.015 * state["reservoir"],
            0.0,
            2.5,
        )
        banking = actions == 0
        pumping = actions == 1
        flushing = actions == 2
        dredging = actions == 3
        releasing = actions == 4
        banked = jnp.where(
            banking,
            jnp.minimum(0.7, jnp.minimum(available, 2.0 - state["cistern"])),
            0.0,
        )
        working_water = available - banked
        drawn = jnp.where(pumping, jnp.minimum(1.15, working_water + state["cistern"]), 0.0)
        from_reservoir = jnp.minimum(drawn, working_water)
        from_cistern = drawn - from_reservoir
        remaining = working_water - from_reservoir
        cistern = state["cistern"] + banked - from_cistern
        pulse = jnp.where(flushing, jnp.minimum(1.35, remaining), 0.0)
        eroded = jnp.minimum(deposit, 0.35 * pulse)
        removed = jnp.where(dredging, jnp.minimum(0.5, deposit), 0.0)
        sediment = jnp.clip(deposit - eroded - removed, 0.0, 2.5)
        remaining = remaining - pulse
        released = jnp.where(releasing, jnp.minimum(1.0, cistern), 0.0)
        cistern = cistern - released
        base_flow = jnp.minimum(remaining, jnp.where(banking, 0.04, 0.16))
        capacity = 6.0 - 1.1 * sediment
        overflow = jnp.maximum(remaining - base_flow - capacity, 0.0)
        reservoir = jnp.clip(remaining - base_flow - overflow, 0.0, capacity)
        outflow = pulse + base_flow + overflow + released
        useful_draw = drawn / (1.0 + 0.7 * deposit)
        utility = jnp.clip(useful_draw / state["demand"], 0.0, 1.0)
        rewards = jnp.clip(
            0.7 * utility
            - 0.05 * banking.astype(jnp.float32)
            - 0.07 * flushing.astype(jnp.float32)
            - 0.11 * dredging.astype(jnp.float32)
            - 0.025 * releasing.astype(jnp.float32)
            - 0.1 * sediment / 2.5
            - 0.08 * jnp.clip(overflow / 2.0, 0.0, 1.0),
            -1.0,
            1.0,
        ).astype(jnp.float32)
        next_t = state["t"] + jnp.array(1, dtype=jnp.int32)
        seasonal_rain = 0.06 + 0.12 * (0.5 + 0.5 * jnp.sin(2.0 * jnp.pi * next_t / 8.0))
        next_state = {
            "t": next_t,
            "reservoir": reservoir.astype(jnp.float32),
            "sediment": sediment.astype(jnp.float32),
            "cistern": jnp.clip(cistern, 0.0, 2.0).astype(jnp.float32),
            "transit_water": (0.88 * jnp.roll(outflow, 1)).astype(jnp.float32),
            "transit_silt": jnp.roll(0.8 * eroded + 0.02 * overflow, 1).astype(jnp.float32),
            "rain": (
                seasonal_rain
                + jax.random.uniform(keys[0], (2,), minval=0.0, maxval=0.07, dtype=jnp.float32)
            ).astype(jnp.float32),
            "demand": jax.random.uniform(keys[1], (2,), minval=0.65, maxval=1.1, dtype=jnp.float32),
            "last_utility": utility.astype(jnp.float32),
            "last_out": outflow.astype(jnp.float32),
            "last_loss": overflow.astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info