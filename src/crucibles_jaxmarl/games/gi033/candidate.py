"""Two reservoirs with delayed pollutant transport and persistent soil debt."""
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
            "phase": jax.random.randint(keys[0], (), 0, 8, dtype=jnp.int32),
            "water": jax.random.uniform(keys[1], (2,), dtype=jnp.float32, minval=0.55, maxval=0.85),
            "pollutant": jax.random.uniform(keys[2], (2,), dtype=jnp.float32, minval=0.08, maxval=0.25),
            "plume": jnp.zeros((2,), dtype=jnp.float32),
            "soil": jnp.zeros((2,), dtype=jnp.float32),
            "gate": jnp.full((2,), 0.10, dtype=jnp.float32),
            "survey": jnp.zeros((2,), dtype=jnp.float32),
            "survey_age": jnp.full((2,), 32, dtype=jnp.int32),
            "last_harvest": jnp.zeros((2,), dtype=jnp.float32),
            "last_arrival": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        phase = jnp.mod(state["t"] + state["phase"], 8).astype(jnp.float32)
        upstream = (jnp.arange(2, dtype=jnp.int32) == (phase >= 4.0).astype(jnp.int32)).astype(jnp.float32)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["water"])
        obs = obs.at[:, 2].set(state["pollutant"] / 1.5)
        obs = obs.at[:, 3].set(state["soil"])
        obs = obs.at[:, 4].set(state["gate"])
        obs = obs.at[:, 5].set(jnp.roll(state["gate"], 1))
        obs = obs.at[:, 6].set(jnp.roll(state["water"], 1))
        obs = obs.at[:, 7].set(upstream)
        obs = obs.at[:, 8].set(jnp.sin(phase * jnp.pi / 4.0))
        obs = obs.at[:, 9].set(jnp.cos(phase * jnp.pi / 4.0))
        obs = obs.at[:, 10].set(state["survey"] / 1.5)
        obs = obs.at[:, 11].set(state["survey_age"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 12].set(state["last_harvest"] / 0.22)
        obs = obs.at[:, 13].set(jnp.roll(state["last_harvest"], 1) / 0.22)
        obs = obs.at[:, 14].set(state["last_arrival"] / 1.5)
        obs = obs.at[:, 15].set(jnp.arange(2, dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        phase = jnp.mod(state["t"] + state["phase"], 8)
        upstream = (jnp.arange(2, dtype=jnp.int32) == (phase >= 4).astype(jnp.int32)).astype(jnp.float32)
        monitor = actions == 0
        extract = actions == 1
        flush = actions == 2
        filter_action = actions == 3
        build_gate = actions == 4

        arrival = state["plume"] * (1.0 - state["gate"])
        load = jnp.clip(state["pollutant"] + arrival, 0.0, 1.5)
        concentration = load / (state["water"] + 0.25)
        harvested = jnp.minimum(state["water"], 0.22) * extract.astype(jnp.float32)
        flush_water = jnp.minimum(jnp.maximum(state["water"] - harvested, 0.0), 0.10) * flush.astype(jnp.float32)
        flushed = jnp.minimum(load, 0.38) * flush.astype(jnp.float32) * (flush_water / 0.10)
        filtered = jnp.minimum(jnp.maximum(load - flushed, 0.0), 0.24) * filter_action.astype(jnp.float32)

        water_after_use = jnp.maximum(state["water"] - harvested - flush_water, 0.0)
        flow = 0.10 * water_after_use * upstream * (1.0 - jnp.roll(state["gate"], 1))
        residual = jnp.maximum(load - flushed - filtered, 0.0)
        carried = jnp.minimum(residual, flow * concentration * 0.40)
        rain = jax.random.uniform(key, (2,), dtype=jnp.float32, minval=0.04, maxval=0.10)
        recharge = rain * (1.0 - 0.65 * state["soil"]) * (1.0 - 0.55 * state["gate"])
        water = jnp.clip(water_after_use - flow + jnp.roll(flow, 1) + recharge, 0.0, 1.0)
        pollutant = jnp.clip(0.96 * residual - carried + 0.25 * harvested, 0.0, 1.5)
        plume = jnp.clip(jnp.roll(flushed * (0.55 + 0.35 * upstream) + carried, 1), 0.0, 1.5)
        soil = jnp.clip(
            0.97 * state["soil"] + 0.045 * jnp.maximum(concentration - 0.50, 0.0)
            + 0.060 * state["gate"] * state["plume"] - 0.10 * filtered,
            0.0, 1.0,
        )
        gate = jnp.where(build_gate, 0.85, jnp.maximum(state["gate"] - 0.08, 0.0))
        utility = (
            2.0 * harvested * (0.70 - 0.80 * concentration) * (1.0 - 0.50 * state["soil"])
            - 0.035 * filter_action.astype(jnp.float32)
            - 0.025 * flush.astype(jnp.float32)
            - 0.030 * build_gate.astype(jnp.float32)
            - 0.030 * state["soil"]
        )
        rewards = jnp.tanh(utility).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "phase": state["phase"],
            "water": water,
            "pollutant": pollutant,
            "plume": plume,
            "soil": soil,
            "gate": gate,
            "survey": jnp.where(monitor, jnp.roll(state["pollutant"], 1), state["survey"]),
            "survey_age": jnp.where(monitor, 0, jnp.minimum(state["survey_age"] + 1, 32)).astype(jnp.int32),
            "last_harvest": harvested,
            "last_arrival": arrival,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
