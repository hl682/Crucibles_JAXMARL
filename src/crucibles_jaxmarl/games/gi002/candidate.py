import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "water": jax.random.uniform(key, (), minval=0.5, maxval=0.9, dtype=jnp.float32),
            "pollution": jnp.array(0.0, dtype=jnp.float32),
            "runoff": jnp.array(0.0, dtype=jnp.float32),
            "roots": jnp.zeros((2,), dtype=jnp.float32),
            "last": jnp.zeros((2,), dtype=jnp.int32),
            "intake": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["water"])
        obs = obs.at[:, 1].set(state["pollution"])
        obs = obs.at[:, 2].set(state["runoff"])
        obs = obs.at[:, 3].set(state["roots"])
        obs = obs.at[:, 4].set(jnp.roll(state["roots"], 1))
        obs = obs.at[:, 5].set(state["intake"])
        obs = obs.at[:, 6].set(jnp.roll(state["intake"], 1))
        obs = obs.at[:, 7].set(state["t"] / 32.0)
        obs = obs.at[:, 8:13].set(jax.nn.one_hot(state["last"], 5))
        obs = obs.at[:, 13].set(jnp.roll(state["last"], 1) / 4.0)
        obs = obs.at[:, 14].set(jnp.arange(2, dtype=jnp.float32))
        obs = obs.at[:, 15].set(1.0 - state["pollution"])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        # 0 idle, 1 gentle pump, 2 intensive pump, 3 plant, 4 filter.
        demand = jnp.where(actions == 1, 0.12, jnp.where(actions == 2, 0.32, 0.0))
        intake = demand * jnp.minimum(1.0, state["water"] / jnp.maximum(jnp.sum(demand), 0.01))
        useful = intake * (1.0 - state["pollution"])
        filter_count = jnp.sum((actions == 4).astype(jnp.float32))
        pollution = jnp.clip(0.85 * state["pollution"] + state["runoff"]
                             - 0.16 * filter_count - 0.04 * jnp.sum(state["roots"]), 0.0, 1.0)
        roots = jnp.clip(0.9 * state["roots"] + 0.3 * (actions == 3)
                         - 0.15 * (actions == 2), 0.0, 1.0)
        water = jnp.clip(state["water"] - jnp.sum(intake) + 0.08
                         + 0.08 * jnp.sum(state["roots"]) * (1.0 - state["pollution"]), 0.0, 1.0)
        runoff = 0.24 * jnp.sum((actions == 2).astype(jnp.float32))
        rewards = useful - 0.07 * (actions == 3) - 0.05 * (actions == 4)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "water": water.astype(jnp.float32),
            "pollution": pollution.astype(jnp.float32),
            "runoff": runoff.astype(jnp.float32),
            "roots": roots.astype(jnp.float32),
            "last": actions.astype(jnp.int32),
            "intake": intake.astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards.astype(jnp.float32), dones, info
