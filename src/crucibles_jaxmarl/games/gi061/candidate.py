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
            "temperature": jax.random.uniform(keys[0], (2,), minval=0.15, maxval=0.35, dtype=jnp.float32),
            "wear": jnp.zeros((2,), dtype=jnp.float32),
            "charge": jax.random.uniform(keys[1], (2,), minval=0.7, maxval=0.95, dtype=jnp.float32),
            "locked": jnp.zeros((2,), dtype=jnp.bool_),
            "transport": jnp.zeros((2,), dtype=jnp.float32),
            "coolant": jnp.array(0.4, dtype=jnp.float32),
            "clog": jnp.array(0.0, dtype=jnp.float32),
            "last_output": jnp.zeros((2,), dtype=jnp.float32),
            "ambient": jax.random.uniform(keys[2], (), minval=0.025, maxval=0.05, dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        temp = state["temperature"] / 1.2
        wear = state["wear"]
        locked = state["locked"].astype(jnp.float32)
        charge = state["charge"]
        transport = state["transport"]
        output = state["last_output"]
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(temp)
        obs = obs.at[:, 2].set(temp[::-1])
        obs = obs.at[:, 3].set(wear)
        obs = obs.at[:, 4].set(wear[::-1])
        obs = obs.at[:, 5].set(locked)
        obs = obs.at[:, 6].set(locked[::-1])
        obs = obs.at[:, 7].set(charge)
        obs = obs.at[:, 8].set(charge[::-1])
        obs = obs.at[:, 9].set(transport)
        obs = obs.at[:, 10].set(transport[::-1])
        obs = obs.at[:, 11].set(state["coolant"])
        obs = obs.at[:, 12].set(output)
        obs = obs.at[:, 13].set(output[::-1])
        obs = obs.at[:, 14].set(state["ambient"] / 0.05)
        obs = obs.at[:, 15].set(state["clog"])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        idle = (actions == 0).astype(jnp.float32)
        service = (actions == 3).astype(jnp.float32)
        replenish = (actions == 4).astype(jnp.float32)
        requested = jnp.where(actions == 1, 0.4, jnp.where(actions == 2, 0.9, 0.0))
        spent = jnp.minimum(state["charge"], requested) * jnp.logical_not(state["locked"]).astype(jnp.float32)
        output = spent * (1.0 - 0.65 * state["wear"])
        cooling = 0.14 * state["coolant"] * (1.0 - 0.6 * state["clog"])
        temperature = jnp.clip(
            0.91 * state["temperature"] + 0.32 * output
            + 0.22 * state["transport"][::-1] + state["ambient"]
            - cooling - 0.08 * idle,
            0.0, 1.2,
        )
        wear = jnp.clip(
            state["wear"] + 0.08 * jnp.maximum(temperature - 0.6, 0.0) - 0.16 * service,
            0.0, 1.0,
        )
        locked = (temperature >= 0.9) | (state["locked"] & (temperature > 0.35))
        coolant = jnp.clip(state["coolant"] + 0.2 * jnp.sum(replenish) - 0.08 * jnp.sum(output) - 0.02, 0.0, 1.0)
        clog = jnp.clip(0.97 * state["clog"] + 0.04 * jnp.sum(output) - 0.12 * jnp.sum(replenish), 0.0, 1.0)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "temperature": temperature.astype(jnp.float32),
            "wear": wear.astype(jnp.float32),
            "charge": jnp.clip(state["charge"] - spent + 0.16 * idle + 0.05 * service, 0.0, 1.0).astype(jnp.float32),
            "locked": locked,
            "transport": output.astype(jnp.float32),
            "coolant": coolant.astype(jnp.float32),
            "clog": clog.astype(jnp.float32),
            "last_output": output.astype(jnp.float32),
            "ambient": state["ambient"],
        }
        rewards = jnp.clip(
            0.65 * output - 0.1 * spent - 0.07 * service - 0.1 * replenish
            - 0.08 * locked.astype(jnp.float32),
            -1.0, 1.0,
        ).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
