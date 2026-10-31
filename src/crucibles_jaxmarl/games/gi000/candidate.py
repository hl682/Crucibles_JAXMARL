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
            "energy": jax.random.uniform(key, (2,), minval=0.2, maxval=0.5, dtype=jnp.float32),
            "packet": jnp.zeros((2,), dtype=jnp.float32),
            "last": jnp.zeros((2,), dtype=jnp.int32),
            "yield": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        energy = state["energy"]
        packet = state["packet"]
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(energy)
        obs = obs.at[:, 1].set(jnp.roll(energy, 1))
        obs = obs.at[:, 2].set(packet)
        obs = obs.at[:, 3].set(jnp.roll(packet, 1))
        obs = obs.at[:, 4].set(state["t"] / 32.0)
        obs = obs.at[:, 5].set(state["yield"])
        obs = obs.at[:, 6].set(jnp.roll(state["yield"], 1))
        obs = obs.at[:, 7].set(jnp.arange(2, dtype=jnp.float32))
        obs = obs.at[:, 8:13].set(jax.nn.one_hot(state["last"], 5))
        obs = obs.at[:, 13].set(jnp.roll(state["last"], 1) / 4.0)
        obs = obs.at[:, 14].set(energy >= 0.6)
        obs = obs.at[:, 15].set(jnp.minimum(energy, 0.25))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        # 0 idle, 1 charge, 2 transmit, 3 harvest, 4 intercept.
        available = jnp.minimum(1.0, state["energy"] + jnp.roll(state["packet"], 1))
        transmitted = jnp.where(actions == 2, jnp.minimum(available, 0.25), 0.0)
        remaining = available - transmitted
        stolen = jnp.where(
            actions == 4,
            jnp.minimum(jnp.roll(remaining, 1), 0.12),
            0.0,
        )
        after_theft = remaining + stolen - jnp.roll(stolen, 1)
        harvested = jnp.where((actions == 3) & (after_theft >= 0.6), 0.6, 0.0)
        recharge = jnp.where(actions == 1, 0.16, 0.02)
        energy = jnp.clip(after_theft - harvested + recharge, 0.0, 1.0)
        rewards = jnp.clip(harvested - 0.04 * (actions == 4) - 0.02 * (actions == 1), -1.0, 1.0)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "energy": energy.astype(jnp.float32),
            "packet": transmitted.astype(jnp.float32),
            "last": actions.astype(jnp.int32),
            "yield": harvested.astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards.astype(jnp.float32), dones, info
