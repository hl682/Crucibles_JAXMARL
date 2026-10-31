import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        keys = jax.random.split(key, 2)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "volume": jnp.full((2,), 2.0, dtype=jnp.float32),
            "dirt": 2.0 * jax.random.uniform(keys[0], (2,), minval=0.15, maxval=0.8, dtype=jnp.float32),
            "capacity": jnp.ones((2,), dtype=jnp.float32),
            "pipe_volume": jnp.zeros((2,), dtype=jnp.float32),
            "pipe_dirt": jnp.zeros((2,), dtype=jnp.float32),
            "inlet_dirt": jax.random.uniform(keys[1], (2,), minval=0.2, maxval=0.8, dtype=jnp.float32),
            "throughput": jnp.zeros((2,), dtype=jnp.float32),
            "overflow": jnp.zeros((2,), dtype=jnp.float32),
            "last_reward": jnp.zeros((2,), dtype=jnp.float32),
            "last_action": jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"] / 32.0)
        obs = obs.at[:, 1].set(state["volume"] / 4.0)
        obs = obs.at[:, 2].set(jnp.roll(state["volume"], 1) / 4.0)
        obs = obs.at[:, 3].set(state["dirt"] / jnp.maximum(state["volume"], 0.001))
        obs = obs.at[:, 4].set(state["capacity"])
        obs = obs.at[:, 5].set(state["pipe_volume"])
        obs = obs.at[:, 6].set(state["pipe_dirt"] / jnp.maximum(state["pipe_volume"], 0.001))
        obs = obs.at[:, 7].set(jnp.roll(state["pipe_volume"], 1))
        obs = obs.at[:, 8].set(state["inlet_dirt"])
        obs = obs.at[:, 9].set(state["last_reward"])
        obs = obs.at[:, 10].set(state["overflow"] / 2.0)
        obs = obs.at[:, 11].set(jnp.roll(state["overflow"], 1) / 2.0)
        obs = obs.at[:, 12].set(state["throughput"] / 32.0)
        obs = obs.at[:, 13].set(jnp.roll(state["throughput"], 1) / 32.0)
        obs = obs.at[:, 14].set(state["last_action"] / 4.0)
        obs = obs.at[:, 15].set(jnp.roll(state["last_action"], 1) / 4.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        admit = (actions == 0).astype(jnp.float32)
        incoming_volume = jnp.roll(state["pipe_volume"], 1)
        incoming_dirt = jnp.roll(state["pipe_dirt"], 1)
        total_volume = state["volume"] + incoming_volume + admit
        total_dirt = state["dirt"] + incoming_dirt + admit * state["inlet_dirt"]
        fraction = jnp.clip(total_dirt / jnp.maximum(total_volume, 0.001), 0.0, 1.0)
        overflow = jnp.maximum(total_volume - 4.0, 0.0)
        volume = jnp.minimum(total_volume, 4.0)
        dirt = volume * fraction
        process = jnp.where(actions == 1, jnp.minimum(volume, 1.0), 0.0)
        forward = jnp.where(actions == 2, jnp.minimum(volume, 1.0), 0.0)
        filtered = jnp.where(actions == 3, jnp.minimum(dirt, state["capacity"]), 0.0)
        regenerate = actions == 4
        wash_volume = (regenerate & (state["capacity"] < 1.0)).astype(jnp.float32)
        wash_dirt = wash_volume * (1.0 - state["capacity"])
        next_volume = jnp.maximum(volume - process - forward, 0.0)
        next_dirt = jnp.clip(dirt - fraction * (process + forward) - filtered, 0.0, next_volume)
        next_capacity = jnp.where(regenerate, 1.0, jnp.maximum(state["capacity"] - filtered, 0.0))
        rewards = 0.8 * process * (1.0 - 2.0 * fraction) + 0.04 * forward
        rewards = rewards - 0.04 * admit - 0.03 * (actions == 3).astype(jnp.float32) - 0.05 * regenerate.astype(jnp.float32)
        rewards = jnp.clip(rewards - 0.10 * jnp.minimum(overflow, 1.0), -1.0, 1.0).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "volume": next_volume.astype(jnp.float32),
            "dirt": next_dirt.astype(jnp.float32),
            "capacity": next_capacity.astype(jnp.float32),
            "pipe_volume": (forward + wash_volume).astype(jnp.float32),
            "pipe_dirt": (forward * fraction + wash_dirt).astype(jnp.float32),
            "inlet_dirt": state["inlet_dirt"],
            "throughput": state["throughput"] + process,
            "overflow": overflow.astype(jnp.float32),
            "last_reward": rewards,
            "last_action": actions.astype(jnp.int32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
