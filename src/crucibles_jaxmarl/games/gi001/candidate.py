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
            "raw": jax.random.randint(key, (2,), 0, 2, dtype=jnp.int32),
            "finish": jnp.zeros((2,), dtype=jnp.int32),
            "heat": jnp.zeros((2,), dtype=jnp.float32),
            "last": jnp.zeros((2,), dtype=jnp.int32),
            "completed": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["raw"] / 3.0)
        obs = obs.at[:, 1].set(jnp.roll(state["raw"], 1) / 3.0)
        obs = obs.at[:, 2].set(state["finish"] / 3.0)
        obs = obs.at[:, 3].set(jnp.roll(state["finish"], 1) / 3.0)
        obs = obs.at[:, 4].set(state["heat"])
        obs = obs.at[:, 5].set(jnp.roll(state["heat"], 1))
        obs = obs.at[:, 6].set(state["t"] / 32.0)
        obs = obs.at[:, 7].set(jnp.arange(2, dtype=jnp.float32))
        obs = obs.at[:, 8:13].set(jax.nn.one_hot(state["last"], 5))
        obs = obs.at[:, 13].set(state["completed"])
        obs = obs.at[:, 14].set(jnp.roll(state["completed"], 1))
        obs = obs.at[:, 15].set(state["heat"] < 0.8)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        # 0 cool, 1 fabricate, 2 forward, 3 finish partner goods, 4 purge.
        operational = state["heat"] < 0.8
        made = ((actions == 1) & operational & (state["raw"] < 3)).astype(jnp.int32)
        sent = ((actions == 2) & operational & (state["raw"] > 0)
                & (jnp.roll(state["finish"], 1) < 3)).astype(jnp.int32)
        completed = ((actions == 3) & operational & (state["finish"] > 0)).astype(jnp.int32)
        purged = ((actions == 4) & (state["finish"] > 0)).astype(jnp.int32)
        raw = state["raw"] + made - sent
        finish = state["finish"] - completed - purged + jnp.roll(sent, 1)
        heat = jnp.clip(state["heat"] + 0.22 * (made + sent + completed)
                        - jnp.where(actions == 0, 0.35, 0.08), 0.0, 1.0)
        rewards = 0.2 * completed + 0.65 * jnp.roll(completed, 1) - 0.04 * made - 0.03 * purged
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "raw": raw.astype(jnp.int32),
            "finish": finish.astype(jnp.int32),
            "heat": heat.astype(jnp.float32),
            "last": actions.astype(jnp.int32),
            "completed": completed.astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards.astype(jnp.float32), dones, info
