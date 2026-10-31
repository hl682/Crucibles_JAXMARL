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
            "cash": jnp.full((2,), 3.0, dtype=jnp.float32),
            "active": jnp.zeros((2,), dtype=jnp.bool_),
            "backed": jnp.zeros((2,), dtype=jnp.bool_),
            "age": jnp.zeros((2,), dtype=jnp.int32),
            "term": jax.random.randint(keys[0], (2,), 3, 6, dtype=jnp.int32),
            "yield": jax.random.uniform(keys[1], (2,), minval=0.5, maxval=1.25, dtype=jnp.float32),
            "successes": jnp.zeros((2,), dtype=jnp.int32),
            "defaults": jnp.zeros((2,), dtype=jnp.int32),
            "last_action": jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"] / 32.0)
        obs = obs.at[:, 1].set(state["cash"] / 8.0)
        obs = obs.at[:, 2].set(jnp.roll(state["cash"], 1) / 8.0)
        obs = obs.at[:, 3].set(state["active"].astype(jnp.float32))
        obs = obs.at[:, 4].set(jnp.roll(state["active"], 1).astype(jnp.float32))
        obs = obs.at[:, 5].set(state["age"] / 8.0)
        obs = obs.at[:, 6].set(jnp.roll(state["age"], 1) / 8.0)
        obs = obs.at[:, 7].set(state["backed"].astype(jnp.float32))
        obs = obs.at[:, 8].set(jnp.roll(state["backed"], 1).astype(jnp.float32))
        obs = obs.at[:, 9].set(state["term"] / 5.0)
        obs = obs.at[:, 10].set(state["yield"] / 1.25)
        obs = obs.at[:, 11].set(state["successes"] / 8.0)
        obs = obs.at[:, 12].set(jnp.roll(state["successes"], 1) / 8.0)
        obs = obs.at[:, 13].set(state["defaults"] / 8.0)
        obs = obs.at[:, 14].set(jnp.roll(state["defaults"], 1) / 8.0)
        obs = obs.at[:, 15].set(state["last_action"] / 4.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        active = state["active"]
        issue = (actions == 1) & (~active) & (state["cash"] >= 2.0)
        guarantee = (actions == 2) & jnp.roll(active, 1) & (~jnp.roll(state["backed"], 1)) & (state["cash"] >= 1.0)
        expedite = (actions == 4) & active & (state["cash"] >= 0.25)
        backed = state["backed"] | jnp.roll(guarantee, 1)
        age = state["age"] + active.astype(jnp.int32) + expedite.astype(jnp.int32)
        mature = active & backed & (age >= state["term"])
        default = active & (~backed) & (age >= state["term"] + 2)
        cancel = active & (actions == 3) & (~mature) & (~default)
        close = mature | default | cancel
        issuer_refund = (2.0 + state["yield"]) * mature.astype(jnp.float32) + default.astype(jnp.float32) + cancel.astype(jnp.float32)
        backer_refund = jnp.roll(backed.astype(jnp.float32) * (close.astype(jnp.float32) + 0.25 * mature.astype(jnp.float32)), 1)
        cash = state["cash"] - 2.0 * issue.astype(jnp.float32) - guarantee.astype(jnp.float32) - 0.25 * expedite.astype(jnp.float32)
        cash = jnp.clip(cash + issuer_refund + backer_refund + 0.2 * (actions == 0).astype(jnp.float32), 0.0, 8.0)
        rewards = 0.65 * mature.astype(jnp.float32) + 0.20 * jnp.roll(mature, 1).astype(jnp.float32)
        rewards = rewards - 0.10 * issue.astype(jnp.float32) - 0.04 * guarantee.astype(jnp.float32) - 0.035 * expedite.astype(jnp.float32)
        rewards = jnp.clip(rewards - 0.25 * default.astype(jnp.float32) - 0.12 * cancel.astype(jnp.float32), -1.0, 1.0).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "cash": cash.astype(jnp.float32),
            "active": (active & (~close)) | issue,
            "backed": backed & (~close),
            "age": jnp.where(active & (~close), age, jnp.zeros((2,), dtype=jnp.int32)),
            "term": state["term"],
            "yield": state["yield"],
            "successes": state["successes"] + mature.astype(jnp.int32),
            "defaults": state["defaults"] + default.astype(jnp.int32),
            "last_action": actions.astype(jnp.int32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
