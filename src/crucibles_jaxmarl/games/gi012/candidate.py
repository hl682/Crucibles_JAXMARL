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
            "stock": jnp.ones((2,), dtype=jnp.int32),
            "money": jnp.full((2,), 2, dtype=jnp.int32),
            "escrow": jnp.zeros((2,), dtype=jnp.int32),
            "far": jnp.zeros((2,), dtype=jnp.int32),
            "near": jnp.zeros((2,), dtype=jnp.int32),
            "taste": jax.random.uniform(keys[0], (2,), dtype=jnp.float32),
            "make_cost": 0.08 + 0.04 * jax.random.uniform(
                keys[1], (2,), dtype=jnp.float32
            ),
            "received": jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["stock"].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 2].set(state["money"].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 3].set(state["escrow"].astype(jnp.float32))
        obs = obs.at[:, 4].set(jnp.roll(state["escrow"], 1).astype(jnp.float32))
        obs = obs.at[:, 5].set(state["far"].astype(jnp.float32))
        obs = obs.at[:, 6].set(state["near"].astype(jnp.float32))
        obs = obs.at[:, 7].set(jnp.roll(state["far"], 1).astype(jnp.float32))
        obs = obs.at[:, 8].set(jnp.roll(state["near"], 1).astype(jnp.float32))
        obs = obs.at[:, 9].set(state["taste"])
        obs = obs.at[:, 10].set(state["make_cost"] / 0.12)
        obs = obs.at[:, 11].set(1.0 - state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 12].set((state["stock"] < 4).astype(jnp.float32))
        obs = obs.at[:, 13].set(
            ((state["money"] > 0) & (state["escrow"] == 0)).astype(jnp.float32)
        )
        obs = obs.at[:, 14].set(
            ((state["stock"] > 0) & (jnp.roll(state["escrow"], 1) > 0)).astype(
                jnp.float32
            )
        )
        obs = obs.at[:, 15].set(state["received"].astype(jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        made = ((actions == 1) & (state["stock"] < 4)).astype(jnp.int32)
        posted = (
            (actions == 2) & (state["money"] > 0) & (state["escrow"] == 0)
        ).astype(jnp.int32)
        cancelled = ((actions == 4) & (state["escrow"] > 0)).astype(jnp.int32)
        offered = state["escrow"] + posted - cancelled
        shipped = (
            (actions == 3) & (state["stock"] > 0) & (jnp.roll(offered, 1) > 0)
        ).astype(jnp.int32)
        received = jnp.roll(state["near"], 1)
        rewards = (
            received.astype(jnp.float32) * (0.6 + 0.2 * state["taste"])
            - made.astype(jnp.float32) * state["make_cost"]
            - 0.015 * posted.astype(jnp.float32)
            - 0.01 * cancelled.astype(jnp.float32)
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "stock": state["stock"] + made - shipped,
            "money": state["money"] - posted + cancelled + state["near"],
            "escrow": offered - jnp.roll(shipped, 1),
            "far": shipped,
            "near": state["far"],
            "taste": state["taste"],
            "make_cost": state["make_cost"],
            "received": received,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {
            "done_all": done_all,
            "legal_mask": self.legal_actions(next_state),
        }
        return self.observe(next_state), next_state, rewards, dones, info
