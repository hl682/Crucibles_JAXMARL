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
            "content": jax.random.randint(keys[0], (2,), 0, 2, dtype=jnp.int32),
            "version": jnp.zeros((2,), dtype=jnp.int32),
            "demand": jax.random.randint(keys[1], (2,), 0, 2, dtype=jnp.int32),
            "battery": jax.random.uniform(keys[2], (2,), minval=0.8, maxval=1.2, dtype=jnp.float32),
            "lock": jnp.zeros((2,), dtype=jnp.int32),
            "cache_content": jnp.zeros((2,), dtype=jnp.int32),
            "cache_version": jnp.full((2,), -1, dtype=jnp.int32),
            "cache_demand": jnp.zeros((2,), dtype=jnp.int32),
            "cache_available": jnp.zeros((2,), dtype=jnp.bool_),
            "cache_lease": jnp.zeros((2,), dtype=jnp.int32),
            "cache_age": jnp.full((2,), 32, dtype=jnp.int32),
            "pipe_content": jnp.zeros((2,), dtype=jnp.int32),
            "pipe_version": jnp.full((2,), -1, dtype=jnp.int32),
            "pipe_demand": jnp.zeros((2,), dtype=jnp.int32),
            "pipe_delay": jnp.zeros((2,), dtype=jnp.int32),
            "pipe_lease": jnp.zeros((2,), dtype=jnp.int32),
            "last_read": jnp.zeros((2,), dtype=jnp.float32),
            "last_peer_use": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["content"])
        obs = obs.at[:, 1].set(state["version"] / 32.0)
        obs = obs.at[:, 2].set(state["demand"])
        obs = obs.at[:, 3].set(state["battery"] / 2.0)
        obs = obs.at[:, 4].set(state["lock"] / 3.0)
        obs = obs.at[:, 5].set(state["cache_content"])
        obs = obs.at[:, 6].set(state["cache_version"] / 32.0)
        obs = obs.at[:, 7].set(state["cache_demand"])
        obs = obs.at[:, 8].set(state["cache_available"].astype(jnp.float32))
        obs = obs.at[:, 9].set(state["cache_lease"] / 2.0)
        obs = obs.at[:, 10].set(state["cache_age"] / 32.0)
        obs = obs.at[:, 11].set(state["pipe_delay"] / 2.0)
        obs = obs.at[:, 12].set(state["last_read"])
        obs = obs.at[:, 13].set(state["last_peer_use"])
        obs = obs.at[:, 14].set(state["t"] / 32.0)
        obs = obs.at[:, 15].set(jnp.array([-1.0, 1.0], dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        channel_free = state["pipe_delay"] == 0
        mask = jnp.zeros((2, 5), dtype=jnp.bool_)
        mask = mask.at[:, 0].set(True)
        mask = mask.at[:, 1].set(channel_free & (state["battery"] >= 0.1))
        mask = mask.at[:, 2].set(state["cache_available"] & (state["battery"] >= 0.08))
        mask = mask.at[:, 3].set((state["lock"] == 0) & (state["battery"] >= 0.15))
        mask = mask.at[:, 4].set(channel_free & (state["battery"] >= 0.3))
        return mask

    def raw_step(self, key, state, actions):
        channel_free = state["pipe_delay"] == 0
        ordinary = (actions == 1) & channel_free & (state["battery"] >= 0.1)
        reading = (actions == 2) & state["cache_available"] & (state["battery"] >= 0.08)
        writing = (actions == 3) & (state["lock"] == 0) & (state["battery"] >= 0.15)
        fencing = (actions == 4) & channel_free & (state["battery"] >= 0.3)
        sending = ordinary | fencing
        content = jnp.where(writing, 1 - state["content"], state["content"]).astype(jnp.int32)
        version = state["version"] + writing.astype(jnp.int32)
        success = (
            reading
            & (state["cache_version"] >= 0)
            & (state["cache_version"] == jnp.roll(version, 1))
            & (state["cache_content"] == state["demand"])
        )
        failure = reading & (~success)
        delivered = jnp.roll(state["pipe_delay"] == 1, 1)
        new_demands = jax.random.randint(key, (2,), 0, 2, dtype=jnp.int32)
        demand = jnp.where(success, new_demands, state["demand"]).astype(jnp.int32)
        peer_use = jnp.roll(success, 1).astype(jnp.float32)
        rewards = jnp.clip(
            0.55 * success.astype(jnp.float32)
            - 0.25 * failure.astype(jnp.float32)
            + 0.2 * peer_use
            - 0.02 * ordinary.astype(jnp.float32)
            - 0.03 * writing.astype(jnp.float32)
            - 0.06 * fencing.astype(jnp.float32),
            -1.0,
            1.0,
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "content": content,
            "version": version.astype(jnp.int32),
            "demand": demand,
            "battery": jnp.clip(
                state["battery"]
                + 0.28 * (actions == 0).astype(jnp.float32)
                - 0.1 * ordinary.astype(jnp.float32)
                - 0.08 * reading.astype(jnp.float32)
                - 0.15 * writing.astype(jnp.float32)
                - 0.3 * fencing.astype(jnp.float32),
                0.0,
                2.0,
            ).astype(jnp.float32),
            "lock": jnp.where(fencing, 3, jnp.maximum(state["lock"] - 1, 0)).astype(jnp.int32),
            "cache_content": jnp.where(delivered, jnp.roll(state["pipe_content"], 1), state["cache_content"]).astype(jnp.int32),
            "cache_version": jnp.where(delivered, jnp.roll(state["pipe_version"], 1), state["cache_version"]).astype(jnp.int32),
            "cache_demand": jnp.where(delivered, jnp.roll(state["pipe_demand"], 1), state["cache_demand"]).astype(jnp.int32),
            "cache_available": jnp.where(delivered, True, state["cache_available"] & (~reading)),
            "cache_lease": jnp.where(
                delivered, jnp.roll(state["pipe_lease"], 1), jnp.maximum(state["cache_lease"] - 1, 0)
            ).astype(jnp.int32),
            "cache_age": jnp.where(delivered, 0, jnp.minimum(state["cache_age"] + 1, 32)).astype(jnp.int32),
            "pipe_content": jnp.where(sending, state["content"], state["pipe_content"]).astype(jnp.int32),
            "pipe_version": jnp.where(sending, state["version"], state["pipe_version"]).astype(jnp.int32),
            "pipe_demand": jnp.where(sending, state["demand"], state["pipe_demand"]).astype(jnp.int32),
            "pipe_delay": jnp.where(
                sending, jnp.where(fencing, 1, 2), jnp.maximum(state["pipe_delay"] - 1, 0)
            ).astype(jnp.int32),
            "pipe_lease": jnp.where(sending, jnp.where(fencing, 2, 0), state["pipe_lease"]).astype(jnp.int32),
            "last_read": (success.astype(jnp.float32) - failure.astype(jnp.float32)),
            "last_peer_use": peer_use,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info