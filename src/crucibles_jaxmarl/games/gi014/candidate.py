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
            "signature": jnp.zeros((2,), dtype=jnp.int32),
            "active": jnp.array(0, dtype=jnp.int32),
            "effort": jnp.zeros((2,), dtype=jnp.int32),
            "age": jnp.array(0, dtype=jnp.int32),
            "preference": jax.random.randint(
                keys[0], (2,), minval=1, maxval=3, dtype=jnp.int32
            ),
            "skill": jax.random.randint(
                keys[1], (2,), minval=1, maxval=3, dtype=jnp.int32
            ),
            "pending_payout": jnp.zeros((2,), dtype=jnp.float32),
            "resolution": jnp.array(0, dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        required = jnp.where(
            state["active"] > 0,
            2 + (state["active"] != state["skill"]).astype(jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
        )
        project_value = jnp.where(
            state["active"] > 0,
            jnp.where(state["preference"] == state["active"], 0.65, 0.25),
            jnp.zeros((2,), dtype=jnp.float32),
        ).astype(jnp.float32)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(1.0 - state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 2].set(state["signature"].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 3].set(
            jnp.roll(state["signature"], 1).astype(jnp.float32) / 2.0
        )
        obs = obs.at[:, 4].set(state["active"].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 5].set(state["effort"].astype(jnp.float32) / 3.0)
        obs = obs.at[:, 6].set(jnp.roll(state["effort"], 1).astype(jnp.float32) / 3.0)
        obs = obs.at[:, 7].set(state["age"].astype(jnp.float32) / 8.0)
        obs = obs.at[:, 8].set(state["preference"].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 9].set(state["skill"].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 10].set(required.astype(jnp.float32) / 3.0)
        obs = obs.at[:, 11].set(state["pending_payout"])
        obs = obs.at[:, 12].set((state["resolution"] == 1).astype(jnp.float32))
        obs = obs.at[:, 13].set((state["resolution"] == 2).astype(jnp.float32))
        obs = obs.at[:, 14].set((state["resolution"] == 3).astype(jnp.float32))
        obs = obs.at[:, 15].set(project_value)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        signing = (actions == 1) | (actions == 2)
        signatures = jnp.where(signing, actions, state["signature"])
        signatures = jnp.where(actions == 4, 0, signatures).astype(jnp.int32)
        active = state["active"] > 0
        launch = (
            jnp.logical_not(active)
            & jnp.all(signatures > 0)
            & (signatures[0] == signatures[1])
        )
        required = 2 + (state["active"] != state["skill"]).astype(jnp.int32)
        working = active & (actions == 3) & (state["effort"] < required)
        progress = state["effort"] + working.astype(jnp.int32)
        ready = active & jnp.all(progress >= required)
        withdrawn = active & jnp.any(actions == 4)
        expired = active & (state["age"] + 1 >= 8) & jnp.logical_not(ready)
        completed = ready & jnp.logical_not(withdrawn)
        ended = completed | withdrawn | expired
        project_value = jnp.where(
            state["preference"] == state["active"], 0.65, 0.25
        ).astype(jnp.float32)
        rewards = (
            state["pending_payout"]
            - 0.07 * working.astype(jnp.float32)
            - 0.01 * signing.astype(jnp.float32)
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "signature": jnp.where(
                launch, jnp.zeros((2,), dtype=jnp.int32), signatures
            ),
            "active": jnp.where(
                ended, 0, jnp.where(launch, signatures[0], state["active"])
            ).astype(jnp.int32),
            "effort": jnp.where(
                launch | ended, jnp.zeros((2,), dtype=jnp.int32), progress
            ),
            "age": jnp.where(
                launch | ended, 0, jnp.where(active, state["age"] + 1, 0)
            ).astype(jnp.int32),
            "preference": state["preference"],
            "skill": state["skill"],
            "pending_payout": jnp.where(
                completed, project_value, jnp.zeros((2,), dtype=jnp.float32)
            ),
            "resolution": jnp.where(
                withdrawn, 2, jnp.where(completed, 1, jnp.where(expired, 3, 0))
            ).astype(jnp.int32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {
            "done_all": done_all,
            "legal_mask": self.legal_actions(next_state),
        }
        return self.observe(next_state), next_state, rewards, dones, info
