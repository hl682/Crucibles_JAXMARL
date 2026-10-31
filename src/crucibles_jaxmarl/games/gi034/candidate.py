"""A capacity-one machine with persistent two-party setup latches."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        keys = jax.random.split(key, 4)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "kind": jax.random.randint(keys[0], (2,), 0, 2, dtype=jnp.int32),
            "deadline": jax.random.randint(keys[1], (2,), 7, 11, dtype=jnp.int32),
            "value": jax.random.uniform(keys[2], (2,), dtype=jnp.float32, minval=0.60, maxval=1.0),
            "progress": jnp.zeros((2,), dtype=jnp.int32),
            "ticket": jnp.zeros((2,), dtype=jnp.int32),
            "latch": jnp.zeros((2,), dtype=jnp.int32),
            "mode": jnp.array(0, dtype=jnp.int32),
            "priority": jax.random.randint(keys[3], (), 0, 2, dtype=jnp.int32),
            "cooldown": jnp.array(0, dtype=jnp.int32),
            "completed": jnp.zeros((2,), dtype=jnp.int32),
            "last_outcome": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        roles = jnp.arange(2, dtype=jnp.int32)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(roles.astype(jnp.float32))
        obs = obs.at[:, 2].set(state["kind"].astype(jnp.float32))
        obs = obs.at[:, 3].set(state["deadline"].astype(jnp.float32) / 10.0)
        obs = obs.at[:, 4].set(state["progress"].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 5].set(state["value"])
        obs = obs.at[:, 6].set(state["ticket"].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 7].set(jnp.roll(state["ticket"], 1).astype(jnp.float32) / 4.0)
        obs = obs.at[:, 8].set(state["latch"].astype(jnp.float32))
        obs = obs.at[:, 9].set(jnp.roll(state["latch"], 1).astype(jnp.float32))
        obs = obs.at[:, 10].set(state["mode"].astype(jnp.float32))
        obs = obs.at[:, 11].set((roles == state["priority"]).astype(jnp.float32))
        obs = obs.at[:, 12].set(state["cooldown"].astype(jnp.float32))
        obs = obs.at[:, 13].set(state["last_outcome"])
        obs = obs.at[:, 14].set(jnp.roll(state["completed"], 1).astype(jnp.float32) / 32.0)
        obs = obs.at[:, 15].set(state["completed"].astype(jnp.float32) / 32.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        keys = jax.random.split(key, 3)
        roles = jnp.arange(2, dtype=jnp.int32)
        latch = jnp.where(actions == 2, 0, jnp.where(actions == 3, 1, state["latch"])).astype(jnp.int32)
        mode = jnp.where(latch[0] == latch[1], latch[0], state["mode"]).astype(jnp.int32)
        changed_mode = mode != state["mode"]
        eligible = (
            (actions == 4) & (state["ticket"] > 0) & (state["kind"] == mode)
            & (state["cooldown"] == 0) & ~changed_mode
        )
        winner = jnp.where(
            eligible[0] & eligible[1], state["priority"],
            jnp.where(eligible[0], 0, jnp.where(eligible[1], 1, -1)),
        ).astype(jnp.int32)
        served = roles == winner
        progress = state["progress"] + served.astype(jnp.int32)
        complete = progress >= 2
        expired = (state["deadline"] <= 1) & ~complete
        replace = complete | expired
        next_kind = jax.random.randint(keys[0], (2,), 0, 2, dtype=jnp.int32)
        next_deadline = jax.random.randint(keys[1], (2,), 7, 11, dtype=jnp.int32)
        next_value = jax.random.uniform(keys[2], (2,), dtype=jnp.float32, minval=0.60, maxval=1.0)

        outcome = 0.70 * state["value"] * complete.astype(jnp.float32) - 0.30 * state["value"] * expired.astype(jnp.float32)
        rewards = jnp.clip(
            outcome - 0.025 * served.astype(jnp.float32)
            - 0.015 * (actions == 1).astype(jnp.float32)
            - 0.025 * ((actions == 2) | (actions == 3)).astype(jnp.float32),
            -1.0, 1.0,
        ).astype(jnp.float32)
        renewed_ticket = jnp.where(actions == 1, 4, jnp.maximum(state["ticket"] - 1, 0)).astype(jnp.int32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "kind": jnp.where(replace, next_kind, state["kind"]),
            "deadline": jnp.where(replace, next_deadline, jnp.maximum(state["deadline"] - 1, 0)).astype(jnp.int32),
            "value": jnp.where(replace, next_value, state["value"]),
            "progress": jnp.where(replace, 0, progress).astype(jnp.int32),
            "ticket": jnp.where(replace, 0, renewed_ticket).astype(jnp.int32),
            "latch": latch,
            "mode": mode,
            "priority": jnp.where(winner >= 0, 1 - winner, state["priority"]).astype(jnp.int32),
            "cooldown": jnp.where(changed_mode, 1, jnp.maximum(state["cooldown"] - 1, 0)).astype(jnp.int32),
            "completed": state["completed"] + complete.astype(jnp.int32),
            "last_outcome": outcome,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
