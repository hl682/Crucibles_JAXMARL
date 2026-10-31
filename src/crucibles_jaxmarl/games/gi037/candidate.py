import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        balance_key, value_key = jax.random.split(key, 2)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "balance": jax.random.uniform(balance_key, (2,), minval=1.0, maxval=1.4, dtype=jnp.float32),
            "locked": jnp.zeros((2,), dtype=jnp.float32),
            "approved": jnp.zeros((2,), dtype=jnp.bool_),
            "age": jnp.zeros((2,), dtype=jnp.int32),
            "trust": jnp.full((2,), 0.6, dtype=jnp.float32),
            "value": jax.random.uniform(value_key, (2,), minval=-0.8, maxval=1.0, dtype=jnp.float32),
            "last_action": jnp.zeros((2,), dtype=jnp.int32),
            "last_settled": jnp.array(0.0, dtype=jnp.float32),
            "last_cancel": jnp.zeros((2,), dtype=jnp.bool_),
            "settlements": jnp.array(0.0, dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        other = jnp.array([1, 0], dtype=jnp.int32)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["balance"] / 2.0)
        obs = obs.at[:, 2].set(state["locked"] / 0.8)
        obs = obs.at[:, 3].set(state["approved"].astype(jnp.float32))
        obs = obs.at[:, 4].set(state["age"].astype(jnp.float32) / 8.0)
        obs = obs.at[:, 5].set(state["trust"])
        obs = obs.at[:, 6].set((state["value"] + 0.8) / 1.8)
        obs = obs.at[:, 7].set(state["locked"][other] / 0.8)
        obs = obs.at[:, 8].set(state["approved"][other].astype(jnp.float32))
        obs = obs.at[:, 9].set(state["age"][other].astype(jnp.float32) / 8.0)
        obs = obs.at[:, 10].set(state["trust"][other])
        obs = obs.at[:, 11].set(state["last_action"].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 12].set(state["last_action"][other].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 13].set(state["last_settled"])
        obs = obs.at[:, 14].set(state["last_cancel"][other].astype(jnp.float32))
        obs = obs.at[:, 15].set(state["settlements"] / 8.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        other = jnp.array([1, 0], dtype=jnp.int32)
        required = 0.4 + 0.4 * (1.0 - state["trust"][other])
        deposit = (actions == 1) & (state["locked"] == 0.0) & (state["balance"] >= required)
        amount = required * deposit.astype(jnp.float32)
        cancel = (actions == 3) & (state["locked"] > 0.0)
        expired = (state["age"] >= 8) & (state["locked"] > 0.0)
        both_locked = jnp.all(state["locked"] > 0.0)
        new_endorse = (actions == 2) & both_locked & ~state["approved"]
        settle = both_locked & jnp.all(state["approved"]) & jnp.all(state["age"] >= 2) & jnp.any(actions == 4) & ~jnp.any(cancel | expired)
        release = cancel | expired | settle
        refund = jnp.where(settle, state["locked"], jnp.where(cancel, 0.75 * state["locked"], jnp.where(expired, 0.9 * state["locked"], 0.0)))
        next_locked = jnp.where(release, 0.0, state["locked"] + amount)
        next_age = jnp.where(release | deposit, 0, jnp.where(state["locked"] > 0.0, jnp.minimum(state["age"] + 1, 8), 0)).astype(jnp.int32)
        next_approved = jnp.where(jnp.any(release), jnp.zeros((2,), dtype=jnp.bool_), state["approved"] | new_endorse)
        next_balance = jnp.clip(state["balance"] - amount + refund + 0.12 * (actions == 0).astype(jnp.float32) + 0.08 * settle.astype(jnp.float32), 0.0, 2.0)
        next_trust = jnp.clip(state["trust"] + 0.08 * settle.astype(jnp.float32) - 0.18 * cancel.astype(jnp.float32), 0.0, 1.0)
        sampled_value = jax.random.uniform(key, (2,), minval=-0.8, maxval=1.0, dtype=jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "balance": next_balance,
            "locked": next_locked.astype(jnp.float32),
            "approved": next_approved,
            "age": next_age,
            "trust": next_trust,
            "value": jnp.where(settle, sampled_value, state["value"]),
            "last_action": actions.astype(jnp.int32),
            "last_settled": settle.astype(jnp.float32),
            "last_cancel": cancel,
            "settlements": state["settlements"] + settle.astype(jnp.float32),
        }
        rewards = jnp.clip(settle.astype(jnp.float32) * (0.25 + 0.6 * state["value"]) - 0.12 * cancel.astype(jnp.float32) - 0.06 * expired.astype(jnp.float32) - 0.02 * deposit.astype(jnp.float32) - 0.01 * new_endorse.astype(jnp.float32), -1.0, 1.0).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info