import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        risk = jax.random.uniform(key, (2,), minval=0.2, maxval=0.8)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "capital": jnp.full((2,), 0.75, dtype=jnp.float32),
            "principal": jnp.zeros((2,), dtype=jnp.float32),
            "age": jnp.zeros((2,), dtype=jnp.int32),
            "risk": risk.astype(jnp.float32),
            "fund": jnp.array(0.15, dtype=jnp.float32),
            "membership": jnp.zeros((2,), dtype=jnp.float32),
            "coverage": jnp.zeros((2,), dtype=jnp.float32),
            "audit": jnp.zeros((2,), dtype=jnp.float32),
            "last_claim": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        observations = jnp.zeros((2, 16), dtype=jnp.float32)
        observations = observations.at[:, 0].set(state["t"] / 32.0)
        observations = observations.at[:, 1].set(state["capital"] / 1.2)
        observations = observations.at[:, 2].set(jnp.roll(state["capital"], 1) / 1.2)
        observations = observations.at[:, 3].set(state["principal"] / 0.35)
        observations = observations.at[:, 4].set(jnp.roll(state["principal"], 1) / 0.35)
        observations = observations.at[:, 5].set(state["age"] / 3.0)
        observations = observations.at[:, 6].set(jnp.roll(state["age"], 1) / 3.0)
        observations = observations.at[:, 7].set(state["risk"])
        observations = observations.at[:, 8].set(state["fund"] / 2.0)
        observations = observations.at[:, 9].set(state["membership"])
        observations = observations.at[:, 10].set(jnp.roll(state["membership"], 1))
        observations = observations.at[:, 11].set(state["coverage"])
        observations = observations.at[:, 12].set(jnp.roll(state["coverage"], 1))
        observations = observations.at[:, 13].set(state["audit"])
        observations = observations.at[:, 14].set(jnp.roll(state["audit"], 1))
        observations = observations.at[:, 15].set(state["last_claim"] / 0.35)
        return observations

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        keys = jax.random.split(key, 2)
        draws = jax.random.uniform(keys[0], (2,), dtype=jnp.float32)
        fresh_risk = jax.random.uniform(keys[1], (2,), minval=0.2, maxval=0.8)
        active = state["principal"] > 0.0
        cancel = active & (actions == 4)
        due = active & (state["age"] >= 3) & (~cancel)
        failed = due & (draws < state["risk"] * (1.0 - 0.6 * state["audit"]))
        succeeded = due & (~failed)
        claims = failed.astype(jnp.float32) * state["principal"] * state["coverage"]
        ration = jnp.minimum(1.0, state["fund"] / (jnp.sum(claims) + 0.000001))
        payments = claims * ration
        settlement = (
            succeeded.astype(jnp.float32) * 1.3 * state["principal"]
            + payments
            + cancel.astype(jnp.float32) * 0.75 * state["principal"]
        )
        contribution = jnp.where(actions == 2, jnp.minimum(state["capital"], 0.16), 0.0)
        audit_fee = jnp.where(
            (actions == 3) & (state["capital"] >= 0.03), 0.03, 0.0
        )
        launched = (actions == 1) & (~active) & (state["capital"] >= 0.05)
        new_principal = jnp.where(
            launched, jnp.minimum(state["capital"], 0.35), 0.0
        )
        closed = due | cancel
        surviving = active & (~closed)
        principal = jnp.where(surviving, state["principal"], new_principal)
        age = jnp.where(
            surviving, state["age"] + 1, jnp.where(launched, 1, 0)
        ).astype(jnp.int32)
        capital = jnp.clip(
            state["capital"] + settlement + 0.025
            - contribution - audit_fee - new_principal,
            0.0, 1.2,
        )
        fund = jnp.clip(
            state["fund"] - jnp.sum(payments) + jnp.sum(contribution), 0.0, 2.0
        )
        membership = jnp.minimum(
            1.0, 0.9 * state["membership"] + contribution / 0.16
        )
        coverage = jnp.where(
            surviving, state["coverage"], jnp.where(launched, state["membership"], 0.0)
        )
        audit = jnp.minimum(
            1.0, 0.65 * state["audit"] + jnp.roll((audit_fee > 0.0).astype(jnp.float32), 1)
        )
        realized_surplus = (
            succeeded.astype(jnp.float32) * 0.3 * state["principal"]
            - failed.astype(jnp.float32) * state["principal"] + payments
            - cancel.astype(jnp.float32) * 0.25 * state["principal"]
            - 0.1 * contribution - audit_fee
        )
        rewards = jnp.tanh(3.0 * realized_surplus).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "capital": capital.astype(jnp.float32),
            "principal": principal.astype(jnp.float32),
            "age": age,
            "risk": jnp.where(closed, fresh_risk, state["risk"]).astype(jnp.float32),
            "fund": fund.astype(jnp.float32),
            "membership": membership.astype(jnp.float32),
            "coverage": coverage.astype(jnp.float32),
            "audit": audit.astype(jnp.float32),
            "last_claim": payments.astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
