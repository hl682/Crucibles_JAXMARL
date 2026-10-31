"""Persistent public claims steer access, with delayed peer credibility audits."""
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
        regime = jax.random.randint(keys[0], (), 0, 2, dtype=jnp.int32)
        accuracy = jax.random.uniform(keys[1], (2,), dtype=jnp.float32, minval=0.70, maxval=0.90)
        regime_sign = 2.0 * regime.astype(jnp.float32) - 1.0
        signal = jnp.where(jax.random.uniform(keys[2], (2,), dtype=jnp.float32) < accuracy, regime_sign, -regime_sign)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "regime": regime,
            "accuracy": accuracy,
            "signal": signal,
            "claim": jnp.zeros((2,), dtype=jnp.float32),
            "claim_valid": jnp.zeros((2,), dtype=jnp.bool_),
            "claim_correct": jnp.zeros((2,), dtype=jnp.float32),
            "claim_age": jnp.zeros((2,), dtype=jnp.int32),
            "trust": jnp.full((2,), 0.5, dtype=jnp.float32),
            "gate": jnp.array(0.0, dtype=jnp.float32),
            "commitment": jnp.zeros((2,), dtype=jnp.float32),
            "audit_pending": jnp.zeros((2,), dtype=jnp.bool_),
            "audit_correct": jnp.zeros((2,), dtype=jnp.float32),
            "audit_result": jnp.zeros((2,), dtype=jnp.float32),
            "last_maturity": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(jnp.arange(2, dtype=jnp.float32))
        obs = obs.at[:, 2].set(state["signal"])
        obs = obs.at[:, 3].set(state["accuracy"])
        obs = obs.at[:, 4].set(state["claim"])
        obs = obs.at[:, 5].set(jnp.roll(state["claim"], 1))
        obs = obs.at[:, 6].set(state["trust"])
        obs = obs.at[:, 7].set(jnp.roll(state["trust"], 1))
        obs = obs.at[:, 8].set(state["gate"])
        obs = obs.at[:, 9].set(state["claim_age"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 10].set(jnp.roll(state["claim_age"], 1).astype(jnp.float32) / 32.0)
        obs = obs.at[:, 11].set(state["commitment"])
        obs = obs.at[:, 12].set(state["audit_pending"].astype(jnp.float32))
        obs = obs.at[:, 13].set(state["audit_result"])
        obs = obs.at[:, 14].set(state["last_maturity"])
        obs = obs.at[:, 15].set(jnp.roll(state["audit_pending"], 1).astype(jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        keys = jax.random.split(key, 2)
        regime_sign = 2.0 * state["regime"].astype(jnp.float32) - 1.0
        publish = (actions == 0) | (actions == 1)
        reported_sign = jnp.where(actions == 0, -1.0, 1.0)
        audit = (actions == 2) & jnp.roll(state["claim_valid"], 1)
        invest = (actions == 3) | (actions == 4)
        commitment = jnp.where(actions == 3, -1.0, jnp.where(actions == 4, 1.0, 0.0))

        active = jnp.abs(state["commitment"]) > 0.0
        correct = state["commitment"] == regime_sign
        access = 0.35 + 0.65 * 0.5 * (1.0 + state["commitment"] * state["gate"])
        maturity = active.astype(jnp.float32) * jnp.where(correct, 0.65 * access, -0.45)
        rewards = jnp.clip(
            maturity - 0.030 * publish.astype(jnp.float32)
            - 0.040 * (actions == 2).astype(jnp.float32)
            - 0.015 * invest.astype(jnp.float32),
            -1.0, 1.0,
        ).astype(jnp.float32)

        reviewed = jnp.roll(state["audit_pending"], 1)
        historical_correctness = jnp.roll(state["audit_correct"], 1)
        trust = jnp.clip(jnp.where(reviewed, 0.80 * state["trust"] + 0.20 * historical_correctness, state["trust"]), 0.05, 0.95)
        claim = jnp.where(publish, reported_sign, state["claim"])
        claim_valid = state["claim_valid"] | publish
        claim_age = jnp.where(publish, 0, jnp.minimum(state["claim_age"] + 1, 32)).astype(jnp.int32)
        weight = trust * claim_valid.astype(jnp.float32) / (1.0 + 0.15 * claim_age.astype(jnp.float32))
        target = jnp.sum(weight * claim) / jnp.maximum(jnp.sum(weight), 0.10)
        gate = jnp.clip(0.75 * state["gate"] + 0.25 * target, -1.0, 1.0)
        flip = jax.random.bernoulli(keys[0], p=0.14)
        regime = jnp.where(flip, 1 - state["regime"], state["regime"]).astype(jnp.int32)
        next_sign = 2.0 * regime.astype(jnp.float32) - 1.0
        signal = jnp.where(jax.random.uniform(keys[1], (2,), dtype=jnp.float32) < state["accuracy"], next_sign, -next_sign)

        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "regime": regime,
            "accuracy": state["accuracy"],
            "signal": signal,
            "claim": claim,
            "claim_valid": claim_valid,
            "claim_correct": jnp.where(publish, (reported_sign == regime_sign).astype(jnp.float32), state["claim_correct"]),
            "claim_age": claim_age,
            "trust": trust,
            "gate": gate,
            "commitment": commitment,
            "audit_pending": audit,
            "audit_correct": jnp.where(audit, jnp.roll(state["claim_correct"], 1), 0.0),
            "audit_result": jnp.where(state["audit_pending"], 2.0 * state["audit_correct"] - 1.0, 0.0),
            "last_maturity": maturity,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
