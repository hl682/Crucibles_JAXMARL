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
            "secret": jax.random.randint(keys[0], (2,), 0, 2, dtype=jnp.int32),
            "report": jnp.full((2,), -1, dtype=jnp.int32),
            "version": jnp.zeros((2,), dtype=jnp.int32),
            "age": jnp.zeros((2,), dtype=jnp.int32),
            "trust": jax.random.uniform(keys[1], (2,), minval=0.35, maxval=0.65, dtype=jnp.float32),
            "evidence": jnp.full((2,), -1, dtype=jnp.int32),
            "evidence_version": jnp.full((2,), -1, dtype=jnp.int32),
            "pending_audit": jnp.zeros((2,), dtype=jnp.bool_),
            "snapshot_truth": jnp.full((2,), -1, dtype=jnp.int32),
            "snapshot_report": jnp.full((2,), -1, dtype=jnp.int32),
            "snapshot_version": jnp.full((2,), -1, dtype=jnp.int32),
            "last_result": jnp.zeros((2,), dtype=jnp.float32),
            "last_fee": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        evidence_valid = (state["evidence"] >= 0) & (state["evidence_version"] == jnp.roll(state["version"], 1))
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["secret"].astype(jnp.float32))
        obs = obs.at[:, 1].set((state["report"].astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 2].set((jnp.roll(state["report"], 1).astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 3].set(state["trust"])
        obs = obs.at[:, 4].set(jnp.roll(state["trust"], 1))
        obs = obs.at[:, 5].set(state["age"].astype(jnp.float32) / 8.0)
        obs = obs.at[:, 6].set(jnp.roll(state["age"], 1).astype(jnp.float32) / 8.0)
        obs = obs.at[:, 7].set((state["evidence"].astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 8].set(evidence_valid.astype(jnp.float32))
        obs = obs.at[:, 9].set(state["pending_audit"].astype(jnp.float32))
        obs = obs.at[:, 10].set(jnp.roll(state["pending_audit"], 1).astype(jnp.float32))
        obs = obs.at[:, 11].set(state["version"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 12].set(jnp.roll(state["version"], 1).astype(jnp.float32) / 32.0)
        obs = obs.at[:, 13].set(state["last_result"])
        obs = obs.at[:, 14].set(state["last_fee"] / 0.24)
        obs = obs.at[:, 15].set(state["t"].astype(jnp.float32) / 32.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        other_report = jnp.roll(state["report"], 1)
        other_secret = jnp.roll(state["secret"], 1)
        other_version = jnp.roll(state["version"], 1)
        reporting = (actions == 1) | (actions == 2)
        audit = (actions == 3) & (other_report >= 0)
        trade = (actions == 4) & (other_report >= 0)
        evidence_valid = (state["evidence"] >= 0) & (state["evidence_version"] == other_version)
        prediction = jnp.where(evidence_valid, state["evidence"], other_report)
        correct = prediction == other_secret
        stake = 0.2 + 0.6 * jnp.roll(state["trust"], 1)
        trading_return = trade.astype(jnp.float32) * jnp.where(correct, 0.55, -0.65) * stake
        sender_fee = jnp.roll(trade.astype(jnp.float32) * jnp.where(correct, 0.06, 0.30) * stake, 1)
        truthful_snapshot = state["snapshot_truth"] == state["snapshot_report"]
        audit_adjustment = state["pending_audit"].astype(jnp.float32) * jnp.where(truthful_snapshot, 0.14, -0.22)
        next_trust = jnp.clip(0.92 * state["trust"] + 0.04 + jnp.roll(audit_adjustment, 1), 0.0, 1.0)
        next_evidence = jnp.where(state["pending_audit"], state["snapshot_truth"], state["evidence"])
        next_evidence_version = jnp.where(state["pending_audit"], state["snapshot_version"], state["evidence_version"])
        refreshed = jnp.roll(trade, 1)
        replacement_secrets = jax.random.randint(key, (2,), 0, 2, dtype=jnp.int32)
        next_secret = jnp.where(refreshed, replacement_secrets, state["secret"])
        next_version = state["version"] + refreshed.astype(jnp.int32)
        next_report = jnp.where(refreshed, -1, jnp.where(reporting, actions - 1, state["report"])).astype(jnp.int32)
        next_age = jnp.where(refreshed | reporting, 0, jnp.minimum(state["age"] + 1, 8)).astype(jnp.int32)
        next_snapshot_truth = jnp.where(audit, other_secret, state["snapshot_truth"])
        next_snapshot_report = jnp.where(audit, other_report, state["snapshot_report"])
        next_snapshot_version = jnp.where(audit, other_version, state["snapshot_version"])
        last_result = jnp.where(trade, jnp.where(correct, 1.0, -1.0), 0.0)
        rewards = jnp.tanh(
            trading_return + sender_fee - 0.035 * audit.astype(jnp.float32)
            - 0.012 * reporting.astype(jnp.float32)
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "secret": next_secret.astype(jnp.int32),
            "report": next_report,
            "version": next_version.astype(jnp.int32),
            "age": next_age,
            "trust": next_trust.astype(jnp.float32),
            "evidence": next_evidence.astype(jnp.int32),
            "evidence_version": next_evidence_version.astype(jnp.int32),
            "pending_audit": audit.astype(jnp.bool_),
            "snapshot_truth": next_snapshot_truth.astype(jnp.int32),
            "snapshot_report": next_snapshot_report.astype(jnp.int32),
            "snapshot_version": next_snapshot_version.astype(jnp.int32),
            "last_result": last_result.astype(jnp.float32),
            "last_fee": sender_fee.astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
