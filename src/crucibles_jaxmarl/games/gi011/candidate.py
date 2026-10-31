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
        truth = jax.random.randint(keys[0], (), 0, 2, dtype=jnp.int32)
        accuracy = jax.random.uniform(keys[1], (2,), minval=0.65, maxval=0.9, dtype=jnp.float32)
        correct = jax.random.uniform(keys[2], (2,), dtype=jnp.float32) < accuracy
        signal = jnp.where(correct, truth, 1 - truth).astype(jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "truth": truth,
            "accuracy": accuracy,
            "signal": signal,
            "signal_age": jnp.zeros((2,), dtype=jnp.int32),
            "claim": jnp.full((2,), -1, dtype=jnp.int32),
            "weight": jnp.full((2,), 0.5, dtype=jnp.float32),
            "pending_age": jnp.array(0, dtype=jnp.int32),
            "pending_choice": jnp.array(0, dtype=jnp.int32),
            "pending_truth": truth,
            "pending_claim": jnp.full((2,), -1, dtype=jnp.int32),
            "pending_active": jnp.zeros((2,), dtype=jnp.bool_),
            "last_truth": jnp.array(-1, dtype=jnp.int32),
            "last_success": jnp.array(0.0, dtype=jnp.float32),
            "project_count": jnp.array(0, dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        active = state["claim"] >= 0
        signed = 2.0 * state["claim"].astype(jnp.float32) - 1.0
        public_score = jnp.sum(state["weight"] * signed * active.astype(jnp.float32))
        visible_choice = jnp.where(state["pending_age"] > 0, state["pending_choice"], -1)
        visible_locked_claim = jnp.where(state["pending_age"] > 0, state["pending_claim"], -1)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["signal"].astype(jnp.float32))
        obs = obs.at[:, 2].set(state["signal_age"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 3].set(state["accuracy"])
        obs = obs.at[:, 4].set((state["claim"].astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 5].set((jnp.roll(state["claim"], 1).astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 6].set(state["weight"])
        obs = obs.at[:, 7].set(jnp.roll(state["weight"], 1))
        obs = obs.at[:, 8].set(public_score / 2.0)
        obs = obs.at[:, 9].set(state["pending_age"].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 10].set((visible_choice.astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 11].set((state["last_truth"].astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 12].set(state["last_success"])
        obs = obs.at[:, 13].set(state["project_count"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 14].set((visible_locked_claim.astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 15].set(jnp.arange(2, dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        keys = jax.random.split(key, 2)
        inspected = actions == 1
        published = (actions == 2) | (actions == 3)
        completed = state["pending_age"] == 1
        correct_choice = state["pending_choice"] == state["pending_truth"]
        settlement = jnp.where(completed, jnp.where(correct_choice, 1.0, -1.0), 0.0)
        credit_correct = state["pending_claim"] == state["pending_truth"]
        adjustment = jnp.where(credit_correct, 0.12, -0.16)
        weight = jnp.clip(
            state["weight"]
            + (completed & state["pending_active"]).astype(jnp.float32) * adjustment,
            0.25,
            1.0,
        )
        active = state["claim"] >= 0
        signed = 2.0 * state["claim"].astype(jnp.float32) - 1.0
        public_score = jnp.sum(state["weight"] * signed * active.astype(jnp.float32))
        launched = (state["pending_age"] == 0) & jnp.any(actions == 4)
        choice = (public_score > 0.0).astype(jnp.int32)
        pending_age = jnp.where(launched, 2, jnp.maximum(state["pending_age"] - 1, 0)).astype(jnp.int32)
        pending_choice = jnp.where(launched, choice, state["pending_choice"]).astype(jnp.int32)
        pending_truth = jnp.where(launched, state["truth"], state["pending_truth"]).astype(jnp.int32)
        pending_claim = jnp.where(launched, state["claim"], state["pending_claim"]).astype(jnp.int32)
        pending_active = jnp.where(launched, active, state["pending_active"]).astype(jnp.bool_)
        changed = completed & jax.random.bernoulli(keys[0], p=0.25)
        truth = jnp.where(changed, 1 - state["truth"], state["truth"]).astype(jnp.int32)
        sample_correct = jax.random.uniform(keys[1], (2,), dtype=jnp.float32) < state["accuracy"]
        sample = jnp.where(sample_correct, truth, 1 - truth).astype(jnp.int32)
        signal = jnp.where(inspected, sample, state["signal"]).astype(jnp.int32)
        signal_age = jnp.where(
            inspected,
            0,
            jnp.minimum(state["signal_age"] + 1, 32),
        ).astype(jnp.int32)
        claim = jnp.where(published, actions - 2, state["claim"]).astype(jnp.int32)
        rewards = (
            jnp.full((2,), 0.75 * settlement, dtype=jnp.float32)
            - 0.03 * inspected.astype(jnp.float32)
            - 0.01 * published.astype(jnp.float32)
            - 0.02 * (launched & (actions == 4)).astype(jnp.float32)
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "truth": truth,
            "accuracy": state["accuracy"],
            "signal": signal,
            "signal_age": signal_age,
            "claim": claim,
            "weight": weight.astype(jnp.float32),
            "pending_age": pending_age,
            "pending_choice": pending_choice,
            "pending_truth": pending_truth,
            "pending_claim": pending_claim,
            "pending_active": pending_active,
            "last_truth": jnp.where(completed, state["pending_truth"], state["last_truth"]).astype(jnp.int32),
            "last_success": jnp.where(completed, settlement, state["last_success"]).astype(jnp.float32),
            "project_count": state["project_count"] + launched.astype(jnp.int32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info