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
        quality = jax.random.uniform(keys[0], (2,), minval=0.45, maxval=0.85, dtype=jnp.float32)
        signal_noise = jax.random.uniform(keys[1], (2,), minval=-0.18, maxval=0.18, dtype=jnp.float32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "cash": jax.random.uniform(keys[2], (2,), minval=0.45, maxval=0.75, dtype=jnp.float32),
            "quality": quality,
            "signal": jnp.clip(quality + signal_noise, 0.0, 1.0),
            "pending": jnp.zeros((2,), dtype=jnp.float32),
            "timer": jnp.zeros((2,), dtype=jnp.int32),
            "bond": jnp.zeros((2,), dtype=jnp.float32),
            "receivable": jnp.zeros((2,), dtype=jnp.float32),
            "last_claim": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["cash"])
        obs = obs.at[:, 1].set(jnp.roll(state["cash"], 1))
        obs = obs.at[:, 2].set(state["pending"] / 0.25)
        obs = obs.at[:, 3].set(jnp.roll(state["pending"], 1) / 0.25)
        obs = obs.at[:, 4].set(state["timer"].astype(jnp.float32) / 3.0)
        obs = obs.at[:, 5].set(jnp.roll(state["timer"], 1).astype(jnp.float32) / 3.0)
        obs = obs.at[:, 6].set(state["bond"] / 0.18)
        obs = obs.at[:, 7].set(jnp.roll(state["bond"], 1) / 0.18)
        obs = obs.at[:, 8].set(state["receivable"])
        obs = obs.at[:, 9].set(jnp.roll(state["receivable"], 1))
        obs = obs.at[:, 10].set(state["quality"])
        obs = obs.at[:, 11].set(jnp.roll(state["signal"], 1))
        obs = obs.at[:, 12].set(state["signal"])
        obs = obs.at[:, 13].set(state["last_claim"] / 0.45)
        obs = obs.at[:, 14].set(jnp.roll(state["last_claim"], 1) / 0.45)
        obs = obs.at[:, 15].set(state["t"].astype(jnp.float32) / 32.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        keys = jax.random.split(key, 2)
        draws = jax.random.uniform(keys[0], (2,), dtype=jnp.float32)
        noise = jax.random.uniform(keys[1], (2,), minval=-0.18, maxval=0.18, dtype=jnp.float32)
        mature = (state["timer"] == 1) & (state["pending"] > 0.0)
        success = mature & (draws < state["quality"])
        other_timer = jnp.roll(state["timer"], 1)
        recall = (actions == 4) & (state["bond"] > 0.0) & (other_timer > 1)
        effective_bond = jnp.where(recall, 0.0, state["bond"])
        insured = (jnp.roll(effective_bond, 1) > 0.0).astype(jnp.float32)
        credit = mature.astype(jnp.float32) * (
            success.astype(jnp.float32) * (0.40 - 0.05 * insured)
            + (1.0 - success.astype(jnp.float32)) * 0.18 * insured
        )
        insurer_return = jnp.roll(success, 1).astype(jnp.float32) * (
            effective_bond + 0.05 * (effective_bond > 0.0).astype(jnp.float32)
        )
        available_cash = state["cash"] + recall.astype(jnp.float32) * state["bond"] + insurer_return
        old_bond_released = jnp.roll(mature, 1) | recall
        remaining_bond = jnp.where(old_bond_released, 0.0, state["bond"])
        underwrite = (
            (actions == 2) & (other_timer > 1)
            & (jnp.roll(state["pending"], 1) > 0.0)
            & (remaining_bond == 0.0) & (available_cash >= 0.18)
        )
        launch = (actions == 1) & (state["pending"] == 0.0) & (available_cash >= 0.25)
        available_credit = state["receivable"] + credit
        claim = jnp.where(actions == 3, jnp.minimum(0.45, available_credit), 0.0)
        new_cash = jnp.clip(
            available_cash - 0.18 * underwrite.astype(jnp.float32)
            - 0.25 * launch.astype(jnp.float32) + 0.75 * claim + 0.015, 0.0, 1.0
        )
        new_receivable = jnp.clip(0.97 * (available_credit - claim), 0.0, 1.0)
        new_bond = jnp.where(underwrite, 0.18, remaining_bond)
        new_pending = jnp.where(launch, 0.25, jnp.where(mature, 0.0, state["pending"]))
        new_timer = jnp.where(launch, 3, jnp.maximum(state["timer"] - 1, 0)).astype(jnp.int32)
        new_signal = jnp.where(launch, jnp.clip(state["quality"] + noise, 0.0, 1.0), state["signal"])
        rewards = jnp.tanh(
            1.6 * claim - 0.02 * launch.astype(jnp.float32)
            - 0.025 * underwrite.astype(jnp.float32) - 0.03 * recall.astype(jnp.float32)
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "cash": new_cash.astype(jnp.float32),
            "quality": state["quality"],
            "signal": new_signal.astype(jnp.float32),
            "pending": new_pending.astype(jnp.float32),
            "timer": new_timer,
            "bond": new_bond.astype(jnp.float32),
            "receivable": new_receivable.astype(jnp.float32),
            "last_claim": claim.astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
