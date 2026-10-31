"""Endogenous field extraction with noisy private sensing and delayed messages."""

import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        draws = jax.random.uniform(key, (5,), dtype=jnp.float32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "field": (-0.65 + 1.30 * draws[0]).astype(jnp.float32),
            "bias": -0.25 + 0.50 * draws[1:3],
            "richness": 0.65 + 0.25 * draws[3:5],
            "signal": jnp.zeros((2,), dtype=jnp.float32),
            "signal_age": jnp.full((2,), 32, dtype=jnp.int32),
            "signal_valid": jnp.zeros((2,), dtype=jnp.bool_),
            "inbox": jnp.zeros((2,), dtype=jnp.float32),
            "inbox_age": jnp.full((2,), 32, dtype=jnp.int32),
            "inbox_valid": jnp.zeros((2,), dtype=jnp.bool_),
            "pending": jnp.zeros((2,), dtype=jnp.float32),
            "pending_age": jnp.full((2,), 32, dtype=jnp.int32),
            "pending_valid": jnp.zeros((2,), dtype=jnp.bool_),
            "interference": jnp.zeros((2,), dtype=jnp.float32),
            "last_extracted": jnp.zeros((2,), dtype=jnp.float32),
            "last_reward": jnp.zeros((2,), dtype=jnp.float32),
            "last_action": jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"] / 32.0)
        obs = obs.at[:, 1].set(state["signal"] / 1.5)
        obs = obs.at[:, 2].set(state["signal_age"] / 32.0)
        obs = obs.at[:, 3].set(state["signal_valid"].astype(jnp.float32))
        obs = obs.at[:, 4].set(state["inbox"] / 1.5)
        obs = obs.at[:, 5].set(state["inbox_age"] / 32.0)
        obs = obs.at[:, 6].set(state["inbox_valid"].astype(jnp.float32))
        obs = obs.at[:, 7].set(state["bias"] / 0.25)
        obs = obs.at[:, 8].set(state["richness"])
        obs = obs.at[:, 9].set(jnp.mean(state["richness"]))
        obs = obs.at[:, 10].set(state["interference"])
        obs = obs.at[:, 11].set(state["pending_valid"].astype(jnp.float32))
        obs = obs.at[:, 12].set(state["last_extracted"] / 0.3)
        obs = obs.at[:, 13].set(state["last_reward"])
        obs = obs.at[:, 14].set(state["last_action"] / 4.0)
        obs = obs.at[:, 15].set(jnp.roll(state["last_action"], 1, axis=0) / 4.0)
        return obs.astype(jnp.float32)

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        sample = actions == 0
        relay = actions == 1
        harvesting = (actions == 2) | (actions == 3)
        restore = actions == 4
        direction = jnp.where(
            actions == 3, 1.0, jnp.where(actions == 2, -1.0, 0.0)
        ).astype(jnp.float32)
        keys = jax.random.split(key, 2)
        sensor_noise = jax.random.uniform(
            keys[0], (2,), minval=-1.0, maxval=1.0, dtype=jnp.float32
        )
        drift = jax.random.uniform(
            keys[1], (), minval=-0.02, maxval=0.02, dtype=jnp.float32
        )
        local_field = jnp.clip(state["field"] + state["bias"], -1.0, 1.0)
        measured = jnp.clip(
            state["field"]
            + state["bias"]
            + (0.04 + 0.24 * state["interference"]) * sensor_noise,
            -1.5,
            1.5,
        )
        signal = jnp.where(sample, measured, state["signal"])
        signal_age = jnp.where(
            sample, 1, jnp.minimum(state["signal_age"] + 1, 32)
        ).astype(jnp.int32)
        delivered = jnp.roll(state["pending_valid"], 1, axis=0)
        inbox = jnp.where(
            delivered, jnp.roll(state["pending"], 1, axis=0), state["inbox"]
        )
        inbox_age = jnp.where(
            delivered,
            jnp.minimum(jnp.roll(state["pending_age"], 1, axis=0) + 1, 32),
            jnp.minimum(state["inbox_age"] + 1, 32),
        ).astype(jnp.int32)
        send = relay & state["signal_valid"]
        pending = jnp.where(send, state["signal"] - state["bias"], 0.0)
        pending_age = jnp.where(
            send, jnp.minimum(state["signal_age"] + 1, 32), 32
        ).astype(jnp.int32)
        extracted = (
            jnp.minimum(state["richness"], 0.30)
            * harvesting.astype(jnp.float32)
        )
        restoration = restore.astype(jnp.float32)
        richness = jnp.clip(
            state["richness"]
            - extracted
            + 0.025
            + 0.22 * restoration
            + 0.12 * jnp.roll(restoration, 1, axis=0),
            0.0,
            1.0,
        )
        field = jnp.clip(
            0.97 * state["field"]
            - 0.18 * jnp.sum(direction * extracted)
            - 0.04 * state["field"] * jnp.sum(restoration)
            + drift,
            -0.75,
            0.75,
        )
        interference = jnp.clip(
            0.50 * state["interference"]
            + 0.35 * jnp.roll(harvesting, 1, axis=0).astype(jnp.float32),
            0.0,
            1.0,
        )
        rewards = jnp.clip(
            0.45 * extracted * direction * local_field
            + 0.10 * extracted
            - 0.015 * sample.astype(jnp.float32)
            - 0.010 * relay.astype(jnp.float32)
            - 0.025 * restoration,
            -1.0,
            1.0,
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "field": field.astype(jnp.float32),
            "bias": state["bias"],
            "richness": richness.astype(jnp.float32),
            "signal": signal.astype(jnp.float32),
            "signal_age": signal_age,
            "signal_valid": state["signal_valid"] | sample,
            "inbox": inbox.astype(jnp.float32),
            "inbox_age": inbox_age,
            "inbox_valid": state["inbox_valid"] | delivered,
            "pending": pending.astype(jnp.float32),
            "pending_age": pending_age,
            "pending_valid": send,
            "interference": interference.astype(jnp.float32),
            "last_extracted": extracted.astype(jnp.float32),
            "last_reward": rewards,
            "last_action": actions.astype(jnp.int32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
