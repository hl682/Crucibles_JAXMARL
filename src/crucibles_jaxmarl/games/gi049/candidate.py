import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "target": jax.random.randint(key, (2,), 0, 2, dtype=jnp.int32),
            "stage": jnp.full((2,), -1, dtype=jnp.int32),
            "locked": jnp.zeros((2,), dtype=jnp.bool_),
            "public": jnp.full((2,), -1, dtype=jnp.int32),
            "pending": jnp.full((2,), -1, dtype=jnp.int32),
            "piece_age": jnp.zeros((2,), dtype=jnp.int32),
            "message_age": jnp.zeros((2,), dtype=jnp.int32),
            "rounds": jnp.array(0, dtype=jnp.int32),
            "last_result": jnp.zeros((2,), dtype=jnp.float32),
            "last_success": jnp.array(0.0, dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        observations = jnp.zeros((2, 16), dtype=jnp.float32)
        observations = observations.at[:, 0].set(state["t"] / 32.0)
        observations = observations.at[:, 1].set(state["target"])
        observations = observations.at[:, 2].set((state["stage"] + 1) / 2.0)
        observations = observations.at[:, 3].set((jnp.roll(state["stage"], 1) + 1) / 2.0)
        observations = observations.at[:, 4].set(state["locked"])
        observations = observations.at[:, 5].set(jnp.roll(state["locked"], 1))
        observations = observations.at[:, 6].set((state["public"] + 1) / 2.0)
        observations = observations.at[:, 7].set((jnp.roll(state["public"], 1) + 1) / 2.0)
        observations = observations.at[:, 8].set(state["pending"] >= 0)
        observations = observations.at[:, 9].set(state["piece_age"] / 32.0)
        observations = observations.at[:, 10].set(jnp.roll(state["piece_age"], 1) / 32.0)
        observations = observations.at[:, 11].set(state["message_age"] / 32.0)
        observations = observations.at[:, 12].set(jnp.roll(state["message_age"], 1) / 32.0)
        observations = observations.at[:, 13].set(state["rounds"] / 16.0)
        observations = observations.at[:, 14].set(state["last_result"])
        observations = observations.at[:, 15].set(state["last_success"])
        return observations

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        fresh_target = jax.random.randint(key, (2,), 0, 2, dtype=jnp.int32)
        write = (~state["locked"]) & ((actions == 1) | (actions == 2))
        stage = jnp.where(write, actions - 1, state["stage"]).astype(jnp.int32)
        new_lock = (actions == 4) & (stage >= 0) & (~state["locked"])
        locked = state["locked"] | new_lock
        piece_age = jnp.where(
            stage >= 0,
            jnp.where(write, 0, jnp.minimum(32, state["piece_age"] + 1)),
            0,
        ).astype(jnp.int32)
        delivered = state["pending"] >= 0
        public = jnp.where(delivered, state["pending"], state["public"])
        message_age = jnp.where(
            public >= 0,
            jnp.where(delivered, 0, jnp.minimum(32, state["message_age"] + 1)),
            0,
        ).astype(jnp.int32)
        pending = jnp.where(actions == 3, state["target"], -1).astype(jnp.int32)
        completed = jnp.all(locked)
        correct = jnp.roll(stage, 1) == state["target"]
        partner_quality = jnp.maximum(0.25, 1.0 - 0.05 * jnp.roll(piece_age, 1))
        successful = jnp.all(correct)
        payout = (
            jnp.where(correct, 0.85, -0.55) * partner_quality
            + 0.1 * successful.astype(jnp.float32)
        )
        rewards = jnp.clip(
            completed.astype(jnp.float32) * payout
            - 0.025 * write.astype(jnp.float32)
            - 0.01 * (actions == 3).astype(jnp.float32)
            - 0.01 * new_lock.astype(jnp.float32),
            -1.0, 1.0,
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "target": jnp.where(completed, fresh_target, state["target"]).astype(jnp.int32),
            "stage": jnp.where(completed, -1, stage).astype(jnp.int32),
            "locked": jnp.where(completed, jnp.zeros((2,), dtype=jnp.bool_), locked),
            "public": jnp.where(completed, -1, public).astype(jnp.int32),
            "pending": jnp.where(completed, -1, pending).astype(jnp.int32),
            "piece_age": jnp.where(completed, 0, piece_age).astype(jnp.int32),
            "message_age": jnp.where(completed, 0, message_age).astype(jnp.int32),
            "rounds": state["rounds"] + completed.astype(jnp.int32),
            "last_result": jnp.where(
                completed, jnp.where(correct, 1.0, -1.0), state["last_result"]
            ).astype(jnp.float32),
            "last_success": jnp.where(
                completed, successful.astype(jnp.float32), state["last_success"]
            ).astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
