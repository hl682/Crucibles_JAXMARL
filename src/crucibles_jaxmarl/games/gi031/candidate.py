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
        target = jax.random.randint(keys[0], (), 0, 2, dtype=jnp.int32)
        reliability = jax.random.uniform(keys[1], (2,), minval=0.65, maxval=0.9, dtype=jnp.float32)
        correct_clue = jax.random.bernoulli(keys[2], p=reliability, shape=(2,))
        clue = jnp.where(correct_clue, target, 1 - target).astype(jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "target": target,
            "reliability": reliability,
            "clue": clue,
            "clue_epoch": jnp.zeros((2,), dtype=jnp.int32),
            "clue_age": jnp.zeros((2,), dtype=jnp.int32),
            "message": jnp.full((2,), -1, dtype=jnp.int32),
            "message_epoch": jnp.full((2,), -1, dtype=jnp.int32),
            "message_age": jnp.zeros((2,), dtype=jnp.int32),
            "commitment": jnp.full((2,), -1, dtype=jnp.int32),
            "commitment_age": jnp.zeros((2,), dtype=jnp.int32),
            "epoch": jnp.array(0, dtype=jnp.int32),
            "phase": jnp.array(0, dtype=jnp.int32),
            "last_outcome": jnp.array(0.0, dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        other_message = jnp.roll(state["message"], 1)
        other_message_epoch = jnp.roll(state["message_epoch"], 1)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"] / 32.0)
        obs = obs.at[:, 1].set(state["clue"])
        obs = obs.at[:, 2].set(state["reliability"])
        obs = obs.at[:, 3].set((state["clue_epoch"] == state["epoch"]).astype(jnp.float32))
        obs = obs.at[:, 4].set(state["clue_age"] / 32.0)
        obs = obs.at[:, 5].set((other_message + 1) / 2.0)
        obs = obs.at[:, 6].set(((other_message >= 0) & (other_message_epoch == state["epoch"])).astype(jnp.float32))
        obs = obs.at[:, 7].set(jnp.roll(state["message_age"], 1) / 32.0)
        obs = obs.at[:, 8].set((state["message"] + 1) / 2.0)
        obs = obs.at[:, 9].set(((state["message"] >= 0) & (state["message_epoch"] == state["epoch"])).astype(jnp.float32))
        obs = obs.at[:, 10].set((state["commitment"] + 1) / 2.0)
        obs = obs.at[:, 11].set((jnp.roll(state["commitment"], 1) + 1) / 2.0)
        obs = obs.at[:, 12].set(state["commitment_age"] / 3.0)
        obs = obs.at[:, 13].set(jnp.roll(state["commitment_age"], 1) / 3.0)
        obs = obs.at[:, 14].set(state["phase"] / 8.0)
        obs = obs.at[:, 15].set(state["last_outcome"])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        keys = jax.random.split(key, 2)
        sample = actions == 0
        announce = (actions == 1) | (actions == 2)
        commit = actions >= 3
        correct_clue = jax.random.bernoulli(keys[0], p=state["reliability"], shape=(2,))
        sampled_clue = jnp.where(correct_clue, state["target"], 1 - state["target"]).astype(jnp.int32)
        clue = jnp.where(sample, sampled_clue, state["clue"])
        clue_epoch = jnp.where(sample, state["epoch"], state["clue_epoch"])
        clue_age = jnp.where(sample, 0, jnp.minimum(state["clue_age"] + 1, 32)).astype(jnp.int32)
        message = jnp.where(announce, actions - 1, state["message"]).astype(jnp.int32)
        message_epoch = jnp.where(announce, state["epoch"], state["message_epoch"])
        message_age = jnp.where(announce, 0, jnp.minimum(state["message_age"] + 1, 32)).astype(jnp.int32)
        aged_commitment = jnp.minimum(state["commitment_age"] + 1, 3)
        expired = (state["commitment"] >= 0) & (aged_commitment >= 3) & (~commit)
        proposed = jnp.where(commit, actions - 3, jnp.where(aged_commitment < 3, state["commitment"], -1)).astype(jnp.int32)
        proposed_age = jnp.where(commit, 0, jnp.where(proposed >= 0, aged_commitment, 0)).astype(jnp.int32)
        paired = (proposed[0] >= 0) & (proposed[1] >= 0)
        agree = proposed[0] == proposed[1]
        correct = proposed[0] == state["target"]
        success = paired & agree & correct
        wrong = paired & agree & (~correct)
        conflict = paired & (~agree)
        outcome = (0.75 * success.astype(jnp.float32) - 0.55 * wrong.astype(jnp.float32) - 0.20 * conflict.astype(jnp.float32)).astype(jnp.float32)
        refresh = paired | (state["phase"] + 1 >= 8)
        sampled_target = jax.random.randint(keys[1], (), 0, 2, dtype=jnp.int32)
        target = jnp.where(refresh, sampled_target, state["target"])
        commitment = jnp.where(refresh, jnp.full((2,), -1, dtype=jnp.int32), proposed)
        commitment_age = jnp.where(refresh, jnp.zeros((2,), dtype=jnp.int32), proposed_age)
        rewards = jnp.full((2,), outcome, dtype=jnp.float32)
        rewards = rewards - 0.035 * sample.astype(jnp.float32) - 0.01 * announce.astype(jnp.float32) - 0.02 * commit.astype(jnp.float32)
        rewards = jnp.clip(rewards - 0.06 * expired.astype(jnp.float32), -1.0, 1.0).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "target": target,
            "reliability": state["reliability"],
            "clue": clue,
            "clue_epoch": clue_epoch,
            "clue_age": clue_age,
            "message": message,
            "message_epoch": message_epoch,
            "message_age": message_age,
            "commitment": commitment,
            "commitment_age": commitment_age,
            "epoch": state["epoch"] + refresh.astype(jnp.int32),
            "phase": jnp.where(refresh, 0, state["phase"] + 1).astype(jnp.int32),
            "last_outcome": outcome,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
