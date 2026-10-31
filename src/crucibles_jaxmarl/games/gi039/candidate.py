import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        mode_key, clue_key = jax.random.split(key, 2)
        mode = jax.random.randint(mode_key, (), 0, 2, dtype=jnp.int32)
        correct_clue = jax.random.bernoulli(clue_key, 0.6, (2,))
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "mode": mode,
            "clue": jnp.where(correct_clue, mode, 1 - mode).astype(jnp.int32),
            "clue_age": jnp.zeros((2,), dtype=jnp.int32),
            "precision": jnp.full((2,), 0.6, dtype=jnp.float32),
            "charge": jnp.full((2,), 0.65, dtype=jnp.float32),
            "vote": jnp.full((2,), -1, dtype=jnp.int32),
            "vote_age": jnp.zeros((2,), dtype=jnp.int32),
            "approved": jnp.zeros((2,), dtype=jnp.bool_),
            "last_decision": jnp.array(-1, dtype=jnp.int32),
            "last_result": jnp.array(0.0, dtype=jnp.float32),
            "resolutions": jnp.array(0.0, dtype=jnp.float32),
            "last_action": jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        other = jnp.array([1, 0], dtype=jnp.int32)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["clue"].astype(jnp.float32))
        obs = obs.at[:, 2].set(state["clue_age"].astype(jnp.float32) / 8.0)
        obs = obs.at[:, 3].set(state["precision"])
        obs = obs.at[:, 4].set(state["charge"])
        obs = obs.at[:, 5].set((state["vote"].astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 6].set(state["vote_age"].astype(jnp.float32) / 8.0)
        obs = obs.at[:, 7].set(state["approved"].astype(jnp.float32))
        obs = obs.at[:, 8].set((state["vote"][other].astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 9].set(state["vote_age"][other].astype(jnp.float32) / 8.0)
        obs = obs.at[:, 10].set(state["approved"][other].astype(jnp.float32))
        obs = obs.at[:, 11].set((state["last_decision"].astype(jnp.float32) + 1.0) / 2.0)
        obs = obs.at[:, 12].set(state["last_result"])
        obs = obs.at[:, 13].set(state["resolutions"] / 8.0)
        obs = obs.at[:, 14].set(state["last_action"].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 15].set(state["last_action"][other].astype(jnp.float32) / 4.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        mode_key, drift_key, clue_key = jax.random.split(key, 3)
        proposing = (actions == 1) | (actions == 2)
        resolve = jnp.all(state["approved"]) & jnp.all(state["vote"] >= 0) & (state["vote"][0] == state["vote"][1]) & jnp.all(state["vote_age"] >= 1) & ~jnp.any(proposing)
        correct = state["vote"][0] == state["mode"]
        sampled_mode = jax.random.randint(mode_key, (), 0, 2, dtype=jnp.int32)
        drift = jax.random.bernoulli(drift_key, 0.06, ())
        next_mode = jnp.where(resolve, sampled_mode, jnp.where(drift, 1 - state["mode"], state["mode"])).astype(jnp.int32)
        inspect = (actions == 3) & (state["charge"] >= 0.35)
        authorize = (actions == 4) & (state["vote"] >= 0) & (state["charge"] >= 0.1) & ~state["approved"] & ~resolve
        refresh = inspect | resolve
        correct_clue = jax.random.bernoulli(clue_key, 0.6, (2,))
        fresh_clue = jnp.where(inspect, next_mode, jnp.where(correct_clue, next_mode, 1 - next_mode)).astype(jnp.int32)
        next_clue = jnp.where(refresh, fresh_clue, state["clue"]).astype(jnp.int32)
        next_clue_age = jnp.where(refresh, 0, jnp.minimum(state["clue_age"] + 1, 8)).astype(jnp.int32)
        next_precision = jnp.where(inspect, 1.0, jnp.where(resolve, 0.6, state["precision"])).astype(jnp.float32)
        next_vote = jnp.where(resolve, -1, jnp.where(proposing, actions - 1, state["vote"])).astype(jnp.int32)
        next_vote_age = jnp.where(resolve | proposing, 0, jnp.where(state["vote"] >= 0, jnp.minimum(state["vote_age"] + 1, 8), 0)).astype(jnp.int32)
        next_approved = jnp.where(resolve | proposing, jnp.zeros((2,), dtype=jnp.bool_), state["approved"] | authorize)
        next_charge = jnp.clip(state["charge"] + 0.16 * (actions == 0).astype(jnp.float32) - 0.35 * inspect.astype(jnp.float32) - 0.1 * authorize.astype(jnp.float32), 0.0, 1.0)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "mode": next_mode,
            "clue": next_clue,
            "clue_age": next_clue_age,
            "precision": next_precision,
            "charge": next_charge,
            "vote": next_vote,
            "vote_age": next_vote_age,
            "approved": next_approved,
            "last_decision": jnp.where(resolve, state["vote"][0], state["last_decision"]).astype(jnp.int32),
            "last_result": jnp.where(resolve, jnp.where(correct, 1.0, -1.0), state["last_result"]).astype(jnp.float32),
            "resolutions": state["resolutions"] + resolve.astype(jnp.float32),
            "last_action": actions.astype(jnp.int32),
        }
        decision_payoff = jnp.where(resolve, jnp.where(correct, 0.8, -0.7), 0.0)
        rewards = jnp.clip(jnp.full((2,), decision_payoff, dtype=jnp.float32) - 0.035 * inspect.astype(jnp.float32) - 0.01 * authorize.astype(jnp.float32), -1.0, 1.0).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info