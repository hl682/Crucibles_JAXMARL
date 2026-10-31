import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        settings = jax.random.randint(key, (3,), 0, 2, dtype=jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "target": settings[:2],
            "switch": settings[2],
            "pending": jnp.array(-1, dtype=jnp.int32),
            "stage": jnp.zeros((2,), dtype=jnp.int32),
            "transit": jnp.zeros((2,), dtype=jnp.int32),
            "route": jnp.array([0, 1], dtype=jnp.int32),
            "energy": jnp.full((2,), 0.65, dtype=jnp.float32),
            "age": jnp.zeros((2,), dtype=jnp.int32),
            "vote": jnp.full((2,), -1, dtype=jnp.int32),
            "vote_age": jnp.zeros((2,), dtype=jnp.int32),
            "last_result": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"] / 32.0)
        obs = obs.at[:, 1].set(state["target"].astype(jnp.float32))
        obs = obs.at[:, 2].set(state["switch"].astype(jnp.float32))
        obs = obs.at[:, 3].set(state["pending"].astype(jnp.float32))
        obs = obs.at[:, 4].set(state["stage"] / 2.0)
        obs = obs.at[:, 5].set(jnp.roll(state["stage"], 1) / 2.0)
        obs = obs.at[:, 6].set(state["age"] / 8.0)
        obs = obs.at[:, 7].set(jnp.roll(state["age"], 1) / 8.0)
        obs = obs.at[:, 8].set(state["energy"])
        obs = obs.at[:, 9].set(state["route"].astype(jnp.float32))
        obs = obs.at[:, 10].set(state["vote"].astype(jnp.float32))
        obs = obs.at[:, 11].set(jnp.roll(state["vote"], 1).astype(jnp.float32))
        obs = obs.at[:, 12].set(state["vote_age"] / 4.0)
        obs = obs.at[:, 13].set(jnp.roll(state["vote_age"], 1) / 4.0)
        obs = obs.at[:, 14].set(state["transit"] / 2.0)
        obs = obs.at[:, 15].set(state["last_result"])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        applying = state["pending"] >= 0
        active_switch = jnp.where(applying, state["pending"], state["switch"])
        energy_start = jnp.minimum(state["energy"] + 0.03, 1.0)
        casting = (actions >= 3) & (energy_start >= 0.03)
        base_votes = jnp.where(applying, jnp.full((2,), -1, dtype=jnp.int32), state["vote"])
        base_vote_age = jnp.where(applying, 0, state["vote_age"])
        votes = jnp.where(casting, actions - 3, base_votes).astype(jnp.int32)
        vote_age = jnp.where(casting, 0, jnp.where(base_votes >= 0, base_vote_age + 1, 0)).astype(jnp.int32)
        votes = jnp.where(vote_age >= 4, -1, votes).astype(jnp.int32)
        agreement = jnp.all(votes >= 0) & (votes[0] == votes[1])
        pending = jnp.where(agreement, votes[0], -1).astype(jnp.int32)
        final_votes = jnp.where(agreement, -1, votes).astype(jnp.int32)
        final_vote_age = jnp.where(final_votes >= 0, vote_age, 0).astype(jnp.int32)

        loading = (actions == 1) & (state["stage"] == 0) & (energy_start >= 0.12)
        committing = (actions == 2) & (state["stage"] == 1) & (energy_start >= 0.08)
        arriving = (state["stage"] == 2) & (state["transit"] == 1)
        success = arriving & (state["route"] == state["target"])
        staged = jnp.where(loading, 1, jnp.where(committing, 2, state["stage"])).astype(jnp.int32)
        active_age = jnp.where(staged > 0, state["age"] + 1, 0).astype(jnp.int32)
        expired = (active_age >= 8) & (staged > 0) & ~arriving
        completed = arriving | expired
        latched_route = jnp.mod(jnp.arange(2, dtype=jnp.int32) + active_switch, 2)
        flight_count = jnp.where(committing, 2, jnp.maximum(state["transit"] - 1, 0)).astype(jnp.int32)
        fresh_targets = jax.random.randint(key, (2,), 0, 2, dtype=jnp.int32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "target": jnp.where(completed, fresh_targets, state["target"]),
            "switch": active_switch.astype(jnp.int32),
            "pending": pending,
            "stage": jnp.where(completed, 0, staged).astype(jnp.int32),
            "transit": jnp.where(completed, 0, flight_count).astype(jnp.int32),
            "route": jnp.where(committing, latched_route, state["route"]),
            "energy": jnp.clip(
                energy_start - 0.12 * loading - 0.08 * committing - 0.03 * casting,
                0.0,
                1.0,
            ),
            "age": jnp.where(completed, 0, active_age).astype(jnp.int32),
            "vote": final_votes,
            "vote_age": final_vote_age,
            "last_result": jnp.where(
                arriving,
                jnp.where(success, 1.0, -1.0),
                jnp.where(expired, -0.50, 0.0),
            ).astype(jnp.float32),
        }
        rewards = (
            0.80 * success.astype(jnp.float32)
            - 0.40 * (arriving & ~success).astype(jnp.float32)
            - 0.45 * expired.astype(jnp.float32)
            - 0.05 * loading.astype(jnp.float32)
            - 0.015 * casting.astype(jnp.float32)
            - 0.01 * (state["stage"] > 0).astype(jnp.float32)
        )
        rewards = jnp.clip(rewards, -1.0, 1.0).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.broadcast_to(done_all, (2,))
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
