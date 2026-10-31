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
        load = jax.random.uniform(keys[0], (2,), minval=0.05, maxval=0.3, dtype=jnp.float32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "load": load,
            "latent": jnp.zeros((2,), dtype=jnp.float32),
            "immunity": jnp.zeros((2,), dtype=jnp.float32),
            "vaccine_stage": jnp.zeros((2,), dtype=jnp.int32),
            "stock": jax.random.uniform(keys[1], (2,), minval=0.8, maxval=1.2, dtype=jnp.float32),
            "susceptibility": jax.random.uniform(keys[2], (2,), minval=0.8, maxval=1.2, dtype=jnp.float32),
            "symptom": jnp.clip((load - 0.12) / 0.88, 0.0, 1.0),
            "report": jnp.zeros((2,), dtype=jnp.float32),
            "report_age": jnp.full((2,), 32, dtype=jnp.int32),
            "test_sample": jnp.zeros((2,), dtype=jnp.float32),
            "test_wait": jnp.zeros((2,), dtype=jnp.int32),
            "last_contact": jnp.zeros((2,), dtype=jnp.float32),
            "last_gain": jnp.zeros((2,), dtype=jnp.float32),
            "last_work": jnp.zeros((2,), dtype=jnp.float32),
            "last_support": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["stock"] / 2.0)
        obs = obs.at[:, 1].set(state["symptom"])
        obs = obs.at[:, 2].set(jnp.roll(state["symptom"], 1))
        obs = obs.at[:, 3].set(state["immunity"])
        obs = obs.at[:, 4].set(state["vaccine_stage"] / 2.0)
        obs = obs.at[:, 5].set(state["report"])
        obs = obs.at[:, 6].set(state["report_age"] / 32.0)
        obs = obs.at[:, 7].set(state["test_wait"] / 2.0)
        obs = obs.at[:, 8].set(state["susceptibility"] / 1.2)
        obs = obs.at[:, 9].set(state["last_contact"])
        obs = obs.at[:, 10].set(state["last_gain"] / 0.35)
        obs = obs.at[:, 11].set(state["last_work"])
        obs = obs.at[:, 12].set(jnp.roll(state["last_work"], 1))
        obs = obs.at[:, 13].set(state["last_support"])
        obs = obs.at[:, 14].set(state["t"] / 32.0)
        obs = obs.at[:, 15].set(jnp.array([-1.0, 1.0], dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        mask = jnp.ones((2, 5), dtype=jnp.bool_)
        mask = mask.at[:, 2].set((state["stock"] >= 0.08) & (state["test_wait"] == 0))
        mask = mask.at[:, 3].set(
            (state["stock"] >= 0.25) & (state["vaccine_stage"] == 0) & (state["immunity"] < 0.9)
        )
        mask = mask.at[:, 4].set(state["stock"] >= 0.18)
        return mask

    def raw_step(self, key, state, actions):
        resting = actions == 0
        working = actions == 1
        screening = (actions == 2) & (state["stock"] >= 0.08) & (state["test_wait"] == 0)
        vaccinating = (
            (actions == 3)
            & (state["stock"] >= 0.25)
            & (state["vaccine_stage"] == 0)
            & (state["immunity"] < 0.9)
        )
        treating = (actions == 4) & (state["stock"] >= 0.18)
        received_support = jnp.roll(treating, 1).astype(jnp.float32)
        other_working = jnp.roll(working, 1)
        contact = jnp.where(working & other_working, 1.0, jnp.where(working | other_working, 0.35, 0.05))
        spark = 0.08 * jax.random.bernoulli(key, p=0.08, shape=(2,)).astype(jnp.float32)
        exposure = (
            0.3 * jnp.roll(state["load"], 1) * contact + spark
        ) * (1.0 - state["immunity"]) * state["susceptibility"]
        latent = jnp.clip(
            exposure + 0.25 * state["latent"] - 0.1 * received_support,
            0.0,
            1.0,
        )
        load = jnp.clip(
            0.9 * state["load"]
            + 0.65 * state["latent"]
            - 0.11 * resting.astype(jnp.float32)
            - 0.18 * received_support
            - 0.05 * state["immunity"],
            0.0,
            1.0,
        )
        immunity = jnp.clip(
            0.94 * state["immunity"]
            + 0.55 * (state["vaccine_stage"] == 1).astype(jnp.float32),
            0.0,
            0.95,
        )
        work_gain = 0.35 * working.astype(jnp.float32) * (1.0 - state["load"])
        delivered = state["test_wait"] == 1
        rewards = jnp.clip(
            0.55 * working.astype(jnp.float32) * (1.0 - state["load"])
            - 0.35 * state["load"]
            - 0.08 * screening.astype(jnp.float32)
            - 0.12 * vaccinating.astype(jnp.float32)
            - 0.1 * treating.astype(jnp.float32)
            + 0.08 * received_support * state["load"],
            -1.0,
            1.0,
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "load": load.astype(jnp.float32),
            "latent": latent.astype(jnp.float32),
            "immunity": immunity.astype(jnp.float32),
            "vaccine_stage": jnp.where(
                vaccinating, 2, jnp.maximum(state["vaccine_stage"] - 1, 0)
            ).astype(jnp.int32),
            "stock": jnp.clip(
                state["stock"]
                + work_gain
                + 0.08 * resting.astype(jnp.float32)
                - 0.02
                - 0.08 * screening.astype(jnp.float32)
                - 0.25 * vaccinating.astype(jnp.float32)
                - 0.18 * treating.astype(jnp.float32),
                0.0,
                2.0,
            ).astype(jnp.float32),
            "susceptibility": state["susceptibility"],
            "symptom": jnp.clip((state["load"] - 0.12) / 0.88, 0.0, 1.0).astype(jnp.float32),
            "report": jnp.where(delivered, state["test_sample"], state["report"]).astype(jnp.float32),
            "report_age": jnp.where(
                delivered, 2, jnp.minimum(state["report_age"] + 1, 32)
            ).astype(jnp.int32),
            "test_sample": jnp.where(screening, state["load"], state["test_sample"]).astype(jnp.float32),
            "test_wait": jnp.where(screening, 2, jnp.maximum(state["test_wait"] - 1, 0)).astype(jnp.int32),
            "last_contact": contact.astype(jnp.float32),
            "last_gain": work_gain.astype(jnp.float32),
            "last_work": working.astype(jnp.float32),
            "last_support": received_support,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info