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
            "latent": jax.random.randint(
                key, (), minval=0, maxval=2, dtype=jnp.int32
            ),
            "battery": jnp.full((2,), 3, dtype=jnp.int32),
            "sample": jnp.full((2,), -1, dtype=jnp.int32),
            "sample_age": jnp.full((2,), 32, dtype=jnp.int32),
            "published": jnp.full((2,), -1, dtype=jnp.int32),
            "published_age": jnp.full((2,), 32, dtype=jnp.int32),
            "far": jnp.full((2,), -1, dtype=jnp.int32),
            "near": jnp.full((2,), -1, dtype=jnp.int32),
            "channel": jnp.array(0, dtype=jnp.int32),
            "last_settlement": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        peer_report = jnp.roll(state["published"], 1)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["battery"].astype(jnp.float32) / 3.0)
        obs = obs.at[:, 2].set(jnp.maximum(state["sample"], 0).astype(jnp.float32))
        obs = obs.at[:, 3].set((state["sample"] >= 0).astype(jnp.float32))
        obs = obs.at[:, 4].set(state["sample_age"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 5].set(
            jnp.maximum(state["published"], 0).astype(jnp.float32)
        )
        obs = obs.at[:, 6].set((state["published"] >= 0).astype(jnp.float32))
        obs = obs.at[:, 7].set(state["published_age"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 8].set(jnp.maximum(peer_report, 0).astype(jnp.float32))
        obs = obs.at[:, 9].set((peer_report >= 0).astype(jnp.float32))
        obs = obs.at[:, 10].set(
            jnp.roll(state["published_age"], 1).astype(jnp.float32) / 32.0
        )
        obs = obs.at[:, 11].set((state["far"] + 1).astype(jnp.float32) / 2.0)
        obs = obs.at[:, 12].set((state["near"] + 1).astype(jnp.float32) / 2.0)
        obs = obs.at[:, 13].set(state["channel"].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 14].set(state["last_settlement"])
        obs = obs.at[:, 15].set(1.0 - state["t"].astype(jnp.float32) / 32.0)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        keys = jax.random.split(key, 2)
        flip = jax.random.bernoulli(keys[0], p=0.12).astype(jnp.int32)
        latent = jnp.mod(state["latent"] + flip, 2).astype(jnp.int32)
        error = jax.random.bernoulli(
            keys[1], p=0.15, shape=(2,)
        ).astype(jnp.int32)
        measurement = jnp.mod(latent + error, 2).astype(jnp.int32)
        scanning = (actions == 1) & (state["battery"] > 0)
        launching = (
            ((actions == 3) | (actions == 4)) & (state["battery"] > 0)
        )
        requesting = (actions == 2) & (state["sample"] >= 0)
        channel = jnp.sum(requesting.astype(jnp.int32))
        transmitted = requesting & (channel == 1)
        settling = state["near"] >= 0
        matched = settling & (state["near"] == latent)
        congested = jnp.all(settling) & (state["near"][0] == state["near"][1])
        correct_value = jnp.where(congested, 0.275, 0.55)
        settlement = jnp.where(
            settling, jnp.where(matched, correct_value, -0.25), 0.0
        ).astype(jnp.float32)
        rewards = (
            settlement
            - 0.025 * scanning.astype(jnp.float32)
            - 0.015 * launching.astype(jnp.float32)
            - 0.005 * requesting.astype(jnp.float32)
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "latent": latent,
            "battery": jnp.minimum(
                3,
                state["battery"]
                + (actions == 0).astype(jnp.int32)
                - scanning.astype(jnp.int32)
                - launching.astype(jnp.int32),
            ),
            "sample": jnp.where(scanning, measurement, state["sample"]),
            "sample_age": jnp.where(
                scanning, 0, jnp.minimum(state["sample_age"] + 1, 32)
            ).astype(jnp.int32),
            "published": jnp.where(
                transmitted, state["sample"], state["published"]
            ),
            "published_age": jnp.where(
                transmitted,
                jnp.minimum(state["sample_age"] + 1, 32),
                jnp.minimum(state["published_age"] + 1, 32),
            ).astype(jnp.int32),
            "far": jnp.where(launching, actions - 3, -1).astype(jnp.int32),
            "near": state["far"],
            "channel": channel,
            "last_settlement": settlement,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {
            "done_all": done_all,
            "legal_mask": self.legal_actions(next_state),
        }
        return self.observe(next_state), next_state, rewards, dones, info
