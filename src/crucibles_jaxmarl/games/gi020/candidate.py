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
            "cash": jax.random.uniform(keys[0], (2,), minval=2.0, maxval=3.0, dtype=jnp.float32),
            "channel": jnp.zeros((2,), dtype=jnp.int32),
            "sealed": jnp.zeros((2,), dtype=jnp.bool_),
            "age": jnp.zeros((2,), dtype=jnp.int32),
            "preference": jax.random.randint(keys[1], (2,), 1, 3, dtype=jnp.int32),
            "last_settlement": jnp.zeros((2,), dtype=jnp.float32),
            "last_revoke": jnp.zeros((2,), dtype=jnp.bool_),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(jnp.array([-1.0, 1.0], dtype=jnp.float32))
        obs = obs.at[:, 2].set(state["cash"] / 6.0)
        obs = obs.at[:, 3].set(jnp.roll(state["cash"], 1) / 6.0)
        obs = obs.at[:, 4].set(state["channel"].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 5].set(jnp.roll(state["channel"], 1).astype(jnp.float32) / 2.0)
        obs = obs.at[:, 6].set(state["sealed"].astype(jnp.float32))
        obs = obs.at[:, 7].set(jnp.roll(state["sealed"], 1).astype(jnp.float32))
        obs = obs.at[:, 8].set(state["age"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 9].set(jnp.roll(state["age"], 1).astype(jnp.float32) / 32.0)
        obs = obs.at[:, 10].set(state["preference"].astype(jnp.float32) / 2.0)
        obs = obs.at[:, 11].set(((state["channel"] > 0) & (state["channel"] == state["preference"])).astype(jnp.float32))
        obs = obs.at[:, 12].set(state["last_settlement"])
        obs = obs.at[:, 13].set(jnp.roll(state["last_settlement"], 1))
        obs = obs.at[:, 14].set(state["last_revoke"].astype(jnp.float32))
        obs = obs.at[:, 15].set(jnp.roll(state["last_revoke"], 1).astype(jnp.float32))
        return obs

    def legal_actions(self, state):
        vacant = state["channel"] == 0
        can_open = vacant & (state["cash"] >= 1.0)
        can_seal = (~vacant) & (~state["sealed"])
        can_revoke = ~vacant
        return jnp.stack((jnp.ones((2,), dtype=jnp.bool_), can_open, can_open, can_seal, can_revoke), axis=1)

    def raw_step(self, key, state, actions):
        legal = self.legal_actions(state)
        opened = ((actions == 1) | (actions == 2)) & legal[:, 1]
        authorized = (actions == 3) & legal[:, 3]
        revoked = (actions == 4) & legal[:, 4]
        paired = (state["channel"][0] > 0) & (state["channel"][0] == state["channel"][1])
        matured = jnp.all(state["age"] >= 2) & jnp.all(state["sealed"])
        release = paired & matured & (~jnp.any(revoked))
        favored = (state["channel"] == state["preference"]).astype(jnp.float32)
        settlement = release.astype(jnp.float32) * (0.35 + 0.45 * favored)
        cash = jnp.clip(
            state["cash"] + 0.06 - opened.astype(jnp.float32)
            + 0.7 * revoked.astype(jnp.float32)
            + release.astype(jnp.float32) * (1.0 + 0.35 + 0.45 * favored),
            0.0, 6.0,
        )
        cleared = revoked | release
        channel = jnp.where(cleared, 0, jnp.where(opened, actions, state["channel"])).astype(jnp.int32)
        sealed = jnp.where(cleared, False, state["sealed"] | authorized)
        age = jnp.where(
            channel > 0,
            jnp.where(opened, 0, jnp.minimum(state["age"] + 1, 32)),
            0,
        ).astype(jnp.int32)
        interrupted = jnp.roll(revoked, 1) & state["sealed"]
        rewards = jnp.clip(
            settlement - 0.04 * opened.astype(jnp.float32)
            - 0.01 * authorized.astype(jnp.float32)
            - 0.07 * revoked.astype(jnp.float32)
            - 0.04 * interrupted.astype(jnp.float32),
            -1.0, 1.0,
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "cash": cash.astype(jnp.float32),
            "channel": channel,
            "sealed": sealed,
            "age": age,
            "preference": state["preference"],
            "last_settlement": settlement.astype(jnp.float32),
            "last_revoke": revoked,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
