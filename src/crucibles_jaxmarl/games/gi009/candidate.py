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
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "water": jax.random.uniform(keys[0], (), minval=0.65, maxval=0.95, dtype=jnp.float32),
            "toxin": jax.random.uniform(keys[1], (), minval=0.05, maxval=0.18, dtype=jnp.float32),
            "collapsed": jnp.array(False, dtype=jnp.bool_),
            "reserve": jax.random.uniform(keys[2], (2,), minval=0.45, maxval=0.8, dtype=jnp.float32),
            "nursery": jnp.zeros((2, 2), dtype=jnp.float32),
            "last_take": jnp.zeros((2,), dtype=jnp.float32),
            "last_private": jnp.zeros((2,), dtype=jnp.float32),
            "last_clean": jnp.zeros((2,), dtype=jnp.float32),
            "last_mature": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        recovery_gate = (state["water"] > 0.55) & (state["toxin"] < 0.22)
        collapse_gate = (state["water"] < 0.16) | (state["toxin"] > 0.72)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["water"])
        obs = obs.at[:, 2].set(state["toxin"])
        obs = obs.at[:, 3].set(state["collapsed"].astype(jnp.float32))
        obs = obs.at[:, 4].set(state["reserve"])
        obs = obs.at[:, 5:7].set(state["nursery"])
        obs = obs.at[:, 7].set(jnp.sum(state["nursery"][:, 1]) / 2.0)
        obs = obs.at[:, 8].set(state["last_take"] / 0.18)
        obs = obs.at[:, 9].set(jnp.roll(state["last_take"], 1) / 0.18)
        obs = obs.at[:, 10].set(state["last_private"] / 0.18)
        obs = obs.at[:, 11].set(jnp.roll(state["last_clean"], 1))
        obs = obs.at[:, 12].set(jnp.sum(state["last_mature"]) / 2.0)
        obs = obs.at[:, 13].set(recovery_gate.astype(jnp.float32))
        obs = obs.at[:, 14].set(collapse_gate.astype(jnp.float32))
        obs = obs.at[:, 15].set(jnp.arange(2, dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        public_request = 0.18 * (actions == 1).astype(jnp.float32)
        allocation = jnp.minimum(
            1.0,
            state["water"] / jnp.maximum(jnp.sum(public_request), 0.18),
        )
        taken = public_request * allocation
        private_take = jnp.minimum(
            state["reserve"],
            0.18 * (actions == 2).astype(jnp.float32),
        )
        planted = ((actions == 3) & (state["reserve"] >= 0.14)).astype(jnp.float32)
        cleaned = (actions == 4).astype(jnp.float32)
        matured = state["nursery"][:, 0]
        nursery = jnp.stack((state["nursery"][:, 1], planted), axis=1)
        toxin = jnp.clip(
            state["toxin"] + 0.1 * jnp.sum(taken) / 0.18
            - 0.08 * jnp.sum(cleaned) - 0.02 - 0.05 * jnp.sum(matured),
            0.0,
            1.0,
        )
        regeneration = jnp.where(state["collapsed"], 0.006, 0.05)
        water = jnp.clip(
            state["water"] - jnp.sum(taken)
            + regeneration * (1.0 - state["water"])
            + 0.09 * jnp.sum(matured),
            0.0,
            1.0,
        )
        recovery_gate = (water > 0.55) & (toxin < 0.22)
        collapse_gate = (water < 0.16) | (toxin > 0.72)
        collapsed = jnp.where(state["collapsed"], ~recovery_gate, collapse_gate)
        reserve = jnp.clip(
            state["reserve"] - private_take - 0.14 * planted
            + 0.025 * water * (1.0 - toxin),
            0.0,
            1.0,
        )
        public_quality = 1.0 - 0.65 * state["collapsed"].astype(jnp.float32)
        rewards = (
            0.7 * public_quality * taken / 0.18
            + 0.45 * private_take / 0.18
            - 0.02 * planted - 0.04 * cleaned
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "water": water.astype(jnp.float32),
            "toxin": toxin.astype(jnp.float32),
            "collapsed": collapsed.astype(jnp.bool_),
            "reserve": reserve.astype(jnp.float32),
            "nursery": nursery.astype(jnp.float32),
            "last_take": taken.astype(jnp.float32),
            "last_private": private_take.astype(jnp.float32),
            "last_clean": cleaned,
            "last_mature": matured,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info