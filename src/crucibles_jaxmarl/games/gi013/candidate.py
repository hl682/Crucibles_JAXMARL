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
            "water": jax.random.uniform(
                keys[0], (2,), minval=0.55, maxval=0.85, dtype=jnp.float32
            ),
            "contamination": jax.random.uniform(
                keys[1], (2,), minval=0.05, maxval=0.2, dtype=jnp.float32
            ),
            "filter": jnp.full((2,), 0.25, dtype=jnp.float32),
            "plume": jnp.zeros((2,), dtype=jnp.float32),
            "gate": jax.random.bernoulli(keys[2], p=0.5, shape=(2,)),
            "last_yield": jnp.zeros((2,), dtype=jnp.float32),
            "last_rain": jnp.zeros((2,), dtype=jnp.float32),
            "last_unfiltered": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        expected = 0.85 * jnp.where(
            jnp.all(state["gate"]), jnp.roll(state["plume"], 1), state["plume"]
        )
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(1.0 - state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 2].set(state["water"])
        obs = obs.at[:, 3].set(state["contamination"])
        obs = obs.at[:, 4].set(state["filter"] / 0.75)
        obs = obs.at[:, 5].set(state["plume"] / 0.28)
        obs = obs.at[:, 6].set(jnp.roll(state["plume"], 1) / 0.28)
        obs = obs.at[:, 7].set(state["gate"].astype(jnp.float32))
        obs = obs.at[:, 8].set(jnp.roll(state["gate"], 1).astype(jnp.float32))
        obs = obs.at[:, 9].set(state["last_yield"])
        obs = obs.at[:, 10].set(state["last_rain"] / 0.11)
        obs = obs.at[:, 11].set(state["last_unfiltered"] / 0.28)
        obs = obs.at[:, 12].set(expected / 0.28)
        obs = obs.at[:, 13].set(jnp.mean(state["water"]))
        obs = obs.at[:, 14].set(jnp.mean(state["contamination"]))
        obs = obs.at[:, 15].set(1.0 - 0.85 * state["contamination"])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        cleaning = (actions == 2).astype(jnp.float32)
        restoring = (actions == 3).astype(jnp.float32)
        toggling = actions == 4
        gates = jnp.where(toggling, jnp.logical_not(state["gate"]), state["gate"])
        incoming = 0.85 * jnp.where(
            jnp.all(gates), jnp.roll(state["plume"], 1), state["plume"]
        )
        filter_available = jnp.minimum(0.75, state["filter"] + 0.25 * restoring)
        absorbed = jnp.minimum(filter_available, incoming)
        unfiltered = incoming - absorbed
        contamination = jnp.clip(
            0.94 * state["contamination"] + unfiltered - 0.24 * cleaning,
            0.0,
            1.0,
        )
        extracted = jnp.where(
            actions == 1, jnp.minimum(state["water"], 0.35), 0.0
        ).astype(jnp.float32)
        useful_yield = (extracted / 0.35) * (1.0 - 0.85 * contamination)
        rainfall = jax.random.uniform(
            key, (2,), minval=0.08, maxval=0.11, dtype=jnp.float32
        )
        rewards = (
            0.6 * useful_yield
            - 0.05 * cleaning
            - 0.055 * restoring
            - 0.01 * toggling.astype(jnp.float32)
        ).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "water": jnp.clip(state["water"] - extracted + rainfall, 0.0, 1.0),
            "contamination": contamination,
            "filter": filter_available - absorbed,
            "plume": 0.8 * extracted,
            "gate": gates,
            "last_yield": useful_yield,
            "last_rain": rainfall,
            "last_unfiltered": unfiltered,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {
            "done_all": done_all,
            "legal_mask": self.legal_actions(next_state),
        }
        return self.observe(next_state), next_state, rewards, dones, info
