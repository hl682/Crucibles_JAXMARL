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
            "temperature": jax.random.uniform(keys[0], (2,), minval=0.2, maxval=0.55, dtype=jnp.float32),
            "backlog": jnp.ones((2,), dtype=jnp.float32),
            "tolerance": jax.random.uniform(keys[1], (2,), minval=0.45, maxval=0.7, dtype=jnp.float32),
            "wear": jnp.zeros((2,), dtype=jnp.float32),
            "valve": jnp.zeros((2,), dtype=jnp.float32),
            "ambient": jnp.array(0.15, dtype=jnp.float32),
            "reservoir": jnp.array(0.0, dtype=jnp.float32),
            "last_exchange": jnp.zeros((2,), dtype=jnp.float32),
            "last_service": jnp.zeros((2,), dtype=jnp.float32),
            "last_dump": jnp.zeros((2,), dtype=jnp.float32),
            "last_grade": jnp.zeros((2,), dtype=jnp.float32),
            "last_process": jnp.zeros((2,), dtype=jnp.float32),
            "last_offers": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        other = jnp.array([1, 0], dtype=jnp.int32)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["temperature"])
        obs = obs.at[:, 2].set(state["backlog"] / 3.0)
        obs = obs.at[:, 3].set(state["tolerance"])
        obs = obs.at[:, 4].set(state["wear"])
        obs = obs.at[:, 5].set(state["valve"])
        obs = obs.at[:, 6].set(state["valve"][other])
        obs = obs.at[:, 7].set(state["ambient"])
        obs = obs.at[:, 8].set(state["reservoir"] / 2.0)
        obs = obs.at[:, 9].set(state["last_exchange"])
        obs = obs.at[:, 10].set(state["last_service"])
        obs = obs.at[:, 11].set(state["last_dump"])
        obs = obs.at[:, 12].set(state["last_grade"])
        obs = obs.at[:, 13].set(state["last_process"][other])
        obs = obs.at[:, 14].set(state["last_offers"])
        obs = obs.at[:, 15].set(jnp.arange(2, dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        other = jnp.array([1, 0], dtype=jnp.int32)
        valve = jnp.where(actions == 3, 1.0, jnp.where(actions == 4, 0.0, state["valve"])).astype(jnp.float32)
        exchange = 0.28 * jnp.prod(valve) * (state["temperature"][other] - state["temperature"])
        work = (actions == 1).astype(jnp.float32) * jnp.minimum(state["backlog"], 1.0)
        extracted = (actions == 2).astype(jnp.float32) * jnp.minimum(state["temperature"], 0.25)
        grade = jnp.clip((state["tolerance"] - state["temperature"] + 0.15) / 0.3, 0.0, 1.0) * (1.0 - 0.5 * state["wear"])
        service = work * grade
        dump = 1.2 * extracted
        temperature = jnp.clip(state["temperature"] + exchange + 0.23 * work - extracted + 0.12 * state["ambient"] - 0.025, 0.0, 1.0)
        recovery = ((actions == 0) & (temperature < 0.4)).astype(jnp.float32)
        wear = jnp.clip(state["wear"] + 0.08 * jnp.maximum(temperature - 0.8, 0.0) - 0.03 * recovery, 0.0, 1.0)
        release = 0.25 * state["reservoir"]
        offers = jax.random.uniform(key, (2,), minval=0.1, maxval=0.3, dtype=jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "temperature": temperature,
            "backlog": jnp.clip(state["backlog"] - work + offers, 0.0, 3.0),
            "tolerance": state["tolerance"],
            "wear": wear,
            "valve": valve,
            "ambient": jnp.clip(0.88 * state["ambient"] + 0.35 * release + 0.02 * jnp.sum(work), 0.05, 1.0),
            "reservoir": jnp.clip(state["reservoir"] - release + jnp.sum(dump), 0.0, 2.0),
            "last_exchange": exchange,
            "last_service": service,
            "last_dump": dump,
            "last_grade": grade,
            "last_process": (actions == 1).astype(jnp.float32),
            "last_offers": offers,
        }
        utility = 0.9 * service - 0.06 * (actions == 2).astype(jnp.float32) - 0.2 * jnp.maximum(temperature - 0.75, 0.0) - 0.1 * (work - service)
        rewards = jnp.tanh(utility).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
