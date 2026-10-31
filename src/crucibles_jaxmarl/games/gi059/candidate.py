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
            "mode": jax.random.randint(keys[0], (), 0, 2, dtype=jnp.int32),
            "tally": jnp.zeros((2,), dtype=jnp.float32),
            "influence": jnp.ones((2,), dtype=jnp.float32),
            "arrears": jnp.zeros((2,), dtype=jnp.float32),
            "stock": jnp.full((2,), 0.7, dtype=jnp.float32),
            "quality": jnp.array(0.7, dtype=jnp.float32),
            "productivity": jax.random.uniform(keys[1], (2,), minval=0.5, maxval=1.0, dtype=jnp.float32),
            "forecast": jax.random.uniform(keys[2], (2,), minval=0.5, maxval=1.0, dtype=jnp.float32),
            "report": jnp.zeros((2,), dtype=jnp.float32),
            "report_age": jnp.full((2,), 32, dtype=jnp.int32),
            "last_allocation": jnp.zeros((2,), dtype=jnp.float32),
            "last_harvest": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(jnp.arange(2, dtype=jnp.float32))
        obs = obs.at[:, 2].set(state["mode"].astype(jnp.float32))
        obs = obs.at[:, 3].set((4.0 - jnp.mod(state["t"], 4).astype(jnp.float32)) / 4.0)
        obs = obs.at[:, 4].set(state["stock"] / 3.0)
        obs = obs.at[:, 5].set(state["quality"])
        obs = obs.at[:, 6].set(state["influence"])
        obs = obs.at[:, 7].set(state["arrears"] / 2.0)
        obs = obs.at[:, 8].set(state["tally"][0] / 8.0)
        obs = obs.at[:, 9].set(state["tally"][1] / 8.0)
        obs = obs.at[:, 10].set(state["productivity"])
        obs = obs.at[:, 11].set(state["forecast"])
        obs = obs.at[:, 12].set(state["report"])
        obs = obs.at[:, 13].set(state["report_age"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 14].set(state["last_allocation"])
        obs = obs.at[:, 15].set(state["last_harvest"])
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        roles = jnp.arange(2, dtype=jnp.int32)
        other = jnp.array([1, 0], dtype=jnp.int32)
        vote = ((actions == 1) | (actions == 2)).astype(jnp.float32)
        spent = vote * jnp.minimum(state["influence"], 0.5)
        weight = spent * (1.0 + state["arrears"] / 3.0)
        ballots = jnp.stack((jnp.sum(weight * (actions == 1).astype(jnp.float32)), jnp.sum(weight * (actions == 2).astype(jnp.float32))))
        tally = state["tally"] + ballots
        choice = jnp.where(tally[0] > tally[1], 0, jnp.where(tally[1] > tally[0], 1, state["mode"])).astype(jnp.int32)
        next_t = state["t"] + jnp.array(1, dtype=jnp.int32)
        boundary = jnp.mod(next_t, 4) == 0
        repair = (actions == 3).astype(jnp.float32)
        survey = actions == 4
        harvest = (actions == 0).astype(jnp.float32) * jnp.minimum(state["stock"], 0.6)
        shares = jnp.where(roles == state["mode"], 0.75, 0.25).astype(jnp.float32)
        allocation = 0.55 * state["quality"] * shares * state["productivity"]
        new_forecast = jax.random.uniform(key, (2,), minval=0.5, maxval=1.0, dtype=jnp.float32)
        next_state = {
            "t": next_t,
            "mode": jnp.where(boundary, choice, state["mode"]).astype(jnp.int32),
            "tally": jnp.where(boundary, jnp.zeros((2,), dtype=jnp.float32), tally),
            "influence": jnp.clip(state["influence"] - spent + 0.035 + 0.12 * repair, 0.0, 1.0),
            "arrears": jnp.clip(0.9 * state["arrears"] + 0.2 - allocation, 0.0, 2.0),
            "stock": jnp.clip(state["stock"] - harvest + allocation, 0.0, 3.0),
            "quality": jnp.clip(0.97 * state["quality"] + 0.12 * jnp.sum(repair) - 0.025 * jnp.sum(harvest), 0.1, 1.0),
            "productivity": jnp.where(boundary, state["forecast"], state["productivity"]),
            "forecast": jnp.where(boundary, new_forecast, state["forecast"]),
            "report": jnp.where(survey, state["forecast"][other], state["report"]),
            "report_age": jnp.where(boundary, 32, jnp.where(survey, 0, jnp.minimum(state["report_age"] + 1, 32))).astype(jnp.int32),
            "last_allocation": allocation,
            "last_harvest": harvest,
        }
        utility = 1.1 * harvest - 0.04 * spent - 0.04 * repair - 0.02 * survey.astype(jnp.float32)
        rewards = jnp.tanh(utility).astype(jnp.float32)
        done_all = next_t >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
