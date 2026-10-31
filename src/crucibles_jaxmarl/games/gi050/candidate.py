import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        draws = jax.random.uniform(key, (2,), dtype=jnp.float32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "feed": (0.45 + 0.15 * draws).astype(jnp.float32),
            "culture": (0.35 + 0.1 * jnp.roll(draws, 1)).astype(jnp.float32),
            "batch": jnp.zeros((2,), dtype=jnp.float32),
            "maturity": jnp.zeros((2,), dtype=jnp.float32),
            "quality": jnp.zeros((2,), dtype=jnp.float32),
            "cloud": jnp.zeros((2,), dtype=jnp.float32),
            "toxicity": jnp.zeros((2,), dtype=jnp.float32),
            "last_yield": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        observations = jnp.zeros((2, 16), dtype=jnp.float32)
        observations = observations.at[:, 0].set(state["t"] / 32.0)
        observations = observations.at[:, 1].set(state["feed"])
        observations = observations.at[:, 2].set(jnp.roll(state["feed"], 1))
        observations = observations.at[:, 3].set(state["culture"])
        observations = observations.at[:, 4].set(jnp.roll(state["culture"], 1))
        observations = observations.at[:, 5].set(state["batch"] / 0.4)
        observations = observations.at[:, 6].set(jnp.roll(state["batch"], 1) / 0.4)
        observations = observations.at[:, 7].set(state["maturity"])
        observations = observations.at[:, 8].set(jnp.roll(state["maturity"], 1))
        observations = observations.at[:, 9].set(state["quality"])
        observations = observations.at[:, 10].set(jnp.roll(state["quality"], 1))
        observations = observations.at[:, 11].set(state["cloud"])
        observations = observations.at[:, 12].set(jnp.roll(state["cloud"], 1))
        observations = observations.at[:, 13].set(state["toxicity"])
        observations = observations.at[:, 14].set(jnp.roll(state["toxicity"], 1))
        observations = observations.at[:, 15].set(state["last_yield"] / 0.4)
        return observations

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        active = state["batch"] > 0.0
        loading = (actions == 1) & (~active) & (state["feed"] > 0.0)
        loaded = jnp.where(loading, jnp.minimum(0.4, state["feed"]), 0.0)
        released = jnp.where(actions == 2, jnp.minimum(0.4, state["culture"]), 0.0)
        harvesting = (actions == 3) & active
        harvested = jnp.where(harvesting, state["batch"], 0.0)
        scrubbing = actions == 4
        incoming = jnp.roll(state["cloud"], 1)
        progress = (0.04 + 0.7 * incoming) / (1.0 + state["toxicity"])
        grown_maturity = jnp.minimum(1.0, state["maturity"] + progress)
        grown_quality = jnp.maximum(
            0.2, state["quality"] * (1.0 - 0.04 * state["toxicity"])
        )
        product = harvested * state["maturity"] * state["quality"]
        remaining = active & (~harvesting)
        batch = jnp.where(remaining, state["batch"], loaded)
        maturity = jnp.where(remaining, grown_maturity, 0.0)
        quality = jnp.where(
            remaining, grown_quality,
            jnp.where(loading, 1.0 - 0.3 * state["toxicity"], 0.0),
        )
        feed = jnp.clip(
            state["feed"] - loaded + 0.04 + 0.04 * (actions == 0).astype(jnp.float32),
            0.0, 1.0,
        )
        culture = jnp.clip(
            state["culture"] - released + 0.02 + 0.05 * (actions == 0).astype(jnp.float32),
            0.0, 1.0,
        )
        cloud = jnp.minimum(1.0, 0.35 * state["cloud"] + released)
        toxicity = jnp.clip(
            0.92 * state["toxicity"] + 0.1 * loaded
            + 0.35 * jnp.roll(harvested, 1) - 0.3 * scrubbing.astype(jnp.float32),
            0.0, 1.0,
        )
        surplus = (
            product - 0.18 * harvested
            - 0.025 * released - 0.02 * scrubbing.astype(jnp.float32)
        )
        rewards = jnp.tanh(4.0 * surplus).astype(jnp.float32)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "feed": feed.astype(jnp.float32),
            "culture": culture.astype(jnp.float32),
            "batch": batch.astype(jnp.float32),
            "maturity": maturity.astype(jnp.float32),
            "quality": quality.astype(jnp.float32),
            "cloud": cloud.astype(jnp.float32),
            "toxicity": toxicity.astype(jnp.float32),
            "last_yield": product.astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
