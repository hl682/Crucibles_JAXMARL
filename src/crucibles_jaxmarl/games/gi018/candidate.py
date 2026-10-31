import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 2
    observation_size = 16
    num_actions = 5
    reward_low = -1.0
    reward_high = 1.0

    def reset(self, key):
        draws = jax.random.uniform(key, (3, 2), dtype=jnp.float32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "position": jnp.array([0, 1], dtype=jnp.int32),
            "reserve": jnp.full((2,), 0.35, dtype=jnp.float32),
            "mature": 0.25 + 0.30 * draws[0],
            "water": 0.50 + 0.30 * draws[1],
            "toxin": 0.15 + 0.20 * draws[2],
            "young": jnp.zeros((2,), dtype=jnp.float32),
            "old": jnp.zeros((2,), dtype=jnp.float32),
            "damaged": jnp.zeros((2,), dtype=jnp.bool_),
            "last_yield": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        here = state["position"]
        there = 1 - here
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"] / 32.0)
        obs = obs.at[:, 1].set(here.astype(jnp.float32))
        obs = obs.at[:, 2].set(state["reserve"])
        obs = obs.at[:, 3].set(jnp.take(state["mature"], here))
        obs = obs.at[:, 4].set(jnp.take(state["mature"], there))
        obs = obs.at[:, 5].set(jnp.take(state["water"], here))
        obs = obs.at[:, 6].set(jnp.take(state["water"], there))
        obs = obs.at[:, 7].set(jnp.take(state["toxin"], here))
        obs = obs.at[:, 8].set(jnp.take(state["toxin"], there))
        obs = obs.at[:, 9].set(jnp.take(state["young"], here))
        obs = obs.at[:, 10].set(jnp.take(state["old"], here))
        obs = obs.at[:, 11].set(jnp.take(state["damaged"], here).astype(jnp.float32))
        obs = obs.at[:, 12].set(jnp.take(state["damaged"], there).astype(jnp.float32))
        obs = obs.at[:, 13].set(jnp.roll(here, 1).astype(jnp.float32))
        obs = obs.at[:, 14].set(state["last_yield"] / 0.22)
        obs = obs.at[:, 15].set(jnp.roll(state["last_yield"], 1) / 0.22)
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        locations = jax.nn.one_hot(state["position"], 2, dtype=jnp.float32)
        harvesting = actions == 1
        demand_by_patch = jnp.sum(locations * harvesting[:, None].astype(jnp.float32), axis=0)
        available = jnp.take(state["mature"], state["position"])
        local_demand = jnp.take(demand_by_patch, state["position"])
        harvest = jnp.where(harvesting, jnp.minimum(0.22, available / jnp.maximum(local_demand, 1.0)), 0.0)
        harvested = jnp.sum(locations * harvest[:, None], axis=0)
        reserve_available = jnp.minimum(state["reserve"] + 0.35 * harvest, 1.0)
        sowing = jnp.where(actions == 2, jnp.minimum(reserve_available, 0.14), 0.0)
        cleaning = jnp.where(actions == 3, jnp.minimum(reserve_available, 0.10), 0.0)
        moving = (actions == 4) & (reserve_available >= 0.03)
        seeds = jnp.sum(locations * (2.0 * sowing)[:, None], axis=0)
        cleanup = jnp.sum(locations * cleaning[:, None], axis=0)
        growth_factor = jnp.where(state["damaged"], 0.25, 1.0)
        maturation = state["old"] * state["water"] * growth_factor
        toxin = jnp.clip(0.97 * state["toxin"] + 0.60 * harvested - 0.90 * cleanup, 0.0, 1.0)
        damaged = jnp.where(toxin > 0.65, True, jnp.where(toxin < 0.25, False, state["damaged"]))
        water = jnp.clip(
            state["water"]
            + 0.06
            - 0.50 * maturation
            - 0.08 * harvested
            + 0.05 * (jnp.roll(state["water"], 1) - state["water"]),
            0.0,
            1.0,
        )
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "position": jnp.where(moving, 1 - state["position"], state["position"]).astype(jnp.int32),
            "reserve": jnp.clip(
                reserve_available - sowing - cleaning - 0.03 * moving + 0.025 * (actions == 0),
                0.0,
                1.0,
            ),
            "mature": jnp.clip(state["mature"] - harvested + maturation, 0.0, 1.0),
            "water": water,
            "toxin": toxin,
            "young": seeds,
            "old": state["young"],
            "damaged": damaged,
            "last_yield": harvest,
        }
        damage_here = jnp.take(state["damaged"], state["position"]).astype(jnp.float32)
        rewards = (
            0.70 * harvest / 0.22
            - 0.15 * (sowing / 0.14 + cleaning / 0.10)
            - 0.03 * moving.astype(jnp.float32)
            - 0.08 * damage_here * harvesting.astype(jnp.float32)
        )
        rewards = jnp.clip(rewards, -1.0, 1.0).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.broadcast_to(done_all, (2,))
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
