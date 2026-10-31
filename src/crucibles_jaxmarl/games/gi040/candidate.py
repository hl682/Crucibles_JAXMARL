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
            "stock": jax.random.uniform(keys[0], (2,), minval=3.0, maxval=4.0, dtype=jnp.float32),
            "cost": jax.random.uniform(keys[1], (2,), minval=0.7, maxval=1.1, dtype=jnp.float32),
            "stage": jnp.zeros((2,), dtype=jnp.int32),
            "quality": jnp.zeros((2,), dtype=jnp.float32),
            "bond": jnp.zeros((2,), dtype=jnp.float32),
            "verified": jnp.zeros((2,), dtype=jnp.bool_),
            "shield": jnp.zeros((2,), dtype=jnp.float32),
            "trust": jnp.full((2,), 0.5, dtype=jnp.float32),
            "last_payoff": jnp.zeros((2,), dtype=jnp.float32),
            "last_delivery": jnp.zeros((2,), dtype=jnp.float32),
            "shipped": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        incoming_stage = jnp.roll(state["stage"], 1)
        incoming_verified = jnp.roll(state["verified"], 1)
        incoming_quality = jnp.roll(state["quality"], 1)
        reported_quality = jnp.where(incoming_verified, incoming_quality, -0.5)
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["stock"] / 6.0)
        obs = obs.at[:, 1].set(state["cost"] / 1.1)
        obs = obs.at[:, 2].set(state["stage"] / 3.0)
        obs = obs.at[:, 3].set(state["quality"])
        obs = obs.at[:, 4].set(state["bond"] / 0.4)
        obs = obs.at[:, 5].set(incoming_stage / 3.0)
        obs = obs.at[:, 6].set(incoming_verified.astype(jnp.float32))
        obs = obs.at[:, 7].set(reported_quality)
        obs = obs.at[:, 8].set(state["shield"] / 1.5)
        obs = obs.at[:, 9].set(state["trust"])
        obs = obs.at[:, 10].set(jnp.roll(state["trust"], 1))
        obs = obs.at[:, 11].set(state["last_payoff"])
        obs = obs.at[:, 12].set(state["last_delivery"])
        obs = obs.at[:, 13].set(state["shipped"] / 16.0)
        obs = obs.at[:, 14].set(state["t"] / 32.0)
        obs = obs.at[:, 15].set(jnp.array([-1.0, 1.0], dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        vacant = state["stage"] == 0
        mask = jnp.zeros((2, 5), dtype=jnp.bool_)
        mask = mask.at[:, 0].set(True)
        mask = mask.at[:, 1].set(vacant & (state["stock"] >= state["cost"] + 0.4))
        mask = mask.at[:, 2].set(vacant & (state["stock"] >= 0.55))
        mask = mask.at[:, 3].set((jnp.roll(state["stage"], 1) > 0) & (state["stock"] >= 0.1))
        mask = mask.at[:, 4].set((state["stock"] >= 0.25) & (state["shield"] < 1.4))
        return mask

    def raw_step(self, key, state, actions):
        vacant = state["stage"] == 0
        honest = (actions == 1) & vacant & (state["stock"] >= state["cost"] + 0.4)
        shortcut = (actions == 2) & vacant & (state["stock"] >= 0.55)
        launch = honest | shortcut
        inspect = (actions == 3) & (jnp.roll(state["stage"], 1) > 0) & (state["stock"] >= 0.1)
        insure = (actions == 4) & (state["stock"] >= 0.25) & (state["shield"] < 1.4)
        checked = state["verified"] | jnp.roll(inspect, 1)
        mature = state["stage"] == 1
        good = state["quality"] > 0.5
        detected = mature & (~good) & checked
        accepted = mature & (~detected)
        delivered_good = jnp.roll(mature & good, 1)
        delivered_bad = jnp.roll(mature & (~good) & (~checked), 1)
        delivered_rejected = jnp.roll(detected, 1)
        available_shield = jnp.minimum(1.5, state["shield"] + 0.65 * insure.astype(jnp.float32))
        damage = 0.55 * delivered_bad.astype(jnp.float32)
        absorbed = jnp.minimum(available_shield, damage)
        receiver_loss = damage - absorbed
        production_cost = honest.astype(jnp.float32) * state["cost"] + 0.15 * shortcut.astype(jnp.float32)
        stock = jnp.clip(
            state["stock"]
            + 0.55 * (actions == 0).astype(jnp.float32)
            - production_cost
            - 0.4 * launch.astype(jnp.float32)
            - 0.1 * inspect.astype(jnp.float32)
            - 0.25 * insure.astype(jnp.float32)
            + state["bond"] * accepted.astype(jnp.float32)
            + 0.35 * accepted.astype(jnp.float32)
            + 0.6 * delivered_good.astype(jnp.float32)
            + 0.2 * delivered_rejected.astype(jnp.float32)
            - receiver_loss,
            0.0,
            6.0,
        )
        issuer_payoff = mature.astype(jnp.float32) * jnp.where(
            good,
            0.45 + 0.2 * state["trust"],
            jnp.where(checked, -0.65, 0.55 + 0.1 * state["trust"]),
        )
        receiver_payoff = (
            0.35 * delivered_good.astype(jnp.float32)
            + 0.08 * delivered_rejected.astype(jnp.float32)
            - receiver_loss
        )
        effort = (
            0.12 * honest.astype(jnp.float32)
            + 0.035 * shortcut.astype(jnp.float32)
            + 0.035 * inspect.astype(jnp.float32)
            + 0.06 * insure.astype(jnp.float32)
        )
        rewards = jnp.clip(issuer_payoff + receiver_payoff - effort, -1.0, 1.0).astype(jnp.float32)
        continuing = state["stage"] > 1
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "stock": stock.astype(jnp.float32),
            "cost": state["cost"],
            "stage": jnp.where(launch, 3, jnp.maximum(state["stage"] - 1, 0)).astype(jnp.int32),
            "quality": jnp.where(launch, honest.astype(jnp.float32), jnp.where(continuing, state["quality"], 0.0)),
            "bond": jnp.where(launch, 0.4, jnp.where(continuing, state["bond"], 0.0)).astype(jnp.float32),
            "verified": jnp.where(launch, False, continuing & checked),
            "shield": jnp.maximum(available_shield - absorbed - 0.08, 0.0).astype(jnp.float32),
            "trust": jnp.clip(
                state["trust"]
                + 0.1 * (mature & good).astype(jnp.float32)
                + 0.03 * (mature & (~good) & (~checked)).astype(jnp.float32)
                - 0.3 * detected.astype(jnp.float32),
                0.0,
                1.0,
            ),
            "last_payoff": rewards,
            "last_delivery": receiver_payoff.astype(jnp.float32),
            "shipped": state["shipped"] + launch.astype(jnp.float32),
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info