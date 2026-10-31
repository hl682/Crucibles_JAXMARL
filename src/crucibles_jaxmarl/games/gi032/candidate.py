"""Delayed bilateral lots with optional assays and sender warranties."""
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
            "stock": jax.random.uniform(keys[0], (2,), dtype=jnp.float32, minval=0.25, maxval=0.60),
            "threshold": jax.random.uniform(keys[1], (2,), dtype=jnp.float32, minval=0.40, maxval=0.75),
            "mail_quality": jnp.zeros((2,), dtype=jnp.float32),
            "mail_present": jnp.zeros((2,), dtype=jnp.bool_),
            "mail_warranty": jnp.zeros((2,), dtype=jnp.bool_),
            "mail_age": jnp.zeros((2,), dtype=jnp.int32),
            "assayed": jnp.zeros((2,), dtype=jnp.bool_),
            "assay": jnp.zeros((2,), dtype=jnp.float32),
            "armed": jnp.zeros((2,), dtype=jnp.bool_),
            "reputation": jnp.full((2,), 0.5, dtype=jnp.float32),
            "last_received": jnp.zeros((2,), dtype=jnp.float32),
            "last_sent": jnp.zeros((2,), dtype=jnp.float32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"].astype(jnp.float32) / 32.0)
        obs = obs.at[:, 1].set(state["stock"])
        obs = obs.at[:, 2].set(state["threshold"])
        obs = obs.at[:, 3].set(state["mail_present"].astype(jnp.float32))
        obs = obs.at[:, 4].set(state["mail_age"].astype(jnp.float32) / 4.0)
        obs = obs.at[:, 5].set(state["assayed"].astype(jnp.float32))
        obs = obs.at[:, 6].set(jnp.where(state["assayed"], state["assay"], 0.0))
        obs = obs.at[:, 7].set(state["mail_warranty"].astype(jnp.float32))
        obs = obs.at[:, 8].set(state["armed"].astype(jnp.float32))
        obs = obs.at[:, 9].set(state["reputation"])
        obs = obs.at[:, 10].set(jnp.roll(state["reputation"], 1))
        obs = obs.at[:, 11].set(jnp.roll(state["mail_present"], 1).astype(jnp.float32))
        obs = obs.at[:, 12].set(jnp.roll(state["mail_age"], 1).astype(jnp.float32) / 4.0)
        obs = obs.at[:, 13].set(state["last_received"])
        obs = obs.at[:, 14].set(state["last_sent"])
        obs = obs.at[:, 15].set(jnp.arange(2, dtype=jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        refine = actions == 0
        accept = (actions == 3) & state["mail_present"]
        inspect = (actions == 2) & state["mail_present"] & ~state["assayed"]
        expire = state["mail_present"] & (state["mail_age"] >= 3) & ~accept
        keep = state["mail_present"] & ~accept & ~expire
        dispatch = (actions == 1) & (state["stock"] > 0.08) & ~jnp.roll(keep, 1)
        incoming = jnp.roll(dispatch, 1)
        new_warranty = (actions == 4) & ~state["armed"]
        satisfactory = state["mail_quality"] >= state["threshold"]
        warranty = state["mail_warranty"].astype(jnp.float32)

        receiver_payoff = accept.astype(jnp.float32) * (
            0.75 * (state["mail_quality"] - state["threshold"])
            + 0.12 * satisfactory.astype(jnp.float32)
            + 0.22 * warranty * (~satisfactory).astype(jnp.float32)
        )
        outgoing_settlement = accept.astype(jnp.float32) * (
            0.20 + 0.15 * state["mail_quality"]
            + 0.04 * warranty * satisfactory.astype(jnp.float32)
            - 0.35 * warranty * (~satisfactory).astype(jnp.float32)
        ) - expire.astype(jnp.float32) * (0.03 + 0.07 * warranty)
        sender_payoff = jnp.roll(outgoing_settlement, 1)
        reward = jnp.clip(
            receiver_payoff + sender_payoff
            - 0.025 * refine.astype(jnp.float32)
            - 0.030 * dispatch.astype(jnp.float32)
            - 0.025 * inspect.astype(jnp.float32)
            - 0.040 * new_warranty.astype(jnp.float32),
            -1.0, 1.0,
        ).astype(jnp.float32)

        sender_reviewed = jnp.roll(accept | expire, 1)
        sender_rating = jnp.roll((accept & satisfactory).astype(jnp.float32), 1)
        reputation = jnp.where(
            sender_reviewed,
            0.80 * state["reputation"] + 0.20 * sender_rating,
            state["reputation"],
        )
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "stock": jnp.where(dispatch, 0.12, jnp.clip(state["stock"] + 0.18 * refine.astype(jnp.float32), 0.0, 1.0)),
            "threshold": state["threshold"],
            "mail_quality": jnp.where(incoming, jnp.roll(state["stock"], 1), jnp.where(keep, state["mail_quality"], 0.0)),
            "mail_present": keep | incoming,
            "mail_warranty": jnp.where(incoming, jnp.roll(state["armed"], 1), keep & state["mail_warranty"]),
            "mail_age": jnp.where(incoming, 0, jnp.where(keep, jnp.minimum(state["mail_age"] + 1, 4), 0)).astype(jnp.int32),
            "assayed": jnp.where(incoming, False, keep & (state["assayed"] | inspect)),
            "assay": jnp.where(incoming, 0.0, jnp.where(keep, jnp.where(inspect, state["mail_quality"], state["assay"]), 0.0)),
            "armed": jnp.where(dispatch, False, state["armed"] | new_warranty),
            "reputation": reputation,
            "last_received": receiver_payoff,
            "last_sent": sender_payoff,
        }
        done_all = next_state["t"] >= 32
        dones = jnp.full((2,), done_all, dtype=jnp.bool_)
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, reward, dones, info
