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
            "wallet": 0.45 + 0.15 * draws[0],
            "bond": jnp.zeros((2,), dtype=jnp.float32),
            "value": 0.30 + 0.70 * draws[1],
            "efficiency": 0.80 + 0.40 * draws[2],
            "progress": jnp.array(0.0, dtype=jnp.float32),
            "signed": jnp.zeros((2,), dtype=jnp.bool_),
            "age": jnp.array(0, dtype=jnp.int32),
            "last_settlement": jnp.array(False, dtype=jnp.bool_),
            "last_breach": jnp.array(False, dtype=jnp.bool_),
            "last_action": jnp.zeros((2,), dtype=jnp.int32),
        }
        return self.observe(state), state

    def observe(self, state):
        obs = jnp.zeros((2, 16), dtype=jnp.float32)
        obs = obs.at[:, 0].set(state["t"] / 32.0)
        obs = obs.at[:, 1].set(state["wallet"])
        obs = obs.at[:, 2].set(state["bond"] / 0.60)
        obs = obs.at[:, 3].set(jnp.roll(state["bond"], 1) / 0.60)
        obs = obs.at[:, 4].set(state["progress"])
        obs = obs.at[:, 5].set(state["signed"].astype(jnp.float32))
        obs = obs.at[:, 6].set(jnp.roll(state["signed"], 1).astype(jnp.float32))
        obs = obs.at[:, 7].set(state["age"] / 8.0)
        obs = obs.at[:, 8].set(state["value"])
        obs = obs.at[:, 9].set(state["efficiency"] / 1.20)
        obs = obs.at[:, 10].set(state["last_settlement"].astype(jnp.float32))
        obs = obs.at[:, 11].set(state["last_breach"].astype(jnp.float32))
        obs = obs.at[:, 12].set(state["last_action"] / 4.0)
        obs = obs.at[:, 13].set(jnp.roll(state["last_action"], 1) / 4.0)
        obs = obs.at[:, 14].set(jnp.all(state["bond"] >= 0.20).astype(jnp.float32))
        obs = obs.at[:, 15].set((state["progress"] >= 0.75).astype(jnp.float32))
        return obs

    def legal_actions(self, state):
        return jnp.ones((2, 5), dtype=jnp.bool_)

    def raw_step(self, key, state, actions):
        wallet_start = jnp.minimum(state["wallet"] + 0.03, 1.0)
        deposit = jnp.where(
            actions == 1,
            jnp.minimum(0.24, jnp.minimum(wallet_start, 0.60 - state["bond"])),
            0.0,
        )
        deposited_bond = state["bond"] + deposit
        withdrawing = (actions == 4) & (deposited_bond > 0.0)
        forfeited = jnp.where(withdrawing, 0.50 * deposited_bond, 0.0)
        refunded = jnp.where(withdrawing, 0.50 * deposited_bond, 0.0)
        held = jnp.where(withdrawing, 0.0, deposited_bond)
        received = jnp.roll(forfeited, 1)
        wallet_available = jnp.clip(wallet_start - deposit + refunded + received, 0.0, 1.0)
        committed = jnp.all(held >= 0.20)
        work_spend = jnp.where(
            (actions == 2) & committed,
            jnp.minimum(wallet_available, 0.12),
            0.0,
        )
        progress_worked = jnp.clip(
            0.96 * state["progress"] + 2.40 * jnp.sum(work_spend * state["efficiency"]),
            0.0,
            1.0,
        )
        breach = jnp.any(withdrawing)
        settlement = jnp.all(state["signed"]) & committed & ~breach
        active = jnp.any(held > 0.0)
        project_age = jnp.where(active, state["age"] + 1, 0).astype(jnp.int32)
        expired = (project_age >= 8) & active & ~settlement
        signed = state["signed"] | ((actions == 3) & (state["progress"] >= 0.75) & committed)
        clearing = settlement | expired
        release = jnp.where(settlement, held, jnp.where(expired, 0.50 * held, 0.0))
        settlement_cash = jnp.where(settlement, 0.20 * state["value"], 0.0)
        next_state = {
            "t": state["t"] + jnp.array(1, dtype=jnp.int32),
            "wallet": jnp.clip(wallet_available - work_spend + release + settlement_cash, 0.0, 1.0),
            "bond": jnp.where(clearing, 0.0, held),
            "value": state["value"],
            "efficiency": state["efficiency"],
            "progress": jnp.where(clearing | breach, 0.0, progress_worked),
            "signed": jnp.where(clearing | breach | ~committed, False, signed),
            "age": jnp.where(clearing, 0, project_age).astype(jnp.int32),
            "last_settlement": settlement,
            "last_breach": breach,
            "last_action": actions.astype(jnp.int32),
        }
        outcome = jnp.where(settlement, 0.65 * (0.40 + 0.60 * state["value"]), 0.0)
        transfers = 0.30 * (received - forfeited)
        costs = 0.25 * work_spend + 0.03 * deposit + jnp.where(expired, 0.35 * held, 0.0)
        rewards = jnp.clip(outcome + transfers - costs, -1.0, 1.0).astype(jnp.float32)
        done_all = next_state["t"] >= 32
        dones = jnp.broadcast_to(done_all, (2,))
        info = {"done_all": done_all, "legal_mask": self.legal_actions(next_state)}
        return self.observe(next_state), next_state, rewards, dones, info
