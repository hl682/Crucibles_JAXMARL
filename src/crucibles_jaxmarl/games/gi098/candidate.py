"""Prism Commons: original deterministic shared optical workshop."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 73
    num_actions = 9
    max_steps = 96

    def reset(self, key):
        s = {
            "t": jnp.array(0, dtype=jnp.int32),
            "done": jnp.array(False, dtype=jnp.bool_),
            "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
            "map": jnp.zeros((9, 9), dtype=jnp.int32),
            "agent_pos": jnp.array([[4, 3], [1, 4], [4, 5]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_vis": jnp.ones((3, 3), dtype=jnp.bool_),
            "mirror_pos": jnp.array([[4, 4], [4, 2], [4, 6], [2, 2], [2, 6], [6, 2], [6, 6]], dtype=jnp.int32),
            "mirror_active": jnp.ones((7,), dtype=jnp.bool_),
            "mirror_vis": jnp.ones((3, 7), dtype=jnp.bool_),
            "mirror_dir": jnp.zeros((7,), dtype=jnp.int32),
            "mirror_lock": jnp.zeros((7,), dtype=jnp.int32),
            "receiver_pos": jnp.array([[4, 0], [8, 4], [4, 8]], dtype=jnp.int32),
            "receiver_active": jnp.ones((3,), dtype=jnp.bool_),
            "receiver_vis": jnp.ones((3, 3), dtype=jnp.bool_),
            "receiver_charge": jnp.zeros((3,), dtype=jnp.int32),
            "emitter_pos": jnp.array([[0, 4]], dtype=jnp.int32),
            "emitter_active": jnp.ones((1,), dtype=jnp.bool_),
            "emitter_vis": jnp.ones((3, 1), dtype=jnp.bool_),
            "energy": jnp.array([12], dtype=jnp.int32),
            "battery_pos": jnp.array([[1, 7]], dtype=jnp.int32),
            "battery_active": jnp.ones((1,), dtype=jnp.bool_),
            "battery_vis": jnp.ones((3, 1), dtype=jnp.bool_),
            "beam_pos": jnp.array([[0, 4]], dtype=jnp.int32),
            "beam_active": jnp.zeros((1,), dtype=jnp.bool_),
            "beam_vis": jnp.ones((3, 1), dtype=jnp.bool_),
            "beam_dir": jnp.zeros((1,), dtype=jnp.int32),
            "beam_ttl": jnp.zeros((1,), dtype=jnp.int32),
            "beam_owner": jnp.array([-1], dtype=jnp.int32),
            "turn_event": jnp.zeros((7,), dtype=jnp.bool_),
            "delivery_event": jnp.zeros((3,), dtype=jnp.bool_),
            "fire_event": jnp.zeros((1,), dtype=jnp.bool_),
            "pump_event": jnp.zeros((3,), dtype=jnp.bool_),
        }
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate((
            jnp.stack((s["t"], s["energy"][0], s["done"].astype(jnp.int32),
                       s["terminated"].astype(jnp.int32), s["truncated"].astype(jnp.int32))),
            s["agent_pos"].reshape(-1), s["mirror_pos"].reshape(-1),
            s["mirror_dir"], s["mirror_lock"], s["receiver_pos"].reshape(-1),
            s["receiver_charge"], s["emitter_pos"].reshape(-1), s["battery_pos"].reshape(-1),
            s["beam_pos"].reshape(-1), s["beam_active"].astype(jnp.int32),
            s["beam_dir"], s["beam_ttl"], s["beam_owner"],
            s["turn_event"].astype(jnp.int32), s["delivery_event"].astype(jnp.int32),
            s["fire_event"].astype(jnp.int32), s["pump_event"].astype(jnp.int32),
        ))
        return jnp.concatenate((jnp.arange(3, dtype=jnp.int32)[:, None],
                                jnp.broadcast_to(public, (3, 72))), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        return jnp.broadcast_to((~s["done"]) | (jnp.arange(9) == 0), (3, 9))

    def advance(self, s, actions):
        n = dict(s)
        a = jnp.clip(actions, 0, 8)
        valid = (actions >= 0) & (actions < 9) & self.legal_actions(s)[jnp.arange(3), a]
        a = jnp.where(valid, a, 0)
        moves = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)

        def move(k, pos):
            i = (k + s["t"]) % 3
            target = pos[i] + moves[a[i]]
            inside = jnp.all((target >= 0) & (target <= 8))
            safe = jnp.clip(target, 0, 8)
            occupied = jnp.any(jnp.all(pos == target, axis=1) & (jnp.arange(3) != i))
            ok = inside & ~occupied & (s["map"][safe[1], safe[0]] == 0)
            return pos.at[i].set(jnp.where(ok, target, pos[i]))

        n["agent_pos"] = jax.lax.fori_loop(0, 3, move, s["agent_pos"])
        n["mirror_lock"] = jnp.maximum(s["mirror_lock"] - 1, 0)
        n["turn_event"] = jnp.zeros((7,), dtype=jnp.bool_)
        n["delivery_event"] = jnp.zeros((3,), dtype=jnp.bool_)
        n["fire_event"] = jnp.zeros((1,), dtype=jnp.bool_)
        n["pump_event"] = jnp.zeros((3,), dtype=jnp.bool_)
        firing_actor = jnp.array(-1, dtype=jnp.int32)

        # A turn spends one shared energy unit and commits the mirror for two ticks.
        def operate(k, carry):
            q, fired_by = carry
            i = (k + s["t"]) % 3
            distance = jnp.sum(jnp.abs(q["mirror_pos"] - q["agent_pos"][i]), axis=1, dtype=jnp.int32)
            target = jnp.argmin(distance)
            unlocked = (s["mirror_lock"][target] == 0) & (q["mirror_lock"][target] == 0)
            near = distance[target] <= 1
            turn = (a[i] == 5) & near & unlocked & (q["energy"][0] > 0)
            clamp = (a[i] == 6) & near & unlocked & (q["energy"][0] > 0)
            q["mirror_dir"] = q["mirror_dir"].at[target].set(
                jnp.where(turn, (q["mirror_dir"][target] + 1) % 3, q["mirror_dir"][target]))
            q["mirror_lock"] = q["mirror_lock"].at[target].set(
                jnp.where(turn, 2, jnp.where(clamp, 5, q["mirror_lock"][target])))
            q["turn_event"] = q["turn_event"].at[target].set(q["turn_event"][target] | turn)
            q["energy"] = q["energy"] - (turn | clamp).astype(jnp.int32)
            at_source = jnp.sum(jnp.abs(q["agent_pos"][i] - q["emitter_pos"][0]), dtype=jnp.int32) <= 1
            fire = (a[i] == 7) & at_source & ~q["beam_active"][0] & (q["energy"][0] >= 3)
            q["beam_active"] = q["beam_active"] | fire
            q["beam_pos"] = jnp.where(fire, q["emitter_pos"], q["beam_pos"])
            q["beam_dir"] = jnp.where(fire, jnp.zeros((1,), dtype=jnp.int32), q["beam_dir"])
            q["beam_ttl"] = jnp.where(fire, jnp.array([40], dtype=jnp.int32), q["beam_ttl"])
            q["beam_owner"] = jnp.where(fire, jnp.array([i], dtype=jnp.int32), q["beam_owner"])
            q["energy"] = q["energy"] - 3 * fire.astype(jnp.int32)
            q["fire_event"] = q["fire_event"] | fire
            fired_by = jnp.where(fire, i, fired_by)
            at_pump = jnp.sum(jnp.abs(q["agent_pos"][i] - q["battery_pos"][0]), dtype=jnp.int32) <= 1
            pump = (a[i] == 8) & at_pump & (q["energy"][0] < 20)
            q["energy"] = jnp.minimum(q["energy"] + 2 * pump.astype(jnp.int32), 20)
            q["pump_event"] = q["pump_event"].at[i].set(pump)
            return q, fired_by

        n, firing_actor = jax.lax.fori_loop(0, 3, operate, (n, firing_actor))
        # One real cell per tick; a pulse is bounded by forty propagation ticks.
        directions = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=jnp.int32)
        proposed = n["beam_pos"][0] + directions[n["beam_dir"][0]]
        inside = jnp.all((proposed >= 0) & (proposed <= 8))
        live = n["beam_active"][0]
        visit = live & inside
        n["beam_pos"] = jnp.where(visit, proposed[None, :], n["beam_pos"])
        mirror_hit = jnp.all(n["mirror_pos"] == proposed, axis=1) & visit
        mirror_index = jnp.argmax(mirror_hit)
        orientation = n["mirror_dir"][mirror_index]
        reflected = jnp.where(orientation == 1, 3 - n["beam_dir"][0],
                              jnp.where(orientation == 2,
                                        jnp.array([1, 0, 3, 2], dtype=jnp.int32)[n["beam_dir"][0]],
                                        n["beam_dir"][0]))
        n["beam_dir"] = n["beam_dir"].at[0].set(jnp.where(jnp.any(mirror_hit), reflected, n["beam_dir"][0]))
        receiver_hit = jnp.all(n["receiver_pos"] == proposed, axis=1) & visit
        n["delivery_event"] = receiver_hit & (n["receiver_charge"] < 3)
        n["receiver_charge"] = jnp.minimum(n["receiver_charge"] + n["delivery_event"].astype(jnp.int32), 3)
        n["beam_ttl"] = jnp.where(live, jnp.maximum(n["beam_ttl"] - 1, 0), 0)
        n["beam_active"] = jnp.array([visit & ~jnp.any(receiver_hit) & (n["beam_ttl"][0] > 0)], dtype=jnp.bool_)
        success = jnp.all(n["receiver_charge"] == 3)
        n["t"] = s["t"] + 1
        n["terminated"] = success
        n["truncated"] = (n["t"] >= self.max_steps) & ~success
        n["done"] = n["terminated"] | n["truncated"]
        rewards = 4 * n["delivery_event"].astype(jnp.int32)
        rewards = rewards + ((jnp.arange(3) == n["beam_owner"][0]) & jnp.any(n["delivery_event"])).astype(jnp.int32)
        rewards = rewards - (jnp.arange(3) == firing_actor).astype(jnp.int32) + 4 * success.astype(jnp.int32)
        return n, rewards.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        proposed, rewards = self.advance(state, actions)
        next_state = jax.tree.map(lambda old, new: jnp.where(state["done"], old, new), state, proposed)
        rewards = jnp.where(state["done"], jnp.zeros((3,), dtype=jnp.float32), rewards)
        dones = jnp.full((3,), next_state["done"], dtype=jnp.bool_)
        info = {"done_all": next_state["done"], "terminated": next_state["terminated"],
                "truncated": next_state["truncated"]}
        return self.observe(next_state), next_state, rewards, dones, info
