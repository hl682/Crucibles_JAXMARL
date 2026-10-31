"""Lantern Courtyard: two wardens, two couriers, scarce shelters and rescue."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 73
    num_actions = 7
    max_steps = 96

    def refresh(self, s):
        n = dict(s)
        teams = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2, dtype=jnp.int32)
        n["agent_vis"] = (teams[:, None] == teams[None, :]) | (distance <= 3)
        flag_distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["flag_pos"][None, :, :]), axis=2, dtype=jnp.int32)
        carrier_team = teams[jnp.clip(s["flag_owner"], 0, 3)]
        friendly_carrier = (s["flag_owner"][None, :] >= 0) & (teams[:, None] == carrier_team[None, :])
        n["flag_vis"] = ((flag_distance <= 3) | friendly_carrier) & s["flag_active"][None, :]
        return n

    def reset(self, key):
        tile = jnp.zeros((7, 9), dtype=jnp.int32)
        tile = tile.at[2:5, 4].set(1)
        s = {
            "t": jnp.array(0, dtype=jnp.int32),
            "done": jnp.array(False, dtype=jnp.bool_),
            "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
            "map": tile,
            "agent_pos": jnp.array([[5, 2], [5, 4], [1, 2], [1, 4]], dtype=jnp.int32),
            "agent_active": jnp.ones((4,), dtype=jnp.bool_),
            "agent_vis": jnp.ones((4, 4), dtype=jnp.bool_),
            "frozen": jnp.zeros((4,), dtype=jnp.int32),
            "grace": jnp.zeros((4,), dtype=jnp.int32),
            "carrying": jnp.full((4,), -1, dtype=jnp.int32),
            "escaped": jnp.zeros((4,), dtype=jnp.bool_),
            "guard": jnp.zeros((4,), dtype=jnp.bool_),
            "flag_pos": jnp.array([[2, 1], [2, 5]], dtype=jnp.int32),
            "flag_active": jnp.ones((2,), dtype=jnp.bool_),
            "flag_vis": jnp.ones((4, 2), dtype=jnp.bool_),
            "flag_owner": jnp.full((2,), -1, dtype=jnp.int32),
            "shelter_pos": jnp.array([[4, 1], [4, 5]], dtype=jnp.int32),
            "shelter_active": jnp.ones((2,), dtype=jnp.bool_),
            "shelter_vis": jnp.ones((4, 2), dtype=jnp.bool_),
            "shelter_charge": jnp.full((2,), 4, dtype=jnp.int32),
            "shelter_seal": jnp.zeros((2,), dtype=jnp.int32),
            "exit_pos": jnp.array([[8, 3]], dtype=jnp.int32),
            "exit_active": jnp.ones((1,), dtype=jnp.bool_),
            "exit_vis": jnp.ones((4, 1), dtype=jnp.bool_),
            "tag_event": jnp.zeros((4,), dtype=jnp.bool_),
            "rescue_event": jnp.zeros((4,), dtype=jnp.bool_),
            "escape_event": jnp.zeros((4,), dtype=jnp.bool_),
            "seal_event": jnp.zeros((2,), dtype=jnp.bool_),
            "cover_event": jnp.zeros((4,), dtype=jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        av = s["agent_vis"]
        fv = s["flag_vis"]
        global_fields = jnp.concatenate((
            jnp.stack((s["t"], s["done"].astype(jnp.int32),
                       s["terminated"].astype(jnp.int32), s["truncated"].astype(jnp.int32))),
            s["shelter_pos"].reshape(-1), s["shelter_charge"], s["shelter_seal"],
            s["exit_pos"].reshape(-1), s["tag_event"].astype(jnp.int32),
            s["rescue_event"].astype(jnp.int32), s["escape_event"].astype(jnp.int32),
            s["seal_event"].astype(jnp.int32), s["cover_event"].astype(jnp.int32),
        ))
        fields = (
            jnp.arange(4, dtype=jnp.int32)[:, None],
            jnp.broadcast_to(global_fields, (4, 32)),
            jnp.where(av[:, :, None], s["agent_pos"][None, :, :], 0).reshape(4, 8),
            av.astype(jnp.int32),
            jnp.where(av, s["frozen"][None, :], 0),
            jnp.where(av, s["grace"][None, :], 0),
            jnp.where(av, s["carrying"][None, :], -2),
            jnp.where(av, s["escaped"][None, :], False).astype(jnp.int32),
            jnp.where(av, s["guard"][None, :], False).astype(jnp.int32),
            jnp.where(fv[:, :, None], s["flag_pos"][None, :, :], 0).reshape(4, 4),
            fv.astype(jnp.int32),
            jnp.where(fv, s["flag_owner"][None, :], -3),
        )
        return jnp.concatenate(fields, axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        able = (s["frozen"] == 0) & ~s["escaped"] & ~s["done"]
        return able[:, None] | (jnp.arange(7)[None, :] == 0)

    def advance(self, s, actions):
        n = dict(s)
        clipped = jnp.clip(actions, 0, 6)
        legal = (actions >= 0) & (actions < 7) & self.legal_actions(s)[jnp.arange(4), clipped]
        a = jnp.where(legal, clipped, 0)
        able = (s["frozen"] == 0) & ~s["escaped"]
        n["frozen"] = jnp.maximum(s["frozen"] - 1, 0)
        n["grace"] = jnp.maximum(s["grace"] - 1, 0)
        n["guard"] = jnp.zeros((4,), dtype=jnp.bool_)
        n["shelter_seal"] = jnp.maximum(s["shelter_seal"] - 1, 0)
        regenerate = (((s["t"] + 1) % 6) == 0) & (s["shelter_seal"] == 0)
        n["shelter_charge"] = jnp.minimum(s["shelter_charge"] + regenerate.astype(jnp.int32), 4)
        n["tag_event"] = jnp.zeros((4,), dtype=jnp.bool_)
        n["rescue_event"] = jnp.zeros((4,), dtype=jnp.bool_)
        n["escape_event"] = jnp.zeros((4,), dtype=jnp.bool_)
        n["seal_event"] = jnp.zeros((2,), dtype=jnp.bool_)
        n["cover_event"] = jnp.zeros((4,), dtype=jnp.bool_)
        moves = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0]], dtype=jnp.int32)

        def move(k, pos):
            i = (k + s["t"]) % 4
            target = pos[i] + moves[a[i]]
            bounds = jnp.all(target >= 0) & (target[0] <= 8) & (target[1] <= 6)
            safe = jnp.minimum(jnp.maximum(target, 0), jnp.array([8, 6], dtype=jnp.int32))
            occupied = jnp.any(jnp.all(pos == target, axis=1) & ~s["escaped"] & (jnp.arange(4) != i))
            ok = able[i] & bounds & ~occupied & (s["map"][safe[1], safe[0]] == 0)
            return pos.at[i].set(jnp.where(ok, target, pos[i]))

        n["agent_pos"] = jax.lax.fori_loop(0, 4, move, s["agent_pos"])
        # Wardens seal first. A courier must spend a remaining charge to shelter this tick.
        def seal(k, q):
            i = (k + s["t"]) % 2
            distance = jnp.sum(jnp.abs(q["shelter_pos"] - q["agent_pos"][i]), axis=1, dtype=jnp.int32)
            target = jnp.argmin(distance)
            use = (a[i] == 6) & (distance[target] <= 1)
            q["shelter_seal"] = q["shelter_seal"].at[target].set(jnp.where(use, 3, q["shelter_seal"][target]))
            q["seal_event"] = q["seal_event"].at[target].set(q["seal_event"][target] | use)
            return q

        n = jax.lax.fori_loop(0, 2, seal, n)

        def cover(k, q):
            i = 2 + (k + s["t"]) % 2
            exact = jnp.all(q["shelter_pos"] == q["agent_pos"][i], axis=1)
            target = jnp.argmax(exact)
            use = able[i] & (a[i] == 6) & jnp.any(exact) & (q["shelter_seal"][target] == 0) & (q["shelter_charge"][target] > 0)
            q["shelter_charge"] = q["shelter_charge"].at[target].add(-use.astype(jnp.int32))
            q["guard"] = q["guard"].at[i].set(use)
            q["cover_event"] = q["cover_event"].at[i].set(use)
            return q

        n = jax.lax.fori_loop(0, 2, cover, n)
        rescue_credit = jnp.zeros((4,), dtype=jnp.int32)

        # Rescue takes precedence over pickup or escape, and grants two-tick grace.
        def courier(k, carry):
            q, credit = carry
            i = 2 + (k + s["t"]) % 2
            mate = 5 - i
            adjacent = jnp.sum(jnp.abs(q["agent_pos"][i] - q["agent_pos"][mate]), dtype=jnp.int32) <= 1
            act = able[i] & (a[i] == 5)
            rescue = act & adjacent & (q["frozen"][mate] > 0) & ~q["escaped"][mate]
            q["frozen"] = q["frozen"].at[mate].set(jnp.where(rescue, 0, q["frozen"][mate]))
            q["grace"] = q["grace"].at[mate].set(jnp.where(rescue, 2, q["grace"][mate]))
            q["rescue_event"] = q["rescue_event"].at[mate].set(q["rescue_event"][mate] | rescue)
            credit = credit.at[i].add(rescue.astype(jnp.int32))
            at_exit = jnp.all(q["agent_pos"][i] == q["exit_pos"][0])
            escape = act & ~rescue & at_exit & (q["carrying"][i] >= 0)
            carried = jnp.maximum(q["carrying"][i], 0)
            q["flag_owner"] = q["flag_owner"].at[carried].set(jnp.where(escape, -2, q["flag_owner"][carried]))
            q["flag_active"] = q["flag_active"].at[carried].set(q["flag_active"][carried] & ~escape)
            q["escaped"] = q["escaped"].at[i].set(q["escaped"][i] | escape)
            q["escape_event"] = q["escape_event"].at[i].set(escape)
            q["carrying"] = q["carrying"].at[i].set(jnp.where(escape, -1, q["carrying"][i]))
            available = q["flag_active"] & (q["flag_owner"] == -1) & jnp.all(q["flag_pos"] == q["agent_pos"][i], axis=1)
            pickup = act & ~rescue & ~escape & (q["carrying"][i] < 0) & jnp.any(available)
            flag = jnp.argmax(available)
            q["flag_owner"] = q["flag_owner"].at[flag].set(jnp.where(pickup, i, q["flag_owner"][flag]))
            q["carrying"] = q["carrying"].at[i].set(jnp.where(pickup, flag, q["carrying"][i]))
            return q, credit

        n, rescue_credit = jax.lax.fori_loop(0, 2, courier, (n, rescue_credit))
        tag_credit = jnp.zeros((2,), dtype=jnp.int32)

        def tag(k, carry):
            q, credit = carry
            i = (k + s["t"]) % 2
            distance = jnp.sum(jnp.abs(q["agent_pos"][2:] - q["agent_pos"][i]), axis=1, dtype=jnp.int32)
            candidates = (distance <= 1) & (q["frozen"][2:] == 0) & (q["grace"][2:] == 0) & ~q["guard"][2:] & ~q["escaped"][2:]
            target = 2 + jnp.argmin(jnp.where(candidates, distance, 99))
            hit = (a[i] == 5) & jnp.any(candidates)
            q["frozen"] = q["frozen"].at[target].set(jnp.where(hit, 6, q["frozen"][target]))
            q["tag_event"] = q["tag_event"].at[target].set(q["tag_event"][target] | hit)
            credit = credit.at[i].add(hit.astype(jnp.int32))
            dropped = (q["flag_owner"] == target) & hit
            q["flag_pos"] = jnp.where(dropped[:, None], q["agent_pos"][target], q["flag_pos"])
            q["flag_owner"] = jnp.where(dropped, -1, q["flag_owner"])
            q["carrying"] = q["carrying"].at[target].set(jnp.where(hit, -1, q["carrying"][target]))
            return q, credit

        n, tag_credit = jax.lax.fori_loop(0, 2, tag, (n, tag_credit))
        n["flag_pos"] = jnp.where((n["flag_owner"] >= 0)[:, None],
                                  n["agent_pos"][jnp.clip(n["flag_owner"], 0, 3)], n["flag_pos"])
        all_escaped = jnp.all(n["escaped"][2:])
        caught = jnp.all(n["escaped"][2:] | (n["frozen"][2:] > 0)) & ~all_escaped
        n["t"] = s["t"] + 1
        n["terminated"] = all_escaped | caught
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["done"] = n["terminated"] | n["truncated"]
        escapes = jnp.sum(n["escape_event"].astype(jnp.int32), dtype=jnp.int32)
        warden_reward = 2 * tag_credit - 2 * escapes + 3 * caught.astype(jnp.int32) - 3 * all_escaped.astype(jnp.int32)
        courier_reward = (-2 * n["tag_event"][2:].astype(jnp.int32)
                          + n["rescue_event"][2:].astype(jnp.int32) + rescue_credit[2:]
                          + 3 * n["escape_event"][2:].astype(jnp.int32) + 2 * escapes
                          + 3 * all_escaped.astype(jnp.int32) - 3 * caught.astype(jnp.int32))
        n = self.refresh(n)
        return n, jnp.concatenate((warden_reward, courier_reward)).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        proposed, rewards = self.advance(state, actions)
        next_state = jax.tree.map(lambda old, new: jnp.where(state["done"], old, new), state, proposed)
        rewards = jnp.where(state["done"], jnp.zeros((4,), dtype=jnp.float32), rewards)
        dones = jnp.full((4,), next_state["done"], dtype=jnp.bool_)
        info = {"done_all": next_state["done"], "terminated": next_state["terminated"],
                "truncated": next_state["truncated"]}
        return self.observe(next_state), next_state, rewards, dones, info
