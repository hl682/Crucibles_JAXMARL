"""Original counterweighted bridge evacuation game; never executed by the author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 109
    num_actions = 9
    max_steps = 120

    def reset(self, key):
        x = jnp.arange(9)[None, :]
        y = jnp.arange(7)[:, None]
        terrain = jnp.where((x == 4) & (y != 1) & (y != 5), 1, 0).astype(jnp.int32)
        terrain = jnp.where((x == 4) & ((y == 1) | (y == 5)), 2, terrain).astype(jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "episode_done": jnp.array(False), "terminated": jnp.array(False), "truncated": jnp.array(False),
            "terrain": terrain,
            "agent_pos": jnp.array([[2, 1], [1, 3], [2, 5]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "role": jnp.arange(3, dtype=jnp.int32),
            "brace_energy": jnp.full((3,), 3, dtype=jnp.int32),
            "evacuated": jnp.zeros((3,), dtype=jnp.bool_),
            "victim_pos": jnp.array([[1, 1], [1, 5], [3, 3], [6, 2]], dtype=jnp.int32),
            "victim_owner": jnp.full((4,), -1, dtype=jnp.int32),
            "victim_status": jnp.zeros((4,), dtype=jnp.int32),
            "victim_active": jnp.ones((4,), dtype=jnp.bool_),
            "victim_visible": jnp.ones((3, 4), dtype=jnp.bool_),
            "weight_pos": jnp.array([[2, 2], [6, 4]], dtype=jnp.int32),
            "weight_owner": jnp.full((2,), -1, dtype=jnp.int32),
            "weight_active": jnp.ones((2,), dtype=jnp.bool_),
            "weight_visible": jnp.ones((3, 2), dtype=jnp.bool_),
            "plate_pos": jnp.array([[2, 1], [6, 5]], dtype=jnp.int32),
            "plate_load": jnp.array([1, 0], dtype=jnp.int32),
            "plate_active": jnp.ones((2,), dtype=jnp.bool_),
            "plate_visible": jnp.ones((3, 2), dtype=jnp.bool_),
            "bridge_pos": jnp.array([[4, 1], [4, 5]], dtype=jnp.int32),
            "bridge_open": jnp.array([True, False]),
            "brace_timer": jnp.zeros((2,), dtype=jnp.int32),
            "bridge_active": jnp.ones((2,), dtype=jnp.bool_),
            "bridge_visible": jnp.ones((3, 2), dtype=jnp.bool_),
            "shelter_pos": jnp.array([[8, 3]], dtype=jnp.int32),
            "shelter_active": jnp.ones((1,), dtype=jnp.bool_),
            "shelter_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "delivered": jnp.array(0, dtype=jnp.int32),
            "departed": jnp.array(0, dtype=jnp.int32),
            "bridge_changes": jnp.array(0, dtype=jnp.int32),
            "transfers": jnp.array(0, dtype=jnp.int32)
        }
        return self.observe(state), state

    def observe(self, state):
        public = jnp.concatenate((
            state["agent_pos"].reshape(-1) / 8.0, state["agent_active"].astype(jnp.float32),
            state["brace_energy"] / 3.0, state["victim_pos"].reshape(-1) / 8.0,
            (state["victim_owner"] + 1) / 3.0, state["victim_status"] / 2.0,
            state["weight_pos"].reshape(-1) / 8.0, (state["weight_owner"] + 1) / 3.0,
            state["plate_load"] / 11.0, state["bridge_open"].astype(jnp.float32),
            state["brace_timer"] / 3.0,
            jnp.stack((state["t"] / 120.0, state["delivered"] / 4.0, state["departed"] / 3.0)),
            state["terrain"].reshape(-1) / 2.0
        )).astype(jnp.float32)
        return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(public, (3, 106))), axis=1)

    def legal_actions(self, state):
        live = state["agent_active"] & ~state["episode_done"]
        return live[:, None] | (jnp.arange(9)[None, :] == 0)

    def raw_step(self, key, state, actions):
        safe = jnp.clip(actions, 0, 8)
        a = jnp.where((actions >= 0) & (actions < 9) &
                      self.legal_actions(state)[jnp.arange(3), safe], safe, 0)
        pos = state["agent_pos"]
        vo = state["victim_owner"]
        wo = state["weight_owner"]
        has_v = jnp.any(vo[None, :] == jnp.arange(3)[:, None], axis=1)
        has_w = jnp.any(wo[None, :] == jnp.arange(3)[:, None], axis=1)
        fast = (~has_v | (jnp.arange(3) == 2)) & (~has_w | (jnp.arange(3) == 1))
        can_walk = fast | (state["t"] % 2 == 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)
        for turn in range(3):
            i = (state["t"] + turn) % 3
            proposed = jnp.clip(pos[i] + delta[a[i]], jnp.array([0, 0]), jnp.array([8, 6]))
            tile = state["terrain"][proposed[1], proposed[0]]
            bridge_here = jnp.all(state["bridge_pos"] == proposed, axis=1)
            entry = (tile != 1) & ((tile != 2) | jnp.any(bridge_here & state["bridge_open"]))
            stay = jnp.all(proposed == pos[i])
            occupied = jnp.any(jnp.all(pos == proposed, axis=1) & state["agent_active"] & (jnp.arange(3) != i))
            move = state["agent_active"][i] & can_walk[i] & (entry | stay) & ~occupied
            pos = pos.at[i].set(jnp.where(move, proposed, pos[i]))
        vp = state["victim_pos"]
        vs = state["victim_status"]
        wp = state["weight_pos"]
        transfers = jnp.array(0, dtype=jnp.int32)
        for i in range(3):
            carrying_v = vo == i
            carrying_w = wo == i
            empty = ~jnp.any(carrying_v) & ~jnp.any(carrying_w)
            use_v = (a[i] == 5) & state["agent_active"][i]
            put_v = carrying_v & use_v
            at_shelter = jnp.all(pos[i] == state["shelter_pos"][0])
            vp = jnp.where(put_v[:, None], pos[i], vp)
            vs = jnp.where(put_v, jnp.where(at_shelter, 2, 0), vs)
            vo = jnp.where(put_v, -1, vo)
            available_v = (vs == 0) & (vo < 0) & jnp.all(vp == pos[i], axis=1)
            take_v = use_v & empty & jnp.any(available_v)
            chosen_v = jnp.argmax(available_v)
            pickup_v = (jnp.arange(4) == chosen_v) & take_v
            vo = jnp.where(pickup_v, i, vo)
            vs = jnp.where(pickup_v, 1, vs)
            use_w = (a[i] == 6) & state["agent_active"][i]
            put_w = carrying_w & use_w
            wp = jnp.where(put_w[:, None], pos[i], wp)
            wo = jnp.where(put_w, -1, wo)
            available_w = (wo < 0) & jnp.all(wp == pos[i], axis=1)
            take_w = use_w & empty & jnp.any(available_w)
            chosen_w = jnp.argmax(available_w)
            pickup_w = (jnp.arange(2) == chosen_w) & take_w
            wo = jnp.where(pickup_w, i, wo)
            transfers = transfers + jnp.sum(put_v | pickup_v).astype(jnp.int32) + jnp.sum(put_w | pickup_w).astype(jnp.int32)
        vp = jnp.where((vo >= 0)[:, None], pos[jnp.clip(vo, 0, 2)], vp)
        wp = jnp.where((wo >= 0)[:, None], pos[jnp.clip(wo, 0, 2)], wp)
        cargo = jnp.any(vo[None, :] == jnp.arange(3)[:, None], axis=1) | jnp.any(wo[None, :] == jnp.arange(3)[:, None], axis=1)
        leave = (a == 8) & ~cargo & jnp.all(pos == state["shelter_pos"][0], axis=1) & state["agent_active"]
        evacuated = state["evacuated"] | leave
        active = ~evacuated
        mass = 1 + jnp.sum(vo[None, :] == jnp.arange(3)[:, None], axis=1) + 2 * jnp.sum(wo[None, :] == jnp.arange(3)[:, None], axis=1)
        on_plate = jnp.all(pos[:, None, :] == state["plate_pos"][None, :, :], axis=2)
        load = jnp.sum(on_plate * (mass * active)[:, None], axis=0)
        ground_w = jnp.all(wp[:, None, :] == state["plate_pos"][None, :, :], axis=2) & (wo < 0)[:, None]
        ground_v = jnp.all(vp[:, None, :] == state["plate_pos"][None, :, :], axis=2) & (vs == 0)[:, None]
        load = (load + 2 * jnp.sum(ground_w, axis=0) + jnp.sum(ground_v, axis=0)).astype(jnp.int32)
        near_bridge = jnp.sum(jnp.abs(pos[:, None, :] - state["bridge_pos"][None, :, :]), axis=2) <= 1
        braces = (a == 7) & (state["brace_energy"] > 0) & active & jnp.any(near_bridge, axis=1)
        durations = jnp.array([3, 1, 1], dtype=jnp.int32)
        fresh_brace = jnp.max(near_bridge * (braces * durations)[:, None], axis=0).astype(jnp.int32)
        timer = jnp.maximum(jnp.maximum(state["brace_timer"] - 1, 0), fresh_brace)
        balanced_open = jnp.stack((load[0] > load[1], load[1] > load[0]))
        bridge_open = balanced_open | (timer > 0)
        energy = jnp.clip(state["brace_energy"] + ((a == 0) & active).astype(jnp.int32) - braces.astype(jnp.int32), 0, 3)
        delivered = jnp.sum(vs == 2).astype(jnp.int32)
        departed = jnp.sum(evacuated).astype(jnp.int32)
        t = state["t"] + jnp.array(1, dtype=jnp.int32)
        terminated = (delivered == 4) & jnp.all(evacuated)
        truncated = (t >= self.max_steps) & ~terminated
        nxt = dict(state)
        nxt["t"] = t
        nxt["agent_pos"] = pos
        nxt["agent_active"] = active
        nxt["evacuated"] = evacuated
        nxt["brace_energy"] = energy
        nxt["victim_pos"] = vp
        nxt["victim_owner"] = vo
        nxt["victim_status"] = vs
        nxt["weight_pos"] = wp
        nxt["weight_owner"] = wo
        nxt["plate_load"] = load
        nxt["bridge_open"] = bridge_open
        nxt["brace_timer"] = timer
        nxt["delivered"] = delivered
        nxt["departed"] = departed
        nxt["transfers"] = transfers
        nxt["bridge_changes"] = jnp.sum(bridge_open != state["bridge_open"]).astype(jnp.int32)
        nxt["terminated"] = terminated
        nxt["truncated"] = truncated
        nxt["episode_done"] = terminated | truncated
        shared = (delivered - state["delivered"]) + 0.3 * (departed - state["departed"]) - 0.005
        reward = jnp.full((3,), shared, dtype=jnp.float32) + 0.4 * leave.astype(jnp.float32)
        nxt = jax.tree.map(lambda new, old: jnp.where(state["episode_done"], old, new), nxt, state)
        reward = jnp.where(state["episode_done"], jnp.zeros((3,), dtype=jnp.float32), reward).astype(jnp.float32)
        dones = jnp.full((3,), nxt["episode_done"], dtype=jnp.bool_)
        info = {"done_all": nxt["episode_done"], "terminated": nxt["terminated"], "truncated": nxt["truncated"]}
        return self.observe(nxt), nxt, reward, dones, info
