"""Two-bank Trestle: original complementary crews and exact grid support rules."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 128
    num_actions = 11
    max_steps = 128

    def support(self, state):
        installed = state["beam_status"] == 2
        near_anchor = jnp.sum(jnp.abs(state["beam_pos"][:, None, :] - state["anchor_pos"][None, :, :]), axis=2) == 1
        west = installed & jnp.any(near_anchor & state["clamp_on"][None, :] & (state["anchor_pos"][:, 0] == 2)[None, :], axis=1)
        east = installed & jnp.any(near_anchor & state["clamp_on"][None, :] & (state["anchor_pos"][:, 0] == 6)[None, :], axis=1)
        adjacency = (jnp.sum(jnp.abs(state["beam_pos"][:, None, :] - state["beam_pos"][None, :, :]), axis=2) == 1) & installed[:, None] & installed[None, :]
        # Three beam nodes: three bounded rounds cover every possible simple path.
        for unused in range(3):
            west = west | jnp.any(adjacency & west[None, :], axis=1)
            east = east | jnp.any(adjacency & east[None, :], axis=1)
        return (west | east).astype(jnp.int32) + (west & east).astype(jnp.int32)

    def terrain(self, points, heavy, state):
        inside = (points[:, 0] >= 0) & (points[:, 0] <= 8) & (points[:, 1] >= 0) & (points[:, 1] <= 6)
        bank = (points[:, 0] <= 2) | (points[:, 0] >= 6)
        on_beam = jnp.all(points[:, None, :] == state["beam_pos"][None, :, :], axis=2)
        need = 1 + heavy.astype(jnp.int32)
        bridge = jnp.any(on_beam & (state["beam_status"] == 2)[None, :] & (state["beam_support"][None, :] >= need[:, None]), axis=1)
        return inside & (bank | bridge)

    def refresh(self, state):
        n = dict(state)
        carried = state["beam_status"] == 1
        n["beam_pos"] = jnp.where(carried[:, None], state["agent_pos"][0], state["beam_pos"])
        n["beam_active"] = state["beam_status"] != 3
        n["beam_support"] = self.support(n)
        n["clamp_stock"] = jnp.asarray(2, dtype=jnp.int32) - jnp.sum(state["clamp_on"].astype(jnp.int32))
        n["cargo_pos"] = jnp.where((state["cargo_owner"][0] == 2), state["agent_pos"][2][None, :], state["cargo_pos"])
        n["cargo_active"] = ~state["delivered"] & ~state["cargo_lost"]
        tiles = jnp.broadcast_to(jnp.where((jnp.arange(9) >= 3) & (jnp.arange(9) <= 5), 1, 0), (7, 9)).astype(jnp.int32)
        for b in range(3):
            x = n["beam_pos"][b, 0]
            y = n["beam_pos"][b, 1]
            value = jnp.where(n["beam_status"][b] == 2, 1 + n["beam_support"][b], tiles[y, x])
            tiles = tiles.at[y, x].set(value)
        n["tiles"] = tiles
        return n

    def reset(self, key):
        state = {
            "t": jnp.asarray(0, dtype=jnp.int32),
            "episode_done": jnp.asarray(False), "terminated": jnp.asarray(False), "truncated": jnp.asarray(False),
            "agent_pos": jnp.array([[2, 1], [1, 2], [1, 3]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_visibility": jnp.ones((3, 3), dtype=jnp.bool_),
            "tool_role": jnp.array([0, 1, 2], dtype=jnp.int32),
            "fallen": jnp.zeros((3,), dtype=jnp.bool_),
            "beam_pos": jnp.array([[2, 1], [2, 3], [2, 5]], dtype=jnp.int32),
            "beam_active": jnp.ones((3,), dtype=jnp.bool_),
            "beam_visibility": jnp.ones((3, 3), dtype=jnp.bool_),
            "beam_status": jnp.zeros((3,), dtype=jnp.int32),
            "beam_owner": jnp.full((3,), -1, dtype=jnp.int32),
            "beam_support": jnp.zeros((3,), dtype=jnp.int32),
            "anchor_pos": jnp.array([[2, 2], [6, 2], [2, 4], [6, 4]], dtype=jnp.int32),
            "anchor_active": jnp.ones((4,), dtype=jnp.bool_),
            "anchor_visibility": jnp.ones((3, 4), dtype=jnp.bool_),
            "clamp_on": jnp.zeros((4,), dtype=jnp.bool_),
            "clamp_stock": jnp.asarray(2, dtype=jnp.int32),
            "cargo_pos": jnp.array([[0, 3]], dtype=jnp.int32),
            "cargo_active": jnp.ones((1,), dtype=jnp.bool_),
            "cargo_visibility": jnp.ones((3, 1), dtype=jnp.bool_),
            "cargo_owner": jnp.full((1,), -1, dtype=jnp.int32),
            "delivered": jnp.zeros((1,), dtype=jnp.bool_),
            "cargo_lost": jnp.zeros((1,), dtype=jnp.bool_),
            "goal_pos": jnp.array([[8, 3]], dtype=jnp.int32),
            "goal_active": jnp.ones((1,), dtype=jnp.bool_),
            "goal_visibility": jnp.ones((3, 1), dtype=jnp.bool_),
            "tiles": jnp.zeros((7, 9), dtype=jnp.int32),
            "tool_event": jnp.zeros((3,), dtype=jnp.bool_),
            "collapse_event": jnp.zeros((3,), dtype=jnp.bool_),
            "fall_event": jnp.zeros((3,), dtype=jnp.bool_),
            "blocked_event": jnp.zeros((3,), dtype=jnp.bool_),
        }
        state = self.refresh(state)
        return self.observe(state), state

    def legal_actions(self, state):
        # Every live command is an intent; unreachable moves and inapplicable tools no-op.
        return jnp.broadcast_to((~state["episode_done"]) | (jnp.arange(11) == 0), (3, 11))

    def advance(self, state, actions):
        n = dict(state)
        a = jnp.where((actions >= 0) & (actions < 11), actions, 0)
        dirs = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        moving = (a >= 1) & (a <= 4)
        p = state["agent_pos"]
        q = p + dirs[jnp.where(moving, a, 0)]
        heavy = (jnp.arange(3) == 2) & (state["cargo_owner"][0] == 2)
        ground_ok = self.terrain(q, heavy, state)
        occupied = jnp.any(jnp.all(q[:, None, :] == p[None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        q = jnp.where((moving & ground_ok & ~occupied)[:, None], q, p)
        conflict = jnp.any(jnp.all(q[:, None, :] == q[None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        n["agent_pos"] = jnp.where(conflict[:, None], p, q)
        n["blocked_event"] = moving & (~ground_ok | occupied | conflict)
        beam_ids = jnp.arange(3, dtype=jnp.int32)
        target_dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        tool_direction = (a >= 6) & (a <= 9)
        target = n["agent_pos"] + target_dirs[jnp.clip(a - 6, 0, 3)]

        # Carpenter: one physical beam in hand, place or recover at a compass target.
        held = n["beam_status"] == 1
        near_stock = (jnp.sum(jnp.abs(n["beam_pos"] - n["agent_pos"][0]), axis=1) <= 1) & (n["beam_status"] == 0)
        pick_id = jnp.argmin(jnp.where(near_stock, beam_ids, 99)).astype(jnp.int32)
        pick = (a[0] == 5) & ~jnp.any(held) & jnp.any(near_stock)
        n["beam_status"] = jnp.where(pick & (beam_ids == pick_id), 1, n["beam_status"])
        n["beam_owner"] = jnp.where(pick & (beam_ids == pick_id), 0, n["beam_owner"])
        held = n["beam_status"] == 1
        held_id = jnp.argmax(held).astype(jnp.int32)
        at_target = jnp.all(n["beam_pos"] == target[0], axis=1) & (n["beam_status"] == 2)
        slot = (target[0, 0] >= 3) & (target[0, 0] <= 5) & ((target[0, 1] == 2) | (target[0, 1] == 4))
        place = tool_direction[0] & jnp.any(held) & slot & ~jnp.any(at_target)
        recover_id = jnp.argmax(at_target).astype(jnp.int32)
        recover = tool_direction[0] & ~jnp.any(held) & jnp.any(at_target)
        n["beam_pos"] = jnp.where((place & (beam_ids == held_id))[:, None], target[0], n["beam_pos"])
        n["beam_status"] = jnp.where(place & (beam_ids == held_id), 2, n["beam_status"])
        n["beam_owner"] = jnp.where(place & (beam_ids == held_id), -1, n["beam_owner"])
        n["beam_status"] = jnp.where(recover & (beam_ids == recover_id), 1, n["beam_status"])
        n["beam_owner"] = jnp.where(recover & (beam_ids == recover_id), 0, n["beam_owner"])

        # Rigger: two reusable clamps shared among four physical bank anchors.
        anchor_match = jnp.all(n["anchor_pos"] == target[1], axis=1)
        anchor_id = jnp.argmax(anchor_match).astype(jnp.int32)
        can_toggle = n["clamp_on"][anchor_id] | (state["clamp_stock"] > 0)
        toggle = tool_direction[1] & jnp.any(anchor_match) & can_toggle
        n["clamp_on"] = jnp.where(toggle & anchor_match, ~n["clamp_on"], n["clamp_on"])
        n["beam_support"] = self.support(n)
        collapse = (n["beam_status"] == 2) & (n["beam_support"] == 0)
        n["beam_status"] = jnp.where(collapse, 3, n["beam_status"])
        n["beam_owner"] = jnp.where(collapse, -1, n["beam_owner"])
        n["beam_support"] = self.support(n)

        # Hauler: heavy cargo requires a chain clamped to both banks, even when dropped.
        near_cargo = jnp.sum(jnp.abs(n["agent_pos"][2] - n["cargo_pos"][0])) <= 1
        lift = (a[2] == 5) & (n["cargo_owner"][0] < 0) & near_cargo & n["cargo_active"][0]
        owner = jnp.where(lift, 2, n["cargo_owner"][0])
        cargo = jnp.where(owner == 2, n["agent_pos"][2], n["cargo_pos"][0])
        drop_clear = ~jnp.any(jnp.all(n["agent_pos"] == target[2], axis=1))
        drop_ground = self.terrain(target[2][None, :], jnp.ones((1,), dtype=jnp.bool_), n)[0]
        drop = tool_direction[2] & (owner == 2) & drop_clear & drop_ground
        cargo = jnp.where(drop, target[2], cargo)
        owner = jnp.where(drop, -1, owner)
        now_heavy = (jnp.arange(3) == 2) & (owner == 2)
        fall = ~self.terrain(n["agent_pos"], now_heavy, n)
        cargo_safe = self.terrain(cargo[None, :], jnp.ones((1,), dtype=jnp.bool_), n)[0]
        lost = n["cargo_active"][0] & ~cargo_safe
        near_goal = jnp.sum(jnp.abs(n["agent_pos"][2] - n["goal_pos"][0])) <= 1
        deliver = (a[2] == 10) & (owner == 2) & near_goal & ~lost & ~fall[2]
        cargo = jnp.where(deliver, n["goal_pos"][0], cargo)
        owner = jnp.where(deliver | lost, -1, owner)
        n["cargo_pos"] = cargo[None, :]
        n["cargo_owner"] = jnp.asarray([owner], dtype=jnp.int32)
        n["cargo_lost"] = jnp.asarray([lost], dtype=jnp.bool_)
        n["delivered"] = jnp.asarray([deliver], dtype=jnp.bool_)
        n["fallen"] = state["fallen"] | fall
        n["tool_event"] = jnp.stack((pick | place | recover, toggle, lift | drop | deliver))
        n["collapse_event"] = collapse
        n["fall_event"] = fall
        failed = jnp.any(fall) | lost
        n["t"] = state["t"] + 1
        n["terminated"] = deliver | failed
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["episode_done"] = n["terminated"] | n["truncated"]
        n = self.refresh(n)
        amount = jnp.where(failed, -1.0, jnp.where(deliver, 1.0, 0.0))
        return n, jnp.full((3,), amount, dtype=jnp.float32)

    def observe(self, state):
        agents = jnp.concatenate((state["agent_pos"] / 16.0, state["tool_role"][:, None] / 4.0,
                                  state["fallen"][:, None].astype(jnp.float32)), axis=1).reshape((12,))
        beams = jnp.concatenate((state["beam_pos"] / 16.0, state["beam_status"][:, None] / 4.0,
                                 (state["beam_owner"][:, None] + 1) / 2.0, state["beam_support"][:, None] / 2.0), axis=1).reshape((15,))
        anchors = jnp.concatenate((state["anchor_pos"] / 16.0, state["clamp_on"][:, None].astype(jnp.float32)), axis=1).reshape((12,))
        cargo = jnp.concatenate((state["cargo_pos"][0] / 16.0, (state["cargo_owner"] + 1) / 4.0,
                                 state["delivered"].astype(jnp.float32), state["cargo_lost"].astype(jnp.float32)))
        events = jnp.concatenate((state["tool_event"], state["collapse_event"], state["fall_event"], state["blocked_event"])).astype(jnp.float32)
        public = jnp.concatenate((agents, beams, anchors, cargo, state["goal_pos"][0] / 16.0, events, state["tiles"].reshape((63,)) / 4.0))
        return jnp.concatenate((jnp.full((3, 1), state["t"] / 128.0, dtype=jnp.float32),
                                jnp.arange(3, dtype=jnp.float32)[:, None] / 4.0,
                                jnp.broadcast_to(public, (3, 121)), jnp.zeros((3, 5), dtype=jnp.float32)), axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        next_state, rewards = jax.lax.cond(state["episode_done"],
            lambda s, a: (s, jnp.zeros((3,), dtype=jnp.float32)), self.advance, state, actions)
        dones = jnp.full((3,), next_state["episode_done"], dtype=jnp.bool_)
        info = {"done_all": next_state["episode_done"], "terminated": next_state["terminated"], "truncated": next_state["truncated"]}
        return self.observe(next_state), next_state, rewards, dones, info
