"""Bridgewrights: pins, held beam ends and conservative load bearing."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    num_actions = 13
    observation_size = 96
    max_steps = 96

    def refresh(self, s):
        s = dict(s)
        ends, cells = self.geometry(s["beam_pos"], s["beam_axis"])
        s["span_pos"] = jnp.clip(cells, jnp.array([0, 0]), jnp.array([8, 6])).reshape((9, 2))
        s["span_active"] = jnp.repeat(s["beam_status"] == 2, 3)
        s["span_visible"] = jnp.broadcast_to(s["span_active"], (3, 9))
        s["span_supported"] = jnp.repeat(s["beam_supported"], 3)
        s["end_pos"] = jnp.clip(ends, jnp.array([0, 0]), jnp.array([8, 6])).reshape((6, 2))
        s["end_active"] = jnp.repeat(s["beam_status"] == 2, 2)
        s["end_visible"] = jnp.broadcast_to(s["end_active"], (3, 6))
        s["end_held"] = s["held_ends"].reshape(6)
        s["end_pinned"] = s["beam_pinned"].reshape(6)
        s["cargo_pos"] = s["agent_pos"]
        s["cargo_active"] = ~s["cargo_done"]
        s["agent_visible"] = jnp.ones((3, 3), dtype=jnp.bool_)
        s["cargo_visible"] = jnp.broadcast_to(s["cargo_active"], (3, 3))
        return s

    def reset(self, key):
        homes = jnp.array([[2, 1], [2, 5], [6, 3]], dtype=jnp.int32)
        s = {
            "t": jnp.array(0, dtype=jnp.int32),
            "episode_done": jnp.array(False), "terminated": jnp.array(False),
            "truncated": jnp.array(False),
            "agent_pos": homes, "home_pos": homes,
            "agent_active": jnp.ones(3, dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "goal_pos": jnp.array([[7, 1], [7, 5], [1, 3]], dtype=jnp.int32),
            "goal_active": jnp.ones(3, dtype=jnp.bool_),
            "goal_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "cargo_pos": homes, "cargo_done": jnp.zeros(3, dtype=jnp.bool_),
            "cargo_active": jnp.ones(3, dtype=jnp.bool_),
            "cargo_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "beam_pos": homes, "beam_status": jnp.zeros(3, dtype=jnp.int32),
            "beam_carrier": jnp.full(3, -1, dtype=jnp.int32),
            "beam_axis": jnp.zeros(3, dtype=jnp.int32),
            "beam_pinned": jnp.zeros((3, 2), dtype=jnp.bool_),
            "held_ends": jnp.zeros((3, 2), dtype=jnp.bool_),
            "beam_supported": jnp.zeros(3, dtype=jnp.bool_),
            "beam_capacity": jnp.zeros(3, dtype=jnp.int32),
            "beam_load": jnp.zeros(3, dtype=jnp.int32),
            "beam_active": jnp.ones(3, dtype=jnp.bool_),
            "beam_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "span_pos": jnp.zeros((9, 2), dtype=jnp.int32),
            "span_active": jnp.zeros(9, dtype=jnp.bool_),
            "span_visible": jnp.zeros((3, 9), dtype=jnp.bool_),
            "span_supported": jnp.zeros(9, dtype=jnp.bool_),
            "end_pos": jnp.zeros((6, 2), dtype=jnp.int32),
            "end_active": jnp.zeros(6, dtype=jnp.bool_),
            "end_visible": jnp.zeros((3, 6), dtype=jnp.bool_),
            "end_held": jnp.zeros(6, dtype=jnp.bool_),
            "end_pinned": jnp.zeros(6, dtype=jnp.bool_),
            "pin_pos": jnp.array([[2, 0], [2, 6], [6, 0], [6, 6]], dtype=jnp.int32),
            "pin_status": jnp.zeros(4, dtype=jnp.int32),
            "pin_carrier": jnp.full(4, -1, dtype=jnp.int32),
            "pin_beam": jnp.full(4, -1, dtype=jnp.int32),
            "pin_side": jnp.zeros(4, dtype=jnp.int32),
            "pin_active": jnp.ones(4, dtype=jnp.bool_),
            "pin_visible": jnp.ones((3, 4), dtype=jnp.bool_),
            "fall_event": jnp.zeros(3, dtype=jnp.bool_),
            "delivery_event": jnp.zeros(3, dtype=jnp.bool_),
            "collapse_event": jnp.zeros(3, dtype=jnp.bool_),
            "place_event": jnp.zeros(3, dtype=jnp.bool_),
            "install_event": jnp.zeros(4, dtype=jnp.bool_),
            "map": jnp.broadcast_to((jnp.arange(9) == 4).astype(jnp.int32), (7, 9)),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        fields = ("agent_pos", "goal_pos", "cargo_done", "beam_pos", "beam_status",
                  "beam_carrier", "beam_axis", "beam_pinned", "held_ends",
                  "beam_supported", "beam_capacity", "beam_load", "pin_pos",
                  "pin_status", "pin_carrier", "pin_beam", "pin_side",
                  "fall_event", "delivery_event", "collapse_event", "place_event", "install_event")
        pub = jnp.concatenate([s[f].ravel().astype(jnp.float32) / 8.0 for f in fields] +
                              [jnp.array([s["t"] / 128.0, s["episode_done"]], dtype=jnp.float32)])
        return jnp.concatenate([jnp.eye(3, dtype=jnp.float32),
                                jnp.broadcast_to(pub, (3, pub.shape[0]))], axis=1)

    def legal_actions(self, s):
        return jnp.broadcast_to((jnp.arange(13) == 0) | ~s["episode_done"], (3, 13))

    def geometry(self, pos, axis):
        direction = jnp.stack([1 - axis, axis], axis=1)
        ends = pos[:, None, :] + jnp.array([-1, 1], dtype=jnp.int32)[None, :, None] * direction[:, None, :]
        cells = pos[:, None, :] + jnp.array([-1, 0, 1], dtype=jnp.int32)[None, :, None] * direction[:, None, :]
        return ends, cells

    def raw_step(self, key, state, actions):
        s = dict(state)
        a = jnp.where((actions >= 0) & (actions < 13), actions, 0)
        bp, bs, bc, ba = s["beam_pos"], s["beam_status"], s["beam_carrier"], s["beam_axis"]
        pp, ps, pc, pb, pe = s["pin_pos"], s["pin_status"], s["pin_carrier"], s["pin_beam"], s["pin_side"]
        placed = jnp.zeros(3, dtype=jnp.bool_)
        installed = jnp.zeros(4, dtype=jnp.bool_)
        # Resource requests resolve in rotating agent order; each material has exactly one state.
        for k in range(3):
            i = (s["t"] + k) % 3
            carry_b = jnp.any((bs == 1) & (bc == i))
            near = jnp.sum(jnp.abs(bp - s["agent_pos"][i]), axis=1) <= 1
            can_pick = (a[i] == 5) & ~carry_b & jnp.any((bs == 0) & near)
            b = jnp.argmin(jnp.where((bs == 0) & near, jnp.arange(3), 9))
            bs = bs.at[b].set(jnp.where(can_pick, 1, bs[b]))
            bc = bc.at[b].set(jnp.where(can_pick, i, bc[b]))
            owned = (bs == 1) & (bc == i)
            ba = jnp.where(owned & (a[i] == 7), 1 - ba, ba)
            b = jnp.argmax(owned)
            shift = jnp.array([jnp.where(s["agent_pos"][i, 0] <= 3, 1, -1),
                               jnp.where(s["agent_pos"][i, 1] <= 3, 1, -1)], dtype=jnp.int32)
            center = s["agent_pos"][i] + shift * jnp.array([1 - ba[b], ba[b]], dtype=jnp.int32)
            direction = jnp.array([1 - ba[b], ba[b]], dtype=jnp.int32)
            e0, e1 = center - direction, center + direction
            bounds = jnp.all(e0 >= 0) & jnp.all(e1 <= jnp.array([8, 6])) & jnp.all(e1 >= 0) & jnp.all(e0 <= jnp.array([8, 6]))
            clear = ~jnp.any((bs == 2) & jnp.all(bp == center, axis=1))
            do_place = (a[i] == 6) & jnp.any(owned) & bounds & clear
            bp = bp.at[b].set(jnp.where(do_place, center, bp[b]))
            bs = bs.at[b].set(jnp.where(do_place, 2, bs[b]))
            bc = bc.at[b].set(jnp.where(do_place, -1, bc[b]))
            placed = placed.at[b].set(placed[b] | do_place)
            near_p = jnp.sum(jnp.abs(pp - s["agent_pos"][i]), axis=1) <= 1
            pick_p = (a[i] == 10) & ~jnp.any((ps == 1) & (pc == i)) & jnp.any((ps == 0) & near_p)
            p = jnp.argmin(jnp.where((ps == 0) & near_p, jnp.arange(4), 9))
            ps = ps.at[p].set(jnp.where(pick_p, 1, ps[p]))
            pc = pc.at[p].set(jnp.where(pick_p, i, pc[p]))
            ends, cells = self.geometry(bp, ba)
            occupied = jnp.any((ps[:, None, None] == 2) &
                               (pb[:, None, None] == jnp.arange(3)[None, :, None]) &
                               (pe[:, None, None] == jnp.arange(2)[None, None, :]), axis=0)
            candidate = (bs[:, None] == 2) & ~occupied & (ends[:, :, 0] != 4) & (jnp.sum(jnp.abs(ends - s["agent_pos"][i]), axis=2) <= 1)
            target = jnp.argmax(candidate.ravel())
            owned_p = (ps == 1) & (pc == i)
            p = jnp.argmax(owned_p)
            install = (a[i] == 11) & jnp.any(owned_p) & jnp.any(candidate)
            ps = ps.at[p].set(jnp.where(install, 2, ps[p]))
            pc = pc.at[p].set(jnp.where(install, -1, pc[p]))
            pb = pb.at[p].set(jnp.where(install, target // 2, pb[p]))
            pe = pe.at[p].set(jnp.where(install, target % 2, pe[p]))
            pp = pp.at[p].set(jnp.where(install, ends[target // 2, target % 2], pp[p]))
            installed = installed.at[p].set(installed[p] | install)
        ends, cells = self.geometry(bp, ba)
        pinned = jnp.any((ps[:, None, None] == 2) &
                         (pb[:, None, None] == jnp.arange(3)[None, :, None]) &
                         (pe[:, None, None] == jnp.arange(2)[None, None, :]), axis=0)
        # Hold and brace are equivalent support requests, cost a whole turn and use exact endpoint cells.
        holders = ((a == 8) | (a == 9))
        held = jnp.any(jnp.all(s["agent_pos"][:, None, None, :] == ends[None, :, :, :], axis=3) &
                       holders[:, None, None], axis=0)
        held = held & (bs[:, None] == 2)
        supported = (bs == 2) & jnp.all(pinned | held, axis=1)
        capacity = jnp.where(supported, 1 + jnp.all(pinned, axis=1).astype(jnp.int32), 0)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        proposal = jnp.clip(s["agent_pos"] + delta[jnp.where(a <= 4, a, 0)], jnp.array([0, 0]), jnp.array([8, 6]))
        covered = jnp.any(jnp.all(proposal[:, None, None, :] == cells[None, :, :, :], axis=3), axis=2)
        allowed = (proposal[:, 0] != 4) | jnp.any(covered & supported[None, :], axis=1)
        pos = jnp.where(allowed[:, None], proposal, s["agent_pos"])
        on_beam = jnp.any(jnp.all(pos[:, None, None, :] == cells[None, :, :, :], axis=3), axis=2) & (pos[:, 0] == 4)[:, None]
        load = jnp.sum(on_beam, axis=0, dtype=jnp.int32)
        collapse = (bs == 2) & (load > capacity) & (load > 0)
        bs = jnp.where(collapse, 3, bs)
        supported = supported & ~collapse
        safe = (pos[:, 0] != 4) | jnp.any(on_beam & supported[None, :], axis=1)
        fall = ~safe
        pos = jnp.where(fall[:, None], s["home_pos"], pos)
        # Falling recovers carried materials at home; installed supports stay bound to their beam.
        bp = jnp.where((bs == 1)[:, None], pos[jnp.clip(bc, 0, 2)], bp)
        pp = jnp.where((ps == 1)[:, None], pos[jnp.clip(pc, 0, 2)], pp)
        delivered = ~s["cargo_done"] & jnp.all(pos == s["goal_pos"], axis=1)
        done_cargo = s["cargo_done"] | delivered
        s["agent_pos"], s["beam_pos"], s["beam_status"], s["beam_carrier"], s["beam_axis"] = pos, bp, bs, bc, ba
        s["pin_pos"], s["pin_status"], s["pin_carrier"], s["pin_beam"], s["pin_side"] = pp, ps, pc, pb, pe
        s["beam_pinned"], s["held_ends"], s["beam_supported"] = pinned, held & ~collapse[:, None], supported
        s["beam_capacity"] = jnp.where(supported, capacity, 0)
        s["beam_load"] = jnp.sum(jnp.any(jnp.all(pos[:, None, None, :] == cells[None, :, :, :], axis=3), axis=2) & (pos[:, 0] == 4)[:, None], axis=0, dtype=jnp.int32)
        s["cargo_done"], s["fall_event"], s["delivery_event"] = done_cargo, fall, delivered
        s["collapse_event"], s["place_event"], s["install_event"] = collapse, placed, installed
        s["t"] = state["t"] + jnp.array(1, dtype=jnp.int32)
        s["terminated"] = jnp.all(done_cargo) | jnp.all(bs == 3)
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        reward = 2.0 * delivered.astype(jnp.float32) + 0.25 * jnp.sum(delivered.astype(jnp.float32)) - fall.astype(jnp.float32)
        s = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, s)
        reward = jnp.where(state["episode_done"], jnp.zeros(3, dtype=jnp.float32), reward)
        info = {"done_all": s["episode_done"], "terminated": s["terminated"], "truncated": s["truncated"]}
        return self.observe(s), s, reward.astype(jnp.float32), jnp.full((3,), s["episode_done"], dtype=jnp.bool_), info
