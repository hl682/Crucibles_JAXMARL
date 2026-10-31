"""Game 00077: Veinbound. Source authored without importing or executing it."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 2196
    num_actions = 15
    max_steps = 240

    def directions(self):
        return jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)

    def clearEvents(self, state):
        n = dict(state)
        n["event_count"] = jnp.array(0, dtype=jnp.int32)
        n["event_active"] = jnp.zeros((40,), dtype=jnp.bool_)
        for name in ("event_kind", "event_value"):
            n[name] = jnp.zeros((40,), dtype=jnp.int32)
        for name in ("event_actor", "event_object", "event_tick"):
            n[name] = jnp.full((40,), -1, dtype=jnp.int32)
        for name in ("event_from_xy", "event_to_xy"):
            n[name] = jnp.full((40, 2), -1, dtype=jnp.int32)
        return n

    def emit(self, state, active, kind, actor, obj, start, end, value):
        n = dict(state)
        k = jnp.clip(n["event_count"], 0, 39)
        n["event_active"] = n["event_active"].at[k].set(n["event_active"][k] | active)
        for name, datum in (("event_kind", kind), ("event_actor", actor), ("event_object", obj),
                            ("event_from_xy", start), ("event_to_xy", end),
                            ("event_value", value), ("event_tick", n["t"])):
            datum = jnp.asarray(datum, dtype=jnp.int32)
            n[name] = n[name].at[k].set(jnp.where(active, datum, n[name][k]))
        n["event_count"] = n["event_count"] + active.astype(jnp.int32)
        return n

    def passable(self, state, xy, team):
        xx = xy[..., 0]
        yy = xy[..., 1]
        in_bounds = (xx >= 0) & (xx < 13) & (yy >= 0) & (yy < 11)
        tx = jnp.clip(xx, 0, 12)
        ty = jnp.clip(yy, 0, 10)
        terrain = state["terrain"][ty, tx]
        owner = state["base_owner"][ty, tx]
        return in_bounds & (terrain != 1) & ((owner < 0) | (owner == team))

    def resolveMoves(self, positions, targets, moving, valid):
        candidates = moving & valid
        same = jnp.all(targets[:, None, :] == targets[None, :, :], axis=2)
        accepted = candidates & (jnp.sum(same & candidates[None, :], axis=1, dtype=jnp.int32) == 1)
        occupies = jnp.all(targets[:, None, :] == positions[None, :, :], axis=2)
        for iteration in range(4):
            accepted = accepted & ~jnp.any(occupies & ~accepted[None, :], axis=1)
        return accepted

    def present(self, state):
        n = dict(state)
        xx = jnp.broadcast_to(jnp.arange(13, dtype=jnp.int32)[None, :], (11, 13))
        yy = jnp.broadcast_to(jnp.arange(11, dtype=jnp.int32)[:, None], (11, 13))
        grid = jnp.stack((xx, yy), axis=2)
        n["map_visibility"] = jnp.max(jnp.abs(grid[None, :, :, :] -
                                              n["agent_xy"][:, None, None, :]), axis=3) <= 4
        n["agent_active"] = jnp.ones((4,), dtype=jnp.bool_)
        n["agent_visibility"] = jnp.max(jnp.abs(n["agent_xy"][None, :, :] -
                                                n["agent_xy"][:, None, :]), axis=2) <= 4
        n["agent_facing_vector"] = self.directions()[n["agent_facing"]]
        n["geode_active"] = n["geode_status"] != 3
        exposed = (n["geode_status"] == 1) | (n["geode_status"] == 2)
        n["geode_visibility"] = exposed[None, :] & (
            jnp.max(jnp.abs(n["geode_xy"][None, :, :] - n["agent_xy"][:, None, :]), axis=2) <= 4)
        n["geode_number"] = jnp.arange(8, dtype=jnp.int32)
        n["bank_active"] = jnp.ones((2,), dtype=jnp.bool_)
        n["bank_visibility"] = jnp.ones((4, 2), dtype=jnp.bool_)
        n["bank_team"] = jnp.arange(2, dtype=jnp.int32)
        n["soil_xy"] = grid.reshape((143, 2))
        n["soil_active"] = (n["terrain"] == 0).reshape((143,))
        n["soil_visibility"] = n["map_visibility"].reshape((4, 143)) & n["soil_active"][None, :]
        n["soil_depth"] = n["dig_depth"].reshape((143,))
        n["record_xy"] = n["scan_record_xy"].reshape((24, 2))
        n["record_active"] = n["scan_record_valid"].reshape((24,))
        n["record_holder"] = jnp.repeat(jnp.arange(4, dtype=jnp.int32), 6)
        n["record_visibility"] = ((jnp.arange(4, dtype=jnp.int32)[:, None] ==
                                    n["record_holder"][None, :]) & n["record_active"][None, :])
        n["record_radius"] = n["scan_record_radius"].reshape((24,))
        n["record_counts"] = n["scan_record_counts"].reshape((24, 2))
        n["record_tick"] = n["scan_record_tick"].reshape((24,))
        n["record_source"] = n["scan_record_source"].reshape((24,))
        n["record_age"] = jnp.where(n["record_active"], n["t"] - n["record_tick"], 0)
        ef = n["event_from_xy"]
        et = n["event_to_xy"]
        from_bounds = (ef[:, 0] >= 0) & (ef[:, 0] < 13) & (ef[:, 1] >= 0) & (ef[:, 1] < 11)
        to_bounds = (et[:, 0] >= 0) & (et[:, 0] < 13) & (et[:, 1] >= 0) & (et[:, 1] < 11)
        fv = from_bounds[None, :] & (jnp.max(jnp.abs(ef[None, :, :] - n["agent_xy"][:, None, :]), axis=2) <= 4)
        tv = to_bounds[None, :] & (jnp.max(jnp.abs(et[None, :, :] - n["agent_xy"][:, None, :]), axis=2) <= 4)
        actor_index = jnp.clip(n["event_actor"], 0, 3)
        av = n["agent_visibility"][:, actor_index] & (n["event_actor"] >= 0)[None, :]
        kinds = n["event_kind"]
        global_event = (kinds == 7) | (kinds == 14)
        ev = (fv | tv | global_event[None, :]) & n["event_active"][None, :]
        ev = ev & jnp.where((kinds == 8)[None, :], av, True)
        ev = ev & jnp.where((kinds == 9)[None, :], fv & tv, True)
        gem_object = (kinds == 4) | (kinds == 5) | (kinds == 6) | (kinds == 7) | (kinds == 12)
        agent_object = (kinds == 9) | (kinds == 11)
        gv = n["geode_visibility"][:, jnp.clip(n["event_object"], 0, 7)]
        ov = n["agent_visibility"][:, jnp.clip(n["event_object"], 0, 3)]
        object_visible = jnp.where(gem_object[None, :], gv,
                                   jnp.where(agent_object[None, :], ov, False))
        participants = ((jnp.arange(4, dtype=jnp.int32)[:, None] == n["event_actor"][None, :]) |
                        (jnp.arange(4, dtype=jnp.int32)[:, None] == n["event_object"][None, :]))
        value_visible = ev & jnp.where((kinds == 9)[None, :], participants, True)
        value_visible = value_visible & jnp.where((kinds == 4)[None, :], gv, True)
        n["event_visibility"] = ev
        n["event_from_visibility"] = fv & ev
        n["event_to_visibility"] = tv & ev
        n["event_actor_visibility"] = av & ev
        n["event_object_visibility"] = object_visible & (n["event_object"] >= 0)[None, :] & ev
        n["event_value_visibility"] = value_visible
        return n

    def reset(self, key):
        xx = jnp.broadcast_to(jnp.arange(13, dtype=jnp.int32)[None, :], (11, 13))
        yy = jnp.broadcast_to(jnp.arange(11, dtype=jnp.int32)[:, None], (11, 13))
        grid = jnp.stack((xx, yy), axis=2)
        rocks = jnp.array([[4, 3], [4, 7], [8, 3], [8, 7], [6, 5]], dtype=jnp.int32)
        wall = (xx == 0) | (xx == 12) | (yy == 0) | (yy == 10)
        wall = wall | jnp.any(jnp.all(grid[:, :, None, :] == rocks[None, None, :, :], axis=3), axis=2)
        bank_xy = jnp.array([[1, 5], [11, 5]], dtype=jnp.int32)
        base0 = (jnp.abs(xx - 1) + jnp.abs(yy - 5) <= 1) & ~wall
        base1 = (jnp.abs(xx - 11) + jnp.abs(yy - 5) <= 1) & ~wall
        bank0 = (xx == 1) & (yy == 5)
        bank1 = (xx == 11) & (yy == 5)
        terrain = jnp.where(wall, 1, jnp.where(bank0, 2, jnp.where(bank1, 3,
                            jnp.where(base0 | base1, 4, 0)))).astype(jnp.int32)
        owner = jnp.where(base0, 0, jnp.where(base1, 1, -1)).astype(jnp.int32)
        left_pairs = jnp.array([
            [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9],
            [4, 1], [4, 2], [4, 4], [4, 5], [4, 6], [4, 8], [4, 9],
            [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9]
        ], dtype=jnp.int32)
        keys = jax.random.split(key, 3)
        selected = jax.random.permutation(keys[0], 25)[:4]
        left = left_pairs[selected]
        right = jnp.stack((12 - left[:, 0], left[:, 1]), axis=1)
        left_colours = jax.random.permutation(keys[1], jnp.array([0, 0, 1, 1], dtype=jnp.int32))
        ids = jax.random.permutation(keys[2], 8)
        positions = jnp.concatenate((left, right), axis=0)[ids]
        colours = jnp.concatenate((left_colours, 1 - left_colours), axis=0)[ids]
        s = {
            "terrain": terrain, "base_owner": owner, "bank_xy": bank_xy,
            "agent_team": jnp.array([0, 0, 1, 1], dtype=jnp.int32),
            "agent_role": jnp.array([0, 1, 0, 1], dtype=jnp.int32),
            "agent_xy": jnp.array([[1, 4], [1, 6], [11, 4], [11, 6]], dtype=jnp.int32),
            "agent_facing": jnp.array([1, 1, 3, 3], dtype=jnp.int32),
            "agent_charge": jnp.full((4,), 8, dtype=jnp.int32),
            "agent_stun": jnp.zeros((4,), dtype=jnp.int32),
            "agent_shove_cooldown": jnp.zeros((4,), dtype=jnp.int32),
            "agent_cargo": jnp.full((4,), -1, dtype=jnp.int32),
            "dig_depth": jnp.zeros((11, 13), dtype=jnp.int32),
            "geode_xy": positions.astype(jnp.int32), "geode_colour": colours.astype(jnp.int32),
            "geode_status": jnp.zeros((8,), dtype=jnp.int32),
            "geode_carrier": jnp.full((8,), -1, dtype=jnp.int32),
            "team_scores": jnp.zeros((2,), dtype=jnp.int32),
            "scan_record_valid": jnp.zeros((4, 6), dtype=jnp.bool_),
            "scan_record_xy": jnp.full((4, 6, 2), -1, dtype=jnp.int32),
            "scan_record_radius": jnp.zeros((4, 6), dtype=jnp.int32),
            "scan_record_counts": jnp.zeros((4, 6, 2), dtype=jnp.int32),
            "scan_record_tick": jnp.full((4, 6), -1, dtype=jnp.int32),
            "scan_record_source": jnp.full((4, 6), -1, dtype=jnp.int32),
            "t": jnp.array(0, dtype=jnp.int32), "done": jnp.array(False, dtype=jnp.bool_),
            "terminated": jnp.array(False, dtype=jnp.bool_), "truncated": jnp.array(False, dtype=jnp.bool_),
            "terminal_reason": jnp.array(0, dtype=jnp.int32), "winner": jnp.array(-2, dtype=jnp.int32),
        }
        s = self.present(self.clearEvents(s))
        return self.observe(s), s

    def legal_actions(self, state):
        p = state["agent_xy"]
        team = state["agent_team"]
        front = p + self.directions()[state["agent_facing"]]
        fx = jnp.clip(front[:, 0], 0, 12)
        fy = jnp.clip(front[:, 1], 0, 10)
        in_bounds = (front[:, 0] >= 0) & (front[:, 0] < 13) & (front[:, 1] >= 0) & (front[:, 1] < 11)
        masks = jnp.zeros((4, 15), dtype=jnp.bool_).at[:, 0].set(True)
        for direction in range(4):
            masks = masks.at[:, direction + 1].set(self.passable(state, p + self.directions()[direction], team))
        masks = masks.at[:, 5].set(True)
        dig = in_bounds & (state["terrain"][fy, fx] == 0) & (state["dig_depth"][fy, fx] < 2) & (state["agent_charge"] >= 1)
        masks = masks.at[:, 6].set(dig)
        occupied_front = jnp.any(jnp.all(front[:, None, :] == p[None, :, :], axis=2), axis=1)
        drop_attempt = self.passable(state, front, team)
        pick_attempt = in_bounds & (state["terrain"][fy, fx] != 1)
        masks = masks.at[:, 7].set(jnp.where(state["agent_cargo"] >= 0, drop_attempt, pick_attempt))
        masks = masks.at[:, 8].set((state["agent_cargo"] >= 0) & jnp.all(p == state["bank_xy"][team], axis=1))
        masks = masks.at[:, 9].set((state["agent_role"] == 0) & (state["agent_charge"] >= 1))
        masks = masks.at[:, 10].set((state["agent_role"] == 0) & (state["agent_charge"] >= 2))
        masks = masks.at[:, 11].set(True)
        masks = masks.at[:, 12].set(True)
        destination = front + self.directions()[state["agent_facing"]]
        neutral_front = in_bounds & (state["terrain"][fy, fx] != 1) & (state["base_owner"][fy, fx] < 0)
        shove_attempt = neutral_front & self.passable(state, destination, 1 - team)
        shove_attempt = shove_attempt & (state["agent_charge"] >= 2) & (state["agent_shove_cooldown"] == 0)
        masks = masks.at[:, 13].set(shove_attempt)
        masks = masks.at[:, 14].set(state["base_owner"][p[:, 1], p[:, 0]] == team)
        idle = jnp.zeros((4, 15), dtype=jnp.bool_).at[:, 0].set(True)
        return jnp.where(state["done"] | (state["agent_stun"] > 0)[:, None], idle, masks)

    def observe(self, state):
        static_map = jax.nn.one_hot(state["terrain"], 5, dtype=jnp.float32).reshape((-1,))
        px = jnp.tile(jnp.arange(-4, 5, dtype=jnp.int32), 9).reshape((9, 9))
        py = jnp.repeat(jnp.arange(-4, 5, dtype=jnp.int32), 9).reshape((9, 9))
        offsets = jnp.stack((px, py), axis=2)
        rows = []
        for agent in range(4):
            origin = state["agent_xy"][agent]
            team = state["agent_team"][agent]
            cells = offsets + origin
            x = cells[:, :, 0]
            y = cells[:, :, 1]
            bounds = (x >= 0) & (x < 13) & (y >= 0) & (y < 11)
            cx = jnp.clip(x, 0, 12)
            cy = jnp.clip(y, 0, 10)
            visible = bounds & state["map_visibility"][agent, cy, cx]
            terrain = state["terrain"][cy, cx]
            occupants = jnp.all(cells[:, :, None, :] == state["agent_xy"][None, None, :, :], axis=3)
            own = jnp.any(occupants & (state["agent_team"] == team)[None, None, :], axis=2)
            rival = jnp.any(occupants & (state["agent_team"] != team)[None, None, :], axis=2)
            located = jnp.all(cells[:, :, None, :] == state["geode_xy"][None, None, :, :], axis=3)
            surface = located & (state["geode_status"] == 1)[None, None, :]
            carried = located & (state["geode_status"] == 2)[None, None, :]
            amber = state["geode_colour"] == 0
            teal = state["geode_colour"] == 1
            surface_a = jnp.sum(surface & amber[None, None, :], axis=2, dtype=jnp.int32) / 8.0
            surface_t = jnp.sum(surface & teal[None, None, :], axis=2, dtype=jnp.int32) / 8.0
            carried_a = jnp.any(carried & amber[None, None, :], axis=2)
            carried_t = jnp.any(carried & teal[None, None, :], axis=2)
            event_cells = jnp.all(cells[:, :, None, :] == state["event_to_xy"][None, None, :, :], axis=3)
            event_cells = event_cells & state["event_visibility"][agent][None, None, :]
            dig_event = jnp.any(event_cells & (state["event_kind"] == 3)[None, None, :], axis=2)
            scan_event = jnp.any(event_cells & (state["event_kind"] == 8)[None, None, :], axis=2)
            shove_event = jnp.any(event_cells & (state["event_kind"] == 11)[None, None, :], axis=2)
            board = jnp.stack((visible, terrain == 1, terrain == 0, terrain == 2 + team,
                               terrain == 3 - team, terrain == 4, state["dig_depth"][cy, cx] / 2.0,
                               own, rival, surface_a, surface_t, carried_a, carried_t,
                               dig_event, scan_event, shove_event), axis=2).astype(jnp.float32)
            board = jnp.where(visible[:, :, None], board, 0.0).reshape((-1,))
            av = state["agent_visibility"][agent]
            delta = (state["agent_xy"] - origin).astype(jnp.float32) / 4.0
            facing = self.directions()[state["agent_facing"]].astype(jnp.float32)
            dynamic = jnp.concatenate((delta, facing, state["agent_charge"][:, None] / 8.0,
                                       state["agent_stun"][:, None], state["agent_shove_cooldown"][:, None] / 2.0), axis=1)
            dynamic = jnp.where(av[:, None], dynamic, 0.0)
            agents = jnp.concatenate((av.astype(jnp.float32)[:, None],
                                      state["agent_team"].astype(jnp.float32)[:, None],
                                      state["agent_role"].astype(jnp.float32)[:, None], dynamic), axis=1).reshape((-1,))
            gv = state["geode_visibility"][agent]
            gdelta = (state["geode_xy"] - origin).astype(jnp.float32) / 4.0
            is_carried = state["geode_status"] == 2
            carrier = jnp.where(is_carried, state["geode_carrier"] / 3.0, 0.0)
            geodes = jnp.concatenate((gv.astype(jnp.float32)[:, None],
                                      jax.nn.one_hot(state["geode_colour"], 2, dtype=jnp.float32),
                                      gdelta, is_carried.astype(jnp.float32)[:, None],
                                      carrier[:, None], jnp.arange(8, dtype=jnp.float32)[:, None] / 7.0), axis=1)
            geodes = jnp.where(gv[:, None], geodes, 0.0).reshape((-1,))
            valid = state["scan_record_valid"][agent]
            records = jnp.concatenate((valid.astype(jnp.float32)[:, None],
                                       state["scan_record_xy"][agent].astype(jnp.float32) / jnp.array([12.0, 10.0], dtype=jnp.float32),
                                       state["scan_record_radius"][agent, :, None] / 4.0,
                                       state["scan_record_counts"][agent].astype(jnp.float32) / 4.0,
                                       state["scan_record_tick"][agent, :, None] / 240.0,
                                       state["scan_record_source"][agent, :, None] / 3.0), axis=1)
            records = jnp.where(valid[:, None], records, 0.0).reshape((-1,))
            cargo = state["agent_cargo"][agent]
            cargo_colour = state["geode_colour"][jnp.clip(cargo, 0, 7)]
            cargo_one_hot = jax.nn.one_hot(cargo_colour, 2, dtype=jnp.float32) * (cargo >= 0).astype(jnp.float32)
            winner_index = jnp.where(state["winner"] == -2, 0, jnp.where(state["winner"] == 0, 1,
                            jnp.where(state["winner"] == 1, 2, jnp.where(state["winner"] == -1, 3, 4))))
            global_row = jnp.concatenate((
                jax.nn.one_hot(jnp.array(agent, dtype=jnp.int32), 4, dtype=jnp.float32),
                jax.nn.one_hot(state["agent_role"][agent], 2, dtype=jnp.float32),
                jax.nn.one_hot(team, 2, dtype=jnp.float32),
                jnp.array([state["agent_charge"][agent] / 8.0, state["agent_stun"][agent],
                           state["agent_shove_cooldown"][agent] / 2.0, (cargo < 0).astype(jnp.float32)], dtype=jnp.float32),
                cargo_one_hot,
                jnp.array([jnp.where(cargo >= 0, cargo / 7.0, 0.0), state["t"] / 240.0,
                           state["team_scores"][team] / 16.0, state["team_scores"][1 - team] / 16.0,
                           jnp.sum(state["geode_status"] == 3, dtype=jnp.int32) / 8.0,
                           state["done"].astype(jnp.float32), state["terminated"].astype(jnp.float32),
                           state["truncated"].astype(jnp.float32)], dtype=jnp.float32),
                jax.nn.one_hot(winner_index, 5, dtype=jnp.float32),
                origin.astype(jnp.float32) / jnp.array([12.0, 10.0], dtype=jnp.float32),
                jax.nn.one_hot(state["agent_facing"][agent], 4, dtype=jnp.float32)
            ))
            rows = rows + [jnp.concatenate((board, static_map, agents, geodes, records, global_row))]
        return jnp.stack(rows, axis=0).astype(jnp.float32)

    def appendRecord(self, state, agent, active, origin, radius, counts):
        n = dict(state)
        valid = jnp.concatenate((n["scan_record_valid"][agent], active.reshape((1,))))
        ticks = jnp.concatenate((n["scan_record_tick"][agent], n["t"].reshape((1,))))
        sources = jnp.concatenate((n["scan_record_source"][agent], jnp.array([agent], dtype=jnp.int32)))
        keys = jnp.where(valid, ticks * 4 + sources, -1)
        order = jnp.argsort(keys)[-6:]
        kept = valid[order]
        n["scan_record_valid"] = n["scan_record_valid"].at[agent].set(kept)
        for name, row, invalid in (
            ("scan_record_xy", origin, -1), ("scan_record_radius", radius, 0),
            ("scan_record_counts", counts, 0), ("scan_record_tick", n["t"], -1),
            ("scan_record_source", jnp.array(agent, dtype=jnp.int32), -1)
        ):
            row = jnp.asarray(row, dtype=jnp.int32)
            combined = jnp.concatenate((n[name][agent], row[None]), axis=0)[order]
            shape = (6,) + (1,) * (len(combined.shape) - 1)
            combined = jnp.where(kept.reshape(shape), combined, invalid)
            n[name] = n[name].at[agent].set(combined)
        return n

    def mergeRecords(self, state, left, right, active):
        n = dict(state)
        valid = jnp.concatenate((n["scan_record_valid"][left], n["scan_record_valid"][right]))
        ticks = jnp.concatenate((n["scan_record_tick"][left], n["scan_record_tick"][right]))
        sources = jnp.concatenate((n["scan_record_source"][left], n["scan_record_source"][right]))
        keys = ticks * 4 + sources
        index = jnp.arange(12, dtype=jnp.int32)
        equal = keys[:, None] == keys[None, :]
        earlier = index[None, :] < index[:, None]
        duplicate = jnp.any(equal & earlier & valid[None, :], axis=1)
        unique = valid & ~duplicate
        order = jnp.argsort(jnp.where(unique, keys, -1))[-6:]
        kept = unique[order]
        for agent in (left, right):
            n["scan_record_valid"] = n["scan_record_valid"].at[agent].set(jnp.where(active, kept, n["scan_record_valid"][agent]))
        for name, invalid in (("scan_record_xy", -1), ("scan_record_radius", 0),
                               ("scan_record_counts", 0), ("scan_record_tick", -1), ("scan_record_source", -1)):
            combined = jnp.concatenate((state[name][left], state[name][right]), axis=0)[order]
            shape = (6,) + (1,) * (len(combined.shape) - 1)
            combined = jnp.where(kept.reshape(shape), combined, invalid)
            for agent in (left, right):
                n[name] = n[name].at[agent].set(jnp.where(active, combined, n[name][agent]))
        return n

    def activeStep(self, state, actions):
        n = self.clearEvents(state)
        n["t"] = state["t"] + jnp.array(1, dtype=jnp.int32)
        raw = jnp.where((actions >= 0) & (actions < 15), actions, 0).astype(jnp.int32)
        a = jnp.where(state["agent_stun"] > 0, 0, raw).astype(jnp.int32)
        n["agent_stun"] = jnp.maximum(state["agent_stun"] - 1, 0)
        n["agent_shove_cooldown"] = jnp.maximum(state["agent_shove_cooldown"] - 1, 0)
        step = (a >= 1) & (a <= 4)
        n["agent_facing"] = jnp.where(step, a - 1,
                                      jnp.where(a == 5, (state["agent_facing"] + 1) % 4,
                                                state["agent_facing"])).astype(jnp.int32)
        old_positions = state["agent_xy"]
        target = old_positions + self.directions()[jnp.clip(a - 1, 0, 3)]
        allowed = self.passable(state, target, state["agent_team"])
        accepted = self.resolveMoves(old_positions, target, step, allowed)
        n["agent_xy"] = jnp.where(accepted[:, None], target, old_positions)
        for agent in range(4):
            n = self.emit(n, accepted[agent], 1, agent, -1, old_positions[agent], target[agent], 1)
            n = self.emit(n, a[agent] == 5, 2, agent, -1, n["agent_xy"][agent], n["agent_xy"][agent], n["agent_facing"][agent])
        before_shove = n["agent_xy"]
        facing = self.directions()[n["agent_facing"]]
        front = before_shove + facing
        matches = jnp.all(front[:, None, :] == before_shove[None, :, :], axis=2)
        victim = jnp.argmax(matches, axis=1).astype(jnp.int32)
        opponent = jnp.any(matches & (n["agent_team"][:, None] != n["agent_team"][None, :]), axis=1)
        destination = before_shove[victim] + facing
        neutral = n["base_owner"][before_shove[victim, 1], before_shove[victim, 0]] < 0
        empty = ~jnp.any(jnp.all(destination[:, None, :] == before_shove[None, :, :], axis=2), axis=1)
        candidate = (a == 13) & (n["agent_charge"] >= 2) & (state["agent_shove_cooldown"] == 0)
        candidate = candidate & opponent & neutral & (a[victim] != 12)
        candidate = candidate & empty & self.passable(n, destination, n["agent_team"][victim])
        same_victim = victim[:, None] == victim[None, :]
        same_dest = jnp.all(destination[:, None, :] == destination[None, :, :], axis=2)
        unique_victim = jnp.sum(same_victim & candidate[None, :], axis=1, dtype=jnp.int32) == 1
        unique_dest = jnp.sum(same_dest & candidate[None, :], axis=1, dtype=jnp.int32) == 1
        ids = jnp.arange(4, dtype=jnp.int32)
        actor_targeted = jnp.any((ids[:, None] == victim[None, :]) & candidate[None, :], axis=1)
        shove = candidate & unique_victim & unique_dest & ~actor_targeted
        hit = jnp.any((ids[:, None] == victim[None, :]) & shove[None, :], axis=1)
        n["agent_charge"] = n["agent_charge"] - 2 * shove.astype(jnp.int32)
        n["agent_shove_cooldown"] = jnp.where(shove, 2, n["agent_shove_cooldown"])
        before_cargo = n["agent_cargo"]
        for agent in range(4):
            target_agent = victim[agent]
            n["agent_xy"] = n["agent_xy"].at[target_agent].set(
                jnp.where(shove[agent], destination[agent], n["agent_xy"][target_agent]))
            n["agent_stun"] = n["agent_stun"].at[target_agent].set(
                jnp.where(shove[agent], 1, n["agent_stun"][target_agent]))
            carried_id = before_cargo[target_agent]
            safe_id = jnp.clip(carried_id, 0, 7)
            forced_drop = shove[agent] & (carried_id >= 0)
            n["geode_status"] = n["geode_status"].at[safe_id].set(jnp.where(forced_drop, 1, n["geode_status"][safe_id]))
            n["geode_carrier"] = n["geode_carrier"].at[safe_id].set(jnp.where(forced_drop, -1, n["geode_carrier"][safe_id]))
            n["geode_xy"] = n["geode_xy"].at[safe_id].set(jnp.where(forced_drop, before_shove[target_agent], n["geode_xy"][safe_id]))
            n["agent_cargo"] = n["agent_cargo"].at[target_agent].set(jnp.where(forced_drop, -1, n["agent_cargo"][target_agent]))
            n = self.emit(n, shove[agent], 11, agent, target_agent, before_shove[target_agent], destination[agent], 1)
            n = self.emit(n, forced_drop, 12, target_agent, safe_id, before_shove[target_agent], before_shove[target_agent], 1)
        a = jnp.where(hit, 0, a).astype(jnp.int32)
        for agent in range(4):
            n = self.emit(n, a[agent] == 12, 10, agent, -1, n["agent_xy"][agent], n["agent_xy"][agent], 1)
        front = n["agent_xy"] + self.directions()[n["agent_facing"]]
        fx = jnp.clip(front[:, 0], 0, 12)
        fy = jnp.clip(front[:, 1], 0, 10)
        front_bounds = (front[:, 0] >= 0) & (front[:, 0] < 13) & (front[:, 1] >= 0) & (front[:, 1] < 11)
        dig = (a == 6) & (n["agent_charge"] >= 1) & front_bounds
        dig = dig & (n["terrain"][fy, fx] == 0) & (n["dig_depth"][fy, fx] < 2)
        work = jnp.where(n["agent_role"] == 1, 2, 1).astype(jnp.int32)
        work_grid = jnp.zeros((11, 13), dtype=jnp.int32)
        for agent in range(4):
            work_grid = work_grid.at[fy[agent], fx[agent]].add(jnp.where(dig[agent], work[agent], 0))
            n = self.emit(n, dig[agent], 3, agent, -1, n["agent_xy"][agent], front[agent], work[agent])
        n["agent_charge"] = n["agent_charge"] - dig.astype(jnp.int32)
        n["dig_depth"] = jnp.minimum(n["dig_depth"] + work_grid, 2)
        gx = jnp.clip(n["geode_xy"][:, 0], 0, 12)
        gy = jnp.clip(n["geode_xy"][:, 1], 0, 10)
        reveal = (n["geode_status"] == 0) & (n["dig_depth"][gy, gx] == 2)
        n["geode_status"] = jnp.where(reveal, 1, n["geode_status"])
        for obj in range(8):
            n = self.emit(n, reveal[obj], 4, -1, obj, n["geode_xy"][obj], n["geode_xy"][obj], n["geode_colour"][obj])
        cargo_at_handle_start = n["agent_cargo"]
        front_empty = ~jnp.any(jnp.all(front[:, None, :] == n["agent_xy"][None, :, :], axis=2), axis=1)
        drop = (a == 7) & (cargo_at_handle_start >= 0) & front_empty & self.passable(n, front, n["agent_team"])
        for agent in range(4):
            obj = jnp.clip(cargo_at_handle_start[agent], 0, 7)
            n["geode_status"] = n["geode_status"].at[obj].set(jnp.where(drop[agent], 1, n["geode_status"][obj]))
            n["geode_carrier"] = n["geode_carrier"].at[obj].set(jnp.where(drop[agent], -1, n["geode_carrier"][obj]))
            n["geode_xy"] = n["geode_xy"].at[obj].set(jnp.where(drop[agent], front[agent], n["geode_xy"][obj]))
            n["agent_cargo"] = n["agent_cargo"].at[agent].set(jnp.where(drop[agent], -1, n["agent_cargo"][agent]))
            n = self.emit(n, drop[agent], 6, agent, obj, n["agent_xy"][agent], front[agent], 1)
        matching_gem = jnp.all(front[:, None, :] == n["geode_xy"][None, :, :], axis=2) & (n["geode_status"] == 1)[None, :]
        chosen = jnp.min(jnp.where(matching_gem, jnp.arange(8, dtype=jnp.int32)[None, :], 8), axis=1)
        pickup = (a == 7) & (cargo_at_handle_start < 0) & (chosen < 8)
        same_pick = chosen[:, None] == chosen[None, :]
        pickup = pickup & (jnp.sum(same_pick & pickup[None, :], axis=1, dtype=jnp.int32) == 1)
        pickup_origins = n["geode_xy"][jnp.clip(chosen, 0, 7)]
        for agent in range(4):
            obj = jnp.clip(chosen[agent], 0, 7)
            n["geode_status"] = n["geode_status"].at[obj].set(jnp.where(pickup[agent], 2, n["geode_status"][obj]))
            n["geode_carrier"] = n["geode_carrier"].at[obj].set(jnp.where(pickup[agent], agent, n["geode_carrier"][obj]))
            n["geode_xy"] = n["geode_xy"].at[obj].set(jnp.where(pickup[agent], n["agent_xy"][agent], n["geode_xy"][obj]))
            n["agent_cargo"] = n["agent_cargo"].at[agent].set(jnp.where(pickup[agent], obj, n["agent_cargo"][agent]))
            n = self.emit(n, pickup[agent], 5, agent, obj, pickup_origins[agent], n["agent_xy"][agent], 1)
        for agent in range(4):
            team = n["agent_team"][agent]
            obj = jnp.clip(n["agent_cargo"][agent], 0, 7)
            bank = (a[agent] == 8) & (n["agent_cargo"][agent] >= 0) & jnp.all(n["agent_xy"][agent] == n["bank_xy"][team])
            points = jnp.where(n["geode_colour"][obj] == team, 2, 1).astype(jnp.int32)
            n["team_scores"] = n["team_scores"].at[team].add(jnp.where(bank, points, 0))
            n["geode_status"] = n["geode_status"].at[obj].set(jnp.where(bank, 3, n["geode_status"][obj]))
            n["geode_carrier"] = n["geode_carrier"].at[obj].set(jnp.where(bank, -1, n["geode_carrier"][obj]))
            n["geode_xy"] = n["geode_xy"].at[obj].set(jnp.where(bank, jnp.array([-1, -1], dtype=jnp.int32), n["geode_xy"][obj]))
            n["agent_cargo"] = n["agent_cargo"].at[agent].set(jnp.where(bank, -1, n["agent_cargo"][agent]))
            n = self.emit(n, bank, 7, agent, obj, n["agent_xy"][agent], n["agent_xy"][agent], points)
            here = n["agent_xy"][agent]
            recharge = (a[agent] == 14) & (n["base_owner"][here[1], here[0]] == team)
            amount = jnp.where(recharge, jnp.minimum(3, 8 - n["agent_charge"][agent]), 0).astype(jnp.int32)
            n["agent_charge"] = n["agent_charge"].at[agent].add(amount)
            n = self.emit(n, amount > 0, 13, agent, -1, here, here, amount)
        for agent in range(4):
            radius = jnp.where(a[agent] == 10, 4, 2).astype(jnp.int32)
            cost = jnp.where(a[agent] == 10, 2, 1).astype(jnp.int32)
            scan = ((a[agent] == 9) | (a[agent] == 10)) & (n["agent_role"][agent] == 0) & (n["agent_charge"][agent] >= cost)
            distance = jnp.sum(jnp.abs(n["geode_xy"] - n["agent_xy"][agent]), axis=1, dtype=jnp.int32)
            sensed = (n["geode_status"] == 0) & (distance <= radius)
            counts = jnp.stack((jnp.sum(sensed & (n["geode_colour"] == 0), dtype=jnp.int32),
                                jnp.sum(sensed & (n["geode_colour"] == 1), dtype=jnp.int32)))
            n["agent_charge"] = n["agent_charge"].at[agent].add(-jnp.where(scan, cost, 0))
            n = self.appendRecord(n, agent, scan, n["agent_xy"][agent], radius, counts)
            n = self.emit(n, scan, 8, agent, -1, n["agent_xy"][agent], n["agent_xy"][agent], radius)
        for team in range(2):
            left = 2 * team
            right = left + 1
            adjacent = jnp.sum(jnp.abs(n["agent_xy"][left] - n["agent_xy"][right]), dtype=jnp.int32) == 1
            share = ((a[left] == 11) | (a[right] == 11)) & adjacent
            initiator = jnp.where(a[left] == 11, left, right).astype(jnp.int32)
            partner = left + right - initiator
            n = self.mergeRecords(n, left, right, share)
            size = jnp.sum(n["scan_record_valid"][left], dtype=jnp.int32)
            n = self.emit(n, share, 9, initiator, partner, n["agent_xy"][initiator], n["agent_xy"][partner], size)
        carrier_positions = n["agent_xy"][jnp.clip(n["geode_carrier"], 0, 3)]
        n["geode_xy"] = jnp.where((n["geode_status"] == 2)[:, None], carrier_positions, n["geode_xy"])
        n["geode_xy"] = jnp.where((n["geode_status"] == 3)[:, None], -1, n["geode_xy"])
        race = jnp.any(n["team_scores"] >= 8)
        exhausted = jnp.all(n["geode_status"] == 3)
        terminated = race | exhausted
        truncated = (n["t"] >= 240) & ~terminated
        compare_winner = jnp.where(n["team_scores"][0] > n["team_scores"][1], 0,
                                   jnp.where(n["team_scores"][1] > n["team_scores"][0], 1, -1))
        n["winner"] = jnp.where(terminated, compare_winner, jnp.where(truncated, -3, -2)).astype(jnp.int32)
        n["terminal_reason"] = jnp.where(race, 1, jnp.where(exhausted, 2, jnp.where(truncated, 3, 0))).astype(jnp.int32)
        n["terminated"] = terminated
        n["truncated"] = truncated
        n["done"] = terminated | truncated
        absent = jnp.array([-1, -1], dtype=jnp.int32)
        n = self.emit(n, n["done"], 14, -1, -1, absent, absent, n["winner"])
        return self.present(n)

    def raw_step(self, key, state, actions):
        active = self.activeStep(state, actions)
        n = jax.tree.map(lambda old, new: jnp.where(state["done"], old, new), state, active)
        delta = n["team_scores"] - state["team_scores"]
        bonus = jnp.where(n["terminated"] & (n["winner"] == 0), 3,
                          jnp.where(n["terminated"] & (n["winner"] == 1), -3, 0))
        r0 = (delta[0] - delta[1] + bonus).astype(jnp.float32)
        r0 = jnp.where(state["done"], 0.0, r0).astype(jnp.float32)
        rewards = jnp.stack((r0, r0, -r0, -r0)).astype(jnp.float32)
        dones = jnp.full((4,), n["done"], dtype=jnp.bool_)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, rewards, dones, info
