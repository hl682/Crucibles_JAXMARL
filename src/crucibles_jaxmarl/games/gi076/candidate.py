"""Game 00076: Ballast Run. Authored source only; never executed by the author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 1939
    num_actions = 14
    max_steps = 192

    def directions(self):
        return jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)

    def tankCells(self):
        return jnp.array([[0, 0], [4, 0], [0, 2], [4, 2]], dtype=jnp.int32)

    def trays(self):
        return jnp.array([[1, 0], [3, 0], [2, 2]], dtype=jnp.int32)

    def stations(self):
        return jnp.array([[2, 1], [2, 0], [2, 2]], dtype=jnp.int32)

    def clearEvents(self, state):
        n = dict(state)
        n["event_active"] = jnp.zeros((32,), dtype=jnp.bool_)
        n["event_count"] = jnp.array(0, dtype=jnp.int32)
        for name in ("event_kind", "event_value"):
            n[name] = jnp.zeros((32,), dtype=jnp.int32)
        for name in ("event_actor", "event_object", "event_tick"):
            n[name] = jnp.full((32,), -1, dtype=jnp.int32)
        for name in ("event_from_space", "event_to_space"):
            n[name] = jnp.full((32,), 2, dtype=jnp.int32)
        for name in ("event_from_xy", "event_to_xy"):
            n[name] = jnp.full((32, 2), -1, dtype=jnp.int32)
        return n

    def emit(self, state, active, kind, actor, obj, start, end, start_space, end_space, value):
        n = dict(state)
        k = jnp.clip(n["event_count"], 0, 31)
        n["event_active"] = n["event_active"].at[k].set(n["event_active"][k] | active)
        for name, datum in (
            ("event_kind", kind), ("event_actor", actor), ("event_object", obj),
            ("event_from_xy", start), ("event_to_xy", end),
            ("event_from_space", start_space), ("event_to_space", end_space),
            ("event_value", value), ("event_tick", n["t"])
        ):
            datum = jnp.asarray(datum, dtype=jnp.int32)
            n[name] = n[name].at[k].set(jnp.where(active, datum, n[name][k]))
        n["event_count"] = n["event_count"] + active.astype(jnp.int32)
        return n

    def resolveMoves(self, positions, targets, moving, valid, active):
        equal_targets = jnp.all(targets[:, None, :] == targets[None, :, :], axis=2)
        candidates = moving & valid & active
        count = jnp.sum(equal_targets & candidates[None, :], axis=1, dtype=jnp.int32)
        accepted = candidates & (count == 1)
        occupied = jnp.all(targets[:, None, :] == positions[None, :, :], axis=2)
        for iteration in range(3):
            blocked = jnp.any(occupied & active[None, :] & ~accepted[None, :], axis=1)
            accepted = accepted & ~blocked
        return accepted

    def torque(self, state):
        centre = jnp.array([2, 1], dtype=jnp.int32)
        crew = jnp.sum(state["crew_deck_xy"] - centre, axis=0, dtype=jnp.int32)
        water = jnp.sum(state["ballast"][:, None] * (self.tankCells() - centre), axis=0, dtype=jnp.int32)
        cargo = jnp.sum(jnp.where((state["capsule_status"] == 1)[:, None],
                                 2 * (state["capsule_deck_xy"] - centre), 0),
                        axis=0, dtype=jnp.int32)
        return crew + water + cargo

    def footprint(self, terrain, centre):
        xx = jnp.tile(jnp.arange(5, dtype=jnp.int32), 3) - 2 + centre[0]
        yy = jnp.repeat(jnp.arange(3, dtype=jnp.int32), 5) - 1 + centre[1]
        bounds = (xx >= 0) & (xx < 23) & (yy >= 0) & (yy < 15)
        tile = terrain[jnp.clip(yy, 0, 14), jnp.clip(xx, 0, 22)]
        return jnp.all(bounds & (tile != 1) & (tile != 2))

    def present(self, state):
        n = dict(state)
        offset = n["boat_xy"] - jnp.array([2, 1], dtype=jnp.int32)
        n["agent_world_xy"] = n["crew_deck_xy"] + offset
        n["agent_active"] = jnp.ones((3,), dtype=jnp.bool_)
        n["agent_visibility"] = jnp.ones((3, 3), dtype=jnp.bool_)
        n["agent_number"] = jnp.arange(3, dtype=jnp.int32)
        n["vessel_xy"] = n["boat_xy"][None, :]
        n["vessel_active"] = jnp.ones((1,), dtype=jnp.bool_)
        n["vessel_visibility"] = jnp.ones((3, 1), dtype=jnp.bool_)
        n["vessel_hull"] = n["hull"].reshape((1,))
        n["vessel_fuel"] = n["fuel"].reshape((1,))
        n["vessel_velocity"] = n["boat_velocity"][None, :]
        n["vessel_torque"] = self.torque(n)[None, :]
        n["tank_world_xy"] = self.tankCells() + offset
        n["tank_active"] = jnp.ones((4,), dtype=jnp.bool_)
        n["tank_visibility"] = jnp.ones((3, 4), dtype=jnp.bool_)
        n["station_world_xy"] = self.stations() + offset
        n["station_active"] = jnp.ones((3,), dtype=jnp.bool_)
        n["station_visibility"] = jnp.ones((3, 3), dtype=jnp.bool_)
        n["station_kind"] = jnp.arange(3, dtype=jnp.int32)
        bank_slots = n["home_xy"] + jnp.array([[-1, 1], [0, 1], [1, 1]], dtype=jnp.int32)
        source_or_bank = jnp.where((n["capsule_status"] == 2)[:, None], bank_slots, n["buoy_xy"])
        n["capsule_world_xy"] = jnp.where((n["capsule_status"] == 1)[:, None],
                                         n["capsule_deck_xy"] + offset, source_or_bank)
        n["capsule_active"] = n["capsule_status"] != 3
        n["capsule_visibility"] = jnp.broadcast_to(n["capsule_active"][None, :], (3, 3))
        n["capsule_number"] = jnp.arange(3, dtype=jnp.int32)
        n["buoy_active"] = jnp.ones((3,), dtype=jnp.bool_)
        n["buoy_visibility"] = jnp.ones((3, 3), dtype=jnp.bool_)
        n["buoy_available"] = n["capsule_status"] == 0
        n["berth_xy"] = n["home_xy"][None, :]
        n["berth_active"] = jnp.ones((1,), dtype=jnp.bool_)
        n["berth_visibility"] = jnp.ones((3, 1), dtype=jnp.bool_)
        n["banked_count"] = jnp.sum(n["capsule_status"] == 2, dtype=jnp.int32).reshape((1,))
        n["current_direction"] = jnp.where(n["terrain"] == 3, 1,
                                           jnp.where(n["terrain"] == 4, -1, 0)).astype(jnp.int32)
        n["current_direction"] = n["current_direction"] * jnp.where((n["t"] // 8) % 2 == 0, 1, -1)
        n["event_visibility"] = jnp.broadcast_to(n["event_active"][None, :], (3, 32))
        return n

    def reset(self, key):
        xx = jnp.broadcast_to(jnp.arange(23, dtype=jnp.int32)[None, :], (15, 23))
        yy = jnp.broadcast_to(jnp.arange(15, dtype=jnp.int32)[:, None], (15, 23))
        wall = (xx == 0) | (xx == 22) | (yy == 0) | (yy == 14)
        reef = (xx >= 10) & (xx <= 12) & (yy >= 4) & (yy <= 9)
        east = (xx >= 1) & (xx <= 9) & (yy >= 6) & (yy <= 8)
        west = (xx >= 14) & (xx <= 21) & (yy >= 6) & (yy <= 8)
        terrain = jnp.where(wall, 1, jnp.where(reef, 2, jnp.where(east, 3, jnp.where(west, 4, 0)))).astype(jnp.int32)
        s = {
            "terrain": terrain,
            "home_xy": jnp.array([3, 12], dtype=jnp.int32),
            "buoy_xy": jnp.array([[7, 2], [18, 3], [18, 12]], dtype=jnp.int32),
            "boat_xy": jnp.array([3, 12], dtype=jnp.int32),
            "boat_velocity": jnp.zeros((2,), dtype=jnp.int32),
            "crew_deck_xy": jnp.array([[2, 1], [1, 1], [3, 1]], dtype=jnp.int32),
            "ballast": jnp.ones((4,), dtype=jnp.int32),
            "capsule_status": jnp.zeros((3,), dtype=jnp.int32),
            "capsule_deck_xy": jnp.full((3, 2), -1, dtype=jnp.int32),
            "capsule_pinned": jnp.zeros((3,), dtype=jnp.bool_),
            "fuel": jnp.array(80, dtype=jnp.int32),
            "hull": jnp.array(12, dtype=jnp.int32),
            "t": jnp.array(0, dtype=jnp.int32),
            "done": jnp.array(False, dtype=jnp.bool_),
            "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
            "success": jnp.array(False, dtype=jnp.bool_),
            "terminal_reason": jnp.array(0, dtype=jnp.int32),
        }
        s = self.present(self.clearEvents(s))
        return self.observe(s), s

    def legal_actions(self, state):
        p = state["crew_deck_xy"]
        masks = jnp.zeros((3, 14), dtype=jnp.bool_).at[:, 0].set(True)
        for direction in range(4):
            dest = p + self.directions()[direction]
            valid = (dest[:, 0] >= 0) & (dest[:, 0] < 5) & (dest[:, 1] >= 0) & (dest[:, 1] < 3)
            masks = masks.at[:, 1 + direction].set(valid)
        helm = jnp.all(p == jnp.array([2, 1], dtype=jnp.int32), axis=1) & (state["fuel"] > 0)
        for action in range(5, 10):
            masks = masks.at[:, action].set(helm)
        tank_match = jnp.all(p[:, None, :] == self.tankCells()[None, :, :], axis=2)
        masks = masks.at[:, 10].set(jnp.any(tank_match & (state["ballast"] < 3)[None, :], axis=1))
        masks = masks.at[:, 11].set(jnp.any(tank_match & (state["ballast"] > 0)[None, :], axis=1))
        work = jnp.all(p == jnp.array([2, 0], dtype=jnp.int32), axis=1) | jnp.all(p == jnp.array([2, 2], dtype=jnp.int32), axis=1)
        masks = masks.at[:, 12].set(work)
        cargo_match = jnp.all(p[:, None, :] == state["capsule_deck_xy"][None, :, :], axis=2)
        masks = masks.at[:, 13].set(jnp.any(cargo_match & (state["capsule_status"] == 1)[None, :], axis=1))
        idle = jnp.zeros((3, 14), dtype=jnp.bool_).at[:, 0].set(True)
        return jnp.where(state["done"], idle, masks)

    def observe(self, state):
        terrain = jax.nn.one_hot(state["terrain"], 5, dtype=jnp.float32).reshape((-1,))
        xx = jnp.broadcast_to(jnp.arange(5, dtype=jnp.int32)[None, :], (3, 5))
        yy = jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32)[:, None], (3, 5))
        cells = jnp.stack((xx, yy), axis=2)
        crew = jnp.all(cells[:, :, None, :] == state["crew_deck_xy"][None, None, :, :], axis=3)
        tanks = jnp.all(cells[:, :, None, :] == self.tankCells()[None, None, :, :], axis=3)
        water = jnp.sum(tanks.astype(jnp.float32) * state["ballast"][None, None, :], axis=2) / 3.0
        capsules = jnp.all(cells[:, :, None, :] == state["capsule_deck_xy"][None, None, :, :], axis=3)
        capsules = capsules & (state["capsule_status"] == 1)[None, None, :]
        cargo = jnp.sum(capsules, axis=2, dtype=jnp.int32).astype(jnp.float32)
        loose = jnp.sum(capsules & ~state["capsule_pinned"][None, None, :], axis=2, dtype=jnp.int32).astype(jnp.float32)
        fixed = jnp.sum(capsules & state["capsule_pinned"][None, None, :], axis=2, dtype=jnp.int32).astype(jnp.float32)
        stations = jnp.all(cells[:, :, None, :] == self.stations()[None, None, :, :], axis=3).astype(jnp.float32)
        deck = jnp.concatenate((crew.astype(jnp.float32), water[:, :, None], cargo[:, :, None],
                                loose[:, :, None], fixed[:, :, None], stations,
                                jnp.any(tanks, axis=2).astype(jnp.float32)[:, :, None]), axis=2).reshape((-1,))
        cp = jnp.where((state["capsule_status"] == 1)[:, None], state["capsule_deck_xy"], 0).astype(jnp.float32)
        cp = cp / jnp.array([4.0, 2.0], dtype=jnp.float32)
        objects = jnp.concatenate((jax.nn.one_hot(state["capsule_status"], 4, dtype=jnp.float32),
                                   cp, state["capsule_pinned"].astype(jnp.float32)[:, None]), axis=1).reshape((-1,))
        buoys = (state["buoy_xy"].astype(jnp.float32) / jnp.array([22.0, 14.0], dtype=jnp.float32)).reshape((-1,))
        torque = self.torque(state).astype(jnp.float32)
        tide = jnp.where((state["t"] // 8) % 2 == 0, 1.0, -1.0)
        common = jnp.concatenate((terrain, deck, objects, buoys))
        rows = []
        for agent in range(3):
            global_row = jnp.concatenate((
                jax.nn.one_hot(jnp.array(agent, dtype=jnp.int32), 3, dtype=jnp.float32),
                state["boat_xy"].astype(jnp.float32) / jnp.array([22.0, 14.0], dtype=jnp.float32),
                state["boat_velocity"].astype(jnp.float32),
                jnp.array([state["fuel"] / 80.0, state["hull"] / 12.0, state["t"] / 192.0,
                           tide, (8 - state["t"] % 8) / 8.0, (state["t"] % 4 == 0).astype(jnp.float32),
                           torque[0] / 30.0, torque[1] / 16.0,
                           (2 - jnp.sum(state["capsule_pinned"], dtype=jnp.int32)) / 2.0,
                           state["done"].astype(jnp.float32), state["success"].astype(jnp.float32),
                           state["terminated"].astype(jnp.float32), state["truncated"].astype(jnp.float32),
                           state["terminal_reason"] / 4.0,
                           jnp.sum(state["capsule_status"] == 2, dtype=jnp.int32) / 3.0], dtype=jnp.float32)
            ))
            rows = rows + [jnp.concatenate((common, global_row))]
        return jnp.stack(rows, axis=0).astype(jnp.float32)

    def activeStep(self, state, actions):
        n = self.clearEvents(state)
        n["t"] = state["t"] + jnp.array(1, dtype=jnp.int32)
        a = jnp.where((actions >= 0) & (actions < 14), actions, 0).astype(jnp.int32)
        old_crew = state["crew_deck_xy"]
        move = (a >= 1) & (a <= 4)
        target = old_crew + self.directions()[jnp.clip(a - 1, 0, 3)]
        in_deck = (target[:, 0] >= 0) & (target[:, 0] < 5) & (target[:, 1] >= 0) & (target[:, 1] < 3)
        accepted = self.resolveMoves(old_crew, target, move, in_deck, jnp.ones((3,), dtype=jnp.bool_))
        n["crew_deck_xy"] = jnp.where(accepted[:, None], target, old_crew)
        for agent in range(3):
            n = self.emit(n, accepted[agent], 1, agent, -1, old_crew[agent], target[agent], 1, 1, 1)
        p = n["crew_deck_xy"]
        tank_match = jnp.all(p[:, None, :] == self.tankCells()[None, :, :], axis=2)
        for agent in range(3):
            tank = jnp.argmax(tank_match[agent]).astype(jnp.int32)
            located = jnp.any(tank_match[agent])
            delta = jnp.where((a[agent] == 10) & (n["ballast"][tank] < 3), 1,
                              jnp.where((a[agent] == 11) & (n["ballast"][tank] > 0), -1, 0))
            delta = jnp.where(located, delta, 0).astype(jnp.int32)
            n["ballast"] = n["ballast"].at[tank].add(delta)
            n = self.emit(n, delta != 0, 2, agent, tank, self.tankCells()[tank], self.tankCells()[tank], 1, 1, delta)
        cargo_match = jnp.all(p[:, None, :] == n["capsule_deck_xy"][None, :, :], axis=2)
        request = (a == 13)[:, None] & cargo_match & (n["capsule_status"] == 1)[None, :]
        released = jnp.any(request, axis=0) & n["capsule_pinned"]
        requested_attach = jnp.any(request, axis=0) & ~n["capsule_pinned"]
        after_release = n["capsule_pinned"] & ~released
        fits = jnp.sum(requested_attach, dtype=jnp.int32) <= 2 - jnp.sum(after_release, dtype=jnp.int32)
        attached = requested_attach & fits
        n["capsule_pinned"] = after_release | attached
        for obj in range(3):
            actor = jnp.argmax(request[:, obj]).astype(jnp.int32)
            n = self.emit(n, released[obj], 12, actor, obj, n["capsule_deck_xy"][obj], n["capsule_deck_xy"][obj], 1, 1, 1)
            n = self.emit(n, attached[obj], 11, actor, obj, n["capsule_deck_xy"][obj], n["capsule_deck_xy"][obj], 1, 1, 1)
        helm = jnp.all(p == jnp.array([2, 1], dtype=jnp.int32), axis=1) & (state["fuel"] > 0)
        thrust_mask = helm & (a >= 5) & (a <= 8)
        brake_mask = helm & (a == 9)
        force = jnp.sum(self.directions()[jnp.clip(a - 5, 0, 3)] * thrust_mask[:, None], axis=0, dtype=jnp.int32)
        terrain_at_centre = state["terrain"][state["boat_xy"][1], state["boat_xy"][0]]
        base = jnp.where(terrain_at_centre == 3, 1, jnp.where(terrain_at_centre == 4, -1, 0))
        tide = jnp.where((state["t"] // 8) % 2 == 0, 1, -1)
        impulse = jnp.array([base * tide * (state["t"] % 4 == 0).astype(jnp.int32), 0], dtype=jnp.int32)
        brake = jnp.any(brake_mask)
        velocity = jnp.where(brake, jnp.zeros((2,), dtype=jnp.int32),
                             jnp.clip(state["boat_velocity"] + force + impulse, -1, 1))
        fuel_used = jnp.sum(thrust_mask | brake_mask, dtype=jnp.int32)
        n["fuel"] = n["fuel"] - fuel_used
        for agent in range(3):
            end = state["boat_xy"] + self.directions()[jnp.clip(a[agent] - 5, 0, 3)]
            n = self.emit(n, thrust_mask[agent], 3, agent, -1, state["boat_xy"], end, 0, 0, 1)
            n = self.emit(n, brake_mask[agent], 4, agent, -1, state["boat_xy"], state["boat_xy"], 0, 0, 1)
        proposed_boat = state["boat_xy"] + velocity
        horizontal = state["boat_xy"] + jnp.array([velocity[0], 0], dtype=jnp.int32)
        vertical = state["boat_xy"] + jnp.array([0, velocity[1]], dtype=jnp.int32)
        navigable = self.footprint(n["terrain"], proposed_boat) & self.footprint(n["terrain"], horizontal) & self.footprint(n["terrain"], vertical)
        n["boat_xy"] = jnp.where(navigable, proposed_boat, state["boat_xy"])
        n["boat_velocity"] = jnp.where(navigable, velocity, 0)
        n["hull"] = n["hull"] - (~navigable).astype(jnp.int32)
        n = self.emit(n, navigable & jnp.any(velocity != 0), 5, -1, -1, state["boat_xy"], proposed_boat, 0, 0, 1)
        n = self.emit(n, ~navigable, 6, -1, -1, state["boat_xy"], proposed_boat, 0, 0, 1)
        scoop_operators = (a == 12) & jnp.all(p == jnp.array([2, 0], dtype=jnp.int32), axis=1)
        scoop_world = n["boat_xy"] + jnp.array([0, -1], dtype=jnp.int32)
        distances = jnp.max(jnp.abs(n["buoy_xy"] - scoop_world), axis=1)
        occupied_tray = jnp.any(jnp.all(self.trays()[:, None, :] == n["capsule_deck_xy"][None, :, :], axis=2) &
                                 (n["capsule_status"] == 1)[None, :], axis=1)
        hypothetical = self.torque(n)[None, :] + 2 * (self.trays() - jnp.array([2, 1], dtype=jnp.int32))
        stable_load = (jnp.abs(hypothetical[:, 0]) <= 3) & (jnp.abs(hypothetical[:, 1]) <= 2)
        eligible = (n["capsule_status"] == 0) & (distances <= 1) & ~occupied_tray & stable_load
        chosen = jnp.argmin(jnp.where(eligible, distances * 3 + jnp.arange(3, dtype=jnp.int32), 999)).astype(jnp.int32)
        load = jnp.any(scoop_operators) & jnp.all(n["boat_velocity"] == 0) & jnp.any(eligible)
        n["capsule_status"] = n["capsule_status"].at[chosen].set(jnp.where(load, 1, n["capsule_status"][chosen]))
        n["capsule_deck_xy"] = n["capsule_deck_xy"].at[chosen].set(jnp.where(load, self.trays()[chosen], n["capsule_deck_xy"][chosen]))
        n = self.emit(n, load, 7, jnp.argmax(scoop_operators).astype(jnp.int32), chosen, n["buoy_xy"][chosen], self.trays()[chosen], 0, 1, 1)
        dispatch = (a == 12) & jnp.all(p == jnp.array([2, 2], dtype=jnp.int32), axis=1)
        tor = self.torque(n)
        bank = jnp.any(dispatch) & jnp.all(n["boat_xy"] == n["home_xy"]) & jnp.all(n["boat_velocity"] == 0) & (jnp.abs(tor[0]) <= 3) & (jnp.abs(tor[1]) <= 2)
        to_bank = bank & (n["capsule_status"] == 1)
        for obj in range(3):
            n = self.emit(n, to_bank[obj], 8, jnp.argmax(dispatch).astype(jnp.int32), obj, n["capsule_deck_xy"][obj], n["home_xy"], 1, 0, 1)
        n["capsule_status"] = jnp.where(to_bank, 2, n["capsule_status"])
        n["capsule_deck_xy"] = jnp.where(to_bank[:, None], -1, n["capsule_deck_xy"])
        n["capsule_pinned"] = n["capsule_pinned"] & ~to_bank
        tor = self.torque(n)
        shift = jnp.array([jnp.where(jnp.abs(tor[0]) >= 4, jnp.sign(tor[0]), 0),
                           jnp.where(jnp.abs(tor[1]) >= 3, jnp.sign(tor[1]), 0)], dtype=jnp.int32)
        old_cargo = n["capsule_deck_xy"]
        aboard = n["capsule_status"] == 1
        loose = aboard & ~n["capsule_pinned"]
        proposed = old_cargo + shift
        cargo_bounds = (proposed[:, 0] >= 0) & (proposed[:, 0] < 5) & (proposed[:, 1] >= 0) & (proposed[:, 1] < 3)
        lost = loose & ~cargo_bounds
        moving = loose & jnp.any(shift != 0) & ~lost
        slide = self.resolveMoves(old_cargo, proposed, moving, cargo_bounds, aboard & ~lost)
        n["capsule_status"] = jnp.where(lost, 3, n["capsule_status"])
        n["capsule_deck_xy"] = jnp.where(slide[:, None], proposed, old_cargo)
        n["capsule_deck_xy"] = jnp.where((n["capsule_status"] == 1)[:, None], n["capsule_deck_xy"], -1)
        n["capsule_pinned"] = n["capsule_pinned"] & (n["capsule_status"] == 1)
        for obj in range(3):
            n = self.emit(n, slide[obj], 9, -1, obj, old_cargo[obj], proposed[obj], 1, 1, 1)
            n = self.emit(n, lost[obj], 10, -1, obj, old_cargo[obj], proposed[obj], 1, 1, 1)
        tor = self.torque(n)
        strain = (jnp.abs(tor[0]) >= 10) | (jnp.abs(tor[1]) >= 6)
        n["hull"] = jnp.maximum(n["hull"] - strain.astype(jnp.int32), 0)
        n = self.emit(n, strain, 13, -1, -1, n["boat_xy"], n["boat_xy"], 0, 0, 1)
        hull_failure = n["hull"] == 0
        lost_failure = jnp.any(n["capsule_status"] == 3)
        success = jnp.all(n["capsule_status"] == 2) & ~hull_failure & ~lost_failure
        terminated = hull_failure | lost_failure | success
        truncated = (n["t"] >= 192) & ~terminated
        n["terminated"] = terminated
        n["truncated"] = truncated
        n["done"] = terminated | truncated
        n["success"] = success
        n["terminal_reason"] = jnp.where(hull_failure, 2, jnp.where(lost_failure, 3,
                                          jnp.where(success, 1, jnp.where(truncated, 4, 0)))).astype(jnp.int32)
        absent = jnp.array([-1, -1], dtype=jnp.int32)
        n = self.emit(n, n["done"], 14, -1, -1, absent, absent, 2, 2, n["terminal_reason"])
        return self.present(n)

    def raw_step(self, key, state, actions):
        active = self.activeStep(state, actions)
        n = jax.tree.map(lambda old, new: jnp.where(state["done"], old, new), state, active)
        banked = jnp.sum((n["capsule_status"] == 2) & (state["capsule_status"] != 2), dtype=jnp.int32)
        failure = (n["terminal_reason"] == 2) | (n["terminal_reason"] == 3)
        reward = banked.astype(jnp.float32) + 2.0 * n["success"].astype(jnp.float32) - failure.astype(jnp.float32)
        reward = reward - 0.02 * (state["hull"] - n["hull"]).astype(jnp.float32) - 0.001
        reward = jnp.where(state["done"], 0.0, reward).astype(jnp.float32)
        rewards = jnp.full((3,), reward, dtype=jnp.float32)
        dones = jnp.full((3,), n["done"], dtype=jnp.bool_)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, rewards, dones, info
