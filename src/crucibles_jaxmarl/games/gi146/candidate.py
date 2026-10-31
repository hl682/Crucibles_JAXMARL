"""Firebreak pumps and shared evacuation. Original author kernel; unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 304
    num_actions = 13
    max_steps = 80

    def reset(self, key):
        # RESET: key is accepted but this world has a deterministic initial state.
        xy = jnp.stack((jnp.arange(63) % 9, jnp.arange(63) // 9), axis=1).astype(jnp.int32)
        x, y = xy[:, 0], xy[:, 1]
        wall = (x == 0) | (x == 8) | (y == 0) | (y == 6) | ((x == 4) & (y != 3))
        tiles = jnp.where(wall, 1, 0).reshape(7, 9).astype(jnp.int32)
        tiles = tiles.at[3, 4].set(2).at[3, 1].set(3).at[2, 3].set(4).at[4, 1].set(5)
        fire = jnp.zeros(63, jnp.int32).at[15].set(1).at[52].set(1)
        state = {
            "t": jnp.array(0, jnp.int32),
            "done": jnp.array(False), "terminated": jnp.array(False),
            "truncated": jnp.array(False), "tiles": tiles,
            "agent_pos": jnp.array([[1, 3], [3, 2], [3, 3]], jnp.int32),
            "agent_active": jnp.ones(3, jnp.bool_),
            "agent_vis": jnp.zeros((3, 3), jnp.bool_),
            "water": jnp.array([0, 2, 2], jnp.int32),
            "has_hose": jnp.zeros(3, jnp.bool_),
            "has_person": jnp.zeros(3, jnp.bool_),
            "pump_pos": jnp.array([[1, 3]], jnp.int32),
            "pump_active": jnp.ones(1, jnp.bool_), "pump_vis": jnp.ones((3, 1), jnp.bool_),
            "pump_water": jnp.array([20], jnp.int32),
            "hose_pos": jnp.array([[2, 3], [3, 3]], jnp.int32),
            "hose_active": jnp.ones(2, jnp.bool_),
            "hose_vis": jnp.zeros((3, 2), jnp.bool_),
            "hose_holder": jnp.full(2, -1, jnp.int32),
            "hose_connected": jnp.ones(2, jnp.bool_),
            "person_pos": jnp.array([[6, 2], [7, 4]], jnp.int32),
            "person_active": jnp.ones(2, jnp.bool_),
            "person_vis": jnp.zeros((3, 2), jnp.bool_),
            "person_holder": jnp.full(2, -1, jnp.int32),
            "person_status": jnp.zeros(2, jnp.int32),
            "person_health": jnp.full(2, 8, jnp.int32),
            "cell_pos": xy, "fire_level": fire,
            "fuel": jnp.where((~wall) & (x >= 4), 4, 0).astype(jnp.int32),
            "fire_active": fire > 0, "cell_vis": jnp.zeros((3, 63), jnp.bool_),
            "gate_pos": jnp.array([[4, 3]], jnp.int32),
            "gate_active": jnp.ones(1, jnp.bool_), "gate_vis": jnp.ones((3, 1), jnp.bool_),
            "gate_open": jnp.zeros(1, jnp.bool_),
            "console_pos": jnp.array([[3, 2]], jnp.int32),
            "console_active": jnp.ones(1, jnp.bool_), "console_vis": jnp.ones((3, 1), jnp.bool_),
            "refuge_pos": jnp.array([[1, 4]], jnp.int32),
            "refuge_active": jnp.ones(1, jnp.bool_), "refuge_vis": jnp.ones((3, 1), jnp.bool_),
            "spent_water": jnp.array(0, jnp.int32), "water_total": jnp.array(24, jnp.int32),
            "rescued": jnp.array(0, jnp.int32), "lost": jnp.array(0, jnp.int32),
            "received": jnp.zeros(3, jnp.int32), "sprayed": jnp.zeros(3, jnp.bool_),
            "held": jnp.zeros(3, jnp.bool_), "new_rescue": jnp.zeros(2, jnp.bool_),
            "new_loss": jnp.zeros(2, jnp.bool_),
        }
        state = self.refresh(state)
        return self.observe(state), state

    def refresh(self, state):
        s = dict(state)
        p = s["agent_pos"]
        s["agent_vis"] = jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2) <= 3
        s["hose_vis"] = jnp.sum(jnp.abs(p[:, None, :] - s["hose_pos"][None, :, :]), axis=2) <= 3
        s["person_vis"] = jnp.sum(jnp.abs(p[:, None, :] - s["person_pos"][None, :, :]), axis=2) <= 3
        s["cell_vis"] = jnp.sum(jnp.abs(p[:, None, :] - s["cell_pos"][None, :, :]), axis=2) <= 3
        s["has_hose"] = jnp.any(s["hose_holder"][None, :] == jnp.arange(3)[:, None], axis=1)
        s["has_person"] = jnp.any(s["person_holder"][None, :] == jnp.arange(3)[:, None], axis=1)
        # HOSE_GRAPH: only ground couplings conduct; adjacency is Manhattan one.
        ground = s["hose_holder"] < 0
        linked = ground & (jnp.sum(jnp.abs(s["hose_pos"] - s["pump_pos"][0]), axis=1) <= 1)
        neighbors = jnp.sum(jnp.abs(s["hose_pos"][:, None, :] - s["hose_pos"][None, :, :]), axis=2) <= 1
        for unused in range(2):
            linked = ground & (linked | jnp.any(neighbors & linked[None, :], axis=1))
        s["hose_connected"] = linked
        s["fire_active"] = s["fire_level"] > 0
        s["person_active"] = s["person_status"] == 0
        return s

    def records(self, position, attributes, visibility):
        values = jnp.concatenate((position, attributes), axis=1)
        visible_values = jnp.where(visibility[:, :, None], values[None, :, :], -1)
        return jnp.concatenate((visibility[:, :, None].astype(jnp.int32), visible_values), axis=2).reshape(3, -1)

    def observe(self, state):
        s = state
        public = jnp.stack((s["t"], s["pump_water"][0], s["spent_water"], s["water_total"],
                            s["rescued"], s["lost"], s["gate_open"][0].astype(jnp.int32),
                            s["done"].astype(jnp.int32)))
        own = jnp.concatenate((jnp.arange(3)[:, None], s["agent_pos"], s["water"][:, None]), axis=1)
        agents = self.records(s["agent_pos"], jnp.stack((s["water"], s["has_hose"], s["has_person"]), axis=1), s["agent_vis"])
        # A visible coupling exposes its physical pressure connection, like a local gauge.
        hoses = self.records(s["hose_pos"], jnp.stack((s["hose_holder"], s["hose_connected"]), axis=1), s["hose_vis"])
        people = self.records(s["person_pos"], jnp.stack((s["person_holder"], s["person_status"], s["person_health"]), axis=1), s["person_vis"])
        cells = jnp.stack((s["cell_vis"].astype(jnp.int32),
                           jnp.where(s["cell_vis"], s["fire_level"][None, :], -1),
                           jnp.where(s["cell_vis"], s["fuel"][None, :], -1)), axis=2).reshape(3, 189)
        return jnp.concatenate((jnp.broadcast_to(public, (3, 8)), own,
                                jnp.broadcast_to(s["tiles"].reshape(1, 63), (3, 63)),
                                agents, hoses, people, cells), axis=1).astype(jnp.float32)

    def legal_actions(self, state):
        s = state
        p = s["agent_pos"]
        d = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]], jnp.int32)
        target = p[:, None, :] + d[None, :, :]
        tx, ty = jnp.clip(target[:, :, 0], 0, 8), jnp.clip(target[:, :, 1], 0, 6)
        inside = (target[:, :, 0] >= 0) & (target[:, :, 0] <= 8) & (target[:, :, 1] >= 0) & (target[:, :, 1] <= 6)
        moves = inside & (s["tiles"][ty, tx] != 1) & (s["fire_level"][ty * 9 + tx] < 2)
        pump_near = jnp.sum(jnp.abs(p - s["pump_pos"][0]), axis=1) <= 1
        hose_near = jnp.any((jnp.sum(jnp.abs(p[:, None, :] - s["hose_pos"][None, :, :]), axis=2) <= 1) & (s["hose_holder"][None, :] < 0), axis=1)
        person_near = jnp.any((jnp.sum(jnp.abs(p[:, None, :] - s["person_pos"][None, :, :]), axis=2) <= 1) & (s["person_holder"][None, :] < 0) & s["person_active"][None, :], axis=1)
        console_near = jnp.sum(jnp.abs(p - s["console_pos"][0]), axis=1) <= 1
        at_refuge = jnp.all(p == s["refuge_pos"][0], axis=1)
        # RECEIVE is an attempt: connectivity and a concurrent donor are transition conditions,
        # never a privileged-information side channel in the mask.
        extra = jnp.stack((pump_near & (s["pump_water"][0] > 0), s["water"] < 4,
                           s["has_hose"] | hose_near, s["water"] > 0,
                           console_near & (s["water"] > 0),
                           s["has_person"] | person_near,
                           s["has_person"] & at_refuge, s["water"] > 0), axis=1)
        mask = jnp.concatenate((jnp.ones((3, 1), jnp.bool_), moves, extra), axis=1)
        return jnp.where(s["done"], jnp.arange(13)[None, :] == 0, mask)

    def raw_step(self, key, state, actions):
        actions = jnp.asarray(actions, jnp.int32)
        clipped = jnp.clip(actions, 0, 12)
        permitted = self.legal_actions(state)[jnp.arange(3), clipped] & (actions == clipped)
        clean = jnp.where(permitted, clipped, 0)
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, clean))

    def absorb(self, bundle):
        key, s, actions = bundle
        info = {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}
        return self.observe(s), s, jnp.zeros(3, jnp.float32), jnp.ones(3, jnp.bool_), info

    def advance(self, bundle):
        key, old, a = bundle
        s = dict(old)
        p = old["agent_pos"]
        # SIMULTANEOUS_GATE: holding spends one stored water and opens this transition.
        held = a == 9
        gate_open = jnp.any(held)
        d = jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], jnp.int32)
        target = p + d[jnp.minimum(a, 4)] * (a <= 4)[:, None]
        entering_gate = jnp.all(target == old["gate_pos"][0], axis=1)
        moving = (a >= 1) & (a <= 4) & (~entering_gate | gate_open)
        target = jnp.where(moving[:, None], target, p)
        occupied = jnp.any(jnp.all(target[:, None, :] == p[None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(target[:, None, :] == target[None, :, :], axis=2), axis=1) > 1
        p = jnp.where((moving & ~occupied & ~contested)[:, None], target, p)
        s["agent_pos"] = p
        rank = (jnp.arange(3) - old["t"]) % 3
        # PORTABLE_HOSES: simultaneous old-state selection, cyclic-priority ties, no cloning.
        holder = old["hose_holder"]
        eligible = (jnp.sum(jnp.abs(p[:, None, :] - old["hose_pos"][None, :, :]), axis=2) <= 1) & (holder[None, :] < 0)
        choice = jnp.argmax(eligible, axis=1)
        for h in range(2):
            wants = (a == 7) & ~old["has_hose"] & eligible[:, h] & (choice == h)
            winner = jnp.argmin(jnp.where(wants, rank, 9))
            drop = (holder[h] >= 0) & (a[jnp.maximum(holder[h], 0)] == 7)
            next_holder = jnp.where(jnp.any(wants), winner, jnp.where(drop, -1, holder[h]))
            drop_pos = jnp.where(drop, p[jnp.maximum(holder[h], 0)], old["hose_pos"][h])
            s["hose_holder"] = s["hose_holder"].at[h].set(next_holder)
            s["hose_pos"] = s["hose_pos"].at[h].set(jnp.where(next_holder >= 0, p[jnp.maximum(next_holder, 0)], drop_pos))
        s = self.refresh(s)
        # PUMP_AND_TRANSFER: one unit per receiver, pump throughput two; bounded tank four.
        water = old["water"]
        pump_water = old["pump_water"][0]
        pumped = jnp.array(0, jnp.int32)
        received = jnp.zeros(3, jnp.int32)
        near_pump = jnp.sum(jnp.abs(p - old["pump_pos"][0]), axis=1) <= 1
        at_hose = jnp.any((jnp.sum(jnp.abs(p[:, None, :] - s["hose_pos"][None, :, :]), axis=2) <= 1) & s["hose_connected"][None, :], axis=1)
        pumping = jnp.any(a == 5)
        for k in range(3):
            i = (old["t"] + k) % 3
            grant = ((a[i] == 6) & (near_pump[i] | at_hose[i]) & pumping & (pump_water > 0) & (pumped < 2) & (water[i] < 4)).astype(jnp.int32)
            water = water.at[i].add(grant)
            pump_water = pump_water - grant
            pumped = pumped + grant
            received = received.at[i].add(grant)
        for k in range(3):
            donor = (old["t"] + k) % 3
            willing = (a == 6) & (water < 4) & (jnp.sum(jnp.abs(p - p[donor]), axis=1) == 1)
            recipient = jnp.argmin(jnp.where(willing, rank, 9))
            grant = ((a[donor] == 12) & (water[donor] > 0) & jnp.any(willing)).astype(jnp.int32)
            water = water.at[donor].add(-grant).at[recipient].add(grant)
            received = received.at[recipient].add(grant)
        sprayed = a == 8
        consumed = sprayed.astype(jnp.int32) + held.astype(jnp.int32)
        water = water - consumed
        s["water"] = water
        s["pump_water"] = s["pump_water"].at[0].set(pump_water)
        s["spent_water"] = old["spent_water"] + jnp.sum(consumed)
        s["water_total"] = pump_water + jnp.sum(water) + s["spent_water"]
        s["received"], s["sprayed"], s["held"] = received, sprayed, held
        s["gate_open"] = gate_open[None]
        # OCCUPANT_HANDLING: one carried target per firefighter; ties use same rotating rank.
        ph = old["person_holder"]
        near = (jnp.sum(jnp.abs(p[:, None, :] - old["person_pos"][None, :, :]), axis=2) <= 1) & (ph[None, :] < 0) & old["person_active"][None, :]
        pick = jnp.argmax(near, axis=1)
        for q in range(2):
            wants = (a == 10) & ~old["has_person"] & near[:, q] & (pick == q)
            winner = jnp.argmin(jnp.where(wants, rank, 9))
            drop = (ph[q] >= 0) & (a[jnp.maximum(ph[q], 0)] == 10)
            nh = jnp.where(jnp.any(wants), winner, jnp.where(drop, -1, ph[q]))
            floor_pos = jnp.where(drop, p[jnp.maximum(ph[q], 0)], old["person_pos"][q])
            s["person_holder"] = s["person_holder"].at[q].set(nh)
            s["person_pos"] = s["person_pos"].at[q].set(jnp.where(nh >= 0, p[jnp.maximum(nh, 0)], floor_pos))
        carried_by = jnp.maximum(s["person_holder"], 0)
        evacuated = s["person_active"] & (s["person_holder"] >= 0) & (a[carried_by] == 11) & jnp.all(s["person_pos"] == s["refuge_pos"][0], axis=1)
        s["person_status"] = jnp.where(evacuated, 1, old["person_status"])
        s["person_holder"] = jnp.where(evacuated, -1, s["person_holder"])
        # FIRE: spraying clears radius one; every eighth tick spreads one graph edge.
        wet = jnp.any((jnp.sum(jnp.abs(p[:, None, :] - s["cell_pos"][None, :, :]), axis=2) <= 1) & sprayed[:, None], axis=0)
        wet = wet | (gate_open & jnp.all(s["cell_pos"] == s["gate_pos"][0], axis=1))
        level = jnp.where(wet, 0, old["fire_level"])
        adjacency = jnp.sum(jnp.abs(s["cell_pos"][:, None, :] - s["cell_pos"][None, :, :]), axis=2) == 1
        spread = jnp.any(adjacency & (level[None, :] > 0), axis=1)
        pulse = ((old["t"] + 1) % 8) == 0
        burning = (level > 0) | (spread & (old["fuel"] > 0) & ~wet)
        next_level = jnp.where((old["fuel"] > 0) & burning & ~wet, jnp.minimum(level + 1, 2), 0)
        next_fuel = jnp.maximum(old["fuel"] - burning.astype(jnp.int32), 0)
        s["fire_level"] = jnp.where(pulse, next_level, level)
        s["fuel"] = jnp.where(pulse, next_fuel, old["fuel"])
        cell = s["person_pos"][:, 1] * 9 + s["person_pos"][:, 0]
        harm = (s["person_status"] == 0) & (s["fire_level"][cell] > 0)
        s["person_health"] = jnp.maximum(old["person_health"] - harm.astype(jnp.int32), 0)
        lost = (s["person_status"] == 0) & (s["person_health"] == 0)
        s["person_status"] = jnp.where(lost, 2, s["person_status"])
        s["person_holder"] = jnp.where(lost, -1, s["person_holder"])
        s["new_rescue"], s["new_loss"] = evacuated, lost
        s["rescued"] = jnp.sum((s["person_status"] == 1).astype(jnp.int32))
        s["lost"] = jnp.sum((s["person_status"] == 2).astype(jnp.int32))
        # TERMINAL: a casualty fails the joint rescue; all two evacuated succeeds.
        s["t"] = old["t"] + 1
        s["terminated"] = (s["rescued"] == 2) | (s["lost"] > 0)
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        s = self.refresh(s)
        r = 4 * jnp.sum(evacuated.astype(jnp.int32)) - 8 * jnp.sum(lost.astype(jnp.int32)) - 1
        reward = jnp.full(3, r, jnp.float32)
        info = {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}
        return self.observe(s), s, reward, jnp.full(3, s["done"], jnp.bool_), info

