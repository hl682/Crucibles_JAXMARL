"""Tide locks and competitive salvage crews. Original author kernel; unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 170
    num_actions = 10
    max_steps = 96

    def reset(self, key):
        # Key is accepted; this rule variant has a deterministic initial tide and layout.
        xy = jnp.stack((jnp.arange(77) % 11, jnp.arange(77) // 11), axis=1)
        x, y = xy[:, 0], xy[:, 1]
        wall = (x == 0) | (x == 10) | (y == 0) | (y == 6) | ((x == 5) & (y != 2) & (y != 4))
        tiles = jnp.where(wall, 1, 0).reshape(7, 11).astype(jnp.int32)
        tiles = tiles.at[2, 5].set(2).at[4, 5].set(2)
        tiles = tiles.at[1, 4].set(3).at[1, 6].set(3).at[5, 4].set(3).at[5, 6].set(3)
        tiles = tiles.at[3, 1].set(4).at[3, 9].set(5)
        s = {
            "t": jnp.array(0, jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "tiles": tiles, "tide": jnp.array(0, jnp.int32),
            "agent_pos": jnp.array([[4, 1], [4, 2], [6, 5], [6, 4]], jnp.int32),
            "agent_active": jnp.ones(4, jnp.bool_), "agent_vis": jnp.zeros((4, 4), jnp.bool_),
            "team": jnp.array([0, 0, 1, 1], jnp.int32),
            "role": jnp.array([0, 1, 0, 1], jnp.int32),
            "cooldown": jnp.zeros(4, jnp.int32), "holding": jnp.zeros(4, jnp.bool_),
            "crate_pos": jnp.array([[6, 2], [4, 4], [6, 3]], jnp.int32),
            "crate_active": jnp.ones(3, jnp.bool_), "crate_vis": jnp.zeros((4, 3), jnp.bool_),
            "crate_holder": jnp.full(3, -1, jnp.int32),
            "crate_delivered_team": jnp.full(3, -1, jnp.int32),
            "crate_weight": jnp.ones(3, jnp.int32),
            "lock_pos": jnp.array([[5, 2], [5, 4]], jnp.int32),
            "lock_active": jnp.ones(2, jnp.bool_), "lock_vis": jnp.ones((4, 2), jnp.bool_),
            "lock_charge": jnp.full(2, 4, jnp.int32),
            "lock_timer": jnp.zeros(2, jnp.int32), "lock_tickets": jnp.zeros(2, jnp.int32),
            "lock_operator": jnp.full(2, -1, jnp.int32),
            "console_pos": jnp.array([[4, 1], [6, 1], [4, 5], [6, 5]], jnp.int32),
            "console_active": jnp.ones(4, jnp.bool_), "console_vis": jnp.ones((4, 4), jnp.bool_),
            "base_pos": jnp.array([[1, 3], [9, 3]], jnp.int32),
            "base_active": jnp.ones(2, jnp.bool_), "base_vis": jnp.ones((4, 2), jnp.bool_),
            "base_team": jnp.array([0, 1], jnp.int32),
            "score": jnp.zeros(2, jnp.int32), "winner": jnp.array(-1, jnp.int32),
            "crate_total": jnp.array(3, jnp.int32),
            "held_event": jnp.zeros(2, jnp.bool_), "crank_event": jnp.zeros(2, jnp.bool_),
            "cross_event": jnp.zeros(4, jnp.bool_), "pickup_event": jnp.zeros(4, jnp.bool_),
            "delivery_event": jnp.zeros(4, jnp.bool_),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def refresh(self, state):
        s = dict(state)
        p = s["agent_pos"]
        near = jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=2) <= 3
        s["agent_vis"] = near | (s["team"][:, None] == s["team"][None, :])
        s["holding"] = jnp.any(s["crate_holder"][None, :] == jnp.arange(4)[:, None], axis=1)
        crate_near = jnp.sum(jnp.abs(p[:, None, :] - s["crate_pos"][None, :, :]), axis=2) <= 3
        friendly_cargo = (s["crate_holder"][None, :] >= 0) & (s["team"][:, None] == s["team"][jnp.maximum(s["crate_holder"], 0)][None, :])
        s["crate_vis"] = crate_near | friendly_cargo
        s["score"] = jnp.sum((s["crate_delivered_team"][None, :] == jnp.arange(2)[:, None]).astype(jnp.int32), axis=1)
        s["crate_total"] = jnp.sum(s["crate_active"].astype(jnp.int32))
        return s

    def records(self, position, attributes, visibility):
        values = jnp.concatenate((position, attributes), axis=1)
        values = jnp.where(visibility[:, :, None], values[None, :, :], -1)
        return jnp.concatenate((visibility[:, :, None].astype(jnp.int32), values), axis=2).reshape(4, -1)

    def observe(self, state):
        s = state
        public = jnp.stack((s["t"], s["tide"], s["score"][0], s["score"][1], s["done"].astype(jnp.int32), s["winner"]))
        own = jnp.concatenate((jnp.arange(4)[:, None], s["team"][:, None], s["role"][:, None],
                               s["agent_pos"], s["cooldown"][:, None], s["holding"][:, None]), axis=1)
        agents = self.records(s["agent_pos"], jnp.stack((s["team"], s["role"], s["cooldown"], s["holding"]), axis=1), s["agent_vis"])
        crates = self.records(s["crate_pos"], jnp.stack((s["crate_holder"], s["crate_delivered_team"], s["crate_weight"]), axis=1), s["crate_vis"])
        locks = self.records(s["lock_pos"], jnp.stack((s["lock_charge"], s["lock_timer"], s["lock_tickets"], s["lock_operator"]), axis=1), s["lock_vis"])
        consoles = self.records(s["console_pos"], jnp.zeros((4, 0), jnp.int32), s["console_vis"])
        bases = self.records(s["base_pos"], s["base_team"][:, None], s["base_vis"])
        return jnp.concatenate((jnp.broadcast_to(public, (4, 6)), own,
                                jnp.broadcast_to(s["tiles"].reshape(1, 77), (4, 77)),
                                agents, crates, locks, consoles, bases), axis=1).astype(jnp.float32)

    def legal_actions(self, state):
        s = state
        p = s["agent_pos"]
        delta = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]], jnp.int32)
        target = p[:, None, :] + delta[None, :, :]
        x, y = jnp.clip(target[:, :, 0], 0, 10), jnp.clip(target[:, :, 1], 0, 6)
        inside = (target[:, :, 0] > 0) & (target[:, :, 0] < 10) & (target[:, :, 1] > 0) & (target[:, :, 1] < 6)
        moves = inside & (s["tiles"][y, x] != 1) & (s["cooldown"][:, None] == 0)
        console_near = jnp.sum(jnp.abs(p[:, None, :] - s["console_pos"][None, :, :]), axis=2) <= 1
        gate_near = jnp.stack((jnp.any(console_near[:, :2], axis=1), jnp.any(console_near[:, 2:], axis=1)), axis=1)
        rigger = s["role"] == 0
        hold = rigger & (s["tide"] != 2) & jnp.any(gate_near & (s["lock_charge"][None, :] > 0), axis=1)
        crank = rigger & jnp.any(gate_near & (s["lock_charge"][None, :] < 4), axis=1)
        eligible_crate = (jnp.sum(jnp.abs(p[:, None, :] - s["crate_pos"][None, :, :]), axis=2) <= 1) & (s["crate_holder"][None, :] < 0) & (s["crate_delivered_team"][None, :] < 0)
        lift = ~rigger & ~s["holding"] & jnp.any(eligible_crate, axis=1)
        drop = ~rigger & s["holding"]
        at_base = jnp.all(p == s["base_pos"][s["team"]], axis=1)
        deposit = drop & at_base
        mask = jnp.concatenate((jnp.ones((4, 1), jnp.bool_), moves, jnp.stack((hold, crank, lift, drop, deposit), axis=1)), axis=1)
        return jnp.where(s["done"], jnp.arange(10)[None, :] == 0, mask)

    def raw_step(self, key, state, actions):
        actions = jnp.asarray(actions, jnp.int32)
        clipped = jnp.clip(actions, 0, 9)
        valid = (actions == clipped) & self.legal_actions(state)[jnp.arange(4), clipped]
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, jnp.where(valid, clipped, 0)))

    def absorb(self, bundle):
        key, s, a = bundle
        return self.observe(s), s, jnp.zeros(4, jnp.float32), jnp.ones(4, jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def advance(self, bundle):
        key, old, a = bundle
        s = dict(old)
        ids = jnp.arange(4)
        rank = (ids - old["t"]) % 4
        p = old["agent_pos"]
        near = jnp.sum(jnp.abs(p[:, None, :] - old["console_pos"][None, :, :]), axis=2) <= 1
        gate_near = jnp.stack((jnp.any(near[:, :2], axis=1), jnp.any(near[:, 2:], axis=1)), axis=1)
        eligible_hold = gate_near & (old["lock_charge"][None, :] > 0)
        eligible_crank = gate_near & (old["lock_charge"][None, :] < 4)
        chosen_hold = jnp.argmax(eligible_hold, axis=1)
        chosen_crank = jnp.argmax(eligible_crank, axis=1)
        charge = old["lock_charge"]
        # Existing opening lasts through this transition if its old timer is positive.
        open_now = old["lock_timer"] > 0
        tickets = old["lock_tickets"]
        timers = jnp.maximum(old["lock_timer"] - 1, 0)
        operators = jnp.full(2, -1, jnp.int32)
        held = jnp.zeros(2, jnp.bool_)
        cranked = jnp.zeros(2, jnp.bool_)
        for g in range(2):
            holders = (a == 5) & eligible_hold[:, g] & (chosen_hold == g)
            winner = jnp.argmin(jnp.where(holders, rank, 9))
            has_holder = jnp.any(holders)
            crankers = (a == 6) & eligible_crank[:, g] & (chosen_crank == g)
            has_crank = jnp.any(crankers)
            # One shared counterweight stroke per gate regardless of the number of crankers.
            charge = charge.at[g].set(jnp.minimum(4, charge[g] - has_holder.astype(jnp.int32) + has_crank.astype(jnp.int32)))
            timers = timers.at[g].set(jnp.where(has_holder, 3, timers[g]))
            tickets = tickets.at[g].set(jnp.where(has_holder, 2, tickets[g]))
            open_now = open_now.at[g].set(open_now[g] | has_holder)
            operators = operators.at[g].set(jnp.where(has_holder, winner, -1))
            held = held.at[g].set(has_holder)
            cranked = cranked.at[g].set(has_crank)
        # LOCK_CROSSING: current teammate support is needed whenever carrying touches a lock.
        delta = jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], jnp.int32)
        target = p + delta[jnp.minimum(a, 4)] * (a <= 4)[:, None]
        enters = jnp.all(target[:, None, :] == old["lock_pos"][None, :, :], axis=2)
        leaves = jnp.all(p[:, None, :] == old["lock_pos"][None, :, :], axis=2)
        touches = enters | leaves
        support = (operators[None, :] >= 0) & (old["team"][:, None] == old["team"][jnp.maximum(operators, 0)][None, :])
        enter_allowed = open_now[None, :] & (tickets[None, :] > 0) & (old["tide"] != 2)
        denied_entry = jnp.any(enters & ~enter_allowed, axis=1)
        denied_load = old["holding"] & jnp.any(touches & (~support | (old["tide"] == 2)), axis=1)
        moving = (a >= 1) & (a <= 4) & ~denied_entry & ~denied_load
        target = jnp.where(moving[:, None], target, p)
        occupied = jnp.any(jnp.all(target[:, None, :] == p[None, :, :], axis=2) & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(target[:, None, :] == target[None, :, :], axis=2).astype(jnp.int32), axis=1) > 1
        moved = moving & ~occupied & ~contested
        p = jnp.where(moved[:, None], target, p)
        spent_ticket = jnp.sum((enters & moved[:, None]).astype(jnp.int32), axis=0)
        tickets = tickets - spent_ticket
        tickets = jnp.where(timers > 0, tickets, 0)
        s["lock_charge"], s["lock_timer"], s["lock_tickets"], s["lock_operator"] = charge, timers, tickets, operators
        s["held_event"], s["crank_event"] = held, cranked
        s["cross_event"] = moved & jnp.any(touches, axis=1)
        s["agent_pos"] = p
        # LIFT_CAPACITY: one weight-one crate per hauler; loaded steps have a recovery tick.
        s["cooldown"] = jnp.where(moved & old["holding"], 1, jnp.maximum(old["cooldown"] - 1, 0))
        holder = old["crate_holder"]
        delivered_team = old["crate_delivered_team"]
        near_crate = (jnp.sum(jnp.abs(p[:, None, :] - old["crate_pos"][None, :, :]), axis=2) <= 1) & (holder[None, :] < 0) & (delivered_team[None, :] < 0)
        choice = jnp.argmax(near_crate, axis=1)
        picked = jnp.zeros(4, jnp.bool_)
        deposited = jnp.zeros(4, jnp.bool_)
        for c in range(3):
            wants = (a == 7) & near_crate[:, c] & (choice == c)
            winner = jnp.argmin(jnp.where(wants, rank, 9))
            pickup = jnp.any(wants)
            previous = jnp.maximum(holder[c], 0)
            drop = (holder[c] >= 0) & (a[previous] == 8)
            deposit = (holder[c] >= 0) & (a[previous] == 9)
            nh = jnp.where(pickup, winner, jnp.where(drop | deposit, -1, holder[c]))
            place = jnp.where((drop | deposit), p[previous], old["crate_pos"][c])
            place = jnp.where(nh >= 0, p[jnp.maximum(nh, 0)], place)
            s["crate_holder"] = s["crate_holder"].at[c].set(nh)
            s["crate_pos"] = s["crate_pos"].at[c].set(place)
            delivered_team = delivered_team.at[c].set(jnp.where(deposit, old["team"][previous], delivered_team[c]))
            picked = picked.at[winner].set(picked[winner] | pickup)
            deposited = deposited.at[previous].set(deposited[previous] | deposit)
        s["crate_delivered_team"] = delivered_team
        s["pickup_event"], s["delivery_event"] = picked, deposited
        s = self.refresh(s)
        s["t"] = old["t"] + 1
        s["tide"] = (s["t"] // 3) % 4
        # TERMINAL: first majority of the three indivisible crates wins; timeout can draw.
        s["terminated"] = jnp.any(s["score"] >= 2)
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        leading = jnp.where(s["score"][0] > s["score"][1], 0, jnp.where(s["score"][1] > s["score"][0], 1, -1))
        s["winner"] = jnp.where(s["done"], leading, -1).astype(jnp.int32)
        gain = s["score"] - old["score"]
        margin = 5 * (gain[0] - gain[1])
        bonus = jnp.where(s["done"] & (s["winner"] >= 0), jnp.where(s["winner"] == 0, 8, -8), 0)
        reward = jnp.where(s["team"] == 0, margin + bonus, -margin - bonus).astype(jnp.float32)
        return self.observe(s), s, reward, jnp.full(4, s["done"], jnp.bool_), {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

