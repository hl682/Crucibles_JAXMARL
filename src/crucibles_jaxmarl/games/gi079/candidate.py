"""Switchback Relay: spatial breach, timed support and physical extraction.

Original authored kernel; no source execution or learning results are claimed.
"""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 414
    num_actions = 13
    max_steps = 160

    def visibility(self, s):
        s = dict(s)
        p = s["agent_pos"]
        s["agent_visibility"] = jnp.sum(jnp.abs(p[:, None, :] - p[None, :, :]), axis=-1) <= 5
        s["turret_visibility"] = (jnp.sum(jnp.abs(p[:, None, :] - s["turret_pos"][None, :, :]), axis=-1) <= 5) | (s["turret_mark"][None, :] > 0)
        s["bolt_visibility"] = (jnp.sum(jnp.abs(p[:, None, :] - s["bolt_pos"][None, :, :]), axis=-1) <= 5) & s["bolt_active"][None, :]
        s["shield_visibility"] = (jnp.sum(jnp.abs(p[:, None, :] - s["shield_pos"][None, :, :]), axis=-1) <= 5) & s["shield_active"][None, :]
        return s

    def reset(self, key):
        grid = jnp.zeros((9, 13), dtype=jnp.int32)
        grid = grid.at[0, :].set(1).at[8, :].set(1)
        grid = grid.at[:, 0].set(1).at[:, 12].set(1)
        grid = grid.at[1:8, 6].set(1).at[4, 6].set(2)
        grid = grid.at[3, 8].set(1).at[5, 8].set(1).at[3, 10].set(1).at[5, 10].set(1)
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "success": jnp.array(False), "map": grid,
            "agent_pos": jnp.array([[2, 2], [2, 4], [2, 6]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_visibility": jnp.ones((3, 3), dtype=jnp.bool_),
            "agent_role": jnp.array([0, 1, 2], dtype=jnp.int32),
            "agent_health": jnp.full((3,), 4, dtype=jnp.int32),
            "agent_ammo": jnp.full((3,), 5, dtype=jnp.int32),
            "agent_noise": jnp.zeros((3,), dtype=jnp.int32),
            "agent_facing": jnp.array([[1, 0], [1, 0], [1, 0]], dtype=jnp.int32),
            "support": jnp.array(6, dtype=jnp.int32),
            "turret_pos": jnp.array([[11, 2], [11, 4], [11, 6]], dtype=jnp.int32),
            "turret_active": jnp.ones((3,), dtype=jnp.bool_),
            "turret_visibility": jnp.zeros((3, 3), dtype=jnp.bool_),
            "turret_health": jnp.full((3,), 3, dtype=jnp.int32),
            "turret_clock": jnp.array([4, 6, 8], dtype=jnp.int32),
            "turret_mark": jnp.zeros((3,), dtype=jnp.int32),
            "turret_jam": jnp.zeros((3,), dtype=jnp.int32),
            "turret_aim": jnp.array([[-1, 0], [-1, 0], [-1, 0]], dtype=jnp.int32),
            "bolt_pos": jnp.zeros((24, 2), dtype=jnp.int32),
            "bolt_active": jnp.zeros((24,), dtype=jnp.bool_),
            "bolt_visibility": jnp.zeros((3, 24), dtype=jnp.bool_),
            "bolt_vel": jnp.zeros((24, 2), dtype=jnp.int32),
            "bolt_origin": jnp.zeros((24, 2), dtype=jnp.int32),
            "bolt_owner": jnp.zeros((24,), dtype=jnp.int32),
            "bolt_ttl": jnp.zeros((24,), dtype=jnp.int32),
            "shield_pos": jnp.zeros((3, 2), dtype=jnp.int32),
            "shield_active": jnp.zeros((3,), dtype=jnp.bool_),
            "shield_visibility": jnp.zeros((3, 3), dtype=jnp.bool_),
            "shield_ttl": jnp.zeros((3,), dtype=jnp.int32),
            "terminal_pos": jnp.array([[4, 2], [4, 6]], dtype=jnp.int32),
            "terminal_active": jnp.ones((2,), dtype=jnp.bool_),
            "terminal_visibility": jnp.ones((3, 2), dtype=jnp.bool_),
            "terminal_charge": jnp.zeros((2,), dtype=jnp.int32),
            "gate_pos": jnp.array([[6, 4]], dtype=jnp.int32),
            "gate_active": jnp.ones((1,), dtype=jnp.bool_),
            "gate_visibility": jnp.ones((3, 1), dtype=jnp.bool_),
            "gate_open": jnp.zeros((1,), dtype=jnp.bool_),
            "gate_ever_open": jnp.array(False),
            "core_pos": jnp.array([[10, 4]], dtype=jnp.int32),
            "core_active": jnp.ones((1,), dtype=jnp.bool_),
            "core_visibility": jnp.ones((3, 1), dtype=jnp.bool_),
            "core_hull": jnp.array([6], dtype=jnp.int32),
            "core_carrier": jnp.array([-1], dtype=jnp.int32),
            "core_ever_picked": jnp.array(False), "core_crossed": jnp.array(False),
            "supply_pos": jnp.array([[2, 4]], dtype=jnp.int32),
            "supply_active": jnp.ones((1,), dtype=jnp.bool_),
            "supply_visibility": jnp.ones((3, 1), dtype=jnp.bool_),
            "supply_stock": jnp.array([6], dtype=jnp.int32),
            "exit_pos": jnp.array([[1, 4]], dtype=jnp.int32),
            "exit_active": jnp.ones((1,), dtype=jnp.bool_),
            "exit_visibility": jnp.ones((3, 1), dtype=jnp.bool_),
            "ev_fire": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_ability": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_channel": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_pickup": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_drop": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_reload": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_hit_agent": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_hit_turret": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_enemy_fire": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_blocked": jnp.zeros((24,), dtype=jnp.bool_),
        }
        s = self.visibility(s)
        return self.observe(s), s

    def observe(self, s):
        carrier = s["core_carrier"][0]
        own = jnp.concatenate([jax.nn.one_hot(s["agent_role"], 3, dtype=jnp.float32),
            s["agent_pos"] / 12.0, s["agent_health"][:, None] / 4.0,
            s["agent_ammo"][:, None] / 5.0, s["agent_noise"][:, None] / 8.0,
            s["agent_facing"].astype(jnp.float32), (jnp.arange(3) == carrier)[:, None].astype(jnp.float32)], axis=1)
        public = jnp.concatenate([jnp.stack([s["t"] / 160.0, s["support"] / 12.0]),
            s["core_pos"].reshape(-1) / 12.0, s["core_hull"] / 6.0,
            (s["core_carrier"] + 1) / 3.0, s["gate_open"].astype(jnp.float32),
            jnp.stack([s["core_ever_picked"], s["core_crossed"]]).astype(jnp.float32),
            s["map"].reshape(-1) / 2.0])
        ag = jnp.concatenate([s["agent_pos"] / 12.0, s["agent_health"][:, None] / 4.0,
            s["agent_ammo"][:, None] / 5.0, s["agent_noise"][:, None] / 8.0,
            s["agent_role"][:, None] / 2.0, s["agent_facing"].astype(jnp.float32),
            s["agent_active"][:, None].astype(jnp.float32)], axis=1)
        tu = jnp.concatenate([s["turret_pos"] / 12.0, s["turret_health"][:, None] / 3.0,
            s["turret_clock"][:, None] / 8.0, s["turret_mark"][:, None] / 8.0,
            s["turret_jam"][:, None] / 4.0, s["turret_aim"].astype(jnp.float32)], axis=1)
        bo = jnp.concatenate([s["bolt_pos"] / 12.0, s["bolt_vel"].astype(jnp.float32),
            s["bolt_owner"][:, None] / 5.0, s["bolt_ttl"][:, None] / 14.0,
            s["bolt_active"][:, None].astype(jnp.float32)], axis=1)
        sh = jnp.concatenate([s["shield_pos"] / 12.0, s["shield_ttl"][:, None] / 4.0,
            s["shield_active"][:, None].astype(jnp.float32)], axis=1)
        tables = []
        for values, visible in [(ag, s["agent_visibility"]), (tu, s["turret_visibility"]),
                                (bo, s["bolt_visibility"]), (sh, s["shield_visibility"])]:
            rows = jnp.concatenate([jnp.where(visible[:, :, None], values[None, :, :], 0.0), visible[:, :, None].astype(jnp.float32)], axis=2)
            tables = tables + [rows.reshape(3, -1)]
        fixed = jnp.concatenate([jnp.concatenate([s["terminal_pos"] / 12.0, s["terminal_charge"][:, None] / 6.0], axis=1).reshape(-1),
            s["supply_pos"].reshape(-1) / 12.0, s["supply_stock"] / 6.0,
            s["gate_pos"].reshape(-1) / 12.0, s["exit_pos"].reshape(-1) / 12.0])
        return jnp.concatenate([own, jnp.broadcast_to(public[None, :], (3, 126))] + tables + [jnp.broadcast_to(fixed[None, :], (3, 13))], axis=1).astype(jnp.float32)

    def move_options(self, s):
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        target = jnp.clip(s["agent_pos"][:, None, :] + dirs[None, :, :], jnp.array([0, 0]), jnp.array([12, 8]))
        occupied = jnp.any(jnp.all(target[:, :, None, :] == s["agent_pos"][None, None, :, :], axis=-1) & s["agent_active"][None, None, :], axis=-1)
        enemy = jnp.any(jnp.all(target[:, :, None, :] == s["turret_pos"][None, None, :, :], axis=-1) & s["turret_active"][None, None, :], axis=-1)
        return target, (s["map"][target[:, :, 1], target[:, :, 0]] == 0) & ~occupied & ~enemy

    def legal_actions(self, s):
        target, moves = self.move_options(s)
        carrying = jnp.arange(3) == s["core_carrier"][0]
        fire = (s["agent_ammo"] > 0) & ~carrying
        front = jnp.clip(s["agent_pos"] + s["agent_facing"], jnp.array([0, 0]), jnp.array([12, 8]))
        ability = (s["support"] >= jnp.where(s["agent_role"] == 2, 2, 1)) & ~carrying & ((s["agent_role"] != 0) | (s["map"][front[:, 1], front[:, 0]] == 0))
        near_panel = jnp.any(jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["terminal_pos"][None, :, :]), axis=-1) <= 1, axis=1)
        near_core = jnp.sum(jnp.abs(s["agent_pos"] - s["core_pos"][0]), axis=1) <= 1
        interact = near_panel | carrying | (near_core & (s["core_carrier"][0] < 0))
        reload = (jnp.sum(jnp.abs(s["agent_pos"] - s["supply_pos"][0]), axis=1) <= 1) & (s["supply_stock"][0] > 0) & ((s["agent_ammo"] < 5) | (s["support"] < 12))
        mask = jnp.concatenate([jnp.ones((3, 1), dtype=jnp.bool_), moves, jnp.broadcast_to(fire[:, None], (3, 4)), ability[:, None], interact[:, None], (~carrying)[:, None], reload[:, None]], axis=1)
        available = s["agent_active"] & ~s["done"]
        return jnp.where(available[:, None], mask, jnp.broadcast_to(jnp.arange(13) == 0, (3, 13)))

    def launch(self, s, origin, velocity, owner, wanted):
        s = dict(s)
        slot = jnp.argmax((~s["bolt_active"]).astype(jnp.int32))
        fired = wanted & jnp.any(~s["bolt_active"])
        s["bolt_pos"] = s["bolt_pos"].at[slot].set(jnp.where(fired, origin, s["bolt_pos"][slot]))
        s["bolt_origin"] = s["bolt_origin"].at[slot].set(jnp.where(fired, origin, s["bolt_origin"][slot]))
        s["bolt_vel"] = s["bolt_vel"].at[slot].set(jnp.where(fired, velocity, s["bolt_vel"][slot]))
        s["bolt_owner"] = s["bolt_owner"].at[slot].set(jnp.where(fired, owner, s["bolt_owner"][slot]))
        s["bolt_ttl"] = s["bolt_ttl"].at[slot].set(jnp.where(fired, 14, s["bolt_ttl"][slot]))
        s["bolt_active"] = s["bolt_active"].at[slot].set(s["bolt_active"][slot] | fired)
        return s, fired

    def absorb(self, key, s, actions):
        return self.observe(s), s, jnp.zeros((3,), dtype=jnp.float32), jnp.ones((3,), dtype=jnp.bool_), {
            "done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def advance(self, key, state, actions):
        s = dict(state)
        clipped = jnp.clip(actions, 0, 12)
        valid = (actions >= 0) & (actions < 13) & jnp.take_along_axis(self.legal_actions(s), clipped[:, None], axis=1)[:, 0]
        act = jnp.where(valid, clipped, 0)
        target, move_ok = self.move_options(s)
        moving = (act >= 1) & (act <= 4)
        chosen = target[jnp.arange(3), jnp.clip(act - 1, 0, 3)]
        priority = jnp.mod(jnp.arange(3) - s["t"], 3)
        contested = jnp.all(chosen[:, None, :] == chosen[None, :, :], axis=-1)
        lose = jnp.any(contested & moving[None, :] & (priority[None, :] < priority[:, None]), axis=1)
        oldpos = s["agent_pos"]
        s["agent_pos"] = jnp.where((moving & ~lose)[:, None], chosen, oldpos)
        s["agent_facing"] = jnp.where((moving & ~lose)[:, None], s["agent_pos"] - oldpos, s["agent_facing"])
        ci = jnp.maximum(0, s["core_carrier"][0])
        s["core_pos"] = jnp.where((s["core_carrier"][0] >= 0), s["agent_pos"][ci][None, :], s["core_pos"])
        s["turret_mark"] = jnp.maximum(0, s["turret_mark"] - 1)
        s["turret_jam"] = jnp.maximum(0, s["turret_jam"] - 1)
        s["shield_ttl"] = jnp.maximum(0, s["shield_ttl"] - 1)
        for name in ["ev_fire", "ev_ability", "ev_channel", "ev_pickup", "ev_drop", "ev_reload", "ev_hit_agent", "ev_hit_turret", "ev_enemy_fire"]:
            s[name] = jnp.zeros((3,), dtype=jnp.bool_)
        channel = jnp.zeros((2,), dtype=jnp.int32)
        for offset in range(3):
            i = jnp.mod(s["t"] + offset, 3)
            panel = jnp.sum(jnp.abs(s["terminal_pos"] - s["agent_pos"][i]), axis=1) <= 1
            operate = (act[i] == 10) & jnp.any(panel)
            pi = jnp.argmax(panel.astype(jnp.int32))
            channel = channel.at[pi].add(operate.astype(jnp.int32) * (2 + (s["agent_role"][i] == 2).astype(jnp.int32)))
            s["ev_channel"] = s["ev_channel"].at[i].set(operate)
            free_interact = (act[i] == 10) & ~jnp.any(panel)
            drop = free_interact & (s["core_carrier"][0] == i)
            near = jnp.sum(jnp.abs(s["agent_pos"][i] - s["core_pos"][0])) <= 1
            pick = free_interact & (s["core_carrier"][0] < 0) & near
            s["core_carrier"] = s["core_carrier"].at[0].set(jnp.where(drop, -1, jnp.where(pick, i, s["core_carrier"][0])))
            s["core_pos"] = jnp.where((drop | pick), s["agent_pos"][i][None, :], s["core_pos"])
            s["ev_pickup"] = s["ev_pickup"].at[i].set(pick)
            s["ev_drop"] = s["ev_drop"].at[i].set(drop)
            cost = jnp.where(s["agent_role"][i] == 2, 2, 1)
            use = (act[i] == 9) & (s["support"] >= cost)
            s["support"] = s["support"] - use.astype(jnp.int32) * cost
            distance = jnp.sum(jnp.abs(s["turret_pos"] - s["agent_pos"][i]), axis=1)
            scan = use & (s["agent_role"][i] == 1)
            jam = use & (s["agent_role"][i] == 2)
            shield = use & (s["agent_role"][i] == 0)
            s["turret_mark"] = jnp.where(scan & (distance <= 6) & s["turret_active"], 8, s["turret_mark"])
            s["turret_jam"] = jnp.where(jam & (distance <= 5) & s["turret_active"], 4, s["turret_jam"])
            front = jnp.clip(s["agent_pos"][i] + s["agent_facing"][i], jnp.array([0, 0]), jnp.array([12, 8]))
            s["shield_pos"] = s["shield_pos"].at[i].set(jnp.where(shield, front, s["shield_pos"][i]))
            s["shield_ttl"] = s["shield_ttl"].at[i].set(jnp.where(shield, 4, s["shield_ttl"][i]))
            s["ev_ability"] = s["ev_ability"].at[i].set(use)
            refill = (act[i] == 12) & (s["supply_stock"][0] > 0) & ((s["agent_ammo"][i] < 5) | (s["support"] < 12))
            s["supply_stock"] = s["supply_stock"].at[0].add(-refill.astype(jnp.int32))
            s["agent_ammo"] = s["agent_ammo"].at[i].set(jnp.minimum(5, s["agent_ammo"][i] + refill.astype(jnp.int32)))
            s["support"] = jnp.minimum(12, s["support"] + refill.astype(jnp.int32))
            s["ev_reload"] = s["ev_reload"].at[i].set(refill)
        s["terminal_charge"] = jnp.where(channel > 0, jnp.minimum(6, s["terminal_charge"] + channel), jnp.maximum(0, s["terminal_charge"] - 1))
        occupied_gate = jnp.any(jnp.all(s["agent_pos"] == s["gate_pos"][0], axis=1) & s["agent_active"]) | jnp.all(s["core_pos"][0] == s["gate_pos"][0])
        opened = jnp.all(s["terminal_charge"] > 0) | occupied_gate
        s["gate_open"] = jnp.array([opened])
        s["map"] = s["map"].at[4, 6].set(jnp.where(opened, 0, 2))
        s["shield_active"] = s["shield_ttl"] > 0
        s["supply_active"] = s["supply_stock"] > 0
        guarded = (act == 11) & s["agent_active"]
        close = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=-1) <= 1
        protected = guarded | jnp.any(close & (guarded & (s["agent_role"] == 0))[None, :], axis=1)
        proposed = s["bolt_pos"] + s["bolt_vel"]
        out = jnp.any((proposed < jnp.array([0, 0])) | (proposed > jnp.array([12, 8])), axis=1)
        bp = jnp.clip(proposed, jnp.array([0, 0]), jnp.array([12, 8]))
        wall = s["map"][bp[:, 1], bp[:, 0]] != 0
        enemy = s["bolt_owner"] >= 3
        blocked = jnp.any(jnp.all(bp[:, None, :] == s["shield_pos"][None, :, :], axis=-1) & s["shield_active"][None, :], axis=1) & enemy
        travelling = s["bolt_active"] & ~out & ~wall & ~blocked
        hits_agent = travelling[:, None] & enemy[:, None] & jnp.all(bp[:, None, :] == s["agent_pos"][None, :, :], axis=-1) & s["agent_active"][None, :]
        hits_turret = travelling[:, None] & ~enemy[:, None] & jnp.all(bp[:, None, :] == s["turret_pos"][None, :, :], axis=-1) & s["turret_active"][None, :]
        origin_near = jnp.sum(jnp.abs(s["bolt_origin"][:, None, :] - s["turret_pos"][None, :, :]), axis=-1) <= 2
        vulnerability = (s["turret_mark"][None, :] > 0) | origin_near
        damage_agent = jnp.sum(hits_agent.astype(jnp.int32), axis=0) * (~protected).astype(jnp.int32)
        damage_turret = jnp.sum((hits_turret & vulnerability).astype(jnp.int32) * jnp.where(s["turret_mark"][None, :] > 0, 2, 1), axis=0)
        hits_core = travelling & enemy & jnp.all(bp == s["core_pos"][0], axis=1) & ~jnp.any(hits_agent, axis=1)
        s["agent_health"] = jnp.maximum(0, s["agent_health"] - damage_agent)
        s["turret_health"] = jnp.maximum(0, s["turret_health"] - damage_turret)
        s["core_hull"] = jnp.maximum(0, s["core_hull"] - jnp.sum(hits_core.astype(jnp.int32)))
        s["ev_hit_agent"] = damage_agent > 0
        s["ev_hit_turret"] = damage_turret > 0
        s["ev_blocked"] = s["bolt_active"] & (blocked | (jnp.any(hits_agent, axis=1) & ~jnp.any(hits_agent & ~protected[None, :], axis=1)))
        impact = jnp.any(hits_agent, axis=1) | jnp.any(hits_turret, axis=1) | hits_core
        s["bolt_pos"] = jnp.where(s["bolt_active"][:, None], bp, s["bolt_pos"])
        s["bolt_ttl"] = jnp.maximum(0, s["bolt_ttl"] - 1)
        s["bolt_active"] = travelling & ~impact & (s["bolt_ttl"] > 0)
        s["agent_active"] = s["agent_health"] > 0
        s["turret_active"] = s["turret_health"] > 0
        ci = jnp.maximum(0, s["core_carrier"][0])
        lost_carrier = (s["core_carrier"][0] >= 0) & ~s["agent_active"][ci]
        s["core_carrier"] = jnp.where(lost_carrier, -jnp.ones((1,), dtype=jnp.int32), s["core_carrier"])
        directions = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        for offset in range(3):
            i = jnp.mod(s["t"] + offset, 3)
            wanted = (act[i] >= 5) & (act[i] <= 8) & s["agent_active"][i] & (s["agent_ammo"][i] > 0)
            velocity = directions[jnp.clip(act[i] - 5, 0, 3)]
            s, fired = self.launch(s, s["agent_pos"][i], velocity, i, wanted)
            s["agent_ammo"] = s["agent_ammo"].at[i].add(-fired.astype(jnp.int32))
            s["ev_fire"] = s["ev_fire"].at[i].set(fired)
            s["agent_facing"] = s["agent_facing"].at[i].set(jnp.where(fired, velocity, s["agent_facing"][i]))
        carrying = jnp.arange(3) == s["core_carrier"][0]
        s["agent_noise"] = jnp.minimum(8, jnp.maximum(0, s["agent_noise"] - 1) + 4 * s["ev_fire"].astype(jnp.int32) + 2 * s["ev_ability"].astype(jnp.int32) + moving.astype(jnp.int32) + carrying.astype(jnp.int32))
        for ti in range(3):
            delta = s["agent_pos"] - s["turret_pos"][ti]
            dist = jnp.sum(jnp.abs(delta), axis=1)
            score = 10 * s["agent_noise"] + 20 * carrying.astype(jnp.int32) - dist
            ai = jnp.argmax(jnp.where(s["agent_active"], score, -999))
            d = delta[ai]
            horizontal = jnp.abs(d[0]) >= jnp.abs(d[1])
            aim = jnp.where(horizontal, jnp.array([jnp.sign(d[0]), 0]), jnp.array([0, jnp.sign(d[1])])).astype(jnp.int32)
            live = s["turret_active"][ti] & (s["turret_jam"][ti] == 0) & jnp.any(s["agent_active"])
            acquire = live & (s["turret_clock"][ti] == 2)
            s["turret_aim"] = s["turret_aim"].at[ti].set(jnp.where(acquire, aim, s["turret_aim"][ti]))
            clock = jnp.where(live, jnp.maximum(0, s["turret_clock"][ti] - 1), s["turret_clock"][ti])
            wanted = live & (clock == 0)
            s, fired = self.launch(s, s["turret_pos"][ti], s["turret_aim"][ti], ti + 3, wanted)
            s["turret_clock"] = s["turret_clock"].at[ti].set(jnp.where(fired, 6, clock))
            s["ev_enemy_fire"] = s["ev_enemy_fire"].at[ti].set(fired)
        picked = jnp.any(s["ev_pickup"])
        crossed = s["core_ever_picked"] & (s["core_pos"][0, 0] < 6)
        first_open = opened & ~s["gate_ever_open"]
        first_pick = picked & ~s["core_ever_picked"]
        first_cross = crossed & ~s["core_crossed"]
        s["gate_ever_open"] = s["gate_ever_open"] | opened
        s["core_ever_picked"] = s["core_ever_picked"] | picked
        s["core_crossed"] = s["core_crossed"] | crossed
        success = jnp.all(s["core_pos"][0] == s["exit_pos"][0]) & (s["core_hull"][0] > 0) & jnp.any(s["agent_active"])
        failed = (s["core_hull"][0] <= 0) | ~jnp.any(s["agent_active"])
        s["core_active"] = s["core_hull"] > 0
        s["t"] = s["t"] + jnp.array(1, dtype=jnp.int32)
        s["success"] = success
        s["terminated"] = success | failed
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        killed = jnp.sum((state["turret_active"] & ~s["turret_active"]).astype(jnp.int32))
        lost = jnp.sum((state["agent_active"] & ~s["agent_active"]).astype(jnp.int32))
        reward = (-0.01 + 0.2 * first_open.astype(jnp.float32) + 0.2 * first_pick.astype(jnp.float32) + 0.5 * first_cross.astype(jnp.float32) + 0.5 * killed - 0.5 * lost + 3.0 * success.astype(jnp.float32) - 3.0 * failed.astype(jnp.float32)).astype(jnp.float32)
        s = self.visibility(s)
        return self.observe(s), s, jnp.full((3,), reward, dtype=jnp.float32), jnp.full((3,), s["done"], dtype=jnp.bool_), {
            "done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, key, state, actions)
