"""Cinderline Commons: original source proposal; AUTHORED_NOT_EXECUTED."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 1471
    num_actions = 10
    max_steps = 144
    terrain_rows = [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,1,1,1,2,1,1,1,1,1,0],[0,1,1,1,1,0,2,0,1,1,1,1,0],[0,1,1,5,1,1,2,1,1,5,1,1,0],[0,1,1,1,1,1,2,1,1,1,1,1,0],[0,3,2,2,2,2,4,2,2,2,2,3,0],[0,1,1,1,1,1,2,1,1,1,1,1,0],[0,1,1,0,1,1,2,1,1,0,1,1,0],[0,1,1,1,1,1,5,1,1,1,1,1,0],[0,1,1,1,1,1,2,1,1,1,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]]
    ignition_xy = [[[3,1],[9,1],[2,8]],[[2,1],[10,1],[4,9]],[[1,2],[11,2],[10,8]],[[4,1],[8,1],[8,9]]]

    def directions(self):
        return jnp.array([[0,-1],[1,0],[0,1],[-1,0]], dtype=jnp.int32)

    def empty_events(self):
        return {
            "event_active": jnp.zeros((160,), dtype=jnp.bool_),
            "event_kind": jnp.zeros((160,), dtype=jnp.int32),
            "event_actor": jnp.full((160,), -1, dtype=jnp.int32),
            "event_from": jnp.full((160,2), -1, dtype=jnp.int32),
            "event_to": jnp.full((160,2), -1, dtype=jnp.int32),
            "event_value": jnp.zeros((160,4), dtype=jnp.int32),
            "event_tick": jnp.full((160,), -1, dtype=jnp.int32)
        }

    def refresh_views(self, state):
        out = dict(state)
        out["agent_visibility"] = jnp.broadcast_to(out["agent_active"][None,:], (3,3))
        out["seedbed_visibility"] = jnp.broadcast_to(out["seedbed_active"][None,:], (3,3))
        out["cell_heat"] = out["heat"].reshape((143,))
        out["cell_fuel"] = out["fuel"].reshape((143,))
        out["cell_wetness"] = out["wetness"].reshape((143,))
        out["flame_active"] = out["cell_heat"] > 0
        out["brush_active"] = (out["terrain"].reshape((143,)) == 1) & (out["cell_fuel"] > 0)
        out["wet_active"] = out["cell_wetness"] > 0
        out["flame_visibility"] = jnp.broadcast_to(out["flame_active"][None,:], (3,143))
        out["brush_visibility"] = jnp.broadcast_to(out["brush_active"][None,:], (3,143))
        out["wet_visibility"] = jnp.broadcast_to(out["wet_active"][None,:], (3,143))
        return out

    def reset(self, key):
        keys = jax.random.split(key, 2)
        preset = jax.random.randint(keys[0], (), 0, 4, dtype=jnp.int32)
        wind = jax.random.randint(keys[1], (), 0, 4, dtype=jnp.int32)
        terrain = jnp.array(self.terrain_rows, dtype=jnp.int32)
        ignition = jnp.array(self.ignition_xy, dtype=jnp.int32)[preset]
        heat = jnp.zeros((11,13), dtype=jnp.int32).at[ignition[:,1], ignition[:,0]].set(2)
        cells = jnp.arange(143, dtype=jnp.int32)
        state = {
            "terrain": terrain,
            "fuel": jnp.where((terrain == 1) | (terrain == 5), 6, 0).astype(jnp.int32),
            "heat": heat,
            "wetness": jnp.zeros((11,13), dtype=jnp.int32),
            "agent_pos": jnp.array([[5,5],[6,5],[7,5]], dtype=jnp.int32),
            "agent_role": jnp.array([0,1,2], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "water": jnp.array([6,0,0], dtype=jnp.int32),
            "repair_kits": jnp.array([0,0,2], dtype=jnp.int32),
            "tool_lock": jnp.zeros((3,), dtype=jnp.int32),
            "last_actions": jnp.zeros((3,), dtype=jnp.int32),
            "seedbed_pos": jnp.array([[3,3],[9,3],[6,8]], dtype=jnp.int32),
            "seedbed_health": jnp.full((3,), 4, dtype=jnp.int32),
            "seedbed_active": jnp.ones((3,), dtype=jnp.bool_),
            "wind_direction": wind,
            "wind_phase": jnp.array(0, dtype=jnp.int32),
            "t": jnp.array(0, dtype=jnp.int32),
            "horizon": jnp.array(144, dtype=jnp.int32),
            "done": jnp.array(False, dtype=jnp.bool_),
            "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
            "outcome": jnp.array(0, dtype=jnp.int32),
            "episode_return": jnp.array(0.0, dtype=jnp.float32),
            "last_reward": jnp.array(0.0, dtype=jnp.float32),
            "last_reward_components": jnp.zeros((5,), dtype=jnp.float32),
            "cell_pos": jnp.stack((cells % 13, cells // 13), axis=1)
        }
        events = self.empty_events()
        for name in events:
            state[name] = events[name]
        state = self.refresh_views(state)
        return self.observe(state), state

    def observe(self, state):
        beds = state["seedbed_pos"]
        health_grid = jnp.zeros((11,13), dtype=jnp.int32).at[beds[:,1],beds[:,0]].set(state["seedbed_health"])
        board = jnp.concatenate((
            jax.nn.one_hot(state["terrain"], 6, dtype=jnp.float32),
            state["fuel"][:,:,None].astype(jnp.float32) / 6.0,
            state["heat"][:,:,None].astype(jnp.float32) / 3.0,
            state["wetness"][:,:,None].astype(jnp.float32) / 3.0,
            health_grid[:,:,None].astype(jnp.float32) / 4.0
        ), axis=2).reshape((1430,))
        pos = state["agent_pos"].astype(jnp.float32)
        agents = jnp.concatenate((
            pos[:,1:2] / 10.0, pos[:,0:1] / 12.0,
            jax.nn.one_hot(state["agent_role"], 3, dtype=jnp.float32),
            state["water"][:,None].astype(jnp.float32) / 6.0,
            state["repair_kits"][:,None].astype(jnp.float32) / 2.0,
            state["tool_lock"][:,None].astype(jnp.float32),
            state["last_actions"][:,None].astype(jnp.float32) / 9.0
        ), axis=1).reshape((27,))
        clock = (state["t"].astype(jnp.float32) / state["horizon"].astype(jnp.float32))[None]
        tail = jnp.stack((
            state["wind_phase"].astype(jnp.float32) / 15.0,
            state["done"].astype(jnp.float32), state["terminated"].astype(jnp.float32),
            state["truncated"].astype(jnp.float32), state["outcome"].astype(jnp.float32),
            state["episode_return"]
        ))
        global_values = jnp.concatenate((clock, jax.nn.one_hot(state["wind_direction"], 4, dtype=jnp.float32), tail))
        common = jnp.concatenate((board, agents, global_values))
        return jnp.concatenate((jnp.broadcast_to(common[None,:], (3,1468)), jnp.eye(3, dtype=jnp.float32)), axis=1)

    def legal_actions(self, state):
        pos = state["agent_pos"]
        targets = pos[:,None,:] + self.directions()[None,:,:]
        x = jnp.clip(targets[:,:,0], 0, 12)
        y = jnp.clip(targets[:,:,1], 0, 10)
        inside = (targets[:,:,0] >= 0) & (targets[:,:,0] < 13) & (targets[:,:,1] >= 0) & (targets[:,:,1] < 11)
        tiles = state["terrain"][y,x]
        other = ~jnp.eye(3, dtype=jnp.bool_)
        occupied = jnp.any(jnp.all(targets[:,:,None,:] == pos[None,None,:,:], axis=3) & other[:,None,:], axis=2)
        moves = inside & (tiles != 0) & (state["heat"][y,x] <= 1) & ~occupied
        unlocked = state["tool_lock"] == 0
        spray = inside[0] & (tiles[0] != 0) & unlocked[0] & (state["water"][0] > 0)
        cut = inside[1] & (tiles[1] == 1) & (state["fuel"][y[1],x[1]] > 0) & unlocked[1]
        matches = jnp.all(targets[2,:,None,:] == state["seedbed_pos"][None,:,:], axis=2)
        bed_index = jnp.argmax(matches.astype(jnp.int32), axis=1)
        bed_health = state["seedbed_health"][bed_index]
        repair = jnp.any(matches, axis=1) & (bed_health > 0) & (bed_health < 4) & unlocked[2] & (state["repair_kits"][2] > 0)
        tools = jnp.stack((spray, cut, repair))
        here = state["terrain"][pos[:,1],pos[:,0]]
        service = jnp.array([True,False,True], dtype=jnp.bool_) & unlocked
        service = service & jnp.array([here[0] == 3, False, here[2] == 4], dtype=jnp.bool_)
        mask = jnp.concatenate((jnp.ones((3,1), dtype=jnp.bool_), moves, tools, service[:,None]), axis=1)
        noop = jnp.broadcast_to((jnp.arange(10) == 0)[None,:], (3,10))
        return jnp.where(state["done"], noop, mask)

    def absorb(self, payload):
        key, state, actions = payload
        info = {"done_all": state["done"], "terminated": state["terminated"], "truncated": state["truncated"]}
        return self.observe(state), state, jnp.zeros((3,), dtype=jnp.float32), jnp.ones((3,), dtype=jnp.bool_), info

    def advance(self, payload):
        key, state, actions = payload
        limits = jnp.array([10,9,10], dtype=jnp.int32)
        action = jnp.where((actions >= 0) & (actions < limits), actions, 0).astype(jnp.int32)
        out = dict(state)
        out["last_actions"] = action
        old_pos = state["agent_pos"]
        move_requested = (action >= 1) & (action <= 4)
        target = old_pos + self.directions()[jnp.clip(action - 1, 0, 3)]
        x = jnp.clip(target[:,0], 0, 12)
        y = jnp.clip(target[:,1], 0, 10)
        inside = (target[:,0] >= 0) & (target[:,0] < 13) & (target[:,1] >= 0) & (target[:,1] < 11)
        other = ~jnp.eye(3, dtype=jnp.bool_)
        occupied = jnp.any(jnp.all(target[:,None,:] == old_pos[None,:,:], axis=2) & other, axis=1)
        valid = move_requested & inside & (state["terrain"][y,x] != 0) & (state["heat"][y,x] <= 1) & ~occupied
        conflict = jnp.any(jnp.all(target[:,None,:] == target[None,:,:], axis=2) & other & valid[None,:], axis=1)
        moved = valid & ~conflict
        pos = jnp.where(moved[:,None], target, old_pos)
        out["agent_pos"] = pos
        tool_target = pos + self.directions()[jnp.clip(action - 5, 0, 3)]
        tx = jnp.clip(tool_target[:,0], 0, 12)
        ty = jnp.clip(tool_target[:,1], 0, 10)
        tool_inside = (tool_target[:,0] >= 0) & (tool_target[:,0] < 13) & (tool_target[:,1] >= 0) & (tool_target[:,1] < 11)
        tool_tiles = state["terrain"][ty,tx]
        here = state["terrain"][pos[:,1],pos[:,0]]
        unlocked = state["tool_lock"] == 0
        directional = (action >= 5) & (action <= 8)
        spray = directional[0] & unlocked[0] & tool_inside[0] & (tool_tiles[0] != 0) & (state["water"][0] > 0)
        refill = (action[0] == 9) & unlocked[0] & (here[0] == 3)
        spray_amount = jnp.where(spray, jnp.minimum(state["heat"][ty[0],tx[0]], 2), 0)
        heat = state["heat"].at[ty[0],tx[0]].add(-spray_amount)
        wet = state["wetness"].at[ty[0],tx[0]].set(jnp.where(spray, 3, state["wetness"][ty[0],tx[0]]))
        water = state["water"].at[0].set(jnp.where(refill, 6, state["water"][0] - spray.astype(jnp.int32)))
        cut = directional[1] & unlocked[1] & tool_inside[1] & (tool_tiles[1] == 1) & (heat[ty[1],tx[1]] <= 1) & (state["fuel"][ty[1],tx[1]] > 0)
        cut_amount = jnp.where(cut, jnp.minimum(state["fuel"][ty[1],tx[1]], 3), 0)
        fuel = state["fuel"].at[ty[1],tx[1]].add(-cut_amount)
        heat = heat.at[ty[1],tx[1]].set(jnp.where(fuel[ty[1],tx[1]] == 0, 0, heat[ty[1],tx[1]]))
        bed_match = jnp.all(state["seedbed_pos"] == tool_target[2], axis=1)
        bed_index = jnp.argmax(bed_match.astype(jnp.int32))
        old_health = state["seedbed_health"]
        repair = directional[2] & unlocked[2] & jnp.any(bed_match) & (old_health[bed_index] > 0) & (old_health[bed_index] < 4) & (heat[ty[2],tx[2]] <= 1) & (state["repair_kits"][2] > 0)
        repair_units = jnp.zeros((3,), dtype=jnp.int32).at[bed_index].set(repair.astype(jnp.int32))
        repaired_health = old_health + repair_units
        restock = (action[2] == 9) & unlocked[2] & (here[2] == 4)
        kits = state["repair_kits"].at[2].set(jnp.where(restock, 2, state["repair_kits"][2] - repair.astype(jnp.int32)))
        source = jnp.where((fuel > 0) & (wet == 0), heat, 0)
        padded = jnp.pad(source, ((1,1),(1,1)))
        neighbors = jnp.stack((padded[:-2,1:-1], padded[1:-1,2:], padded[2:,1:-1], padded[1:-1,:-2]))
        push = jnp.array([2,3,0,1], dtype=jnp.int32)
        bonus = ((neighbors > 0) & (push[:,None,None] == state["wind_direction"])).astype(jnp.int32)
        exposure = jnp.max(neighbors + bonus, axis=0)
        burning = (fuel > 0) & (heat > 0) & (wet == 0)
        final_fuel = jnp.maximum(fuel - burning.astype(jnp.int32), 0)
        dry_heat = jnp.where(heat > 0, jnp.minimum(heat + 1, 3), jnp.where(exposure >= 3, 1, 0))
        final_heat = jnp.where(final_fuel == 0, 0, jnp.where(wet > 0, jnp.maximum(heat - 1, 0), dry_heat)).astype(jnp.int32)
        final_wet = jnp.where(state["terrain"] == 0, 0, jnp.maximum(wet - 1, 0)).astype(jnp.int32)
        beds = state["seedbed_pos"]
        damage = ((final_heat[beds[:,1],beds[:,0]] >= 2) & (repaired_health > 0)).astype(jnp.int32)
        health = jnp.maximum(repaired_health - damage, 0)
        locks = jnp.maximum(jnp.maximum(state["tool_lock"] - 1, 0), (final_heat[pos[:,1],pos[:,0]] >= 2).astype(jnp.int32))
        tick = state["t"] + jnp.array(1, dtype=jnp.int32)
        wind_phase = (state["wind_phase"] + 1) % 16
        wind = (state["wind_direction"] + (state["wind_phase"] == 15).astype(jnp.int32)) % 4
        failure = jnp.any(health == 0)
        success = ~failure & jnp.all(final_heat == 0)
        terminated = failure | success
        truncated = (tick >= self.max_steps) & ~terminated
        done = terminated | truncated
        outcome = jnp.where(failure, -1, jnp.where(success, 1, 0)).astype(jnp.int32)
        components = jnp.stack((
            jnp.where(success, 5.0, jnp.where(failure, -5.0, 0.0)).astype(jnp.float32),
            -jnp.array(0.10, dtype=jnp.float32) * jnp.sum(damage).astype(jnp.float32),
            -jnp.array(0.001, dtype=jnp.float32) * jnp.sum(burning).astype(jnp.float32),
            -jnp.array(0.0005, dtype=jnp.float32) * cut_amount.astype(jnp.float32),
            jnp.array(-0.002, dtype=jnp.float32)
        ))
        reward = jnp.sum(components)
        out["fuel"] = final_fuel
        out["heat"] = final_heat
        out["wetness"] = final_wet
        out["water"] = water
        out["repair_kits"] = kits
        out["tool_lock"] = locks
        out["seedbed_health"] = health
        out["seedbed_active"] = health > 0
        out["wind_direction"] = wind
        out["wind_phase"] = wind_phase
        out["t"] = tick
        out["done"] = done
        out["terminated"] = terminated
        out["truncated"] = truncated
        out["outcome"] = outcome
        out["episode_return"] = state["episode_return"] + reward
        out["last_reward"] = reward
        out["last_reward_components"] = components
        events = self.empty_events()
        ea = events["event_active"].at[:3].set(move_requested)
        ek = events["event_kind"].at[:3].set(jnp.where(moved, 1, 2))
        er = events["event_actor"].at[:3].set(jnp.arange(3, dtype=jnp.int32))
        ef = events["event_from"].at[:3].set(old_pos)
        et = events["event_to"].at[:3].set(pos)
        ev = events["event_value"].at[:3,0].set(jnp.clip(action - 1, 0, 3))
        attempted_tool = action >= 5
        accepted_tool = jnp.stack((spray | refill, cut, repair | restock))
        tool_kind = jnp.stack((jnp.where(refill, 6, 3), jnp.array(4, dtype=jnp.int32), jnp.where(restock, 6, 5)))
        tool_main = jnp.stack((jnp.where(refill, water[0] - state["water"][0], spray_amount), cut_amount, jnp.where(restock, kits[2] - state["repair_kits"][2], repair.astype(jnp.int32))))
        before_resource = jnp.stack((state["water"][0], jnp.array(0, dtype=jnp.int32), state["repair_kits"][2]))
        after_resource = jnp.stack((water[0], jnp.array(0, dtype=jnp.int32), kits[2]))
        tool_end = jnp.where((action == 9)[:,None], pos, jnp.where(tool_inside[:,None], tool_target, -1))
        ea = ea.at[3:6].set(attempted_tool)
        ek = ek.at[3:6].set(jnp.where(accepted_tool, tool_kind, 7))
        er = er.at[3:6].set(jnp.arange(3, dtype=jnp.int32))
        ef = ef.at[3:6].set(pos)
        et = et.at[3:6].set(tool_end)
        ev = ev.at[3:6].set(jnp.stack((tool_main, before_resource, after_resource, (~accepted_tool).astype(jnp.int32)), axis=1))
        cell_changed = ((state["heat"] != final_heat) | (state["fuel"] != final_fuel) | (state["wetness"] != final_wet)).reshape((143,))
        ea = ea.at[6:149].set(cell_changed)
        ek = ek.at[6:149].set(8)
        ef = ef.at[6:149].set(state["cell_pos"])
        et = et.at[6:149].set(state["cell_pos"])
        ev = ev.at[6:149].set(jnp.stack((state["heat"].reshape((143,)), final_heat.reshape((143,)), state["fuel"].reshape((143,)), final_fuel.reshape((143,))), axis=1))
        ea = ea.at[149:152].set((repair_units > 0) | (damage > 0))
        ek = ek.at[149:152].set(9)
        ef = ef.at[149:152].set(beds)
        et = et.at[149:152].set(beds)
        ev = ev.at[149:152].set(jnp.stack((old_health, repair_units, damage, health), axis=1))
        ea = ea.at[152:155].set(locks != state["tool_lock"])
        ek = ek.at[152:155].set(10)
        er = er.at[152:155].set(jnp.arange(3, dtype=jnp.int32))
        ef = ef.at[152:155].set(pos)
        et = et.at[152:155].set(pos)
        ev = ev.at[152:155,0].set(state["tool_lock"])
        ev = ev.at[152:155,1].set(locks)
        ea = ea.at[155].set(wind != state["wind_direction"])
        ek = ek.at[155].set(11)
        ev = ev.at[155].set(jnp.array([state["wind_direction"],wind,0,0], dtype=jnp.int32))
        ea = ea.at[156].set(done)
        ek = ek.at[156].set(12)
        ev = ev.at[156].set(jnp.array([outcome,terminated.astype(jnp.int32),truncated.astype(jnp.int32),tick], dtype=jnp.int32))
        out["event_active"] = ea
        out["event_kind"] = jnp.where(ea, ek, 0)
        out["event_actor"] = jnp.where(ea, er, -1)
        out["event_from"] = jnp.where(ea[:,None], ef, -1)
        out["event_to"] = jnp.where(ea[:,None], et, -1)
        out["event_value"] = jnp.where(ea[:,None], ev, 0)
        out["event_tick"] = jnp.where(ea, tick, -1).astype(jnp.int32)
        out = self.refresh_views(out)
        info = {"done_all": done, "terminated": terminated, "truncated": truncated}
        return self.observe(out), out, jnp.full((3,), reward, dtype=jnp.float32), jnp.full((3,), done, dtype=jnp.bool_), info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, actions))
