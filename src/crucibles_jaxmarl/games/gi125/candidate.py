"""Common Kiln: private orders share heat, cooling, material and one pair of tongs.
Original authored kernel. No transition, script, learning run or replay executed.
"""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 351
    num_actions = 11
    max_steps = 144

    def refresh(self, state):
        owner = state["item_owner"]
        carried = (owner >= 0) & (owner <= 2)
        item_xy = jnp.where(carried[:, None], state["agent_xy"][jnp.clip(owner, 0, 2)], state["item_xy"])
        item_xy = jnp.where((owner == 3)[:, None], state["station_xy"][0], item_xy)
        item_xy = jnp.where((owner == 4)[:, None], state["station_xy"][2], item_xy)
        tool_owner = state["tool_owner"][0]
        tool_xy = jnp.where((tool_owner >= 0) & (tool_owner <= 2), state["agent_xy"][jnp.clip(tool_owner, 0, 2)], state["tool_xy"][0])
        tool_xy = jnp.where(tool_owner == 3, state["station_xy"][0], tool_xy)
        tool_xy = jnp.where(tool_owner == 4, state["station_xy"][2], tool_xy)
        return dict(state, item_xy=item_xy, item_active=owner >= -1, tool_xy=tool_xy[None, :])

    def reset(self, key):
        xx = jnp.arange(13, dtype=jnp.int32)[None, :]
        yy = jnp.arange(11, dtype=jnp.int32)[:, None]
        tiles = ((xx == 0) | (xx == 12) | (yy == 0) | (yy == 10) |
                 ((xx == 6) & ((yy <= 3) | (yy == 6)))).astype(jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32), "tiles": tiles,
            "agent_xy": jnp.array([[2, 8], [6, 8], [10, 8]], dtype=jnp.int32),
            "agent_active": jnp.ones(3, dtype=jnp.bool_), "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "carrying": jnp.full(3, -1, dtype=jnp.int32), "completed": jnp.zeros(3, dtype=jnp.bool_),
            "order_kind": jax.random.permutation(key, jnp.arange(3, dtype=jnp.int32)),
            "item_xy": jnp.array([[2, 7], [3, 8], [4, 7], [9, 7], [10, 7], [9, 8]], dtype=jnp.int32),
            "item_owner": jnp.full(6, -1, dtype=jnp.int32), "item_stage": jnp.zeros(6, dtype=jnp.int32),
            "item_heat": jnp.zeros(6, dtype=jnp.int32), "item_peak": jnp.zeros(6, dtype=jnp.int32),
            "item_mass": jnp.ones(6, dtype=jnp.int32), "item_active": jnp.ones(6, dtype=jnp.bool_),
            "item_visible": jnp.ones((3, 6), dtype=jnp.bool_),
            "station_xy": jnp.array([[3, 3], [9, 3], [6, 7], [6, 9]], dtype=jnp.int32),
            "station_timer": jnp.zeros(4, dtype=jnp.int32), "station_item": jnp.full(4, -1, dtype=jnp.int32),
            "station_active": jnp.ones(4, dtype=jnp.bool_), "station_visible": jnp.ones((3, 4), dtype=jnp.bool_),
            "temperature": jnp.array([3], dtype=jnp.int32), "fuel": jnp.array([6], dtype=jnp.int32),
            "coolant": jnp.array([4], dtype=jnp.int32),
            "tool_xy": jnp.array([[6, 5]], dtype=jnp.int32), "tool_owner": jnp.array([-1], dtype=jnp.int32),
            "tool_active": jnp.ones(1, dtype=jnp.bool_), "tool_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "heat_event": jnp.array(False, dtype=jnp.bool_), "quench_event": jnp.array(False, dtype=jnp.bool_),
            "assembly_event": jnp.array(False, dtype=jnp.bool_), "delivery_event": jnp.zeros(3, dtype=jnp.bool_),
            "episode_done": jnp.array(False, dtype=jnp.bool_), "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
        }
        return self.observe(state), state

    def legal_actions(self, state):
        return jnp.broadcast_to((jnp.arange(11) == 0) | ~state["episode_done"], (3, 11))

    def observe(self, state):
        public = jnp.concatenate([
            state["agent_xy"].reshape(-1).astype(jnp.float32) / 12.0,
            jax.nn.one_hot(state["carrying"] + 1, 7, dtype=jnp.float32).reshape(-1),
            state["completed"].astype(jnp.float32), state["tool_xy"].reshape(-1).astype(jnp.float32) / 12.0,
            jax.nn.one_hot(state["tool_owner"][0] + 1, 6, dtype=jnp.float32),
            state["station_xy"].reshape(-1).astype(jnp.float32) / 12.0,
            state["station_timer"].astype(jnp.float32) / 3.0,
            jax.nn.one_hot(state["station_item"] + 1, 7, dtype=jnp.float32).reshape(-1),
            state["temperature"].astype(jnp.float32) / 8.0, state["fuel"].astype(jnp.float32) / 6.0,
            state["coolant"].astype(jnp.float32) / 4.0,
            state["item_xy"].reshape(-1).astype(jnp.float32) / 12.0,
            jax.nn.one_hot(state["item_owner"] + 3, 8, dtype=jnp.float32).reshape(-1),
            jax.nn.one_hot(state["item_stage"], 5, dtype=jnp.float32).reshape(-1),
            state["item_heat"].astype(jnp.float32) / 8.0, state["item_peak"].astype(jnp.float32) / 8.0,
            state["item_mass"].astype(jnp.float32) / 2.0, state["item_active"].astype(jnp.float32),
        ])
        tail = jnp.concatenate([jnp.array([state["t"]], dtype=jnp.float32) / 144.0,
                               state["tiles"].reshape(-1).astype(jnp.float32),
                               jnp.array([state["heat_event"], state["quench_event"], state["assembly_event"]], dtype=jnp.float32),
                               state["delivery_event"].astype(jnp.float32)])
        return jnp.concatenate([jnp.broadcast_to(public, (3, 195)),
                                jax.nn.one_hot(state["order_kind"], 3, dtype=jnp.float32),
                                jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(tail, (3, 150))], axis=1)

    def raw_step(self, key, state, actions):
        actions = jnp.asarray(actions, dtype=jnp.int32)
        actions = jnp.where((actions >= 0) & (actions < 11), actions, 0)
        clock = state["t"] + jnp.array(1, dtype=jnp.int32)
        pos = state["agent_xy"]
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)
        target = pos + directions[actions]
        inside = (target[:, 0] >= 1) & (target[:, 0] <= 11) & (target[:, 1] >= 1) & (target[:, 1] <= 9)
        floor = state["tiles"][jnp.clip(target[:, 1], 0, 10), jnp.clip(target[:, 0], 0, 12)] == 0
        occupied = jnp.any(jnp.all(target[:, None, :] == pos[None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(target[:, None, :] == target[None, :, :], axis=2), axis=1) > 1
        new_pos = jnp.where((inside & floor & ~occupied & ~contested)[:, None], target, pos)
        carrying = state["carrying"]
        owner = state["item_owner"]
        item_xy = jnp.where(((owner >= 0) & (owner <= 2))[:, None], new_pos[jnp.clip(owner, 0, 2)], state["item_xy"])
        stage = state["item_stage"]
        insulated = (owner >= 0) & (owner <= 2) & (state["tool_owner"][0] == owner)
        cooling_tick = jnp.where(insulated, jnp.mod(clock, 6) == 0, jnp.mod(clock, 3) == 0)
        heat = jnp.maximum(state["item_heat"] - ((stage == 1) & (owner != 3) & cooling_tick).astype(jnp.int32), 0)
        stage = jnp.where((stage == 1) & (owner != 3) & (heat == 0), 2, stage)
        peak = state["item_peak"]
        mass = state["item_mass"]
        temperature = jnp.maximum(state["temperature"][0] - (jnp.mod(clock, 4) == 0).astype(jnp.int32), 0)
        fuel = state["fuel"][0]
        coolant = jnp.minimum(state["coolant"][0] + (jnp.mod(clock, 8) == 0).astype(jnp.int32), 4)
        timer = jnp.maximum(state["station_timer"] - 1, 0)
        station_item = state["station_item"]
        tool_owner = state["tool_owner"][0]
        tool_xy = state["tool_xy"][0]
        furnace_id = jnp.clip(station_item[0], 0, 5)
        in_furnace = station_item[0] >= 0
        heat = heat.at[furnace_id].set(jnp.where(in_furnace, jnp.maximum(heat[furnace_id], temperature), heat[furnace_id]))
        peak = peak.at[furnace_id].set(jnp.where(in_furnace, jnp.maximum(peak[furnace_id], heat[furnace_id]), peak[furnace_id]))
        furnace_done = in_furnace & (timer[0] == 0)
        owner = owner.at[furnace_id].set(jnp.where(furnace_done, -1, owner[furnace_id]))
        item_xy = item_xy.at[furnace_id].set(jnp.where(furnace_done, state["station_xy"][0], item_xy[furnace_id]))
        release_tongs = furnace_done & (tool_owner == 3)
        tool_xy = jnp.where(release_tongs, state["station_xy"][0], tool_xy)
        tool_owner = jnp.where(release_tongs, -1, tool_owner)
        station_item = station_item.at[0].set(jnp.where(furnace_done, -1, station_item[0]))
        press_id = jnp.clip(station_item[2], 0, 5)
        press_done = (station_item[2] >= 0) & (timer[2] == 0)
        stage = stage.at[press_id].set(jnp.where(press_done, 4, stage[press_id]))
        owner = owner.at[press_id].set(jnp.where(press_done, -1, owner[press_id]))
        item_xy = item_xy.at[press_id].set(jnp.where(press_done, state["station_xy"][2], item_xy[press_id]))
        release_tongs = press_done & (tool_owner == 4)
        tool_xy = jnp.where(release_tongs, state["station_xy"][2], tool_xy)
        tool_owner = jnp.where(release_tongs, -1, tool_owner)
        station_item = station_item.at[2].set(jnp.where(press_done, -1, station_item[2]))
        completed = state["completed"]
        delivery = jnp.zeros(3, dtype=jnp.bool_)
        fuel_used = jnp.zeros(3, dtype=jnp.bool_)
        heat_event = jnp.array(False, dtype=jnp.bool_)
        quench_event = jnp.array(False, dtype=jnp.bool_)
        assembly_event = jnp.array(False, dtype=jnp.bool_)
        for offset in range(3):
            i = jnp.mod(state["t"] + offset, 3)
            action = actions[i]
            near_station = jnp.sum(jnp.abs(new_pos[i] - state["station_xy"]), axis=1) <= 1
            neighbours = (jnp.sum(jnp.abs(new_pos - new_pos[i]), axis=1) <= 1) & (jnp.arange(3) != i) & (actions == 0)
            recipient_ok = neighbours & (carrying < 0)
            recipient = jnp.argmin(jnp.where(recipient_ok, jnp.arange(3), 3))
            item = jnp.clip(carrying[i], 0, 5)
            give_item = (action == 9) & (carrying[i] >= 0) & jnp.any(recipient_ok)
            old_item = carrying[i]
            carrying = carrying.at[recipient].set(jnp.where(give_item, old_item, carrying[recipient]))
            carrying = carrying.at[i].set(jnp.where(give_item, -1, carrying[i]))
            owner = owner.at[item].set(jnp.where(give_item, recipient, owner[item]))
            tool_recipient = jnp.argmin(jnp.where(neighbours, jnp.arange(3), 3))
            give_tool = (action == 9) & (old_item < 0) & (tool_owner == i) & jnp.any(neighbours)
            tool_owner = jnp.where(give_tool, tool_recipient, tool_owner)
            had_item = carrying[i] >= 0
            item = jnp.clip(carrying[i], 0, 5)
            drop = (action == 5) & had_item
            owner = owner.at[item].set(jnp.where(drop, -1, owner[item]))
            item_xy = item_xy.at[item].set(jnp.where(drop, new_pos[i], item_xy[item]))
            candidates = (owner == -1) & (jnp.sum(jnp.abs(item_xy - new_pos[i]), axis=1) <= 1)
            pickup_id = jnp.argmin(jnp.where(candidates, jnp.arange(6), 6))
            pickup = (action == 5) & ~had_item & jnp.any(candidates)
            owner = owner.at[pickup_id].set(jnp.where(pickup, i, owner[pickup_id]))
            carrying = carrying.at[i].set(jnp.where(drop, -1, jnp.where(pickup, pickup_id, carrying[i])))
            drop_tool = (action == 10) & (tool_owner == i)
            take_tool = (action == 10) & (tool_owner == -1) & (jnp.sum(jnp.abs(tool_xy - new_pos[i])) <= 1)
            tool_xy = jnp.where(drop_tool, new_pos[i], tool_xy)
            tool_owner = jnp.where(drop_tool, -1, jnp.where(take_tool, i, tool_owner))
            stoke = (action == 7) & near_station[0] & (fuel > 0)
            vent = (action == 8) & near_station[0]
            fuel = fuel - stoke.astype(jnp.int32)
            temperature = jnp.clip(temperature + 3 * stoke.astype(jnp.int32) - 2 * vent.astype(jnp.int32), 0, 8)
            fuel_used = fuel_used.at[i].set(stoke)
            heat_event = heat_event | stoke | vent
            item = jnp.clip(carrying[i], 0, 5)
            holding = carrying[i] >= 0
            has_tongs = tool_owner == i
            load = (action == 6) & near_station[0] & holding & (stage[item] == 0) & (station_item[0] < 0) & (temperature >= 3) & (has_tongs | (temperature <= 4))
            owner = owner.at[item].set(jnp.where(load, 3, owner[item]))
            stage = stage.at[item].set(jnp.where(load, 1, stage[item]))
            heat = heat.at[item].set(jnp.where(load, temperature, heat[item]))
            peak = peak.at[item].set(jnp.where(load, temperature, peak[item]))
            station_item = station_item.at[0].set(jnp.where(load, item, station_item[0]))
            timer = timer.at[0].set(jnp.where(load, jnp.where(has_tongs, 2, 3), timer[0]))
            tool_owner = jnp.where(load & has_tongs, 3, tool_owner)
            carrying = carrying.at[i].set(jnp.where(load, -1, carrying[i]))
            item = jnp.clip(carrying[i], 0, 5)
            holding = carrying[i] >= 0
            quench = (action == 6) & near_station[1] & holding & (stage[item] == 1) & (coolant > 0) & (timer[1] == 0)
            hard = (peak[item] >= 6) & (heat[item] >= 3)
            stage = stage.at[item].set(jnp.where(quench, jnp.where(hard, 3, 2), stage[item]))
            heat = heat.at[item].set(jnp.where(quench, 0, heat[item]))
            coolant = coolant - quench.astype(jnp.int32)
            timer = timer.at[1].set(jnp.where(quench, 2, timer[1]))
            quench_event = quench_event | quench
            second_ok = (owner == -1) & (stage == 2) & (mass == 1) & (jnp.arange(6) != item) & (jnp.sum(jnp.abs(item_xy - state["station_xy"][2]), axis=1) <= 1)
            second = jnp.argmin(jnp.where(second_ok, jnp.arange(6), 6))
            has_tongs = tool_owner == i
            assemble = (action == 6) & near_station[2] & holding & (stage[item] == 2) & (mass[item] == 1) & (station_item[2] < 0) & jnp.any(second_ok)
            combined_mass = mass[item] + mass[second]
            mass = mass.at[item].set(jnp.where(assemble, combined_mass, mass[item]))
            mass = mass.at[second].set(jnp.where(assemble, 0, mass[second]))
            owner = owner.at[item].set(jnp.where(assemble, 4, owner[item]))
            owner = owner.at[second].set(jnp.where(assemble, -2, owner[second]))
            station_item = station_item.at[2].set(jnp.where(assemble, item, station_item[2]))
            timer = timer.at[2].set(jnp.where(assemble, jnp.where(has_tongs, 2, 3), timer[2]))
            tool_owner = jnp.where(assemble & has_tongs, 4, tool_owner)
            carrying = carrying.at[i].set(jnp.where(assemble, -1, carrying[i]))
            assembly_event = assembly_event | assemble
            item = jnp.clip(carrying[i], 0, 5)
            deliver = (action == 6) & near_station[3] & (carrying[i] >= 0) & ~completed[i] & (stage[item] == state["order_kind"][i] + 2)
            owner = owner.at[item].set(jnp.where(deliver, -3, owner[item]))
            item_xy = item_xy.at[item].set(jnp.where(deliver, state["station_xy"][3], item_xy[item]))
            carrying = carrying.at[i].set(jnp.where(deliver, -1, carrying[i]))
            completed = completed.at[i].set(completed[i] | deliver)
            delivery = delivery.at[i].set(deliver)
        terminated = jnp.all(completed)
        truncated = (clock >= 144) & ~terminated
        next_state = dict(state, t=clock, agent_xy=new_pos, carrying=carrying, completed=completed,
                          item_owner=owner, item_xy=item_xy, item_stage=stage, item_heat=heat, item_peak=peak, item_mass=mass,
                          station_item=station_item, station_timer=timer, temperature=temperature[None], fuel=fuel[None], coolant=coolant[None],
                          tool_owner=tool_owner[None], tool_xy=tool_xy[None, :], heat_event=heat_event, quench_event=quench_event,
                          assembly_event=assembly_event, delivery_event=delivery,
                          terminated=terminated, truncated=truncated, episode_done=terminated | truncated)
        next_state = self.refresh(next_state)
        rewards = delivery.astype(jnp.float32) + 0.2 * jnp.sum(delivery.astype(jnp.float32)) - 0.02 * fuel_used.astype(jnp.float32)
        next_state = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, jnp.asarray(new, dtype=old.dtype)), state, next_state)
        rewards = jnp.where(state["episode_done"], jnp.zeros(3, dtype=jnp.float32), rewards)
        done = next_state["episode_done"]
        info = {"done_all": done, "terminated": next_state["terminated"], "truncated": next_state["truncated"]}
        return self.observe(next_state), next_state, rewards, jnp.full(3, done, dtype=jnp.bool_), info
