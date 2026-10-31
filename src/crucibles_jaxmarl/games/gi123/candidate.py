"""Echo Passage: an acoustic field couples lures, patrols and teammate exposure.
Original authored kernel; scripts and transitions have not been executed.
"""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 356
    num_actions = 8
    max_steps = 112

    def passable(self, tiles, xy, gate_open):
        inside = (xy[:, 0] >= 1) & (xy[:, 0] <= 11) & (xy[:, 1] >= 1) & (xy[:, 1] <= 9)
        tile = tiles[jnp.clip(xy[:, 1], 0, 10), jnp.clip(xy[:, 0], 0, 12)]
        return inside & ((tile == 0) | ((tile == 2) & gate_open))

    def acoustic(self, xy, source_xy, strength):
        distance = jnp.sum(jnp.abs(xy[:, None, :] - source_xy[None, :, :]), axis=2)
        return jnp.max(jnp.maximum(strength[None, :] - distance, 0), axis=1)

    def refresh(self, state):
        pos = state["agent_xy"]
        live_radio = state["radio_ttl"] > 0
        local_agents = jnp.max(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=2) <= 3
        patrol_local = jnp.max(jnp.abs(pos[:, None, :] - state["patrol_xy"][None, :, :]), axis=2) <= 3
        patrol_shared = jnp.any(live_radio[:, None] & patrol_local, axis=0)
        gate_local = jnp.max(jnp.abs(pos[:, None, :] - state["gate_xy"][None, :, :]), axis=2) <= 3
        gate_shared = jnp.any(live_radio[:, None] & gate_local, axis=0)
        agent_visible = local_agents | live_radio[None, :]
        object_local = jnp.max(jnp.abs(pos[:, None, :] - state["object_xy"][None, :, :]), axis=2) <= 3
        owner = jnp.clip(state["object_owner"], 0, 2)
        object_visible = object_local | ((state["object_owner"] >= 0)[None, :] & agent_visible[:, owner])
        noise_distance = jnp.sum(jnp.abs(pos[:, None, :] - state["noise_xy"][None, :, :]), axis=2)
        noise_visible = (state["noise_strength"][None, :] > noise_distance) | jnp.eye(3, dtype=jnp.bool_)
        field_value = self.acoustic(state["field_xy"], state["noise_xy"], state["noise_strength"])
        field_visible = jnp.max(jnp.abs(pos[:, None, :] - state["field_xy"][None, :, :]), axis=2) <= 2
        return dict(state, agent_visible=agent_visible, patrol_visible=patrol_local | patrol_shared[None, :],
                    gate_visible=gate_local | gate_shared[None, :], object_visible=object_visible,
                    object_active=state["object_owner"] != -2, noise_visible=noise_visible,
                    noise_active=state["noise_strength"] > 0, field_value=field_value.astype(jnp.int32),
                    field_active=field_value > 0, field_visible=field_visible)

    def reset(self, key):
        xx = jnp.arange(13, dtype=jnp.int32)[None, :]
        yy = jnp.arange(11, dtype=jnp.int32)[:, None]
        walls = (xx == 0) | (xx == 12) | (yy == 0) | (yy == 10) | ((xx == 6) & (yy != 2) & (yy != 5) & (yy != 8))
        tiles = walls.astype(jnp.int32).at[5, 6].set(2)
        field_xy = jnp.stack([jnp.tile(jnp.arange(13, dtype=jnp.int32), 11),
                              jnp.repeat(jnp.arange(11, dtype=jnp.int32), 13)], axis=1)
        state = {
            "t": jnp.array(0, dtype=jnp.int32), "tiles": tiles,
            "agent_xy": jnp.array([[2, 4], [2, 5], [2, 6]], dtype=jnp.int32),
            "agent_active": jnp.ones(3, dtype=jnp.bool_), "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "carrying": jnp.full(3, -1, dtype=jnp.int32),
            "lure_charges": jnp.full(3, 3, dtype=jnp.int32), "radio_charges": jnp.full(3, 2, dtype=jnp.int32),
            "tool_remaining": jnp.array([1, 0, 0], dtype=jnp.int32), "radio_ttl": jnp.zeros(3, dtype=jnp.int32),
            "patrol_xy": jnp.array([[8, 2], [8, 8]], dtype=jnp.int32),
            "patrol_active": jnp.ones(2, dtype=jnp.bool_), "patrol_visible": jnp.zeros((3, 2), dtype=jnp.bool_),
            "patrol_target": jnp.array([[9, 2], [9, 8]], dtype=jnp.int32), "patrol_attention": jnp.zeros(2, dtype=jnp.int32),
            "gate_xy": jnp.array([[6, 5]], dtype=jnp.int32), "gate_open": jnp.zeros(1, dtype=jnp.bool_),
            "gate_active": jnp.ones(1, dtype=jnp.bool_), "gate_visible": jnp.zeros((3, 1), dtype=jnp.bool_),
            "object_xy": jnp.array([[10, 3], [10, 7]], dtype=jnp.int32), "object_owner": jnp.full(2, -1, dtype=jnp.int32),
            "object_active": jnp.ones(2, dtype=jnp.bool_), "object_visible": jnp.zeros((3, 2), dtype=jnp.bool_),
            "exit_xy": jnp.array([[1, 5]], dtype=jnp.int32), "exit_active": jnp.ones(1, dtype=jnp.bool_),
            "exit_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "noise_xy": jnp.array([[2, 4], [2, 5], [2, 6]], dtype=jnp.int32), "noise_strength": jnp.zeros(3, dtype=jnp.int32),
            "noise_active": jnp.zeros(3, dtype=jnp.bool_), "noise_visible": jnp.zeros((3, 3), dtype=jnp.bool_),
            "field_xy": field_xy, "field_value": jnp.zeros(143, dtype=jnp.int32),
            "field_active": jnp.zeros(143, dtype=jnp.bool_), "field_visible": jnp.zeros((3, 143), dtype=jnp.bool_),
            "alarm": jnp.zeros(1, dtype=jnp.int32), "exposed": jnp.zeros(3, dtype=jnp.bool_),
            "lure_event": jnp.zeros(3, dtype=jnp.bool_), "radio_event": jnp.zeros(3, dtype=jnp.bool_),
            "handoff_event": jnp.zeros(3, dtype=jnp.bool_), "delivery_event": jnp.zeros(2, dtype=jnp.bool_),
            "gate_event": jnp.array(False, dtype=jnp.bool_),
            "episode_done": jnp.array(False, dtype=jnp.bool_), "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
        }
        state = self.refresh(state)
        return self.observe(state), state

    def legal_actions(self, state):
        return jnp.broadcast_to((jnp.arange(8) == 0) | ~state["episode_done"], (3, 8))

    def observe(self, state):
        agent_data = jnp.concatenate([state["agent_xy"].astype(jnp.float32) / 12.0,
                                     jnp.ones((3, 1), dtype=jnp.float32),
                                     jax.nn.one_hot(state["carrying"] + 1, 3, dtype=jnp.float32)], axis=1)
        agents = jnp.where(state["agent_visible"][:, :, None], agent_data[None, :, :], 0.0).reshape(3, 18)
        patrol_data = jnp.concatenate([state["patrol_xy"].astype(jnp.float32) / 12.0,
                                      jnp.ones((2, 1), dtype=jnp.float32), state["patrol_attention"][:, None].astype(jnp.float32) / 3.0], axis=1)
        patrols = jnp.where(state["patrol_visible"][:, :, None], patrol_data[None, :, :], 0.0).reshape(3, 8)
        noise_data = jnp.concatenate([state["noise_xy"].astype(jnp.float32) / 12.0,
                                     state["noise_strength"][:, None].astype(jnp.float32) / 6.0,
                                     jnp.ones((3, 1), dtype=jnp.float32)], axis=1)
        noises = jnp.where((state["noise_visible"] & state["noise_active"][None, :])[:, :, None], noise_data[None, :, :], 0.0).reshape(3, 12)
        object_data = jnp.concatenate([state["object_xy"].astype(jnp.float32) / 12.0,
                                      jnp.ones((2, 1), dtype=jnp.float32), (state["object_owner"][:, None].astype(jnp.float32) + 2.0) / 4.0], axis=1)
        objects = jnp.where((state["object_visible"] & state["object_active"][None, :])[:, :, None], object_data[None, :, :], 0.0).reshape(3, 8)
        gate_data = jnp.concatenate([state["gate_xy"].astype(jnp.float32) / 12.0,
                                    jnp.ones((1, 1), dtype=jnp.float32), state["gate_open"][:, None].astype(jnp.float32)], axis=1)
        gate = jnp.where(state["gate_visible"][:, :, None], gate_data[None, :, :], 0.0).reshape(3, 4)
        resources = jnp.stack([state["lure_charges"].astype(jnp.float32) / 3.0,
                               state["radio_charges"].astype(jnp.float32) / 2.0,
                               state["tool_remaining"].astype(jnp.float32)], axis=1)
        public = jnp.concatenate([state["exit_xy"].reshape(-1).astype(jnp.float32) / 12.0])
        tail = jnp.concatenate([state["alarm"].astype(jnp.float32) / 5.0,
                               jnp.array([state["t"]], dtype=jnp.float32) / 112.0,
                               (state["object_owner"] == -2).astype(jnp.float32),
                               state["radio_ttl"].astype(jnp.float32) / 3.0,
                               state["tiles"].reshape(-1).astype(jnp.float32) / 2.0])
        fields = jnp.where(state["field_visible"], state["field_value"][None, :], 0).astype(jnp.float32) / 6.0
        return jnp.concatenate([state["agent_xy"].astype(jnp.float32) / 12.0, resources,
                                jax.nn.one_hot(state["carrying"] + 1, 3, dtype=jnp.float32), agents, patrols,
                                noises, objects, gate, jnp.broadcast_to(public, (3, 2)),
                                jnp.eye(3, dtype=jnp.float32), jnp.broadcast_to(tail, (3, 150)), fields], axis=1)

    def raw_step(self, key, state, actions):
        actions = jnp.asarray(actions, dtype=jnp.int32)
        actions = jnp.where((actions >= 0) & (actions < 8), actions, 0)
        pos = state["agent_xy"]
        gate_event = (actions[0] == 6) & (state["tool_remaining"][0] > 0) & ~state["gate_open"][0] & (jnp.sum(jnp.abs(pos[0] - state["gate_xy"][0])) <= 1)
        gate_open = state["gate_open"] | gate_event
        tool_remaining = state["tool_remaining"].at[0].set(state["tool_remaining"][0] - gate_event.astype(jnp.int32))
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)
        target = pos + directions[actions]
        occupied = jnp.any(jnp.all(target[:, None, :] == pos[None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(target[:, None, :] == target[None, :, :], axis=2), axis=1) > 1
        move_ok = self.passable(state["tiles"], target, gate_open[0]) & ~occupied & ~contested
        new_pos = jnp.where(move_ok[:, None], target, pos)
        carrying = state["carrying"]
        owner = state["object_owner"]
        original_carry = carrying
        adjacent = jnp.sum(jnp.abs(new_pos[:, None, :] - new_pos[None, :, :]), axis=2) <= 1
        offers = (actions == 6)[:, None] & (original_carry >= 0)[:, None] & (actions == 6)[None, :] & (original_carry < 0)[None, :] & adjacent & ~jnp.eye(3, dtype=jnp.bool_)
        recipient = jnp.argmin(jnp.where(offers, jnp.arange(3)[None, :], 3), axis=1)
        chosen = offers & (jnp.arange(3)[None, :] == recipient[:, None])
        donor = jnp.argmin(jnp.where(chosen, jnp.arange(3)[:, None], 3), axis=0)
        handoff = jnp.any(chosen, axis=1) & (donor[recipient] == jnp.arange(3))
        for i in range(3):
            receiver = recipient[i]
            item = jnp.clip(original_carry[i], 0, 1)
            carrying = carrying.at[i].set(jnp.where(handoff[i], -1, carrying[i]))
            carrying = carrying.at[receiver].set(jnp.where(handoff[i], original_carry[i], carrying[receiver]))
            owner = owner.at[item].set(jnp.where(handoff[i], receiver, owner[item]))
        for item in range(2):
            eligible = (actions == 6) & (carrying < 0) & (owner[item] == -1) & (jnp.sum(jnp.abs(new_pos - state["object_xy"][item]), axis=1) <= 1)
            picker = jnp.argmin(jnp.where(eligible, jnp.arange(3), 3))
            picked = jnp.any(eligible)
            carrying = carrying.at[picker].set(jnp.where(picked, item, carrying[picker]))
            owner = owner.at[item].set(jnp.where(picked, picker, owner[item]))
        delivery = jnp.zeros(2, dtype=jnp.bool_)
        for i in range(3):
            item = jnp.clip(carrying[i], 0, 1)
            delivered = (actions[i] == 6) & (carrying[i] >= 0) & (jnp.sum(jnp.abs(new_pos[i] - state["exit_xy"][0])) <= 1)
            owner = owner.at[item].set(jnp.where(delivered, -2, owner[item]))
            delivery = delivery.at[item].set(delivery[item] | delivered)
            carrying = carrying.at[i].set(jnp.where(delivered, -1, carrying[i]))
        object_xy = jnp.where((owner >= 0)[:, None], new_pos[jnp.clip(owner, 0, 2)], state["object_xy"])
        object_xy = jnp.where((owner == -2)[:, None], state["exit_xy"], object_xy)
        lure = (actions == 5) & (state["lure_charges"] > 0)
        radio = (actions == 7) & (state["radio_charges"] > 0)
        radio_ttl = jnp.where(radio, 3, jnp.maximum(state["radio_ttl"] - 1, 0)).astype(jnp.int32)
        footsteps = jnp.any(new_pos != pos, axis=1) & (state["carrying"] >= 0)
        emission = jnp.maximum(lure.astype(jnp.int32) * 6, footsteps.astype(jnp.int32) * 4)
        emission = jnp.maximum(emission, radio.astype(jnp.int32) * 3)
        emission = emission.at[0].set(jnp.maximum(emission[0], gate_event.astype(jnp.int32) * 5))
        strength = jnp.where(emission > 0, emission, jnp.maximum(state["noise_strength"] - 1, 0))
        noise_xy = jnp.where((emission > 0)[:, None], new_pos, state["noise_xy"])
        hearing = strength[None, :] - jnp.sum(jnp.abs(state["patrol_xy"][:, None, :] - noise_xy[None, :, :]), axis=2)
        heard = jnp.max(hearing, axis=1) > 0
        source = jnp.argmax(hearing, axis=1)
        waypoints = jnp.array([[9, 2], [10, 5], [9, 8], [7, 5]], dtype=jnp.int32)
        waypoint = waypoints[jnp.mod(state["t"] // 6 + jnp.array([0, 2], dtype=jnp.int32), 4)]
        patrol_target = jnp.where(heard[:, None], noise_xy[source], jnp.where((state["patrol_attention"] > 0)[:, None], state["patrol_target"], waypoint))
        attention = jnp.where(heard, 3, jnp.maximum(state["patrol_attention"] - 1, 0)).astype(jnp.int32)
        gap = jnp.sign(patrol_target - state["patrol_xy"])
        xstep = jnp.stack([gap[:, 0], jnp.zeros(2, dtype=jnp.int32)], axis=1)
        ystep = jnp.stack([jnp.zeros(2, dtype=jnp.int32), gap[:, 1]], axis=1)
        first = state["patrol_xy"] + jnp.where((gap[:, 0] != 0)[:, None], xstep, ystep)
        second = state["patrol_xy"] + jnp.where((gap[:, 0] != 0)[:, None], ystep, xstep)
        patrol_xy = jnp.where(self.passable(state["tiles"], first, gate_open[0])[:, None], first,
                              jnp.where(self.passable(state["tiles"], second, gate_open[0])[:, None], second, state["patrol_xy"]))
        nearby_patrol = jnp.min(jnp.sum(jnp.abs(new_pos[:, None, :] - patrol_xy[None, :, :]), axis=2), axis=1)
        audible_here = self.acoustic(new_pos, noise_xy, strength)
        exposed = (nearby_patrol <= 1) | ((nearby_patrol <= 3) & (audible_here >= 2))
        quiet = jnp.all(actions == 0) & ~jnp.any(exposed)
        alarm = jnp.clip(state["alarm"] + jnp.sum(exposed.astype(jnp.int32)) - quiet.astype(jnp.int32), 0, 5)
        success = jnp.all(owner == -2)
        failure = alarm[0] >= 5
        clock = state["t"] + jnp.array(1, dtype=jnp.int32)
        terminated = success | failure
        truncated = (clock >= 112) & ~terminated
        next_state = dict(state, t=clock, agent_xy=new_pos, carrying=carrying, object_xy=object_xy, object_owner=owner,
                          gate_open=gate_open, tool_remaining=tool_remaining, patrol_xy=patrol_xy, patrol_target=patrol_target,
                          patrol_attention=attention, lure_charges=state["lure_charges"] - lure.astype(jnp.int32),
                          radio_charges=state["radio_charges"] - radio.astype(jnp.int32), radio_ttl=radio_ttl,
                          noise_xy=noise_xy, noise_strength=strength, alarm=alarm, exposed=exposed,
                          lure_event=lure, radio_event=radio, handoff_event=handoff, delivery_event=delivery, gate_event=gate_event,
                          terminated=terminated, truncated=truncated, episode_done=terminated | truncated)
        next_state = self.refresh(next_state)
        shared = 0.5 * jnp.sum(delivery.astype(jnp.float32)) - 0.03 * jnp.sum(exposed.astype(jnp.float32))
        shared = shared + success.astype(jnp.float32) - failure.astype(jnp.float32)
        rewards = jnp.full(3, shared, dtype=jnp.float32) - 0.01 * lure.astype(jnp.float32)
        next_state = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, jnp.asarray(new, dtype=old.dtype)), state, next_state)
        rewards = jnp.where(state["episode_done"], jnp.zeros(3, dtype=jnp.float32), rewards)
        done = next_state["episode_done"]
        info = {"done_all": done, "terminated": next_state["terminated"], "truncated": next_state["truncated"]}
        return self.observe(next_state), next_state, rewards, jnp.full(3, done, dtype=jnp.bool_), info
