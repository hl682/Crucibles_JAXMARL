"""Tether Turn: three complementary ports, a bulky load, and two passages.
Original authored kernel. No candidate execution has been performed.
"""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 171
    num_actions = 9
    max_steps = 96

    def geometry(self, center, orientation):
        axes = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=jnp.int32)
        axis = axes[orientation]
        side = axes[jnp.mod(orientation + 3, 4)]
        cells = center[None, :] + jnp.array([-1, 0, 1], dtype=jnp.int32)[:, None] * axis[None, :]
        anchors = center[None, :] + jnp.stack([axis, -axis, side])
        return cells, anchors

    def free_cells(self, tiles, xy):
        bounded = (xy[:, 0] >= 1) & (xy[:, 0] <= 11) & (xy[:, 1] >= 1) & (xy[:, 1] <= 9)
        tile = tiles[jnp.clip(xy[:, 1], 0, 10), jnp.clip(xy[:, 0], 0, 12)]
        return bounded & (tile == 0)

    def agent_moves(self, tiles, pos, delta, load_cells):
        target = pos + delta
        occupied = jnp.any(jnp.all(target[:, None, :] == pos[None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(target[:, None, :] == target[None, :, :], axis=2), axis=1) > 1
        in_load = jnp.any(jnp.all(target[:, None, :] == load_cells[None, :, :], axis=2), axis=1)
        allowed = self.free_cells(tiles, target) & ~occupied & ~contested & ~in_load
        return jnp.where(allowed[:, None], target, pos)

    def refresh(self, state):
        cells, anchors = self.geometry(state["load_xy"][0], state["load_orientation"][0])
        distance = jnp.sum(jnp.abs(state["agent_xy"] - anchors), axis=1)
        return dict(state, load_cells=cells, anchor_xy=anchors,
                    anchor_active=state["attached"],
                    tension=jnp.where(state["attached"], distance, 0).astype(jnp.int32))

    def reset(self, key):
        xx = jnp.arange(13, dtype=jnp.int32)[None, :]
        yy = jnp.arange(11, dtype=jnp.int32)[:, None]
        border = (xx == 0) | (xx == 12) | (yy == 0) | (yy == 10)
        partition = (xx == 6) & (yy != 2) & ~((yy >= 7) & (yy <= 9))
        tiles = (border | partition).astype(jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "tiles": tiles,
            "agent_xy": jnp.array([[5, 5], [1, 5], [3, 3]], dtype=jnp.int32),
            "agent_active": jnp.ones(3, dtype=jnp.bool_),
            "agent_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "attached": jnp.zeros(3, dtype=jnp.bool_),
            "tension": jnp.zeros(3, dtype=jnp.int32),
            "load_xy": jnp.array([[3, 5]], dtype=jnp.int32),
            "load_orientation": jnp.array([0], dtype=jnp.int32),
            "load_active": jnp.ones(1, dtype=jnp.bool_),
            "load_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "load_cells": jnp.zeros((3, 2), dtype=jnp.int32),
            "cell_active": jnp.ones(3, dtype=jnp.bool_),
            "cell_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "anchor_xy": jnp.zeros((3, 2), dtype=jnp.int32),
            "anchor_active": jnp.zeros(3, dtype=jnp.bool_),
            "anchor_visible": jnp.ones((3, 3), dtype=jnp.bool_),
            "wear": jnp.zeros(1, dtype=jnp.int32),
            "goal_xy": jnp.array([[10, 5]], dtype=jnp.int32),
            "goal_active": jnp.ones(1, dtype=jnp.bool_),
            "goal_visible": jnp.ones((3, 1), dtype=jnp.bool_),
            "move_event": jnp.array(False, dtype=jnp.bool_),
            "turn_event": jnp.array(False, dtype=jnp.bool_),
            "jam_event": jnp.array(False, dtype=jnp.bool_),
            "episode_done": jnp.array(False, dtype=jnp.bool_),
            "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
        }
        state = self.refresh(state)
        return self.observe(state), state

    def legal_actions(self, state):
        return jnp.broadcast_to((jnp.arange(9) == 0) | ~state["episode_done"], (3, 9))

    def observe(self, state):
        public = jnp.concatenate([
            state["agent_xy"].reshape(-1).astype(jnp.float32) / 12.0,
            state["attached"].astype(jnp.float32),
            state["tension"].astype(jnp.float32) / 2.0,
            state["load_xy"].reshape(-1).astype(jnp.float32) / 12.0,
            jax.nn.one_hot(state["load_orientation"][0], 4, dtype=jnp.float32),
            state["wear"].astype(jnp.float32) / 6.0,
            state["goal_xy"].reshape(-1).astype(jnp.float32) / 12.0,
        ])
        tail = jnp.concatenate([
            state["tiles"].reshape(-1).astype(jnp.float32),
            jnp.array([state["t"]], dtype=jnp.float32) / 96.0,
            jnp.array([state["move_event"], state["turn_event"], state["jam_event"]], dtype=jnp.float32),
        ])
        return jnp.concatenate([jnp.broadcast_to(public, (3, 21)),
                                jnp.eye(3, dtype=jnp.float32),
                                jnp.broadcast_to(tail, (3, 147))], axis=1)

    def raw_step(self, key, state, actions):
        actions = jnp.asarray(actions, dtype=jnp.int32)
        actions = jnp.where((actions >= 0) & (actions < 9), actions, 0)
        pos = state["agent_xy"]
        center = state["load_xy"][0]
        orientation = state["load_orientation"][0]
        old_cells, old_anchors = self.geometry(center, orientation)
        reach = jnp.sum(jnp.abs(pos - old_anchors), axis=1) <= 2
        attached = jnp.where(actions == 5, ~state["attached"] & reach, state["attached"])
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                                [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)
        delta = directions[actions]
        pulling = attached & (actions >= 1) & (actions <= 4)
        force = jnp.sum(delta * attached[:, None].astype(jnp.int32), axis=0)
        axis = jnp.argmax(jnp.abs(force))
        drive = jax.nn.one_hot(axis, 2, dtype=jnp.int32) * jnp.sign(force[axis])
        enough_force = jnp.max(jnp.abs(force)) >= 2
        braked = jnp.any(attached & (actions == 6))
        torque = jnp.sum(attached.astype(jnp.int32) * ((actions == 7).astype(jnp.int32) - (actions == 8).astype(jnp.int32)))
        request_turn = (jnp.abs(torque) >= 2) & ~braked
        request_move = enough_force & ~braked & ~request_turn
        proposed_center = center + jnp.where(request_move, drive, jnp.zeros(2, dtype=jnp.int32))
        proposed_orientation = jnp.mod(orientation + jnp.where(request_turn, jnp.sign(torque), 0), 4)
        proposed_cells, proposed_anchors = self.geometry(proposed_center, proposed_orientation)
        proposed_agents = self.agent_moves(state["tiles"], pos, delta, proposed_cells)
        ropes_fit = jnp.all(~attached | (jnp.sum(jnp.abs(proposed_agents - proposed_anchors), axis=1) <= 2))
        body_clear = ~jnp.any(jnp.all(proposed_agents[:, None, :] == proposed_cells[None, :, :], axis=2))
        convoy_ok = jnp.all(self.free_cells(state["tiles"], proposed_cells)) & ropes_fit & body_clear
        fallback_agents = self.agent_moves(state["tiles"], pos, delta, old_cells)
        fallback_fit = ~attached | (jnp.sum(jnp.abs(fallback_agents - old_anchors), axis=1) <= 2)
        fallback_agents = jnp.where(fallback_fit[:, None], fallback_agents, pos)
        new_pos = jnp.where(convoy_ok, proposed_agents, fallback_agents)
        new_center = jnp.where(convoy_ok, proposed_center, center)
        new_orientation = jnp.where(convoy_ok, proposed_orientation, orientation)
        moved = jnp.any(new_center != center)
        turned = new_orientation != orientation
        competing = (jnp.sum(pulling.astype(jnp.int32)) >= 2) & ~enough_force
        jam = ((request_move | request_turn) & ~convoy_ok) | (competing & ~braked)
        repair = jnp.sum((attached & (actions == 6)).astype(jnp.int32)) >= 2
        wear = jnp.clip(state["wear"] + jam.astype(jnp.int32) - repair.astype(jnp.int32), 0, 6)
        success = jnp.all(new_center == state["goal_xy"][0]) & (jnp.mod(new_orientation, 2) == 0)
        failure = wear[0] >= 6
        clock = state["t"] + jnp.array(1, dtype=jnp.int32)
        terminated = success | failure
        truncated = (clock >= 96) & ~terminated
        next_state = dict(state, t=clock, agent_xy=new_pos, attached=attached,
                          load_xy=new_center[None, :], load_orientation=new_orientation[None],
                          wear=wear, move_event=moved, turn_event=turned, jam_event=jam,
                          terminated=terminated, truncated=truncated, episode_done=terminated | truncated)
        next_state = self.refresh(next_state)
        old_distance = jnp.sum(jnp.abs(center - state["goal_xy"][0]))
        new_distance = jnp.sum(jnp.abs(new_center - state["goal_xy"][0]))
        shared = 0.01 * (old_distance - new_distance).astype(jnp.float32) - 0.02 * jam.astype(jnp.float32)
        shared = shared + success.astype(jnp.float32) - failure.astype(jnp.float32)
        rewards = jnp.full(3, shared, dtype=jnp.float32)
        next_state = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, jnp.asarray(new, dtype=old.dtype)), state, next_state)
        rewards = jnp.where(state["episode_done"], jnp.zeros(3, dtype=jnp.float32), rewards)
        done = next_state["episode_done"]
        info = {"done_all": done, "terminated": next_state["terminated"], "truncated": next_state["truncated"]}
        return self.observe(next_state), next_state, rewards, jnp.full(3, done, dtype=jnp.bool_), info
