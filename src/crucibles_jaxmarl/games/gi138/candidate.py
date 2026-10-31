"""Switchback Relay: an original four-player spatial passing contest.

Authored source, not an execution receipt. Integer rules; dyadic observations.
"""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 64
    num_actions = 11
    max_steps = 64

    def reset(self, key):
        tiles = jnp.zeros((7, 11), dtype=jnp.int32)
        tiles = tiles.at[1, 5].set(1).at[5, 5].set(1)
        tiles = tiles.at[2:5, 10].set(2).at[2:5, 0].set(3)
        state = {
            "t": jnp.asarray(0, dtype=jnp.int32),
            "episode_done": jnp.asarray(False),
            "terminated": jnp.asarray(False),
            "truncated": jnp.asarray(False),
            "tiles": tiles,
            "agent_pos": jnp.array([[2, 3], [4, 3], [7, 3], [7, 4]], dtype=jnp.int32),
            "agent_active": jnp.ones((4,), dtype=jnp.bool_),
            "agent_visibility": jnp.ones((4, 4), dtype=jnp.bool_),
            "team": jnp.array([0, 0, 1, 1], dtype=jnp.int32),
            "stamina": jnp.full((4,), 6, dtype=jnp.int32),
            "cooldown": jnp.zeros((4,), dtype=jnp.int32),
            "bracing": jnp.zeros((4,), dtype=jnp.bool_),
            "carrying": jnp.array([True, False, False, False]),
            "ball_pos": jnp.array([[2, 3]], dtype=jnp.int32),
            "ball_active": jnp.ones((1,), dtype=jnp.bool_),
            "ball_visibility": jnp.ones((4, 1), dtype=jnp.bool_),
            "ball_owner": jnp.array([0], dtype=jnp.int32),
            "flight_dir": jnp.zeros((1, 2), dtype=jnp.int32),
            "flight_remaining": jnp.zeros((1,), dtype=jnp.int32),
            "last_thrower": jnp.full((1,), -1, dtype=jnp.int32),
            "relay_ready": jnp.zeros((1,), dtype=jnp.bool_),
            "goal_pos": jnp.array([[10, 3], [0, 3]], dtype=jnp.int32),
            "goal_active": jnp.ones((2,), dtype=jnp.bool_),
            "goal_visibility": jnp.ones((4, 2), dtype=jnp.bool_),
            "score": jnp.zeros((2,), dtype=jnp.int32),
            "pass_event": jnp.zeros((4,), dtype=jnp.bool_),
            "catch_event": jnp.zeros((4,), dtype=jnp.bool_),
            "interception_event": jnp.zeros((4,), dtype=jnp.bool_),
            "tackle_event": jnp.zeros((4,), dtype=jnp.bool_),
            "collision_event": jnp.zeros((4,), dtype=jnp.bool_),
            "score_event": jnp.zeros((2,), dtype=jnp.bool_),
        }
        return self.observe(state), state

    def legal_actions(self, state):
        p = state["agent_pos"]
        directions = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        q = p[:, None, :] + directions[None, :, :]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] <= 10) & (q[:, :, 1] >= 0) & (q[:, :, 1] <= 6)
        wall = state["tiles"][jnp.clip(q[:, :, 1], 0, 6), jnp.clip(q[:, :, 0], 0, 10)] == 1
        occupied = jnp.any(jnp.all(q[:, :, None, :] == p[None, None, :, :], axis=3), axis=2)
        move = inside & ~wall & ~occupied & (state["stamina"][:, None] >= 1)
        ready = (state["stamina"] >= 2) & (state["cooldown"] == 0)
        throw = ready & state["carrying"]
        mask = jnp.concatenate((jnp.ones((4, 1), dtype=jnp.bool_), move,
                                jnp.broadcast_to(throw[:, None], (4, 4)),
                                ready[:, None], (state["stamina"] >= 1)[:, None]), axis=1)
        return jnp.where(state["episode_done"], jnp.arange(11)[None, :] == 0, mask)

    def advance(self, state, actions):
        n = dict(state)
        ids = jnp.arange(4, dtype=jnp.int32)
        clipped = jnp.clip(actions, 0, 10)
        valid = (actions >= 0) & (actions < 11) & self.legal_actions(state)[ids, clipped]
        a = jnp.where(valid, clipped, 0)
        p = state["agent_pos"]
        dirs = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        moving = (a >= 1) & (a <= 4)
        proposed = p + dirs[jnp.where(moving, a, 0)]
        same = jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2)
        conflict = jnp.any(same & ~jnp.eye(4, dtype=jnp.bool_), axis=1) & moving
        p = jnp.where(conflict[:, None], p, proposed)
        rank = jnp.mod(ids - state["t"], 4)
        bracing = a == 10
        throwing = (a >= 5) & (a <= 8)
        tackling = a == 9
        cost = moving.astype(jnp.int32) + 2 * throwing.astype(jnp.int32) + 2 * tackling.astype(jnp.int32) + bracing.astype(jnp.int32)
        stamina = jnp.clip(state["stamina"] - cost + 2 * (a == 0).astype(jnp.int32), 0, 6)
        cooldown = jnp.where(throwing | tackling, 2, jnp.maximum(state["cooldown"] - 1, 0))

        # Movement precedes tackles. Bracing sacrifices movement and wards tackles.
        owner = state["ball_owner"][0]
        owner_index = jnp.maximum(owner, 0)
        near_owner = jnp.sum(jnp.abs(p - p[owner_index]), axis=1) == 1
        tackle_ok = tackling & near_owner & (state["team"] != state["team"][owner_index]) & (owner >= 0) & ~bracing[owner_index]
        tackler = jnp.argmin(jnp.where(tackle_ok, rank, 9))
        took = jnp.any(tackle_ok)
        owner = jnp.where(took, tackler, owner)
        relay = state["relay_ready"][0] & ~took
        last = jnp.where(took, -1, state["last_thrower"][0])
        remaining = jnp.where(took, 0, state["flight_remaining"][0])
        direction = state["flight_dir"][0]
        ball = jnp.where(owner >= 0, p[jnp.maximum(owner, 0)], state["ball_pos"][0])

        # A pass is a three-cell flight, one cell per raw transition.
        holder = jnp.maximum(owner, 0)
        launch = (owner >= 0) & throwing[holder]
        pass_event = (ids == owner) & launch
        pass_dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        direction = jnp.where(launch, pass_dirs[jnp.clip(a[holder] - 5, 0, 3)], direction)
        last = jnp.where(launch, owner, last)
        remaining = jnp.where(launch, 3, remaining)
        owner = jnp.where(launch, -1, owner)
        in_flight = (owner < 0) & (remaining > 0)
        next_ball = ball + direction
        inside = (next_ball[0] >= 0) & (next_ball[0] <= 10) & (next_ball[1] >= 0) & (next_ball[1] <= 6)
        blocked = ~inside | (state["tiles"][jnp.clip(next_ball[1], 0, 6), jnp.clip(next_ball[0], 0, 10)] == 1)
        ball = jnp.where(in_flight & ~blocked, next_ball, ball)
        remaining = jnp.where(in_flight, jnp.where(blocked, 0, remaining - 1), remaining)

        # Standing on the ball catches it; a brace reaches one adjacent cell.
        distance = jnp.sum(jnp.abs(p - ball), axis=1)
        catch_ok = (owner < 0) & ((distance == 0) | (in_flight & ~blocked & bracing & (distance == 1)))
        catch_ok = catch_ok & (~in_flight | (ids != last))
        catcher = jnp.argmin(jnp.where(catch_ok, distance * 4 + rank, 99))
        caught = jnp.any(catch_ok)
        enemy_catch = caught & (last >= 0) & (state["team"][catcher] != state["team"][jnp.maximum(last, 0)])
        relay_catch = caught & (last >= 0) & (catcher != last) & (state["team"][catcher] == state["team"][jnp.maximum(last, 0)])
        relay = jnp.where(caught, relay_catch, relay)
        owner = jnp.where(caught, catcher, owner)
        remaining = jnp.where(caught, 0, remaining)
        ball = jnp.where(owner >= 0, p[jnp.maximum(owner, 0)], ball)
        direction = jnp.where(remaining > 0, direction, jnp.zeros((2,), dtype=jnp.int32))
        holder_team = state["team"][jnp.maximum(owner, 0)]
        scoring_x = jnp.where(holder_team == 0, 10, 0)
        scored = (owner >= 0) & relay & (ball[0] == scoring_x) & (ball[1] >= 2) & (ball[1] <= 4)
        score_event = scored & (jnp.arange(2, dtype=jnp.int32) == holder_team)
        nt = state["t"] + 1
        terminated = scored
        truncated = (nt >= self.max_steps) & ~terminated
        n["t"] = nt
        n["episode_done"] = terminated | truncated
        n["terminated"] = terminated
        n["truncated"] = truncated
        n["agent_pos"] = p
        n["stamina"] = stamina
        n["cooldown"] = cooldown
        n["bracing"] = bracing
        n["carrying"] = ids == owner
        n["ball_pos"] = ball[None, :]
        n["ball_owner"] = jnp.asarray([owner], dtype=jnp.int32)
        n["flight_dir"] = direction[None, :]
        n["flight_remaining"] = jnp.asarray([remaining], dtype=jnp.int32)
        n["last_thrower"] = jnp.asarray([last], dtype=jnp.int32)
        n["relay_ready"] = jnp.asarray([relay], dtype=jnp.bool_)
        n["score"] = state["score"] + score_event.astype(jnp.int32)
        n["pass_event"] = pass_event
        n["catch_event"] = (ids == catcher) & caught
        n["interception_event"] = (ids == catcher) & enemy_catch
        n["tackle_event"] = (ids == tackler) & took
        n["collision_event"] = conflict
        n["score_event"] = score_event
        reward = jnp.where(scored, jnp.where(state["team"] == holder_team, 1.0, -1.0), 0.0).astype(jnp.float32)
        return n, reward

    def observe(self, state):
        agents = jnp.concatenate((state["agent_pos"].astype(jnp.float32) / 16.0,
                                  state["stamina"][:, None] / 8.0,
                                  state["cooldown"][:, None] / 4.0,
                                  state["bracing"][:, None].astype(jnp.float32),
                                  state["carrying"][:, None].astype(jnp.float32),
                                  state["team"][:, None].astype(jnp.float32)), axis=1).reshape((28,))
        ball = jnp.concatenate((state["ball_pos"][0] / 16.0, (state["ball_owner"] + 1) / 4.0,
                                state["flight_dir"][0].astype(jnp.float32), state["flight_remaining"] / 4.0,
                                (state["last_thrower"] + 1) / 4.0, state["relay_ready"].astype(jnp.float32)))
        events = jnp.concatenate((state["pass_event"], state["catch_event"], state["interception_event"],
                                  state["tackle_event"], state["collision_event"], state["score_event"])).astype(jnp.float32)
        public = jnp.concatenate((agents, ball, state["score"].astype(jnp.float32), events))
        return jnp.concatenate((jnp.full((4, 1), state["t"] / 64.0, dtype=jnp.float32),
                                jnp.arange(4, dtype=jnp.float32)[:, None] / 4.0,
                                jnp.broadcast_to(public, (4, 60)), jnp.zeros((4, 2), dtype=jnp.float32)), axis=1).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        next_state, rewards = jax.lax.cond(
            state["episode_done"],
            lambda s, a: (s, jnp.zeros((4,), dtype=jnp.float32)),
            self.advance, state, actions)
        dones = jnp.full((4,), next_state["episode_done"], dtype=jnp.bool_)
        info = {"done_all": next_state["episode_done"], "terminated": next_state["terminated"], "truncated": next_state["truncated"]}
        return self.observe(next_state), next_state, rewards, dones, info
