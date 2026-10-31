import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 126
    num_actions = 8
    max_steps = 128

    def reset(self, key):
        # EVIDENCE: original_world
        board = jnp.zeros((7, 11), dtype=jnp.int32)
        board = board.at[jnp.array([0, 1, 5, 6]), 5].set(1)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
            "map": board,
            "agent_pos": jnp.array([[2, 2], [2, 4], [8, 2], [8, 4]], dtype=jnp.int32),
            "agent_active": jnp.ones((4,), dtype=jnp.bool_),
            "agent_visible": jnp.ones((4, 4), dtype=jnp.bool_),
            "team": jnp.array([0, 0, 1, 1], dtype=jnp.int32),
            "stamina": jnp.full((4,), 4, dtype=jnp.int32),
            "carrying": jnp.zeros((4,), dtype=jnp.bool_),
            "ball_pos": jnp.array([[5, 3]], dtype=jnp.int32),
            "ball_active": jnp.ones((1,), dtype=jnp.bool_),
            "ball_visible": jnp.ones((4, 1), dtype=jnp.bool_),
            "ball_owner": jnp.array([-1], dtype=jnp.int32),
            "ball_velocity": jnp.zeros((1, 2), dtype=jnp.int32),
            "flight_left": jnp.zeros((1,), dtype=jnp.int32),
            "ball_team": jnp.array([-1], dtype=jnp.int32),
            "relay_ready": jnp.zeros((1,), dtype=jnp.bool_),
            "hold": jnp.zeros((1,), dtype=jnp.int32),
            "pass_from": jnp.array([-1], dtype=jnp.int32),
            "goal_pos": jnp.array([[10, 3], [0, 3]], dtype=jnp.int32),
            "goal_active": jnp.ones((2,), dtype=jnp.bool_),
            "goal_visible": jnp.ones((4, 2), dtype=jnp.bool_),
            "goal_team": jnp.array([0, 1], dtype=jnp.int32),
            "events": jnp.zeros((5,), dtype=jnp.bool_),
            "episode_done": jnp.array(False),
            "terminated": jnp.array(False),
            "truncated": jnp.array(False),
        }
        return self.observe(state), state

    def legal_actions(self, state):
        # EVIDENCE: action_attempts
        live = jnp.ones((4, 8), dtype=jnp.bool_)
        resting = jnp.broadcast_to(jnp.arange(8)[None, :] == 0, (4, 8))
        return jnp.where(state["episode_done"], resting, live)

    def observe(self, state):
        # EVIDENCE: public_observation
        public = jnp.concatenate((
            state["agent_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["stamina"].astype(jnp.float32) / 4.0,
            state["team"].astype(jnp.float32),
            state["carrying"].astype(jnp.float32),
            state["ball_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            (state["ball_owner"] + 1).astype(jnp.float32) / 4.0,
            state["ball_velocity"].reshape(-1).astype(jnp.float32),
            state["flight_left"].astype(jnp.float32) / 4.0,
            state["ball_team"].astype(jnp.float32),
            state["relay_ready"].astype(jnp.float32),
            state["hold"].astype(jnp.float32) / 4.0,
            (state["pass_from"] + 1).astype(jnp.float32) / 4.0,
            state["goal_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["goal_team"].astype(jnp.float32),
            state["t"][None].astype(jnp.float32) / 128.0,
            state["events"].astype(jnp.float32),
            jnp.stack((state["episode_done"], state["terminated"], state["truncated"])).astype(jnp.float32),
            state["map"].reshape(-1).astype(jnp.float32),
        ))
        return jnp.concatenate((jnp.eye(4, dtype=jnp.float32), jnp.broadcast_to(public, (4, 122))), axis=1)

    def raw_step(self, key, state, actions):
        a = jnp.where((actions >= 0) & (actions < 8), actions, 0).astype(jnp.int32)
        old = state["agent_pos"]
        ids = jnp.arange(4, dtype=jnp.int32)
        rank = jnp.mod(ids - state["t"], 4)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)
        # EVIDENCE: movement_and_fatigue
        target = jnp.clip(old + delta[a], jnp.array([0, 0]), jnp.array([10, 6]))
        moving = (a >= 1) & (a <= 4) & (state["stamina"] > 0)
        target = jnp.where(moving[:, None], target, old)
        wall = state["map"][target[:, 1], target[:, 0]] == 1
        occupied = jnp.any(jnp.all(target[:, None, :] == old[None, :, :], axis=2) & (ids[:, None] != ids[None, :]), axis=1)
        duplicate = jnp.sum(jnp.all(target[:, None, :] == target[None, :, :], axis=2), axis=1) > 1
        moved = moving & ~wall & ~occupied & ~duplicate & jnp.any(target != old, axis=1)
        pos = jnp.where(moved[:, None], target, old)
        stamina = jnp.minimum(4, state["stamina"] - moved.astype(jnp.int32) + ((a == 0) | (a == 7)).astype(jnp.int32))
        tackle_paid = (a == 6) & (stamina >= 2)
        stamina = stamina - 2 * tackle_paid.astype(jnp.int32)

        # EVIDENCE: real_ball_flight_and_interception
        owner = state["ball_owner"][0]
        ball = state["ball_pos"][0]
        velocity = state["ball_velocity"][0]
        left = state["flight_left"][0]
        team = state["ball_team"][0]
        ready = state["relay_ready"][0]
        source = state["pass_from"][0]
        flying = (owner < 0) & (left > 0)
        proposed_ball = ball + velocity
        bounded_ball = jnp.clip(proposed_ball, jnp.array([0, 0]), jnp.array([10, 6]))
        blocked_ball = jnp.any(proposed_ball != bounded_ball) | (state["map"][bounded_ball[1], bounded_ball[0]] == 1)
        ball = jnp.where(flying & ~blocked_ball, bounded_ball, ball)
        left = jnp.where(flying, jnp.where(blocked_ball, 0, left - 1), left)
        candidates = (owner < 0) & jnp.all(pos == ball[None, :], axis=1)
        catcher = jnp.argmin(jnp.where(candidates, rank, 99))
        caught = jnp.any(candidates)
        catch_team = state["team"][catcher]
        intercepted = caught & (source >= 0) & (catch_team != team)
        relay_catch = caught & (source >= 0) & (catch_team == team) & (catcher != source)
        ready = jnp.where(caught, (ready & (catch_team == team)) | relay_catch, ready)
        owner = jnp.where(caught, catcher, owner)
        team = jnp.where(caught, catch_team, team)
        left = jnp.where(caught, 0, left)
        source = jnp.where(caught, -1, source)
        holder = jnp.clip(owner, 0, 3)
        guarded = (a[holder] == 7) & (state["stamina"][holder] > 0)
        close = jnp.sum(jnp.abs(pos - pos[holder]), axis=1) <= 1
        tacklers = (owner >= 0) & tackle_paid & close & (state["team"] != team) & ~guarded
        tackler = jnp.argmin(jnp.where(tacklers, rank, 99))
        stolen = jnp.any(tacklers)
        owner = jnp.where(stolen, tackler, owner)
        team = jnp.where(stolen, state["team"][tackler], team)
        ready = ready & ~stolen
        source = jnp.where(stolen, -1, source)
        intercepted = intercepted | stolen

        # EVIDENCE: teammate_pass_and_hold_limit
        holder = jnp.clip(owner, 0, 3)
        teammate = jnp.array([1, 0, 3, 2], dtype=jnp.int32)[holder]
        difference = pos[teammate] - pos[holder]
        distance = jnp.sum(jnp.abs(difference))
        aligned = (difference[0] == 0) | (difference[1] == 0)
        launch = (owner >= 0) & (a[holder] == 5) & (stamina[holder] > 0) & aligned & (distance >= 1) & (distance <= 4)
        ball = jnp.where((owner >= 0), pos[holder], ball)
        velocity = jnp.where(launch, jnp.sign(difference), jnp.where(left > 0, velocity, jnp.zeros((2,), dtype=jnp.int32)))
        left = jnp.where(launch, distance, left)
        source = jnp.where(launch, holder, source)
        stamina = stamina - ((ids == holder) & launch).astype(jnp.int32)
        previous_same = owner == state["ball_owner"][0]
        hold = jnp.where(owner >= 0, jnp.where(previous_same, state["hold"][0] + 1, 0), 0)
        spill = (owner >= 0) & (hold >= 4) & ~launch
        owner = jnp.where(launch | spill, -1, owner)
        hold = jnp.where(owner < 0, 0, hold)
        ready = ready & ~spill
        source = jnp.where(spill, -1, source)
        velocity = jnp.where(spill, jnp.zeros((2,), dtype=jnp.int32), velocity)
        left = jnp.where(spill, 0, left)
        holder = jnp.clip(owner, 0, 3)
        scoring_team = state["team"][holder]
        score = (owner >= 0) & ready & jnp.all(ball == state["goal_pos"][scoring_team])
        rewards = jnp.where(score, jnp.where(state["team"] == scoring_team, 1.0, -1.0), 0.0).astype(jnp.float32)

        # EVIDENCE: natural_priority_absorbing_terminal
        n = dict(state)
        n["t"] = state["t"] + 1
        n["agent_pos"] = pos
        n["stamina"] = stamina
        n["carrying"] = ids == owner
        n["ball_pos"] = ball[None, :]
        n["ball_owner"] = owner[None].astype(jnp.int32)
        n["ball_velocity"] = velocity[None, :]
        n["flight_left"] = left[None].astype(jnp.int32)
        n["ball_team"] = team[None].astype(jnp.int32)
        n["relay_ready"] = ready[None]
        n["hold"] = hold[None].astype(jnp.int32)
        n["pass_from"] = source[None].astype(jnp.int32)
        n["events"] = jnp.stack((launch, caught, intercepted, spill, score))
        n["terminated"] = score
        n["truncated"] = (n["t"] >= self.max_steps) & ~score
        n["episode_done"] = n["terminated"] | n["truncated"]
        n = jax.tree.map(lambda new, previous: jnp.where(state["episode_done"], previous, new), n, state)
        rewards = jnp.where(state["episode_done"], jnp.zeros((4,), dtype=jnp.float32), rewards)
        done = n["episode_done"]
        info = {"done_all": done, "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, rewards, jnp.full((4,), done, dtype=jnp.bool_), info