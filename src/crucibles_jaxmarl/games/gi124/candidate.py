"""Bankline: a four-player two-team possession and ricochet arena.
Original authored kernel; numerical and gameplay validation remain unexecuted.
"""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 169
    num_actions = 12
    max_steps = 120

    def floor(self, tiles, xy):
        inside = (xy[:, 0] >= 1) & (xy[:, 0] <= 11) & (xy[:, 1] >= 1) & (xy[:, 1] <= 7)
        return inside & (tiles[jnp.clip(xy[:, 1], 0, 8), jnp.clip(xy[:, 0], 0, 12)] == 0)

    def reset(self, key):
        xx = jnp.arange(13, dtype=jnp.int32)[None, :]
        yy = jnp.arange(9, dtype=jnp.int32)[:, None]
        tiles = ((xx == 0) | (xx == 12) | (yy == 0) | (yy == 8) |
                 ((xx == 6) & ((yy == 2) | (yy == 6))) |
                 ((yy == 4) & ((xx == 4) | (xx == 8)))).astype(jnp.int32)
        state = {
            "t": jnp.array(0, dtype=jnp.int32), "tiles": tiles,
            "agent_xy": jnp.array([[2, 3], [3, 5], [10, 5], [9, 3]], dtype=jnp.int32),
            "agent_active": jnp.ones(4, dtype=jnp.bool_), "agent_visible": jnp.ones((4, 4), dtype=jnp.bool_),
            "team": jnp.array([0, 0, 1, 1], dtype=jnp.int32),
            "cooldown": jnp.zeros(4, dtype=jnp.int32), "blocking": jnp.zeros(4, dtype=jnp.bool_),
            "ball_xy": jnp.array([[3, 3]], dtype=jnp.int32), "ball_velocity": jnp.zeros((1, 2), dtype=jnp.int32),
            "ball_owner": jnp.array([-1], dtype=jnp.int32), "ball_timer": jnp.zeros(1, dtype=jnp.int32),
            "ball_active": jnp.ones(1, dtype=jnp.bool_), "ball_visible": jnp.ones((4, 1), dtype=jnp.bool_),
            "last_team": jnp.array([-1], dtype=jnp.int32), "last_holder": jnp.array([-1], dtype=jnp.int32),
            "goal_xy": jnp.array([[1, 4], [11, 4]], dtype=jnp.int32),
            "goal_active": jnp.ones(2, dtype=jnp.bool_), "goal_visible": jnp.ones((4, 2), dtype=jnp.bool_),
            "goal_defender": jnp.array([0, 1], dtype=jnp.int32), "score": jnp.zeros(2, dtype=jnp.int32),
            "pass_event": jnp.array(False, dtype=jnp.bool_), "catch_event": jnp.array(False, dtype=jnp.bool_),
            "intercept_event": jnp.array(False, dtype=jnp.bool_), "bounce_event": jnp.array(False, dtype=jnp.bool_),
            "score_event": jnp.zeros(2, dtype=jnp.bool_),
            "episode_done": jnp.array(False, dtype=jnp.bool_), "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
        }
        return self.observe(state), state

    def legal_actions(self, state):
        return jnp.broadcast_to((jnp.arange(12) == 0) | ~state["episode_done"], (4, 12))

    def observe(self, state):
        public = jnp.concatenate([
            state["agent_xy"].reshape(-1).astype(jnp.float32) / 12.0,
            jax.nn.one_hot(state["team"], 2, dtype=jnp.float32).reshape(-1),
            state["cooldown"].astype(jnp.float32) / 2.0, state["blocking"].astype(jnp.float32),
            state["ball_xy"].reshape(-1).astype(jnp.float32) / 12.0,
            state["ball_velocity"].reshape(-1).astype(jnp.float32),
            jax.nn.one_hot(state["ball_owner"][0] + 1, 5, dtype=jnp.float32),
            state["ball_timer"].astype(jnp.float32) / 6.0,
            jax.nn.one_hot(state["last_team"][0] + 1, 3, dtype=jnp.float32),
            state["score"].astype(jnp.float32) / 3.0,
            state["goal_xy"].reshape(-1).astype(jnp.float32) / 12.0,
            state["tiles"].reshape(-1).astype(jnp.float32),
        ])
        tail = jnp.concatenate([jnp.array([state["t"]], dtype=jnp.float32) / 120.0,
                               jnp.array([state["pass_event"], state["catch_event"], state["intercept_event"], state["bounce_event"]], dtype=jnp.float32)])
        return jnp.concatenate([jnp.broadcast_to(public, (4, 160)), jnp.eye(4, dtype=jnp.float32),
                                jnp.broadcast_to(tail, (4, 5))], axis=1)

    def raw_step(self, key, state, actions):
        actions = jnp.asarray(actions, dtype=jnp.int32)
        actions = jnp.where((actions >= 0) & (actions < 12), actions, 0)
        directions = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)
        pos = state["agent_xy"]
        target = pos + directions[actions]
        occupied = jnp.any(jnp.all(target[:, None, :] == pos[None, :, :], axis=2) & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        contested = jnp.sum(jnp.all(target[:, None, :] == target[None, :, :], axis=2), axis=1) > 1
        new_pos = jnp.where((self.floor(state["tiles"], target) & ~occupied & ~contested)[:, None], target, pos)
        ready = state["cooldown"] == 0
        blocking = (actions == 11) & ready
        cooldown = jnp.maximum(state["cooldown"] - 1, 0)
        priority = jnp.mod(jnp.arange(4, dtype=jnp.int32) - state["t"], 4)
        owner = state["ball_owner"][0]
        safe_owner = jnp.clip(owner, 0, 3)
        ball = jnp.where(owner >= 0, new_pos[safe_owner], state["ball_xy"][0])
        velocity = state["ball_velocity"][0]
        timer = state["ball_timer"][0]
        last_team = state["last_team"][0]
        last_holder = state["last_holder"][0]
        takers = (owner >= 0) & (actions == 5) & ready & (state["team"] != state["team"][safe_owner]) & (jnp.sum(jnp.abs(new_pos - ball), axis=1) <= 1) & ~blocking[safe_owner]
        tackler = jnp.argmin(jnp.where(takers, priority, 4))
        stolen = jnp.any(takers)
        owner = jnp.where(stolen, tackler, owner)
        ball = jnp.where(stolen, new_pos[tackler], ball)
        velocity = jnp.where(stolen, jnp.zeros(2, dtype=jnp.int32), velocity)
        timer = jnp.where(stolen, 0, timer)
        last_team = jnp.where(stolen, state["team"][tackler], last_team)
        last_holder = jnp.where(stolen, tackler, last_holder)
        safe_owner = jnp.clip(owner, 0, 3)
        teammate = jnp.array([1, 0, 3, 2], dtype=jnp.int32)[safe_owner]
        pass_request = (owner >= 0) & (actions[safe_owner] == 6) & ready[safe_owner] & ~stolen
        shoot_request = (owner >= 0) & (actions[safe_owner] >= 7) & (actions[safe_owner] <= 10) & ready[safe_owner] & ~stolen
        release = pass_request | shoot_request
        pass_direction = jnp.sign(new_pos[teammate] - ball)
        shot_directions = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        shot_direction = shot_directions[jnp.clip(actions[safe_owner] - 7, 0, 3)]
        launch_direction = jnp.where(pass_request, pass_direction, shot_direction)
        velocity = jnp.where(release, launch_direction, velocity)
        timer = jnp.where(release, 6, timer)
        last_team = jnp.where(release, state["team"][safe_owner], last_team)
        last_holder = jnp.where(release, safe_owner, last_holder)
        releasing_owner = safe_owner
        owner = jnp.where(release, -1, owner)
        loose_flight = (owner < 0) & (timer > 0)
        x_trial = ball + jnp.array([1, 0], dtype=jnp.int32) * velocity
        y_trial = ball + jnp.array([0, 1], dtype=jnp.int32) * velocity
        diagonal = ball + velocity
        hit_x = ~self.floor(state["tiles"], x_trial[None, :])[0]
        hit_y = ~self.floor(state["tiles"], y_trial[None, :])[0]
        hit_corner = ~self.floor(state["tiles"], diagonal[None, :])[0] & ~hit_x & ~hit_y
        reflected = velocity * jnp.where(jnp.array([hit_x | hit_corner, hit_y | hit_corner]), -1, 1)
        advance = ball + reflected
        can_advance = self.floor(state["tiles"], advance[None, :])[0]
        moving_ball = jnp.where(can_advance, advance, ball)
        moving_velocity = jnp.where(can_advance, reflected, jnp.zeros(2, dtype=jnp.int32))
        ball_after_wall = jnp.where(loose_flight, moving_ball, ball)
        velocity = jnp.where(loose_flight, moving_velocity, velocity)
        timer = jnp.where(loose_flight, jnp.maximum(timer - 1, 0), timer)
        blockers = (owner < 0) & blocking & (jnp.sum(jnp.abs(new_pos - ball_after_wall), axis=1) <= 1)
        blocker = jnp.argmin(jnp.where(blockers, priority, 4))
        deflected = jnp.any(blockers)
        ball_after_block = jnp.where(deflected, ball, ball_after_wall)
        velocity = jnp.where(deflected, -velocity, velocity)
        last_team = jnp.where(deflected, state["team"][blocker], last_team)
        last_holder = jnp.where(deflected, blocker, last_holder)
        distance = jnp.sum(jnp.abs(new_pos - ball_after_block), axis=1)
        catchers = (owner < 0) & ~blocking & ((distance == 0) | ((distance <= 1) & (actions == 5) & ready))
        catchers = catchers & ~(release & (jnp.arange(4) == releasing_owner)) & ~deflected
        catcher = jnp.argmin(jnp.where(catchers, priority, 4))
        caught = jnp.any(catchers)
        intercept = stolen | (caught & (last_team >= 0) & (state["team"][catcher] != last_team))
        owner = jnp.where(caught, catcher, owner)
        ball = jnp.where(caught, new_pos[catcher], ball_after_block)
        velocity = jnp.where(caught, jnp.zeros(2, dtype=jnp.int32), velocity)
        timer = jnp.where(caught, 0, timer)
        last_team = jnp.where(caught, state["team"][catcher], last_team)
        last_holder = jnp.where(caught, catcher, last_holder)
        mouth = (ball[1] >= 3) & (ball[1] <= 5)
        score_event = jnp.array([(owner < 0) & (ball[0] == 11) & mouth,
                                 (owner < 0) & (ball[0] == 1) & mouth], dtype=jnp.bool_)
        score = state["score"] + score_event.astype(jnp.int32)
        scored = jnp.any(score_event)
        ball = jnp.where(scored, jnp.array([6, 4], dtype=jnp.int32), ball)
        velocity = jnp.where(scored | (timer == 0), jnp.zeros(2, dtype=jnp.int32), velocity)
        timer = jnp.where(scored, 0, timer)
        owner = jnp.where(scored, -1, owner)
        last_team = jnp.where(scored, -1, last_team)
        last_holder = jnp.where(scored, -1, last_holder)
        used = blocking | (stolen & (jnp.arange(4) == tackler)) | (release & (jnp.arange(4) == releasing_owner)) | (caught & (actions == 5) & (jnp.arange(4) == catcher))
        cooldown = jnp.where(used, 2, cooldown).astype(jnp.int32)
        bounce = (loose_flight & (hit_x | hit_y | hit_corner | ~can_advance)) | deflected
        clock = state["t"] + jnp.array(1, dtype=jnp.int32)
        terminated = jnp.any(score >= 3)
        truncated = (clock >= 120) & ~terminated
        next_state = dict(state, t=clock, agent_xy=new_pos, cooldown=cooldown, blocking=blocking,
                          ball_xy=ball[None, :], ball_velocity=velocity[None, :], ball_owner=owner[None], ball_timer=timer[None],
                          last_team=last_team[None], last_holder=last_holder[None], score=score,
                          pass_event=pass_request, catch_event=caught, intercept_event=intercept, bounce_event=bounce,
                          score_event=score_event, terminated=terminated, truncated=truncated, episode_done=terminated | truncated)
        team_reward = score_event.astype(jnp.float32) - score_event[::-1].astype(jnp.float32)
        rewards = team_reward[state["team"]]
        next_state = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, jnp.asarray(new, dtype=old.dtype)), state, next_state)
        rewards = jnp.where(state["episode_done"], jnp.zeros(4, dtype=jnp.float32), rewards)
        done = next_state["episode_done"]
        info = {"done_all": done, "terminated": next_state["terminated"], "truncated": next_state["truncated"]}
        return self.observe(next_state), next_state, rewards, jnp.full(4, done, dtype=jnp.bool_), info
