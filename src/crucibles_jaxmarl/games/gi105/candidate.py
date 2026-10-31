"""Original Two-Stride Relay Arena kernel; not imported or executed by its author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 182
    num_actions = 11
    max_steps = 96

    def reset(self, key):
        # Fixed kickoff: team zero has the token. Side-swapping is not implemented here.
        grid = jnp.zeros((9, 13), jnp.int32)
        grid = grid.at[0, :].set(1)
        grid = grid.at[8, :].set(1)
        grid = grid.at[:, 0].set(1)
        grid = grid.at[:, 12].set(1)
        grid = grid.at[3:6, 11].set(2)
        grid = grid.at[3:6, 1].set(3)
        s = {
            "t": jnp.array(0, jnp.int32), "episode_done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False), "map": grid,
            "agent_pos": jnp.array([[3, 3], [5, 5], [9, 3], [7, 5]], jnp.int32),
            "agent_active": jnp.ones(4, jnp.bool_), "agent_vis": jnp.ones((4, 4), jnp.bool_),
            "agent_team": jnp.array([0, 0, 1, 1], jnp.int32), "agent_cooldown": jnp.zeros(4, jnp.int32),
            "ball_pos": jnp.array([[3, 3]], jnp.int32), "ball_active": jnp.ones(1, jnp.bool_),
            "ball_vis": jnp.ones((4, 1), jnp.bool_), "ball_carrier": jnp.array([0], jnp.int32),
            "ball_velocity": jnp.zeros((1, 2), jnp.int32), "ball_flight": jnp.zeros(1, jnp.int32),
            "ball_relay": jnp.zeros(1, jnp.bool_), "ball_age": jnp.zeros(1, jnp.int32),
            "ball_travel": jnp.zeros(1, jnp.int32), "ball_last_thrower": jnp.full(1, -1, jnp.int32),
            "ball_last_team": jnp.full(1, -1, jnp.int32),
            "goal_pos": jnp.array([[11, 4], [1, 4]], jnp.int32),
            "goal_active": jnp.ones(2, jnp.bool_), "goal_vis": jnp.ones((4, 2), jnp.bool_),
            "goal_team": jnp.array([0, 1], jnp.int32), "score": jnp.zeros(2, jnp.int32),
            "event_throw": jnp.zeros(4, jnp.bool_), "event_catch": jnp.zeros(4, jnp.bool_),
            "event_intercept": jnp.zeros(4, jnp.bool_), "event_tackle": jnp.zeros(4, jnp.bool_),
            "event_drop": jnp.zeros(4, jnp.bool_), "event_goal": jnp.zeros(2, jnp.bool_)}
        return self.observe(s), s

    def legal_actions(self, s):
        ids = jnp.arange(4)
        owns = ids == s["ball_carrier"][0]
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s["agent_pos"][:, None, :] + delta[None, :, :]
        tile = s["map"][jnp.clip(p[:, :, 1], 0, 8), jnp.clip(p[:, :, 0], 0, 12)]
        moves = ((tile != 1) & jnp.all((p >= jnp.array([0, 0]))
                  & (p <= jnp.array([12, 8])), axis=-1)
                  & (~owns | (s["ball_travel"][0] < 2))[:, None])
        one = jnp.ones((4, 1), jnp.bool_)
        mask = jnp.concatenate((one, moves, jnp.broadcast_to(owns[:, None], (4, 4)), one, one), axis=1)
        return mask & (~s["episode_done"] | (jnp.arange(11)[None, :] == 0))

    def move(self, s, a):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        proposed = s["agent_pos"] + delta[jnp.where(a <= 4, a, 0)]
        equal = jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=-1)
        occupied = jnp.all(proposed[:, None, :] == s["agent_pos"][None, :, :], axis=-1)
        blocked = jnp.any((equal | occupied) & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        return jnp.where(blocked[:, None], s["agent_pos"], proposed)

    def advance(self, s, a):
        n = dict(s)
        ids = jnp.arange(4)
        n["t"] = s["t"] + jnp.array(1, jnp.int32)
        n["agent_pos"] = self.move(s, a)
        n["agent_cooldown"] = jnp.maximum(s["agent_cooldown"] - 1, 0)
        owner = s["ball_carrier"][0]
        safe_owner = jnp.clip(owner, 0, 3)
        held = owner >= 0
        pos = jnp.where(held, n["agent_pos"][safe_owner], s["ball_pos"][0])
        moved = jnp.any(n["agent_pos"][safe_owner] != s["agent_pos"][safe_owner])
        age = jnp.where(held, s["ball_age"][0] + 1, 0)
        travel = jnp.where(held, s["ball_travel"][0] + moved.astype(jnp.int32), 0)
        thrown = held & (a[safe_owner] >= 5) & (a[safe_owner] <= 8)
        directions = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        velocity = jnp.where(thrown, directions[jnp.clip(a[safe_owner] - 5, 0, 3)], s["ball_velocity"][0])
        flight = jnp.where(thrown, 6, s["ball_flight"][0])
        last_thrower = jnp.where(thrown, owner, s["ball_last_thrower"][0])
        last_team = jnp.where(thrown, n["agent_team"][safe_owner], s["ball_last_team"][0])
        relay = s["ball_relay"][0]
        n["event_throw"] = thrown & (ids == owner)
        n["agent_cooldown"] = jnp.where(thrown & (ids == owner), 2, n["agent_cooldown"])
        owner = jnp.where(thrown, -1, owner)
        age = jnp.where(thrown, 0, age)
        travel = jnp.where(thrown, 0, travel)

        # One tile of actual flight per transition, including the throw transition.
        flying = (owner < 0) & (flight > 0)
        next_pos = pos + velocity
        inside = jnp.all((next_pos >= jnp.array([0, 0])) & (next_pos <= jnp.array([12, 8])))
        open_tile = s["map"][jnp.clip(next_pos[1], 0, 8), jnp.clip(next_pos[0], 0, 12)] != 1
        advances = flying & inside & open_tile
        pos = jnp.where(advances, next_pos, pos)
        flight = jnp.where(flying, jnp.where(advances, flight - 1, 0), flight)
        velocity = jnp.where((flight > 0), velocity, jnp.zeros(2, jnp.int32))

        # Catch resolution: nearest contact, then hostile guard, then rotating role priority.
        distance = jnp.sum(jnp.abs(n["agent_pos"] - pos), axis=1)
        hostile = (last_team >= 0) & (n["agent_team"] != last_team)
        guard = (a == 9) & hostile & (distance <= 1)
        receive = (a == 10) & ~hostile & (distance <= 1)
        eligible = ((n["agent_cooldown"] == 0) & ((distance == 0) | guard | receive)
                    & ((ids != last_thrower) | ~flying))
        rank = jnp.mod(ids - s["t"], 4)
        costs = distance * 16 + jnp.where(guard, 0, 4) + rank
        catcher = jnp.argmin(jnp.where(eligible, costs, 1000))
        caught = (owner < 0) & jnp.any(eligible)
        allied_pass = flying & (last_thrower >= 0) & (last_thrower != catcher)
        allied_pass = allied_pass & (last_team == n["agent_team"][catcher])
        n["event_catch"] = caught & (ids == catcher)
        n["event_intercept"] = caught & flying & hostile[catcher] & (ids == catcher)
        owner = jnp.where(caught, catcher, owner)
        relay = jnp.where(caught, allied_pass, relay)
        pos = jnp.where(caught, n["agent_pos"][catcher], pos)
        flight = jnp.where(caught, 0, flight)
        velocity = jnp.where(caught, jnp.zeros(2, jnp.int32), velocity)
        age = jnp.where(caught, 0, age)
        travel = jnp.where(caught, 0, travel)

        # A guard can steal from an adjacent held carrier; newly caught balls are protected this tick.
        safe_owner = jnp.clip(owner, 0, 3)
        near_carrier = jnp.sum(jnp.abs(n["agent_pos"] - n["agent_pos"][safe_owner]), axis=1) <= 1
        can_tackle = ((a == 9) & near_carrier & (n["agent_team"] != n["agent_team"][safe_owner])
                      & (n["agent_cooldown"] == 0))
        tackler = jnp.argmin(jnp.where(can_tackle, rank, 9))
        tackled = (owner >= 0) & ~caught & ~thrown & jnp.any(can_tackle)
        n["event_tackle"] = tackled & (ids == tackler)
        owner = jnp.where(tackled, tackler, owner)
        pos = jnp.where(tackled, n["agent_pos"][tackler], pos)
        relay = relay & ~tackled
        age = jnp.where(tackled, 0, age)
        travel = jnp.where(tackled, 0, travel)

        forced_drop = (owner >= 0) & (age >= 4)
        n["event_drop"] = forced_drop & (ids == owner)
        n["agent_cooldown"] = jnp.where(forced_drop & (ids == owner), 2, n["agent_cooldown"])
        owner = jnp.where(forced_drop, -1, owner)
        relay = relay & ~forced_drop
        age = jnp.where(forced_drop, 0, age)
        travel = jnp.where(forced_drop, 0, travel)
        n["ball_pos"] = pos[None, :].astype(jnp.int32)
        n["ball_carrier"] = owner[None].astype(jnp.int32)
        n["ball_velocity"] = velocity[None, :].astype(jnp.int32)
        n["ball_flight"] = flight[None].astype(jnp.int32)
        n["ball_relay"] = relay[None]
        n["ball_age"] = age[None].astype(jnp.int32)
        n["ball_travel"] = travel[None].astype(jnp.int32)
        n["ball_last_thrower"] = last_thrower[None].astype(jnp.int32)
        n["ball_last_team"] = last_team[None].astype(jnp.int32)
        scoring_team = n["agent_team"][jnp.clip(owner, 0, 3)]
        goal_x = jnp.where(scoring_team == 0, 11, 1)
        goal = (owner >= 0) & relay & (pos[0] == goal_x) & (pos[1] >= 3) & (pos[1] <= 5)
        scored = goal & (jnp.arange(2) == scoring_team)
        n["event_goal"] = scored
        n["score"] = s["score"] + scored.astype(jnp.int32)
        n["terminated"] = goal
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["episode_done"] = n["terminated"] | n["truncated"]
        team_reward = scored.astype(jnp.int32) - scored[::-1].astype(jnp.int32)
        return n, team_reward[n["agent_team"]].astype(jnp.float32)

    def observe(self, s):
        clock = jnp.array([s["t"], s["episode_done"], s["terminated"], s["truncated"]], jnp.int32)
        agents = jnp.concatenate((s["agent_pos"], s["agent_team"][:, None],
                                  s["agent_cooldown"][:, None]), axis=1).reshape(-1)
        ball = jnp.concatenate((s["ball_pos"].reshape(-1), s["ball_carrier"], s["ball_velocity"].reshape(-1),
            s["ball_flight"], s["ball_relay"], s["ball_age"], s["ball_travel"],
            s["ball_last_thrower"], s["ball_last_team"])).astype(jnp.int32)
        goals = jnp.concatenate((s["goal_pos"], s["goal_team"][:, None]), axis=1).reshape(-1)
        shared = jnp.concatenate((clock, agents, ball, goals, s["map"].reshape(-1), s["score"],
            s["event_throw"], s["event_catch"], s["event_intercept"], s["event_tackle"],
            s["event_drop"], s["event_goal"])).astype(jnp.int32)
        obs = jnp.concatenate((jnp.eye(4, dtype=jnp.int32), jnp.broadcast_to(shared, (4, 178))), axis=1)
        return obs.astype(jnp.float32) / 128.0

    def raw_step(self, key, state, actions):
        clipped = jnp.clip(actions, 0, self.num_actions - 1)
        allowed = self.legal_actions(state)[jnp.arange(4), clipped]
        valid = (actions >= 0) & (actions < self.num_actions) & allowed
        a = jnp.where(valid, clipped, 0).astype(jnp.int32)
        proposed, reward = self.advance(state, a)
        n = jax.tree.map(lambda new, old: jnp.where(state["episode_done"], old, new), proposed, state)
        reward = jnp.where(state["episode_done"], jnp.zeros(4, jnp.float32), reward)
        dones = jnp.full((4,), n["episode_done"], jnp.bool_)
        info = {"done_all": n["episode_done"], "terminated": n["terminated"], "truncated": n["truncated"]}
        return self.observe(n), n, reward, dones, info
