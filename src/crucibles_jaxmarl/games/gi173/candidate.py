"""Relay Arena: original prospective source; never executed by author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 66
    num_actions = 11
    max_steps = 80

    def reset(self, key):
        s = {
            't': jnp.array(0, jnp.int32), 'episode_done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False),
            'agent_pos': jnp.array([[3, 3], [3, 4], [7, 3], [7, 2]], jnp.int32),
            'agent_active': jnp.ones((4,), jnp.bool_),
            'agent_visible': jnp.ones((4, 4), jnp.bool_),
            'team': jnp.array([0, 0, 1, 1], jnp.int32),
            'energy': jnp.full((4,), 8, jnp.int32),
            'cooldown': jnp.zeros((4,), jnp.int32),
            'screen': jnp.zeros((4,), jnp.bool_),
            'possession': jnp.array([True, False, False, False]),
            'ball_pos': jnp.array([[3, 3]], jnp.int32),
            'ball_active': jnp.ones((1,), jnp.bool_),
            'ball_visible': jnp.ones((4, 1), jnp.bool_),
            'carrier': jnp.array([0], jnp.int32),
            'velocity': jnp.zeros((1, 2), jnp.int32),
            'flight': jnp.zeros((1,), jnp.int32),
            'last_passer': jnp.array([-1], jnp.int32),
            'goal_pos': jnp.array([[10, 3], [0, 3]], jnp.int32),
            'goal_active': jnp.ones((2,), jnp.bool_),
            'goal_visible': jnp.ones((4, 2), jnp.bool_),
            'score': jnp.zeros((2,), jnp.int32),
            'passed': jnp.zeros((4,), jnp.bool_),
            'caught': jnp.zeros((4,), jnp.bool_),
            'intercepted': jnp.zeros((4,), jnp.bool_),
            'stolen': jnp.zeros((4,), jnp.bool_),
            'goal_event': jnp.zeros((2,), jnp.bool_),
            'kickoff': jnp.array(False),
        }
        return self.observe(s), s

    def legal_actions(self, s):
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = s['agent_pos'][:, None, :] + dirs[None, :, :]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] <= 10) & (q[:, :, 1] >= 0) & (q[:, :, 1] <= 6)
        holding = s['carrier'][0] == jnp.arange(4)
        walk = inside & (s['energy'] >= 1 + holding.astype(jnp.int32))[:, None]
        ready = s['cooldown'] == 0
        catch = ~holding & ready & (s['energy'] >= 1)
        pass_ok = holding & ready & (s['energy'] >= 2)
        screen = ~holding & ready & (s['energy'] >= 2)
        mask = jnp.concatenate([jnp.ones((4, 1), jnp.bool_), walk, catch[:, None], jnp.broadcast_to(pass_ok[:, None], (4, 4)), screen[:, None]], axis=1)
        return mask & (~s['episode_done'] | (jnp.arange(11)[None, :] == 0))

    def observe(self, s):
        public = jnp.concatenate([
            s['t'][None], s['agent_pos'].reshape(-1), s['energy'], s['cooldown'], s['team'],
            s['screen'].astype(jnp.int32), s['possession'].astype(jnp.int32),
            s['ball_pos'].reshape(-1), s['carrier'], s['velocity'].reshape(-1), s['flight'], s['last_passer'],
            s['goal_pos'].reshape(-1), s['score'], s['passed'].astype(jnp.int32),
            s['caught'].astype(jnp.int32), s['intercepted'].astype(jnp.int32), s['stolen'].astype(jnp.int32),
            s['goal_event'].astype(jnp.int32), s['kickoff'][None].astype(jnp.int32), s['episode_done'][None].astype(jnp.int32),
        ]).astype(jnp.float32)
        return jnp.concatenate([jnp.eye(4, dtype=jnp.float32), jnp.broadcast_to(public[None, :], (4, 62))], axis=1)

    def advance(self, s, actions):
        safe = jnp.clip(actions, 0, 10)
        a = jnp.where((safe == actions) & self.legal_actions(s)[jnp.arange(4), safe], safe, 0)
        dirs = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        old_carrier = s['carrier'][0]
        holding = old_carrier == jnp.arange(4)
        walking = (a >= 1) & (a <= 4)
        screen = a == 10
        desired = s['agent_pos'] + dirs[jnp.where(walking, a, 0)]
        different = ~jnp.eye(4, dtype=jnp.bool_)
        occupied = jnp.any(jnp.all(desired[:, None, :] == s['agent_pos'][None, :, :], axis=-1) & different, axis=1)
        contested = jnp.any(jnp.all(desired[:, None, :] == desired[None, :, :], axis=-1) & different & walking[None, :], axis=1)
        enemy_screen = jnp.any((jnp.sum(jnp.abs(desired[:, None, :] - s['agent_pos'][None, :, :]), axis=-1) <= 1) & screen[None, :] & (s['team'][:, None] != s['team'][None, :]), axis=1)
        move = walking & ~occupied & ~contested & ~enemy_screen
        p = jnp.where(move[:, None], desired, s['agent_pos'])
        cost = walking.astype(jnp.int32) * (1 + holding.astype(jnp.int32)) + (a == 5).astype(jnp.int32) + 2 * (a >= 6).astype(jnp.int32)
        energy = jnp.clip(s['energy'] - cost + 2 * (a == 0).astype(jnp.int32), 0, 8)
        cooldown = jnp.where(a >= 5, 2, jnp.maximum(s['cooldown'] - 1, 0))
        carrier = old_carrier
        ball = jnp.where(old_carrier >= 0, p[jnp.maximum(old_carrier, 0)], s['ball_pos'][0])
        holder = jnp.maximum(old_carrier, 0)
        distance = jnp.sum(jnp.abs(p - ball[None, :]), axis=1)
        protected = jnp.any(screen & (s['team'] == s['team'][holder]) & (jnp.sum(jnp.abs(p - p[holder]), axis=1) <= 1))
        tackle = (old_carrier >= 0) & (a == 5) & (s['team'] != s['team'][holder]) & (distance <= 1) & ~protected
        # Distance wins contested possession; rotating public role order breaks equal-distance ties.
        priority = (jnp.arange(4) - (s['t'] % 4)) % 4
        taker = jnp.argmin(jnp.where(tackle, 4 * distance + priority, 99)).astype(jnp.int32)
        taken = jnp.any(tackle)
        carrier = jnp.where(taken, taker, carrier)
        ball = jnp.where(taken, p[taker], ball)
        stolen = taken & (jnp.arange(4) == taker)
        launch = (old_carrier >= 0) & ~taken & (a[holder] >= 6) & (a[holder] <= 9)
        velocity = jnp.where(launch, dirs[jnp.clip(a[holder] - 5, 0, 4)], s['velocity'][0])
        flight = jnp.where(launch, 4, s['flight'][0])
        passer = jnp.where(launch, old_carrier, s['last_passer'][0])
        carrier = jnp.where(launch, -1, carrier)
        flying = (carrier < 0) & (flight > 0)
        proposed_ball = ball + jnp.where(flying, velocity, 0)
        inside = (proposed_ball[0] >= 0) & (proposed_ball[0] <= 10) & (proposed_ball[1] >= 0) & (proposed_ball[1] <= 6)
        ball = jnp.clip(proposed_ball, jnp.array([0, 0], jnp.int32), jnp.array([10, 6], jnp.int32))
        flight = jnp.where(flying, jnp.maximum(flight - 1, 0), 0)
        flight = jnp.where(inside, flight, 0)
        catch_distance = jnp.sum(jnp.abs(p - ball[None, :]), axis=1)
        receiver = (carrier < 0) & (a == 5) & (catch_distance <= 1)
        winner = jnp.argmin(jnp.where(receiver, 4 * catch_distance + priority, 99)).astype(jnp.int32)
        caught_any = jnp.any(receiver)
        caught = caught_any & (jnp.arange(4) == winner)
        intercepted = caught & (passer >= 0) & (s['team'] != s['team'][jnp.maximum(passer, 0)])
        carrier = jnp.where(caught_any, winner, carrier)
        ball = jnp.where(caught_any, p[winner], ball)
        flight = jnp.where(carrier >= 0, 0, flight)
        velocity = jnp.where(flight > 0, velocity, jnp.zeros((2,), jnp.int32))
        passer = jnp.where(carrier >= 0, -1, passer)
        lane = (ball[1] >= 2) & (ball[1] <= 4)
        goal = jnp.array([(ball[0] == 10) & lane, (ball[0] == 0) & lane], jnp.bool_)
        score = s['score'] + goal.astype(jnp.int32)
        terminal = jnp.any(score >= 2)
        kickoff = jnp.any(goal) & ~terminal & (s['t'] + 1 < self.max_steps)
        start = jnp.array([[3, 3], [3, 4], [7, 3], [7, 2]], jnp.int32)
        kickoff_carrier = jnp.where(goal[0], 2, 0).astype(jnp.int32)
        p = jnp.where(kickoff, start, p)
        carrier = jnp.where(kickoff, kickoff_carrier, carrier)
        ball = jnp.where(kickoff, start[kickoff_carrier], ball)
        velocity = jnp.where(kickoff, jnp.zeros((2,), jnp.int32), velocity)
        flight = jnp.where(kickoff, 0, flight)
        passer = jnp.where(kickoff, -1, passer)
        n = dict(s)
        n['t'] = s['t'] + 1
        n['agent_pos'], n['energy'], n['cooldown'] = p, energy, cooldown
        n['screen'] = screen & ~kickoff
        n['possession'] = carrier == jnp.arange(4)
        n['ball_pos'], n['carrier'], n['velocity'], n['flight'], n['last_passer'] = ball[None, :], carrier[None], velocity[None, :], flight[None], passer[None]
        n['score'], n['passed'], n['caught'], n['intercepted'], n['stolen'] = score, launch & holding, caught, intercepted, stolen
        n['goal_event'], n['kickoff'] = goal, kickoff
        n['terminated'] = terminal
        n['truncated'] = (n['t'] >= self.max_steps) & ~terminal
        n['episode_done'] = terminal | n['truncated']
        own = goal[s['team']].astype(jnp.int32)
        opponent = goal[1 - s['team']].astype(jnp.int32)
        reward = (10 * (own - opponent)).astype(jnp.float32)
        return n, reward

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['episode_done'], lambda _: (state, jnp.zeros((4,), jnp.float32)), lambda _: self.advance(state, actions), operand=None)
        info = {'done_all': n['episode_done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full((4,), n['episode_done'], jnp.bool_), info
