"""Original Signal Veil Relay kernel; authored, never executed by the author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 181
    num_actions = 11
    max_steps = 96

    def distances(self, a, b):
        return jnp.sum(jnp.abs(a[:, None, :] - b[None, :, :]), axis=-1)

    def sight(self, s, points, concealable):
        d = self.distances(s["agent_pos"], points)
        radius = jnp.where(s["agent_role"] == 0, 4, 2)
        around_decoy = self.distances(s["decoy_pos"], points) <= 1
        hostile = s["agent_team"][:, None] != s["decoy_team"][None, :]
        jam = jnp.any(hostile[:, :, None] & s["decoy_active"][None, :, None]
                      & around_decoy[None, :, :], axis=1)
        local = (d <= radius[:, None]) & ~(concealable & jam & (d > 1))
        relay = ((s["agent_team"][:, None] == s["agent_team"][None, :])
                 & (s["agent_signal"][None, :] > 0))
        return local | jnp.any(relay[:, :, None] & local[None, :, :], axis=1)

    def refresh(self, s):
        n = dict(s)
        carrier = s["beacon_carrier"]
        n["beacon_pos"] = jnp.where((carrier >= 0)[:, None],
                                    s["agent_pos"][jnp.clip(carrier, 0, 3)], s["beacon_pos"])
        n["agent_carry"] = jnp.max(jnp.where(
            carrier[None, :] == jnp.arange(4)[:, None], jnp.arange(2)[None, :], -1), axis=1)
        n["decoy_active"] = s["decoy_ttl"] > 0
        n["agent_vis"] = (self.sight(n, n["agent_pos"], True)
                          | (n["agent_signal"][None, :] > 0) | jnp.eye(4, dtype=jnp.bool_))
        n["beacon_vis"] = self.sight(n, n["beacon_pos"], True)
        n["decoy_vis"] = self.sight(n, n["decoy_pos"], False) & n["decoy_active"][None, :]
        return n

    def reset(self, key):
        # Deterministic public layout; the supplied key is deliberately unused.
        grid = jnp.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 2, 0, 0, 0, 1, 0, 0, 0, 3, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=jnp.int32)
        s = {
            "t": jnp.array(0, jnp.int32), "episode_done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False), "map": grid,
            "agent_pos": jnp.array([[2, 2], [2, 4], [8, 2], [8, 4]], jnp.int32),
            "agent_active": jnp.ones(4, jnp.bool_), "agent_vis": jnp.zeros((4, 4), jnp.bool_),
            "agent_team": jnp.array([0, 0, 1, 1], jnp.int32),
            "agent_role": jnp.array([0, 1, 0, 1], jnp.int32),
            "agent_stun": jnp.zeros(4, jnp.int32), "agent_cooldown": jnp.zeros(4, jnp.int32),
            "agent_stock": jnp.ones(4, jnp.int32), "agent_signal": jnp.zeros(4, jnp.int32),
            "agent_carry": jnp.full(4, -1, jnp.int32),
            "beacon_pos": jnp.array([[1, 3], [9, 3]], jnp.int32),
            "beacon_active": jnp.ones(2, jnp.bool_), "beacon_vis": jnp.zeros((4, 2), jnp.bool_),
            "beacon_owner": jnp.array([0, 1], jnp.int32), "beacon_carrier": jnp.full(2, -1, jnp.int32),
            "base_pos": jnp.array([[1, 3], [9, 3]], jnp.int32),
            "base_active": jnp.ones(2, jnp.bool_), "base_vis": jnp.ones((4, 2), jnp.bool_),
            "decoy_pos": jnp.array([[2, 2], [2, 4], [8, 2], [8, 4]], jnp.int32),
            "decoy_active": jnp.zeros(4, jnp.bool_), "decoy_vis": jnp.zeros((4, 4), jnp.bool_),
            "decoy_team": jnp.array([0, 0, 1, 1], jnp.int32), "decoy_ttl": jnp.zeros(4, jnp.int32),
            "score": jnp.zeros(2, jnp.int32),
            "event_hit": jnp.zeros(4, jnp.bool_), "event_deploy": jnp.zeros(4, jnp.bool_),
            "event_signal": jnp.zeros(4, jnp.bool_), "event_pickup": jnp.zeros(2, jnp.bool_),
            "event_pass": jnp.zeros(2, jnp.bool_), "event_return": jnp.zeros(2, jnp.bool_),
            "event_capture": jnp.zeros(2, jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def legal_actions(self, s):
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        dest = s["agent_pos"][:, None, :] + delta[None, :, :]
        tiles = s["map"][jnp.clip(dest[:, :, 1], 0, 6), jnp.clip(dest[:, :, 0], 0, 10)]
        moves = (tiles != 1) & jnp.all((dest >= jnp.array([0, 0]))
                                     & (dest <= jnp.array([10, 6])), axis=-1)
        one = jnp.ones(4, jnp.bool_)
        mask = jnp.concatenate((one[:, None], moves, one[:, None],
            (s["agent_cooldown"] == 0)[:, None], one[:, None],
            ((s["agent_stock"] > 0) & ~s["decoy_active"])[:, None],
            (s["agent_carry"] >= 0)[:, None], one[:, None]), axis=1)
        ready = (s["agent_stun"] == 0) & ~s["episode_done"]
        return mask & (ready[:, None] | (jnp.arange(11)[None, :] == 0))

    def move(self, s, a):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        proposed = s["agent_pos"] + delta[jnp.where(a <= 4, a, 0)]
        same = jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=-1)
        occupied = jnp.all(proposed[:, None, :] == s["agent_pos"][None, :, :], axis=-1)
        blocked = jnp.any((same | occupied) & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        return jnp.where(blocked[:, None], s["agent_pos"], proposed)

    def advance(self, s, a):
        n = dict(s)
        n["t"] = s["t"] + jnp.array(1, jnp.int32)
        n["agent_pos"] = self.move(s, a)
        n["agent_signal"] = jnp.where(a == 7, 3, jnp.maximum(s["agent_signal"] - 1, 0))
        n["event_signal"] = a == 7
        deploy = a == 8
        n["event_deploy"] = deploy
        n["agent_stock"] = s["agent_stock"] - deploy.astype(jnp.int32)
        n["decoy_pos"] = jnp.where(deploy[:, None], n["agent_pos"], s["decoy_pos"])
        n["decoy_ttl"] = jnp.where(deploy, 4, jnp.maximum(s["decoy_ttl"] - 1, 0))
        n = self.refresh(n)

        # All beams see the same post-move, post-deploy snapshot. Decoys win distance ties.
        points = jnp.concatenate((n["decoy_pos"], n["agent_pos"]), axis=0)
        teams = jnp.concatenate((n["decoy_team"], n["agent_team"]))
        active = jnp.concatenate((n["decoy_active"], n["agent_active"]))
        offset = points[None, :, :] - n["agent_pos"][:, None, :]
        dist = jnp.sum(jnp.abs(offset), axis=-1)
        aligned = (offset[:, :, 0] == 0) | (offset[:, :, 1] == 0)
        middle = n["agent_pos"][:, None, :] + jnp.sign(offset)
        clear = n["map"][jnp.clip(middle[:, :, 1], 0, 6),
                         jnp.clip(middle[:, :, 0], 0, 10)] != 1
        eligible = (active[None, :] & (teams[None, :] != n["agent_team"][:, None])
                    & aligned & (dist > 0) & (dist <= 2) & clear)
        costs = jnp.where(eligible, 2 * dist + (jnp.arange(8)[None, :] >= 4), 30)
        target = jnp.argmin(costs, axis=1)
        fires = (a == 6) & (jnp.min(costs, axis=1) < 30)
        hit_decoy = jnp.any(fires[:, None] & (target[:, None] == jnp.arange(4)[None, :]), axis=0)
        hit_agent = jnp.any(fires[:, None] & (target[:, None] == (4 + jnp.arange(4))[None, :]), axis=0)
        n["event_hit"] = hit_agent
        n["agent_stun"] = jnp.where(hit_agent, 2, jnp.maximum(s["agent_stun"] - 1, 0))
        n["agent_cooldown"] = jnp.where(a == 6, 3, jnp.maximum(s["agent_cooldown"] - 1, 0))
        n["decoy_ttl"] = jnp.where(hit_decoy, 0, n["decoy_ttl"])
        drop = (n["beacon_carrier"] >= 0) & hit_agent[jnp.clip(n["beacon_carrier"], 0, 3)]
        n["beacon_carrier"] = jnp.where(drop, -1, n["beacon_carrier"])
        n["event_pickup"] = jnp.zeros(2, jnp.bool_)
        n["event_return"] = jnp.zeros(2, jnp.bool_)
        n["event_pass"] = jnp.zeros(2, jnp.bool_)

        # Adjacent object contests use ascending agent id; tags have already resolved.
        for i in range(4):
            own = n["agent_team"][i]
            enemy = 1 - own
            usable = (a[i] == 5) & (n["agent_stun"][i] == 0)
            nearby = jnp.sum(jnp.abs(n["beacon_pos"] - n["agent_pos"][i]), axis=1) <= 1
            return_own = usable & nearby[own] & (n["beacon_carrier"][own] < 0)
            return_own = return_own & jnp.any(n["beacon_pos"][own] != n["base_pos"][own])
            n["beacon_pos"] = n["beacon_pos"].at[own].set(jnp.where(
                return_own, n["base_pos"][own], n["beacon_pos"][own]))
            n["event_return"] = n["event_return"].at[own].set(n["event_return"][own] | return_own)
            pickup = usable & nearby[enemy] & (n["beacon_carrier"][enemy] < 0)
            n["beacon_carrier"] = n["beacon_carrier"].at[enemy].set(jnp.where(
                pickup, i, n["beacon_carrier"][enemy]))
            n["event_pickup"] = n["event_pickup"].at[enemy].set(n["event_pickup"][enemy] | pickup)

        for i in range(4):
            peer = i ^ 1
            enemy = 1 - n["agent_team"][i]
            handoff = ((a[i] == 9) & (a[peer] == 10) & (n["beacon_carrier"][enemy] == i)
                       & (n["agent_stun"][i] == 0) & (n["agent_stun"][peer] == 0)
                       & (jnp.sum(jnp.abs(n["agent_pos"][i] - n["agent_pos"][peer])) <= 1))
            n["beacon_carrier"] = n["beacon_carrier"].at[enemy].set(jnp.where(
                handoff, peer, n["beacon_carrier"][enemy]))
            n["event_pass"] = n["event_pass"].at[enemy].set(n["event_pass"][enemy] | handoff)

        n = self.refresh(n)
        home = (n["beacon_carrier"] < 0) & jnp.all(n["beacon_pos"] == n["base_pos"], axis=1)
        capture = jnp.zeros(2, jnp.bool_)
        for team in range(2):
            enemy = 1 - team
            carrier = n["beacon_carrier"][enemy]
            at_base = jnp.all(n["agent_pos"][jnp.clip(carrier, 0, 3)] == n["base_pos"][team])
            capture = capture.at[team].set((carrier >= 0) & at_base & home[team])
        n["event_capture"] = capture
        n["score"] = s["score"] + capture.astype(jnp.int32)
        n["terminated"] = jnp.any(capture)
        n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
        n["episode_done"] = n["terminated"] | n["truncated"]
        # The captured beacon remains at the carrier's actual terminal position.
        reward_team = capture.astype(jnp.int32) - capture[::-1].astype(jnp.int32)
        reward = reward_team[n["agent_team"]].astype(jnp.float32)
        return n, reward

    def observe(self, s):
        agent_values = jnp.concatenate((s["agent_pos"], s["agent_team"][:, None],
            s["agent_role"][:, None], s["agent_stun"][:, None], s["agent_cooldown"][:, None],
            s["agent_stock"][:, None], s["agent_signal"][:, None], s["agent_carry"][:, None]), axis=1)
        beacon_values = jnp.concatenate((s["beacon_pos"], s["beacon_owner"][:, None],
                                        s["beacon_carrier"][:, None]), axis=1)
        decoy_values = jnp.concatenate((s["decoy_pos"], s["decoy_team"][:, None],
                                       s["decoy_ttl"][:, None]), axis=1)
        agents = jnp.concatenate((s["agent_vis"][:, :, None].astype(jnp.int32),
            jnp.where(s["agent_vis"][:, :, None], agent_values[None, :, :], 0)), axis=2).reshape((4, 40))
        beacons = jnp.concatenate((s["beacon_vis"][:, :, None].astype(jnp.int32),
            jnp.where(s["beacon_vis"][:, :, None], beacon_values[None, :, :], 0)), axis=2).reshape((4, 10))
        decoys = jnp.concatenate((s["decoy_vis"][:, :, None].astype(jnp.int32),
            jnp.where(s["decoy_vis"][:, :, None], decoy_values[None, :, :], 0)), axis=2).reshape((4, 20))
        clock = jnp.array([s["t"], s["episode_done"], s["terminated"], s["truncated"]], jnp.int32)
        public = jnp.concatenate((s["base_pos"].reshape(-1), s["map"].reshape(-1), s["score"],
            s["event_hit"], s["event_deploy"], s["event_signal"], s["event_pickup"],
            s["event_pass"], s["event_return"], s["event_capture"])).astype(jnp.int32)
        obs = jnp.concatenate((jnp.eye(4, dtype=jnp.int32), jnp.broadcast_to(clock, (4, 4)),
            agents, beacons, decoys, jnp.broadcast_to(public, (4, 103))), axis=1)
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
