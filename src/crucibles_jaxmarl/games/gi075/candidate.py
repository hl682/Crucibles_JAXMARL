"""Scentbank Scramble: original source proposal; AUTHORED_NOT_EXECUTED."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 868
    num_actions = 9
    max_steps = 128
    terrain_rows = [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,1,1,1,1,1,1,1,1,1,0],[0,1,1,1,0,1,1,1,0,1,1,1,0],[0,2,1,1,1,1,1,1,1,1,1,3,0],[0,4,1,1,1,1,1,1,1,1,1,5,0],[0,2,1,1,1,1,1,1,1,1,1,3,0],[0,1,1,1,0,1,1,1,0,1,1,1,0],[0,1,1,1,1,1,1,1,1,1,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]]
    spawn_xy = [[5,2],[7,2],[5,3],[6,3],[7,3],[5,5],[6,5],[7,5],[5,6],[7,6]]

    def directions(self):
        return jnp.array([[0,-1],[1,0],[0,1],[-1,0]], dtype=jnp.int32)

    def candidate_directions(self):
        return jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]], dtype=jnp.int32)

    def distance_relax(self, pivot, distances):
        return jnp.minimum(distances, distances[:,pivot,None] + distances[None,pivot,:])

    def distances(self, terrain):
        index = jnp.arange(117, dtype=jnp.int32)
        pos = jnp.stack((index % 13, index // 13), axis=1)
        walk = terrain.reshape((117,)) != 0
        adjacency = (jnp.sum(jnp.abs(pos[:,None,:] - pos[None,:,:]), axis=2) == 1) & walk[:,None] & walk[None,:]
        diagonal = jnp.eye(117, dtype=jnp.bool_) & walk[:,None]
        initial = jnp.where(diagonal, 0, jnp.where(adjacency, 1, 999)).astype(jnp.int32)
        return jax.lax.fori_loop(0, 117, self.distance_relax, initial)

    def empty_events(self):
        return {
            "event_active": jnp.zeros((32,), dtype=jnp.bool_),
            "event_kind": jnp.zeros((32,), dtype=jnp.int32),
            "event_actor": jnp.full((32,), -1, dtype=jnp.int32),
            "event_from": jnp.full((32,2), -1, dtype=jnp.int32),
            "event_to": jnp.full((32,2), -1, dtype=jnp.int32),
            "event_value": jnp.zeros((32,4), dtype=jnp.int32),
            "event_tick": jnp.full((32,), -1, dtype=jnp.int32)
        }

    def refresh_views(self, state):
        out = dict(state)
        out["agent_visibility"] = jnp.broadcast_to(out["agent_active"][None,:], (4,4))
        out["puck_visibility"] = jnp.broadcast_to(out["puck_active"][None,:], (4,4))
        out["grazer_visibility"] = jnp.broadcast_to(out["grazer_active"][None,:], (4,6))
        out["agent_bank_score"] = out["bank_score"][out["agent_team"]]
        return out

    def reset(self, key):
        keys = jax.random.split(key, 2)
        permutation = jax.random.permutation(keys[0], 10)
        phases = jax.random.randint(keys[1], (6,), 0, 4, dtype=jnp.int32)
        terrain = jnp.array(self.terrain_rows, dtype=jnp.int32)
        state = {
            "terrain": terrain,
            "map_distance": self.distances(terrain),
            "agent_pos": jnp.array([[3,4],[1,4],[9,4],[11,4]], dtype=jnp.int32),
            "agent_team": jnp.array([0,0,1,1], dtype=jnp.int32),
            "agent_active": jnp.ones((4,), dtype=jnp.bool_),
            "last_actions": jnp.zeros((4,), dtype=jnp.int32),
            "whistle_active": jnp.zeros((4,), dtype=jnp.bool_),
            "puck_pos": jnp.full((4,2), -1, dtype=jnp.int32),
            "puck_active": jnp.zeros((4,), dtype=jnp.bool_),
            "puck_kind": jnp.zeros((4,), dtype=jnp.int32),
            "puck_lifetime": jnp.zeros((4,), dtype=jnp.int32),
            "puck_cooldown": jnp.zeros((4,), dtype=jnp.int32),
            "puck_owner": jnp.arange(4, dtype=jnp.int32),
            "grazer_pos": jnp.array(self.spawn_xy, dtype=jnp.int32)[permutation[:6]],
            "grazer_active": jnp.ones((6,), dtype=jnp.bool_),
            "grazer_capture_team": jnp.full((6,), -1, dtype=jnp.int32),
            "grazer_settle_progress": jnp.zeros((6,), dtype=jnp.int32),
            "grazer_wander_phase": phases,
            "grazer_heading": phases,
            "bank_score": jnp.zeros((2,), dtype=jnp.int32),
            "t": jnp.array(0, dtype=jnp.int32),
            "horizon": jnp.array(128, dtype=jnp.int32),
            "done": jnp.array(False, dtype=jnp.bool_),
            "terminated": jnp.array(False, dtype=jnp.bool_),
            "truncated": jnp.array(False, dtype=jnp.bool_),
            "winner_team": jnp.array(-1, dtype=jnp.int32),
            "last_team_reward": jnp.zeros((2,), dtype=jnp.float32),
            "episode_team_return": jnp.zeros((2,), dtype=jnp.float32)
        }
        events = self.empty_events()
        for name in events:
            state[name] = events[name]
        state = self.refresh_views(state)
        return self.observe(state), state

    def observe(self, state):
        board = jax.nn.one_hot(state["terrain"], 6, dtype=jnp.float32).reshape((702,))
        pos = state["agent_pos"].astype(jnp.float32)
        agents = jnp.concatenate((
            pos[:,1:2] / 8.0, pos[:,0:1] / 12.0,
            jax.nn.one_hot(state["agent_team"], 2, dtype=jnp.float32),
            state["whistle_active"][:,None].astype(jnp.float32),
            state["last_actions"][:,None].astype(jnp.float32) / 8.0
        ), axis=1).reshape((24,))
        ppos = jnp.where(state["puck_active"][:,None], state["puck_pos"], 0).astype(jnp.float32)
        pucks = jnp.concatenate((
            ppos[:,1:2] / 8.0, ppos[:,0:1] / 12.0,
            state["puck_active"][:,None].astype(jnp.float32),
            (state["puck_kind"] == 1)[:,None].astype(jnp.float32),
            (state["puck_kind"] == 2)[:,None].astype(jnp.float32),
            state["puck_lifetime"][:,None].astype(jnp.float32) / 24.0,
            state["puck_cooldown"][:,None].astype(jnp.float32) / 8.0,
            jax.nn.one_hot(state["agent_team"], 2, dtype=jnp.float32)
        ), axis=1).reshape((36,))
        active = state["grazer_active"]
        gpos = jnp.where(active[:,None], state["grazer_pos"], 0).astype(jnp.float32)
        grazers = jnp.concatenate((
            gpos[:,1:2] / 8.0, gpos[:,0:1] / 12.0,
            active[:,None].astype(jnp.float32),
            jax.nn.one_hot(state["grazer_capture_team"] + 1, 3, dtype=jnp.float32),
            state["grazer_settle_progress"][:,None].astype(jnp.float32) / 3.0,
            jax.nn.one_hot(state["grazer_wander_phase"], 4, dtype=jnp.float32) * active[:,None],
            jax.nn.one_hot(state["grazer_heading"], 4, dtype=jnp.float32) * active[:,None]
        ), axis=1).reshape((90,))
        first = jnp.stack((
            state["t"].astype(jnp.float32) / state["horizon"].astype(jnp.float32),
            state["bank_score"][0].astype(jnp.float32) / 6.0,
            state["bank_score"][1].astype(jnp.float32) / 6.0,
            jnp.sum(active).astype(jnp.float32) / 6.0,
            state["done"].astype(jnp.float32), state["terminated"].astype(jnp.float32),
            state["truncated"].astype(jnp.float32)
        ))
        last = ((state["t"] % 6).astype(jnp.float32) / 5.0)[None]
        global_values = jnp.concatenate((first, jax.nn.one_hot(state["winner_team"] + 1, 4, dtype=jnp.float32), last))
        common = jnp.concatenate((board, agents, pucks, grazers, global_values))
        return jnp.concatenate((jnp.broadcast_to(common[None,:], (4,864)), jnp.eye(4, dtype=jnp.float32)), axis=1)

    def legal_actions(self, state):
        pos = state["agent_pos"]
        target = pos[:,None,:] + self.directions()[None,:,:]
        x = jnp.clip(target[:,:,0], 0, 12)
        y = jnp.clip(target[:,:,1], 0, 8)
        inside = (target[:,:,0] >= 0) & (target[:,:,0] < 13) & (target[:,:,1] >= 0) & (target[:,:,1] < 9)
        tile = state["terrain"][y,x]
        team = state["agent_team"][:,None]
        bank_ok = ~(((team == 0) & ((tile == 3) | (tile == 5))) | ((team == 1) & ((tile == 2) | (tile == 4))))
        other = ~jnp.eye(4, dtype=jnp.bool_)
        occupied_agent = jnp.any(jnp.all(target[:,:,None,:] == pos[None,None,:,:], axis=3) & other[:,None,:], axis=2)
        occupied_grazer = jnp.any(jnp.all(target[:,:,None,:] == state["grazer_pos"][None,None,:,:], axis=3) & state["grazer_active"][None,None,:], axis=2)
        move = inside & (tile != 0) & bank_ok & ~occupied_agent & ~occupied_grazer
        recover = state["puck_active"] & (jnp.sum(jnp.abs(pos - state["puck_pos"]), axis=1) <= 1)
        puck_occupants = jnp.all(pos[:,None,:] == state["puck_pos"][None,:,:], axis=2) & state["puck_active"][None,:]
        unrecoverable_occupant = jnp.any(puck_occupants & ~recover[None,:], axis=1)
        deploy = ~state["puck_active"] & (state["puck_cooldown"] == 0) & ~unrecoverable_occupant
        benches = jnp.array([[1,4],[11,4]], dtype=jnp.int32)
        settle = jnp.all(pos == benches[state["agent_team"]], axis=1)
        mask = jnp.concatenate((jnp.ones((4,1), dtype=jnp.bool_), move, deploy[:,None], deploy[:,None], recover[:,None], settle[:,None]), axis=1)
        noop = jnp.broadcast_to((jnp.arange(9) == 0)[None,:], (4,9))
        return jnp.where(state["done"], noop, mask)

    def absorb(self, payload):
        key, state, actions = payload
        info = {"done_all": state["done"], "terminated": state["terminated"], "truncated": state["truncated"]}
        return self.observe(state), state, jnp.zeros((4,), dtype=jnp.float32), jnp.ones((4,), dtype=jnp.bool_), info

    def advance(self, payload):
        key, state, actions = payload
        action = jnp.where((actions >= 0) & (actions < 9), actions, 0).astype(jnp.int32)
        out = dict(state)
        out["last_actions"] = action
        old_pos = state["agent_pos"]
        target = old_pos + self.directions()[jnp.clip(action - 1, 0, 3)]
        x = jnp.clip(target[:,0], 0, 12)
        y = jnp.clip(target[:,1], 0, 8)
        inside = (target[:,0] >= 0) & (target[:,0] < 13) & (target[:,1] >= 0) & (target[:,1] < 9)
        tile = state["terrain"][y,x]
        teams = state["agent_team"]
        bank_ok = ~(((teams == 0) & ((tile == 3) | (tile == 5))) | ((teams == 1) & ((tile == 2) | (tile == 4))))
        other = ~jnp.eye(4, dtype=jnp.bool_)
        occupied_agent = jnp.any(jnp.all(target[:,None,:] == old_pos[None,:,:], axis=2) & other, axis=1)
        occupied_grazer = jnp.any(jnp.all(target[:,None,:] == state["grazer_pos"][None,:,:], axis=2) & state["grazer_active"][None,:], axis=1)
        requested_move = (action >= 1) & (action <= 4)
        valid_move = requested_move & inside & (tile != 0) & bank_ok & ~occupied_agent & ~occupied_grazer
        contest = jnp.any(jnp.all(target[:,None,:] == target[None,:,:], axis=2) & other & valid_move[None,:], axis=1)
        moved = valid_move & ~contest
        pos = jnp.where(moved[:,None], target, old_pos)
        old_active = state["puck_active"]
        recover = (action == 7) & old_active & (jnp.sum(jnp.abs(pos - state["puck_pos"]), axis=1) <= 1)
        remaining_active = old_active & ~recover
        puck_pos = jnp.where(remaining_active[:,None], state["puck_pos"], -1)
        puck_kind = jnp.where(remaining_active, state["puck_kind"], 0)
        life = jnp.where(remaining_active, state["puck_lifetime"], 0)
        cooldown = jnp.where(old_active, 0, jnp.maximum(state["puck_cooldown"] - 1, 0))
        cooldown = jnp.where(recover, 4, cooldown).astype(jnp.int32)
        occupied_puck = jnp.any(jnp.all(pos[:,None,:] == puck_pos[None,:,:], axis=2) & remaining_active[None,:], axis=1)
        place = ((action == 5) | (action == 6)) & ~old_active & (state["puck_cooldown"] == 0) & ~occupied_puck
        puck_active = remaining_active | place
        puck_pos = jnp.where(place[:,None], pos, puck_pos)
        puck_kind = jnp.where(place, action - 4, puck_kind).astype(jnp.int32)
        life = jnp.where(place, 24, life).astype(jnp.int32)
        cooldown = jnp.where(place, 0, cooldown).astype(jnp.int32)
        benches = jnp.array([[1,4],[11,4]], dtype=jnp.int32)
        whistle = (action == 8) & jnp.all(pos == benches[teams], axis=1)
        team_whistle = jnp.stack((jnp.any(whistle & (teams == 0)), jnp.any(whistle & (teams == 1))))
        old_gpos = state["grazer_pos"]
        gactive = state["grazer_active"]
        candidates = old_gpos[:,None,:] + self.candidate_directions()[None,:,:]
        cx = jnp.clip(candidates[:,:,0], 0, 12)
        cy = jnp.clip(candidates[:,:,1], 0, 8)
        cin = (candidates[:,:,0] >= 0) & (candidates[:,:,0] < 13) & (candidates[:,:,1] >= 0) & (candidates[:,:,1] < 9)
        c_agent = jnp.any(jnp.all(candidates[:,:,None,:] == pos[None,None,:,:], axis=3), axis=2)
        grazer_other = ~jnp.eye(6, dtype=jnp.bool_)
        c_grazer = jnp.any(jnp.all(candidates[:,:,None,:] == old_gpos[None,None,:,:], axis=3) & gactive[None,None,:] & grazer_other[:,None,:], axis=2)
        valid_candidate = cin & (state["terrain"][cy,cx] != 0) & ~c_agent & ~c_grazer & gactive[:,None]
        valid_candidate = valid_candidate.at[:,0].set(True)
        cindex = cy * 13 + cx
        pindex = jnp.clip(puck_pos[:,1], 0, 8) * 13 + jnp.clip(puck_pos[:,0], 0, 12)
        aindex = pos[:,1] * 13 + pos[:,0]
        gindex = jnp.clip(old_gpos[:,1], 0, 8) * 13 + jnp.clip(old_gpos[:,0], 0, 12)
        dp = state["map_distance"][cindex[:,:,None],pindex[None,None,:]].astype(jnp.float32)
        da = state["map_distance"][cindex[:,:,None],aindex[None,None,:]].astype(jnp.float32)
        dg = state["map_distance"][cindex[:,:,None],gindex[None,None,:]].astype(jnp.float32)
        signs = jnp.where(puck_kind == 1, 1.0, -1.0).astype(jnp.float32)
        scent = jnp.sum(puck_active[None,None,:] * signs[None,None,:] * 8.0 * (life.astype(jnp.float32)[None,None,:] / 24.0) / (1.0 + dp), axis=2)
        pressure = jnp.sum(((~whistle)[None,None,:] & (da <= 2.0)).astype(jnp.float32) * jnp.array(6.0, dtype=jnp.float32) / jnp.square(1.0 + da), axis=2)
        crowd = jnp.sum((gactive[None,None,:] & grazer_other[:,None,:] & (dg <= 2.0)).astype(jnp.float32) / jnp.square(1.0 + dg), axis=2)
        desired = (state["grazer_wander_phase"] + state["t"] // 4) % 4
        cactions = jnp.arange(5, dtype=jnp.int32)[None,:]
        scores = scent - pressure - crowd + jnp.array(0.12, dtype=jnp.float32) * (cactions == (desired[:,None] + 1)) + jnp.array(0.08, dtype=jnp.float32) * (cactions == 0)
        scores = jnp.where(valid_candidate, scores, -jnp.inf)
        order = jnp.stack((desired + 1, (desired + 1) % 4 + 1, (desired + 3) % 4 + 1, (desired + 2) % 4 + 1, jnp.zeros((6,), dtype=jnp.int32)), axis=1)
        ordered = jnp.take_along_axis(scores, order, axis=1)
        choice_rank = jnp.argmax(ordered, axis=1)
        choice = jnp.take_along_axis(order, choice_rank[:,None], axis=1)[:,0]
        old_tile = state["terrain"][jnp.clip(old_gpos[:,1],0,8),jnp.clip(old_gpos[:,0],0,12)]
        old_pen = (old_tile == 2) | (old_tile == 3)
        calm = gactive & old_pen & team_whistle[jnp.clip(old_tile - 2, 0, 1)]
        choice = jnp.where(calm | ~gactive, 0, choice).astype(jnp.int32)
        proposal = old_gpos + self.candidate_directions()[choice]
        moving_grazer = gactive & (choice != 0)
        rank = (jnp.arange(6, dtype=jnp.int32) - state["t"] % 6 + 6) % 6
        same_target = jnp.all(proposal[:,None,:] == proposal[None,:,:], axis=2)
        defeated = jnp.any(same_target & moving_grazer[None,:] & (rank[None,:] < rank[:,None]), axis=1)
        grazer_moved = moving_grazer & ~defeated
        arrived = jnp.where(grazer_moved[:,None], proposal, old_gpos)
        heading = jnp.where(grazer_moved, choice - 1, state["grazer_heading"]).astype(jnp.int32)
        arrival_tile = state["terrain"][jnp.clip(arrived[:,1],0,8),jnp.clip(arrived[:,0],0,12)]
        arrival_team = jnp.clip(arrival_tile - 2, 0, 1)
        in_pen = (arrival_tile == 2) | (arrival_tile == 3)
        settles = gactive & in_pen & team_whistle[arrival_team]
        previous_progress = state["grazer_settle_progress"]
        continuation = (state["grazer_capture_team"] == arrival_team) & (previous_progress > 0)
        pending_progress = jnp.where(settles, jnp.where(continuation, previous_progress + 1, 1), 0).astype(jnp.int32)
        progress = jnp.where(gactive, pending_progress, previous_progress).astype(jnp.int32)
        capture_team = jnp.where(gactive, jnp.where(settles, arrival_team, -1), state["grazer_capture_team"]).astype(jnp.int32)
        captured = gactive & (progress == 3)
        capture_counts = jnp.stack((jnp.sum(captured & (capture_team == 0), dtype=jnp.int32), jnp.sum(captured & (capture_team == 1), dtype=jnp.int32)))
        grazer_active = gactive & ~captured
        grazer_pos = jnp.where(captured[:,None], -1, arrived).astype(jnp.int32)
        bank = state["bank_score"] + capture_counts
        expired = puck_active & (life == 1)
        final_puck_active = puck_active & ~expired
        final_life = jnp.where(final_puck_active, life - 1, 0).astype(jnp.int32)
        final_cooldown = jnp.where(expired, 8, cooldown).astype(jnp.int32)
        tick = state["t"] + jnp.array(1, dtype=jnp.int32)
        terminated = ~jnp.any(grazer_active)
        truncated = (tick >= self.max_steps) & ~terminated
        done = terminated | truncated
        winner = jnp.where(terminated, jnp.where(bank[0] > bank[1], 0, jnp.where(bank[1] > bank[0], 1, 2)), -1).astype(jnp.int32)
        difference = (capture_counts[0] - capture_counts[1]).astype(jnp.float32)
        team_reward = jnp.stack((difference, -difference))
        out["agent_pos"] = pos
        out["whistle_active"] = whistle
        out["puck_pos"] = jnp.where(final_puck_active[:,None], puck_pos, -1).astype(jnp.int32)
        out["puck_active"] = final_puck_active
        out["puck_kind"] = jnp.where(final_puck_active, puck_kind, 0).astype(jnp.int32)
        out["puck_lifetime"] = final_life
        out["puck_cooldown"] = final_cooldown
        out["grazer_pos"] = grazer_pos
        out["grazer_active"] = grazer_active
        out["grazer_capture_team"] = capture_team
        out["grazer_settle_progress"] = progress
        out["grazer_heading"] = heading
        out["bank_score"] = bank
        out["t"] = tick
        out["done"] = done
        out["terminated"] = terminated
        out["truncated"] = truncated
        out["winner_team"] = winner
        out["last_team_reward"] = team_reward
        out["episode_team_return"] = state["episode_team_return"] + team_reward
        events = self.empty_events()
        ea = events["event_active"].at[:4].set(requested_move)
        ek = events["event_kind"].at[:4].set(jnp.where(moved, 1, 2))
        er = events["event_actor"].at[:4].set(jnp.arange(4, dtype=jnp.int32))
        ef = events["event_from"].at[:4].set(old_pos)
        et = events["event_to"].at[:4].set(pos)
        ev = events["event_value"].at[:4,0].set(jnp.clip(action - 1, 0, 3))
        bad_tool = (action >= 5) & ~(place | recover | whistle)
        tool_event = place | recover | bad_tool
        ea = ea.at[4:8].set(tool_event)
        ek = ek.at[4:8].set(jnp.where(place, jnp.where(action == 5, 3, 4), jnp.where(recover, 5, 6)))
        er = er.at[4:8].set(jnp.arange(4, dtype=jnp.int32))
        ef = ef.at[4:8].set(jnp.where(recover[:,None], state["puck_pos"], pos))
        et = et.at[4:8].set(pos)
        tv0 = jnp.where(place, action - 4, jnp.where(recover, state["puck_kind"], 0))
        tv1 = jnp.where(place, 24, jnp.where(recover, state["puck_lifetime"], 0))
        ev = ev.at[4:8].set(jnp.stack((tv0, tv1, jnp.where(recover, 4, 0), bad_tool.astype(jnp.int32)), axis=1))
        ea = ea.at[8:12].set(whistle != state["whistle_active"])
        ek = ek.at[8:12].set(7)
        er = er.at[8:12].set(jnp.arange(4, dtype=jnp.int32))
        ef = ef.at[8:12].set(pos)
        et = et.at[8:12].set(pos)
        ev = ev.at[8:12].set(jnp.stack((state["whistle_active"].astype(jnp.int32), whistle.astype(jnp.int32), teams, jnp.zeros((4,), dtype=jnp.int32)), axis=1))
        ea = ea.at[12:18].set(grazer_moved)
        ek = ek.at[12:18].set(8)
        er = er.at[12:18].set(jnp.arange(6, dtype=jnp.int32))
        ef = ef.at[12:18].set(old_gpos)
        et = et.at[12:18].set(arrived)
        ev = ev.at[12:18,0].set(state["grazer_heading"])
        ev = ev.at[12:18,1].set(heading)
        settle_event = gactive & ((progress != previous_progress) | (capture_team != state["grazer_capture_team"]))
        ea = ea.at[18:24].set(settle_event)
        ek = ek.at[18:24].set(jnp.where(captured, 10, 9))
        er = er.at[18:24].set(jnp.arange(6, dtype=jnp.int32))
        ef = ef.at[18:24].set(old_gpos)
        et = et.at[18:24].set(arrived)
        ev = ev.at[18:24].set(jnp.stack((previous_progress, progress, capture_team, captured.astype(jnp.int32)), axis=1))
        ea = ea.at[24:28].set(expired)
        ek = ek.at[24:28].set(11)
        er = er.at[24:28].set(jnp.arange(4, dtype=jnp.int32))
        ef = ef.at[24:28].set(puck_pos)
        et = et.at[24:28].set(puck_pos)
        ev = ev.at[24:28].set(jnp.stack((puck_kind, jnp.ones((4,), dtype=jnp.int32), jnp.full((4,), 8, dtype=jnp.int32), jnp.zeros((4,), dtype=jnp.int32)), axis=1))
        ea = ea.at[28].set(done)
        ek = ek.at[28].set(12)
        ev = ev.at[28].set(jnp.array([winner,terminated.astype(jnp.int32),truncated.astype(jnp.int32),jnp.sum(grazer_active,dtype=jnp.int32)], dtype=jnp.int32))
        out["event_active"] = ea
        out["event_kind"] = jnp.where(ea, ek, 0)
        out["event_actor"] = jnp.where(ea, er, -1)
        out["event_from"] = jnp.where(ea[:,None], ef, -1)
        out["event_to"] = jnp.where(ea[:,None], et, -1)
        out["event_value"] = jnp.where(ea[:,None], ev, 0)
        out["event_tick"] = jnp.where(ea, tick, -1).astype(jnp.int32)
        out = self.refresh_views(out)
        info = {"done_all": done, "terminated": terminated, "truncated": truncated}
        return self.observe(out), out, team_reward[teams], jnp.full((4,), done, dtype=jnp.bool_), info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, actions))
