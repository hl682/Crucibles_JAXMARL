"""Lane Relay: original four-player, two-team gridball; authored, unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 116
    num_actions = 9
    max_steps = 160

    def reset(self, key):
        board = jnp.array([
            [1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1],
            [2,0,0,0,0,0,0,0,0,0,3],
            [1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1]],jnp.int32)
        s = {
            "t": jnp.array(0,jnp.int32), "episode_done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "map": board, "agent_pos": jnp.array([[3,3],[3,2],[7,3],[7,4]],jnp.int32),
            "agent_active": jnp.ones(4,jnp.bool_), "agent_visible": jnp.ones((4,4),jnp.bool_),
            "team": jnp.array([0,0,1,1],jnp.int32), "stamina": jnp.full(4,5,jnp.int32),
            "guard": jnp.zeros(4,jnp.bool_),
            "ball_pos": jnp.array([[3,3]],jnp.int32), "ball_active": jnp.ones(1,jnp.bool_),
            "ball_visible": jnp.ones((4,1),jnp.bool_), "ball_carrier": jnp.array([0],jnp.int32),
            "ball_velocity": jnp.zeros((1,2),jnp.int32), "flight": jnp.zeros(1,jnp.int32),
            "touch_team": jnp.zeros(1,jnp.int32),
            "goal_pos": jnp.array([[10,3],[0,3]],jnp.int32), "goal_active": jnp.ones(2,jnp.bool_),
            "goal_visible": jnp.ones((4,2),jnp.bool_), "score": jnp.zeros(2,jnp.int32),
            "goal_team": jnp.array([0,1],jnp.int32),
            "restart_pending": jnp.array(False), "restart_team": jnp.array(0,jnp.int32),
            "pass_event": jnp.array(False), "catch_event": jnp.array(False),
            "intercept_event": jnp.array(False), "tackle_event": jnp.array(False),
            "goal_event": jnp.zeros(2,jnp.bool_)}
        return self.observe(s),s

    def legal_actions(self, s):
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        target = jnp.clip(s["agent_pos"][:,None,:] + delta[None,:,:],jnp.array([0,0],jnp.int32),jnp.array([10,6],jnp.int32))
        walk = (s["map"][target[:,:,1],target[:,:,0]] == 0) & (s["stamina"] >= 1)[:,None]
        mask = jnp.zeros((4,9),jnp.bool_).at[:,:5].set(walk).at[:,0].set(True)
        owns = s["ball_carrier"][0] == jnp.arange(4)
        mask = mask.at[:,5].set(owns & (s["stamina"] >= 1))
        mask = mask.at[:,6].set(owns & (s["stamina"] >= 2))
        mask = mask.at[:,7].set(s["stamina"] >= 1)
        mask = mask.at[:,8].set(~owns & (s["stamina"] >= 2))
        return mask & (~(s["episode_done"] | s["restart_pending"]) | (jnp.arange(9)[None,:] == 0))

    def move(self, s, actions):
        ids = jnp.arange(4,dtype=jnp.int32)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32)
        proposed = s["agent_pos"] + delta[jnp.clip(actions,0,4)] * ((actions >= 1) & (actions <= 4))[:,None]
        proposed = jnp.clip(proposed,jnp.array([0,0],jnp.int32),jnp.array([10,6],jnp.int32))
        occupied = jnp.any(jnp.all(proposed[:,None,:] == s["agent_pos"][None,:,:],axis=2) & (ids[:,None] != ids[None,:]),axis=1)
        possible = (actions >= 1) & (actions <= 4) & (s["map"][proposed[:,1],proposed[:,0]] == 0) & ~occupied
        rank = jnp.mod(ids-s["t"],4)
        same = jnp.all(proposed[:,None,:] == proposed[None,:,:],axis=2)
        loses = jnp.any(same & possible[None,:] & (rank[None,:] < rank[:,None]),axis=1)
        return jnp.where((possible & ~loses)[:,None],proposed,s["agent_pos"])

    def raw_step(self, key, state, actions):
        mask = self.legal_actions(state)
        safe = jnp.clip(actions,0,8)
        actions = jnp.where((actions >= 0) & (actions < 9) & mask[jnp.arange(4),safe],safe,0)
        ids = jnp.arange(4,dtype=jnp.int32)
        rank = jnp.mod(ids-state["t"],4)
        s = dict(state)
        s["agent_pos"] = self.move(state,actions)
        cost = jnp.where(actions == 0,0,jnp.where((actions == 6) | (actions == 8),2,1)).astype(jnp.int32)
        s["stamina"] = jnp.clip(state["stamina"] - cost + 2 * (actions == 0).astype(jnp.int32),0,5)
        s["guard"] = actions == 7
        carrier = state["ball_carrier"][0]
        holder = jnp.clip(carrier,0,3)
        # Contact steals precede launches; guarding protects the carrier for this tick.
        tacklers = (actions == 8) & (carrier >= 0) & (s["team"] != s["team"][holder]) & (jnp.sum(jnp.abs(s["agent_pos"]-s["agent_pos"][holder]),axis=1) == 1) & ~s["guard"][holder]
        thief = jnp.argmin(jnp.where(tacklers,rank,99))
        stolen = jnp.any(tacklers)
        carrier = jnp.where(stolen,thief,carrier).astype(jnp.int32)
        s["tackle_event"] = stolen
        holder = jnp.clip(carrier,0,3)
        team = s["team"][holder]
        mate = jnp.array([1,0,3,2],jnp.int32)[holder]
        separation = s["agent_pos"][mate] - s["agent_pos"][holder]
        distance = jnp.sum(jnp.abs(separation))
        aligned = (separation[0] == 0) | (separation[1] == 0)
        can_launch = (carrier >= 0) & ~stolen
        passing = can_launch & (actions[holder] == 5) & aligned & (distance >= 1) & (distance <= 5)
        shooting = can_launch & (actions[holder] == 6) & (s["agent_pos"][holder,1] == 3)
        launch = passing | shooting
        pass_velocity = jnp.sign(separation).astype(jnp.int32)
        shot_velocity = jnp.array([1-2*team,0],jnp.int32)
        velocity = jnp.where(launch,jnp.where(passing,pass_velocity,shot_velocity),state["ball_velocity"][0])
        shot_length = jnp.where(team == 0,10-s["agent_pos"][holder,0],s["agent_pos"][holder,0])
        flight = jnp.where(launch,jnp.where(passing,distance,shot_length),state["flight"][0]).astype(jnp.int32)
        origin = jnp.where(carrier >= 0,s["agent_pos"][holder],state["ball_pos"][0])
        touch = jnp.where((carrier >= 0) | launch,team,state["touch_team"][0]).astype(jnp.int32)
        free = (carrier < 0) | launch
        travelling = free & (flight > 0)
        proposed = jnp.clip(origin + velocity * travelling.astype(jnp.int32),jnp.array([0,0],jnp.int32),jnp.array([10,6],jnp.int32))
        blocked = travelling & (s["map"][proposed[1],proposed[0]] == 1)
        point = jnp.where(blocked,origin,proposed)
        # Every tick of travel is interceptable. Bracing extends reach by one cell.
        ball_distance = jnp.sum(jnp.abs(s["agent_pos"]-point),axis=1)
        catchers = free & ((ball_distance == 0) | (s["guard"] & (ball_distance <= 1)))
        catch_rank = 8 * (ball_distance != 0).astype(jnp.int32) + rank
        catcher = jnp.argmin(jnp.where(catchers,catch_rank,99))
        caught = jnp.any(catchers)
        goals = jnp.all(s["goal_pos"] == point,axis=1) & travelling & ~blocked & ~caught
        goal = jnp.any(goals)
        carrier = jnp.where(caught,catcher,jnp.where(free,-1,carrier)).astype(jnp.int32)
        carrier = jnp.where(goal,-1,carrier).astype(jnp.int32)
        holder = jnp.clip(carrier,0,3)
        s["ball_carrier"] = jnp.full(1,carrier,jnp.int32)
        s["ball_pos"] = jnp.where((carrier >= 0),s["agent_pos"][holder],point)[None,:]
        remaining = jnp.where(travelling,jnp.maximum(flight-1,0),flight)
        remaining = jnp.where(caught | blocked | goal | ~free,0,remaining).astype(jnp.int32)
        s["flight"] = jnp.full(1,remaining,jnp.int32)
        s["ball_velocity"] = jnp.where(remaining > 0,velocity,jnp.zeros(2,jnp.int32))[None,:]
        s["touch_team"] = jnp.full(1,jnp.where(caught,s["team"][catcher],touch),jnp.int32)
        s["pass_event"] = passing
        s["catch_event"] = caught
        s["intercept_event"] = caught & (s["team"][catcher] != touch)
        s["goal_event"] = goals
        s["score"] = state["score"] + goals.astype(jnp.int32)
        s["restart_team"] = jnp.where(goal,1-jnp.argmax(goals),state["restart_team"]).astype(jnp.int32)
        s["restart_pending"] = goal & ~jnp.any(s["score"] >= 3)
        # Scoring frame is retained. A separate forced-wait tick performs an in-match kickoff.
        r = dict(state)
        r["agent_pos"] = jnp.array([[3,3],[3,2],[7,3],[7,4]],jnp.int32)
        r["stamina"] = jnp.full(4,5,jnp.int32)
        r["guard"] = jnp.zeros(4,jnp.bool_)
        kickoff = 2 * state["restart_team"]
        r["ball_pos"] = r["agent_pos"][kickoff][None,:]
        r["ball_carrier"] = jnp.full(1,kickoff,jnp.int32)
        r["touch_team"] = jnp.full(1,state["restart_team"],jnp.int32)
        r["ball_velocity"] = jnp.zeros((1,2),jnp.int32)
        r["flight"] = jnp.zeros(1,jnp.int32)
        r["restart_pending"] = jnp.array(False)
        r["pass_event"] = jnp.array(False)
        r["catch_event"] = jnp.array(False)
        r["intercept_event"] = jnp.array(False)
        r["tackle_event"] = jnp.array(False)
        r["goal_event"] = jnp.zeros(2,jnp.bool_)
        s = jax.tree.map(lambda restart,live: jnp.where(state["restart_pending"],restart,live),r,s)
        s["t"] = state["t"] + jnp.array(1,jnp.int32)
        s["terminated"] = jnp.any(s["score"] >= 3)
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["episode_done"] = s["terminated"] | s["truncated"]
        reward = (s["goal_event"][s["team"]].astype(jnp.float32) - s["goal_event"][1-s["team"]].astype(jnp.float32))
        s = jax.tree.map(lambda old,new: jnp.where(state["episode_done"],old,new),state,s)
        reward = jnp.where(state["episode_done"],jnp.zeros(4,jnp.float32),reward)
        return self.observe(s),s,reward,jnp.full(4,s["episode_done"],jnp.bool_),{"done_all":s["episode_done"],"terminated":s["terminated"],"truncated":s["truncated"]}

    def observe(self, s):
        public = jnp.array([s["t"],s["score"][0],s["score"][1],s["restart_pending"],s["goal_event"][0],s["goal_event"][1],s["pass_event"],s["catch_event"],s["intercept_event"],s["tackle_event"],s["episode_done"]],jnp.float32)
        agents = jnp.concatenate([s["agent_pos"],s["stamina"][:,None],s["team"][:,None],s["guard"][:,None]],axis=1).reshape(20)
        ball = jnp.concatenate([s["ball_pos"].reshape(2),s["ball_carrier"],s["ball_velocity"].reshape(2),s["flight"],s["touch_team"]])
        data = jnp.concatenate([public,agents,ball,s["map"].reshape(77)]).astype(jnp.float32)
        return jnp.concatenate([jnp.arange(4,dtype=jnp.float32)[:,None],jnp.broadcast_to(data[None,:],(4,115))],axis=1)
