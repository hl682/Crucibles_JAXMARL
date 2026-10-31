"""Lantern Flag Rescue. Original Astra rules; authored, never executed by author."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 131
    num_actions = 8
    max_steps = 72

    def reset(self, key):
        state = {
            "t": jnp.array(0, jnp.int32),
            "episode_done": jnp.array(False, jnp.bool_),
            "terminated": jnp.array(False, jnp.bool_),
            "truncated": jnp.array(False, jnp.bool_),
            "terrain": jnp.array([[0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0]], jnp.int32),
            "agent_pos": jnp.array([[1,2],[1,4],[9,2],[8,4]], jnp.int32),
            "agent_active": jnp.ones((4,), jnp.bool_),
            "agent_visible": jnp.ones((4,4), jnp.bool_),
            "team": jnp.array([0,0,1,1], jnp.int32),
            "capture_timer": jnp.zeros((4,), jnp.int32),
            "capture_active": jnp.zeros((4,), jnp.bool_),
            "carrying": jnp.zeros((4,), jnp.bool_),
            "guard": jnp.zeros((4,), jnp.bool_),
            "tagged": jnp.zeros((4,), jnp.bool_),
            "released": jnp.zeros((4,), jnp.bool_),
            "rescued": jnp.zeros((4,), jnp.bool_),
            "flag_pos": jnp.array([[1,3],[9,3]], jnp.int32),
            "flag_active": jnp.ones((2,), jnp.bool_),
            "flag_visible": jnp.ones((4,2), jnp.bool_),
            "flag_carrier": jnp.full((2,), -1, jnp.int32),
            "flag_team": jnp.array([0,1], jnp.int32),
            "base_pos": jnp.array([[1,3],[9,3]], jnp.int32),
            "base_active": jnp.ones((2,), jnp.bool_),
            "base_visible": jnp.ones((4,2), jnp.bool_),
            "score": jnp.zeros((2,), jnp.int32),
        }
        return self.observe(state), state

    def legal_actions(self, state):
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]], jnp.int32)
        target = state["agent_pos"][:,None,:] + delta[None,:,:]
        inside = jnp.all((target >= jnp.array([0,0])) & (target <= jnp.array([10,6])), axis=2)
        safe = jnp.clip(target, jnp.array([0,0]), jnp.array([10,6]))
        floor = state["terrain"][safe[:,:,1], safe[:,:,0]] == 0
        free = (state["capture_timer"] == 0) & ~state["episode_done"]
        mask = jnp.concatenate((inside & floor, jnp.ones((4,3), jnp.bool_)), axis=1)
        return (mask & free[:,None]).at[:,0].set(True)

    def observe(self, state):
        agents = jnp.concatenate((state["agent_pos"].astype(jnp.float32) / jnp.array([10.0,6.0]),
                                  state["capture_timer"][:,None] / 4.0,
                                  state["team"][:,None].astype(jnp.float32),
                                  state["carrying"][:,None].astype(jnp.float32),
                                  state["guard"][:,None].astype(jnp.float32)), axis=1).reshape(-1)
        flags = jnp.concatenate((state["flag_pos"].astype(jnp.float32) / jnp.array([10.0,6.0]),
                                 (state["flag_carrier"][:,None] + 1) / 4.0), axis=1).reshape(-1)
        common = jnp.concatenate((agents, flags,
                                  (state["base_pos"].astype(jnp.float32) / jnp.array([10.0,6.0])).reshape(-1),
                                  state["score"].astype(jnp.float32)))
        tail = jnp.concatenate((jnp.array([state["t"] / 72.0, state["episode_done"]], jnp.float32),
                                state["terrain"].reshape(-1).astype(jnp.float32),
                                state["tagged"].astype(jnp.float32),
                                state["released"].astype(jnp.float32),
                                state["rescued"].astype(jnp.float32)))
        return jnp.concatenate((jnp.broadcast_to(common, (4,36)), jnp.eye(4, dtype=jnp.float32),
                                jnp.broadcast_to(tail, (4,91))), axis=1).astype(jnp.float32)

    def advance(self, state, actions):
        mask = self.legal_actions(state)
        clipped = jnp.clip(actions, 0, 7)
        actions = jnp.where((actions >= 0) & (actions < 8) & mask[jnp.arange(4),clipped], clipped, 0)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0],[0,0],[0,0],[0,0]], jnp.int32)
        old_pos = state["agent_pos"]
        proposed = old_pos + delta[actions]
        others = ~jnp.eye(4, dtype=jnp.bool_)
        occupied = jnp.any(jnp.all(proposed[:,None,:] == old_pos[None,:,:], axis=2) & others, axis=1)
        contested = jnp.any(jnp.all(proposed[:,None,:] == proposed[None,:,:], axis=2) & others, axis=1)
        pos = jnp.where((occupied | contested)[:,None], old_pos, proposed)
        distance = jnp.sum(jnp.abs(pos[:,None,:] - pos[None,:,:]), axis=2)
        ally = state["team"][:,None] == state["team"][None,:]
        was_free = state["capture_timer"] == 0
        guard = (actions == 7) & was_free
        protected = jnp.any((distance <= 1) & ally & others & guard[None,:], axis=1)
        attack = (actions == 5) & was_free
        tagged = jnp.any((distance <= 1) & ~ally & attack[:,None], axis=0) & was_free & ~protected
        rescuer = (actions == 6) & was_free
        rescued = jnp.any((distance <= 1) & ally & others & rescuer[:,None], axis=0) & ~was_free
        timer = jnp.maximum(state["capture_timer"] - 1, 0)
        timer = jnp.where(rescued, 0, timer)
        timer = jnp.where(tagged, 4, timer).astype(jnp.int32)
        released = (state["capture_timer"] > 0) & (timer == 0)
        carrier = state["flag_carrier"]
        flag_pos = state["flag_pos"]
        for f in range(2):
            old_owner = carrier[f]
            owner_index = jnp.maximum(old_owner, 0)
            moved_flag = jnp.where(old_owner >= 0, pos[owner_index], flag_pos[f])
            dropped = (old_owner >= 0) & (timer[owner_index] > 0)
            owner = jnp.where(dropped, -1, old_owner)
            touching = jnp.all(pos == moved_flag[None,:], axis=1) & (timer == 0)
            returned = (owner < 0) & jnp.any(touching & (state["team"] == f))
            moved_flag = jnp.where(returned, state["base_pos"][f], moved_flag)
            touching = jnp.all(pos == moved_flag[None,:], axis=1) & (timer == 0)
            can_pick = touching & (state["team"] != f)
            chosen = jnp.min(jnp.where(can_pick, jnp.arange(4), 4))
            owner = jnp.where((owner < 0) & (chosen < 4), chosen, owner)
            carrier = carrier.at[f].set(owner)
            flag_pos = flag_pos.at[f].set(moved_flag)
        home = (carrier < 0) & jnp.all(flag_pos == state["base_pos"], axis=1)
        scored = jnp.zeros((2,), jnp.int32)
        for f in range(2):
            returning_team = 1 - f
            owner = jnp.maximum(carrier[f], 0)
            at_base = jnp.all(pos[owner] == state["base_pos"][returning_team])
            scored = scored.at[returning_team].set(((carrier[f] >= 0) & at_base & home[returning_team]).astype(jnp.int32))
        ns = dict(state)
        ns["t"] = state["t"] + jnp.array(1, jnp.int32)
        ns["agent_pos"] = pos
        ns["capture_timer"] = timer
        ns["capture_active"] = timer > 0
        ns["guard"] = guard
        ns["tagged"] = tagged
        ns["released"] = released
        ns["rescued"] = rescued
        ns["flag_carrier"] = carrier
        ns["flag_pos"] = flag_pos
        ns["carrying"] = jnp.any(carrier[:,None] == jnp.arange(4)[None,:], axis=0)
        ns["score"] = state["score"] + scored
        ns["terminated"] = jnp.any(scored > 0)
        ns["truncated"] = (ns["t"] >= 72) & ~ns["terminated"]
        ns["episode_done"] = ns["terminated"] | ns["truncated"]
        team_reward = (scored - scored[::-1]).astype(jnp.float32)
        rewards = team_reward[state["team"]] - jnp.array(0.001, jnp.float32)
        return ns, rewards

    def raw_step(self, key, state, actions):
        proposed, reward = self.advance(state, actions)
        ns = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, proposed)
        reward = jnp.where(state["episode_done"], jnp.zeros((4,), jnp.float32), reward)
        dones = jnp.full((4,), ns["episode_done"], jnp.bool_)
        info = {"done_all": ns["episode_done"], "terminated": ns["terminated"], "truncated": ns["truncated"]}
        return self.observe(ns), ns, reward.astype(jnp.float32), dones, info
