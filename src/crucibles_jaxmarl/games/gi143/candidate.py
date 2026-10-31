"""Crosswake Ice Doubles. Original discrete momentum game, author-unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 125
    num_actions = 10
    max_steps = 72

    def reset(self, key):
        terrain = jnp.zeros((7,11), jnp.int32)
        terrain = terrain.at[2:5,0].set(1).at[2:5,10].set(2)
        state = {
            "t": jnp.array(0, jnp.int32),
            "episode_done": jnp.array(False, jnp.bool_),
            "terminated": jnp.array(False, jnp.bool_),
            "truncated": jnp.array(False, jnp.bool_),
            "terrain": terrain,
            "agent_pos": jnp.array([[2,2],[4,4],[8,4],[6,2]], jnp.int32),
            "agent_active": jnp.ones((4,), jnp.bool_),
            "agent_visible": jnp.ones((4,4), jnp.bool_),
            "team": jnp.array([0,0,1,1], jnp.int32),
            "energy": jnp.full((4,), 4, jnp.int32),
            "hit": jnp.zeros((4,), jnp.bool_),
            "stone_pos": jnp.array([[3,3],[7,3]], jnp.int32),
            "stone_active": jnp.ones((2,), jnp.bool_),
            "stone_visible": jnp.ones((4,2), jnp.bool_),
            "velocity": jnp.zeros((2,2), jnp.int32),
            "momentum": jnp.zeros((2,), jnp.int32),
            "pushed": jnp.zeros((2,), jnp.bool_),
            "braked": jnp.zeros((2,), jnp.bool_),
            "collided": jnp.zeros((2,), jnp.bool_),
            "scored_stone": jnp.zeros((2,), jnp.bool_),
            "goal_pos": jnp.array([[0,3],[10,3]], jnp.int32),
            "goal_active": jnp.ones((2,), jnp.bool_),
            "goal_visible": jnp.ones((4,2), jnp.bool_),
            "goal_team": jnp.array([1,0], jnp.int32),
            "score": jnp.zeros((2,), jnp.int32),
        }
        return self.observe(state), state

    def legal_actions(self, state):
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]], jnp.int32)
        target = state["agent_pos"][:,None,:] + delta[None,:,:]
        inside = jnp.all((target >= jnp.array([0,0])) & (target <= jnp.array([10,6])), axis=2)
        distance = jnp.sum(jnp.abs(state["agent_pos"][:,None,:] - state["stone_pos"][None,:,:]), axis=2)
        adjacent = jnp.any((distance == 1) & state["stone_active"][None,:], axis=1)
        push = adjacent & (state["energy"] >= 2)
        brake = adjacent & (state["energy"] >= 1)
        mask = jnp.concatenate((inside, jnp.broadcast_to(push[:,None], (4,4)), brake[:,None]), axis=1)
        return (mask & ~state["episode_done"]).at[:,0].set(True)

    def observe(self, state):
        agents = jnp.concatenate((state["agent_pos"].astype(jnp.float32) / jnp.array([10.0,6.0]),
                                  state["energy"][:,None] / 4.0,
                                  state["team"][:,None].astype(jnp.float32),
                                  state["hit"][:,None].astype(jnp.float32)), axis=1).reshape(-1)
        stones = jnp.concatenate((state["stone_pos"].astype(jnp.float32) / jnp.array([10.0,6.0]),
                                  state["velocity"].astype(jnp.float32), state["momentum"][:,None] / 3.0,
                                  state["stone_active"][:,None].astype(jnp.float32)), axis=1).reshape(-1)
        common = jnp.concatenate((agents, stones, state["score"].astype(jnp.float32) / 2.0))
        tail = jnp.concatenate((jnp.array([state["t"] / 72.0, state["episode_done"]], jnp.float32),
                                state["terrain"].reshape(-1).astype(jnp.float32) / 2.0,
                                state["pushed"].astype(jnp.float32), state["braked"].astype(jnp.float32),
                                state["scored_stone"].astype(jnp.float32), state["collided"].astype(jnp.float32)))
        return jnp.concatenate((jnp.broadcast_to(common, (4,34)), jnp.eye(4, dtype=jnp.float32),
                                jnp.broadcast_to(tail, (4,87))), axis=1).astype(jnp.float32)

    def advance(self, state, actions):
        clipped = jnp.clip(actions, 0, 9)
        mask = self.legal_actions(state)
        actions = jnp.where((actions >= 0) & (actions < 10) & mask[jnp.arange(4),clipped], clipped, 0)
        delta = jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0],[0,0],[0,0],[0,0],[0,0],[0,0]], jnp.int32)
        proposed = state["agent_pos"] + delta[actions]
        other_agent = ~jnp.eye(4, dtype=jnp.bool_)
        occupied = jnp.any(jnp.all(proposed[:,None,:] == state["agent_pos"][None,:,:], axis=2) & other_agent, axis=1)
        contested = jnp.any(jnp.all(proposed[:,None,:] == proposed[None,:,:], axis=2) & other_agent, axis=1)
        stone_block = jnp.any(jnp.all(proposed[:,None,:] == state["stone_pos"][None,:,:], axis=2) & state["stone_active"][None,:], axis=1)
        pos = jnp.where((occupied | contested | stone_block)[:,None], state["agent_pos"], proposed)
        distance = jnp.sum(jnp.abs(pos[:,None,:] - state["stone_pos"][None,:,:]), axis=2)
        nearby = (distance == 1) & state["stone_active"][None,:]
        target = jnp.min(jnp.where(nearby, jnp.arange(2)[None,:], 2), axis=1)
        pushing = (actions >= 5) & (actions <= 8) & (target < 2)
        braking = (actions == 9) & (target < 2)
        push_dirs = jnp.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,-1],[1,0],[0,1],[-1,0],[0,0]], jnp.int32)
        targeting = target[:,None] == jnp.arange(2)[None,:]
        pushed = jnp.any(targeting & pushing[:,None], axis=0)
        braked = jnp.any(targeting & braking[:,None], axis=0)
        force = jnp.sum(push_dirs[actions][:,None,:] * (targeting & pushing[:,None])[:,:,None], axis=0)
        horizontal = jnp.abs(force[:,0]) > jnp.abs(force[:,1])
        vertical = jnp.abs(force[:,1]) > jnp.abs(force[:,0])
        direction = jnp.stack((jnp.where(horizontal, jnp.sign(force[:,0]), 0),
                               jnp.where(vertical, jnp.sign(force[:,1]), 0)), axis=1).astype(jnp.int32)
        velocity = jnp.where(pushed[:,None], direction, state["velocity"])
        momentum = jnp.where(pushed, jnp.where(jnp.any(direction != 0, axis=1), 3, 0), state["momentum"])
        momentum = jnp.where(braked, 0, momentum)
        moving = (momentum > 0) & state["stone_active"]
        endpoint = state["stone_pos"] + velocity * moving[:,None]
        outside = jnp.any((endpoint < jnp.array([0,0])) | (endpoint > jnp.array([10,6])), axis=1)
        contact = jnp.all(endpoint[:,None,:] == pos[None,:,:], axis=2) & moving[:,None]
        player_block = jnp.any(contact, axis=1)
        other_stone = ~jnp.eye(2, dtype=jnp.bool_)
        old_stone_block = jnp.any(jnp.all(endpoint[:,None,:] == state["stone_pos"][None,:,:], axis=2)
                                  & other_stone & state["stone_active"][None,:], axis=1)
        same_endpoint = jnp.any(jnp.all(endpoint[:,None,:] == endpoint[None,:,:], axis=2)
                                & other_stone & state["stone_active"][None,:], axis=1)
        collided = moving & (outside | player_block | old_stone_block | same_endpoint)
        stone_pos = jnp.where((moving & ~collided)[:,None], endpoint, state["stone_pos"])
        momentum = jnp.where(collided, 0, jnp.maximum(momentum - 1, 0)).astype(jnp.int32)
        cell = state["terrain"][stone_pos[:,1],stone_pos[:,0]]
        scored_stone = state["stone_active"] & (cell > 0)
        active = state["stone_active"] & ~scored_stone
        momentum = jnp.where(active, momentum, 0)
        velocity = jnp.where((momentum > 0)[:,None], velocity, 0).astype(jnp.int32)
        scored = jnp.stack((jnp.sum((scored_stone & (cell == 2)).astype(jnp.int32)),
                             jnp.sum((scored_stone & (cell == 1)).astype(jnp.int32)))).astype(jnp.int32)
        hit = jnp.any(contact, axis=0)
        spent = 2 * pushing.astype(jnp.int32) + braking.astype(jnp.int32)
        energy = jnp.clip(state["energy"] + (actions == 0).astype(jnp.int32) - spent - hit.astype(jnp.int32), 0, 4)
        ns = dict(state)
        ns["t"] = state["t"] + jnp.array(1, jnp.int32)
        ns["agent_pos"] = pos
        ns["energy"] = energy.astype(jnp.int32)
        ns["hit"] = hit
        ns["stone_pos"] = stone_pos
        ns["stone_active"] = active
        ns["velocity"] = velocity
        ns["momentum"] = momentum
        ns["pushed"] = pushed
        ns["braked"] = braked
        ns["collided"] = collided
        ns["scored_stone"] = scored_stone
        ns["score"] = state["score"] + scored
        ns["terminated"] = ~jnp.any(active)
        ns["truncated"] = (ns["t"] >= 72) & ~ns["terminated"]
        ns["episode_done"] = ns["terminated"] | ns["truncated"]
        reward = (scored - scored[::-1])[state["team"]].astype(jnp.float32) - spent.astype(jnp.float32) * 0.01
        return ns, reward

    def raw_step(self, key, state, actions):
        proposed, reward = self.advance(state, actions)
        ns = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, proposed)
        reward = jnp.where(state["episode_done"], jnp.zeros((4,), jnp.float32), reward)
        dones = jnp.full((4,), ns["episode_done"], jnp.bool_)
        info = {"done_all": ns["episode_done"], "terminated": ns["terminated"], "truncated": ns["truncated"]}
        return self.observe(ns), ns, reward.astype(jnp.float32), dones, info
