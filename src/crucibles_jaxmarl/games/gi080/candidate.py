"""Canal Charters: an original turn-based spatial infrastructure commons."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 435
    num_actions = 13
    max_steps = 120

    def reset(self, key):
        grid = jnp.zeros((9, 9), dtype=jnp.int32)
        grid = grid.at[0, :].set(1).at[8, :].set(1)
        grid = grid.at[:, 0].set(1).at[:, 8].set(1)
        farms = jnp.array([[2, 6], [4, 6], [6, 6]], dtype=jnp.int32)
        grid = grid.at[farms[:, 1], farms[:, 0]].set(2).at[1, 4].set(3).at[7, 4].set(4)
        ids = jnp.arange(81, dtype=jnp.int32)
        positions = jnp.stack([jnp.mod(ids, 9), ids // 9], axis=1)
        direction = -jnp.ones((81,), dtype=jnp.int32)
        direction = direction.at[jnp.array([13, 22, 31, 40, 49])].set(2)
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False),
            "success": jnp.array(False), "map": grid,
            "agent_pos": jnp.array([[2, 4], [4, 4], [6, 4]], dtype=jnp.int32),
            "agent_active": jnp.ones((3,), dtype=jnp.bool_),
            "agent_visibility": jnp.ones((3, 3), dtype=jnp.bool_),
            "agent_coins": jnp.full((3,), 8, dtype=jnp.int32),
            "agent_kits": jnp.full((3,), 6, dtype=jnp.int32),
            "agent_facing": jnp.full((3,), 2, dtype=jnp.int32),
            "agent_harvest": jnp.zeros((3,), dtype=jnp.int32),
            "tile_pos": positions, "tile_active": direction >= 0,
            "tile_visibility": jnp.ones((3, 81), dtype=jnp.bool_),
            "tile_direction": direction,
            "tile_owner": -jnp.ones((81,), dtype=jnp.int32),
            "tile_open": jnp.ones((81,), dtype=jnp.bool_),
            "tile_water": jnp.zeros((81,), dtype=jnp.int32),
            "tile_fixed": ids == 13,
            "tile_flow": jnp.zeros((81,), dtype=jnp.int32),
            "farm_pos": farms, "farm_active": jnp.ones((3,), dtype=jnp.bool_),
            "farm_visibility": jnp.ones((3, 3), dtype=jnp.bool_),
            "farm_owner": jnp.arange(3, dtype=jnp.int32),
            "farm_water": jnp.zeros((3,), dtype=jnp.int32),
            "drain_pos": jnp.array([[4, 7]], dtype=jnp.int32),
            "drain_active": jnp.ones((1,), dtype=jnp.bool_),
            "drain_visibility": jnp.ones((3, 1), dtype=jnp.bool_),
            "total_spill": jnp.array(0, dtype=jnp.int32),
            "spill_step": jnp.array(0, dtype=jnp.int32),
            "ev_build": jnp.zeros((81,), dtype=jnp.bool_),
            "ev_rotate": jnp.zeros((81,), dtype=jnp.bool_),
            "ev_shutter": jnp.zeros((81,), dtype=jnp.bool_),
            "ev_charter": jnp.zeros((81,), dtype=jnp.bool_),
            "ev_harvest": jnp.zeros((3,), dtype=jnp.bool_),
            "ev_toll": jnp.zeros((3,), dtype=jnp.int32),
            "ev_bail": jnp.zeros((3,), dtype=jnp.int32),
        }
        return self.observe(s), s

    def observe(self, s):
        common = jnp.concatenate([
            jnp.array([s["t"] / 120.0]), jax.nn.one_hot(jnp.mod(s["t"], 3), 3, dtype=jnp.float32),
            jnp.array([s["total_spill"] / 100.0, s["done"].astype(jnp.float32)]),
            s["map"].reshape(-1) / 4.0, s["agent_pos"].reshape(-1) / 8.0,
            s["agent_coins"] / 30.0, s["agent_kits"] / 6.0,
            s["agent_facing"] / 3.0, s["agent_harvest"] / 40.0,
            (s["tile_direction"] + 1) / 4.0, (s["tile_owner"] + 1) / 3.0,
            s["tile_open"].astype(jnp.float32), s["tile_water"] / 6.0,
            s["farm_water"] / 6.0,
        ]).astype(jnp.float32)
        return jnp.concatenate([jnp.broadcast_to(common[None, :], (3, 432)), jnp.eye(3, dtype=jnp.float32)], axis=1)

    def legal_actions(self, s):
        active = jnp.mod(s["t"], 3)
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        target = jnp.clip(s["agent_pos"][:, None, :] + dirs[None, :, :], jnp.array([0, 0]), jnp.array([8, 8]))
        ti = target[:, :, 1] * 9 + target[:, :, 0]
        owners = s["tile_owner"][ti]
        toll = (owners >= 0) & (owners != jnp.arange(3)[:, None])
        occupied = jnp.any(jnp.all(target[:, :, None, :] == s["agent_pos"][None, None, :, :], axis=-1), axis=-1)
        moves = (s["map"][target[:, :, 1], target[:, :, 0]] != 1) & ~occupied & ((~toll) | (s["agent_coins"][:, None] >= 1))
        at = s["agent_pos"][:, 1] * 9 + s["agent_pos"][:, 0]
        terrain = s["map"][s["agent_pos"][:, 1], s["agent_pos"][:, 0]]
        owner = s["tile_owner"][at]
        build = (terrain == 0) & (s["tile_direction"][at] < 0) & (s["agent_kits"] > 0) & (s["agent_coins"] > 0)
        access = (owner < 0) | (owner == jnp.arange(3)) | (s["agent_coins"] > 0)
        rotate = (s["tile_direction"][at] >= 0) & ~s["tile_fixed"][at] & access
        shutter = (s["tile_direction"][at] >= 0) & ~s["tile_fixed"][at] & ((owner < 0) | (owner == jnp.arange(3)))
        charter = (owner == jnp.arange(3)) & (s["tile_direction"][at] >= 0)
        bail = (s["tile_water"][at] > 0) & (s["agent_coins"] > 0)
        rows = jnp.concatenate([jnp.ones((3, 1), dtype=jnp.bool_), moves, jnp.broadcast_to(build[:, None], (3, 4)), rotate[:, None], shutter[:, None], charter[:, None], bail[:, None]], axis=1)
        enabled = (jnp.arange(3) == active) & ~s["done"]
        return jnp.where(enabled[:, None], rows, jnp.broadcast_to(jnp.arange(13) == 0, (3, 13)))

    def absorb(self, key, s, actions):
        return self.observe(s), s, jnp.zeros((3,), dtype=jnp.float32), jnp.ones((3,), dtype=jnp.bool_), {
            "done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def advance(self, key, state, actions):
        s = dict(state)
        i = jnp.mod(s["t"], 3)
        a = jnp.clip(actions[i], 0, 12)
        valid = (actions[i] >= 0) & (actions[i] < 13) & self.legal_actions(s)[i, a]
        a = jnp.where(valid, a, 0)
        for n in ["ev_build", "ev_rotate", "ev_shutter", "ev_charter"]:
            s[n] = jnp.zeros((81,), dtype=jnp.bool_)
        s["ev_toll"] = jnp.zeros((3,), dtype=jnp.int32)
        s["ev_bail"] = jnp.zeros((3,), dtype=jnp.int32)
        dirs = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        moving = (a >= 1) & (a <= 4)
        dst = jnp.clip(s["agent_pos"][i] + dirs[jnp.clip(a - 1, 0, 3)], jnp.array([0, 0]), jnp.array([8, 8]))
        dest_id = dst[1] * 9 + dst[0]
        owner = s["tile_owner"][dest_id]
        toll = moving & (owner >= 0) & (owner != i)
        s["agent_coins"] = s["agent_coins"].at[i].add(-toll.astype(jnp.int32))
        s["agent_coins"] = s["agent_coins"].at[jnp.maximum(0, owner)].add(toll.astype(jnp.int32))
        s["ev_toll"] = s["ev_toll"].at[jnp.maximum(0, owner)].add(toll.astype(jnp.int32))
        s["agent_pos"] = s["agent_pos"].at[i].set(jnp.where(moving, dst, s["agent_pos"][i]))
        s["agent_facing"] = s["agent_facing"].at[i].set(jnp.where(moving, a - 1, s["agent_facing"][i]))
        at = s["agent_pos"][i, 1] * 9 + s["agent_pos"][i, 0]
        owner = s["tile_owner"][at]
        build = (a >= 5) & (a <= 8)
        rotate = a == 9
        permit = rotate & (owner >= 0) & (owner != i)
        s["agent_coins"] = s["agent_coins"].at[i].add(-build.astype(jnp.int32) - permit.astype(jnp.int32))
        s["agent_coins"] = s["agent_coins"].at[jnp.maximum(0, owner)].add(permit.astype(jnp.int32))
        s["ev_toll"] = s["ev_toll"].at[jnp.maximum(0, owner)].add(permit.astype(jnp.int32))
        s["agent_kits"] = s["agent_kits"].at[i].add(-build.astype(jnp.int32))
        direction = jnp.where(build, a - 5, jnp.where(rotate, jnp.mod(s["tile_direction"][at] + 1, 4), s["tile_direction"][at]))
        s["tile_direction"] = s["tile_direction"].at[at].set(direction)
        s["tile_owner"] = s["tile_owner"].at[at].set(jnp.where(build, i, jnp.where(a == 11, -1, owner)))
        s["tile_open"] = s["tile_open"].at[at].set(jnp.where(a == 10, ~s["tile_open"][at], s["tile_open"][at]))
        s["ev_build"] = s["ev_build"].at[at].set(build)
        s["ev_rotate"] = s["ev_rotate"].at[at].set(rotate)
        s["ev_shutter"] = s["ev_shutter"].at[at].set(a == 10)
        s["ev_charter"] = s["ev_charter"].at[at].set(a == 11)
        bailed = jnp.where(a == 12, jnp.minimum(2, s["tile_water"][at]), 0)
        s["tile_water"] = s["tile_water"].at[at].add(-bailed)
        s["agent_coins"] = s["agent_coins"].at[i].add(-(a == 12).astype(jnp.int32))
        s["ev_bail"] = s["ev_bail"].at[i].set(bailed)
        s["tile_active"] = s["tile_direction"] >= 0
        pulse = jnp.mod(s["t"] + 1, 3) == 0
        wet = s["tile_water"].at[13].add(2)
        dest = s["tile_pos"] + dirs[jnp.clip(s["tile_direction"], 0, 3)]
        inside = jnp.all((dest >= 0) & (dest <= 8), axis=1)
        dest = jnp.clip(dest, 0, 8)
        target = dest[:, 1] * 9 + dest[:, 0]
        ground = s["map"][dest[:, 1], dest[:, 0]]
        connected = s["tile_active"][target] | (ground == 2) | (ground == 4)
        sending = s["tile_active"] & s["tile_open"] & inside & connected
        flow = jnp.where(sending, jnp.minimum(2, wet), 0)
        wet = (wet - flow).at[target].add(flow)
        wet = wet.at[67].set(0)
        spill = jnp.sum(jnp.maximum(0, wet - 6))
        wet = jnp.minimum(6, wet)
        farm_id = s["farm_pos"][:, 1] * 9 + s["farm_pos"][:, 0]
        harvest = wet[farm_id] >= 2
        wet = wet.at[farm_id].add(-2 * harvest.astype(jnp.int32))
        s["tile_water"] = jnp.where(pulse, wet, s["tile_water"])
        s["tile_flow"] = jnp.where(pulse, flow, jnp.zeros((81,), dtype=jnp.int32))
        s["spill_step"] = jnp.where(pulse, spill, 0)
        s["total_spill"] = s["total_spill"] + s["spill_step"]
        s["ev_harvest"] = harvest & pulse
        gains = s["ev_harvest"].astype(jnp.int32)
        s["agent_harvest"] = s["agent_harvest"] + gains
        s["agent_coins"] = jnp.minimum(30, s["agent_coins"] + 2 * gains)
        s["farm_water"] = s["tile_water"][farm_id]
        success = jnp.all(s["agent_harvest"] >= 3)
        failed = s["total_spill"] >= 18
        s["t"] = s["t"] + jnp.array(1, dtype=jnp.int32)
        s["success"] = success & ~failed
        s["terminated"] = success | failed
        s["truncated"] = (s["t"] >= self.max_steps) & ~s["terminated"]
        s["done"] = s["terminated"] | s["truncated"]
        civic = 0.05 * jnp.sum(gains) - 0.02 * jnp.minimum(10, s["spill_step"]) + s["success"].astype(jnp.float32) - failed.astype(jnp.float32)
        rewards = (gains.astype(jnp.float32) + civic - 0.005 * (jnp.arange(3) == i).astype(jnp.float32)).astype(jnp.float32)
        return self.observe(s), s, rewards, jnp.full((3,), s["done"], dtype=jnp.bool_), {
            "done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, key, state, actions)
