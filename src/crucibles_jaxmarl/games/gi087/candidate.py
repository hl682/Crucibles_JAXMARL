"""Spillway Orchard: four bank-side foragers share a downstream-moving commons."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 73
    num_actions = 11
    max_steps = 96

    def direction(self, a):
        return jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                          [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)[a]

    def refresh(self, s):
        ns = dict(s)
        aa = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2)
        ap = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["bed_pos"][None, :, :]), axis=2)
        ns["agent_visible"] = aa <= 3
        ns["bed_visible"] = ap <= 3
        return ns

    def reset(self, key):
        x = jnp.arange(11, dtype=jnp.int32)[None, :]
        y = jnp.arange(9, dtype=jnp.int32)[:, None]
        walls = (x == 0) | (x == 10) | (y == 0) | (y == 8)
        tiles = walls.astype(jnp.int32)
        tiles = jnp.where((y == 4) & (x >= 1) & (x <= 9), 2, tiles)
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "tiles": tiles,
            "agent_pos": jnp.array([[2, 3], [4, 3], [6, 3], [8, 3]], dtype=jnp.int32),
            "agent_active": jnp.ones(4, dtype=jnp.bool_),
            "agent_visible": jnp.ones((4, 4), dtype=jnp.bool_),
            "bag": jnp.zeros(4, dtype=jnp.int32),
            "banked": jnp.zeros(4, dtype=jnp.int32),
            "bed_pos": jnp.array([[2, 4], [4, 4], [6, 4], [8, 4]], dtype=jnp.int32),
            "bed_active": jnp.ones(4, dtype=jnp.bool_),
            "bed_visible": jnp.ones((4, 4), dtype=jnp.bool_),
            "stock": jnp.full(4, 3, dtype=jnp.int32),
            "water": jnp.array([3, 2, 1, 1], dtype=jnp.int32),
            "pollution": jnp.zeros(4, dtype=jnp.int32),
            "seedlings": jnp.zeros(4, dtype=jnp.int32),
            "claim_owner": jnp.full(4, -1, dtype=jnp.int32),
            "claim_ttl": jnp.zeros(4, dtype=jnp.int32),
            "pump_pos": jnp.array([[1, 4]], dtype=jnp.int32),
            "pump_active": jnp.ones(1, dtype=jnp.bool_),
            "pump_visible": jnp.ones((4, 1), dtype=jnp.bool_),
            "market_pos": jnp.array([[5, 1]], dtype=jnp.int32),
            "market_active": jnp.ones(1, dtype=jnp.bool_),
            "market_visible": jnp.ones((4, 1), dtype=jnp.bool_),
            "pulse_event": jnp.array(False),
            "bank_event": jnp.zeros(4, dtype=jnp.int32),
            "total_banked": jnp.array(0, dtype=jnp.int32),
            "success": jnp.array(False), "collapse": jnp.array(False),
            "done": jnp.array(False), "terminated": jnp.array(False),
            "truncated": jnp.array(False),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        scale = jnp.array([10.0, 8.0], dtype=jnp.float32)
        agent_data = jnp.concatenate((s["agent_pos"] / scale, s["bag"][:, None] / 4.0), axis=1)
        agents = jnp.where(s["agent_visible"][:, :, None], agent_data[None, :, :], 0.0)
        agents = jnp.concatenate((agents, s["agent_visible"][:, :, None].astype(jnp.float32)), axis=2)
        bed_data = jnp.concatenate((s["bed_pos"] / scale, s["stock"][:, None] / 5.0,
                                    s["water"][:, None] / 3.0, s["pollution"][:, None] / 6.0,
                                    s["seedlings"][:, None] / 2.0, s["claim_ttl"][:, None] / 6.0,
                                    (s["claim_owner"][:, None] + 1) / 4.0), axis=1)
        beds = jnp.where(s["bed_visible"][:, :, None], bed_data[None, :, :], 0.0)
        beds = jnp.concatenate((beds, s["bed_visible"][:, :, None].astype(jnp.float32)), axis=2)
        public = jnp.concatenate((
            jnp.stack((s["t"] / 96.0, s["done"], s["success"], s["collapse"], s["pulse_event"])),
            s["bank_event"] / 4.0, s["banked"] / 8.0,
            (s["pump_pos"] / scale).reshape(-1), (s["market_pos"] / scale).reshape(-1),
        )).astype(jnp.float32)
        return jnp.concatenate((jax.nn.one_hot(jnp.arange(4), 4),
                                jnp.broadcast_to(public, (4, 17)),
                                agents.reshape(4, 16), beds.reshape(4, 36)), axis=1).astype(jnp.float32)

    def local_bed(self, s):
        distances = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["bed_pos"][None, :, :]), axis=2)
        target = jnp.argmin(distances, axis=1).astype(jnp.int32)
        return target, jnp.min(distances, axis=1) <= 1

    def legal_actions(self, s):
        target, near = self.local_bed(s)
        own = (s["claim_ttl"][target] == 0) | (s["claim_owner"][target] == jnp.arange(4))
        pump = jnp.sum(jnp.abs(s["agent_pos"] - s["pump_pos"][0]), axis=1) <= 1
        market = jnp.sum(jnp.abs(s["agent_pos"] - s["market_pos"][0]), axis=1) <= 1
        mask = jnp.ones((4, 11), dtype=jnp.bool_)
        mask = mask.at[:, 5].set(near & own & (s["stock"][target] > 0) & (s["water"][target] > 0) & (s["bag"] < 4))
        mask = mask.at[:, 6].set(near & (s["bag"] > 0) & (s["pollution"][target] > 0))
        mask = mask.at[:, 7].set(near & (s["bag"] > 0) & (s["seedlings"][target] < 2))
        mask = mask.at[:, 8].set(pump & (s["bag"] > 0))
        mask = mask.at[:, 9].set(near & (s["bag"] > 0))
        mask = mask.at[:, 10].set(market & (s["bag"] > 0) & (s["banked"] < 8))
        return jnp.where(s["done"], jnp.broadcast_to(jnp.arange(11) == 0, (4, 11)), mask)

    def advance(self, payload):
        key, s, actions = payload
        safe = jnp.clip(actions, 0, 10)
        valid = (actions >= 0) & (actions < 11) & self.legal_actions(s)[jnp.arange(4), safe]
        a = jnp.where(valid, safe, 0)
        target, near = self.local_bed(s)
        # One local work slot per bed per turn; priority rotates through all four identities.
        work = (a >= 5) & (a <= 9) & (a != 8)
        ranks = jnp.mod(jnp.arange(4) - s["t"], 4)
        competing = (target[:, None] == target[None, :]) & work[None, :]
        beaten = jnp.any(competing & (ranks[None, :] < ranks[:, None]), axis=1)
        accepted = work & ~beaten
        harvest = accepted & (a == 5)
        clean = accepted & (a == 6)
        sow = accepted & (a == 7)
        stake = accepted & (a == 9)
        pulse = a == 8
        at = jax.nn.one_hot(target, 4, dtype=jnp.int32)
        harvested = jnp.sum(at * harvest[:, None].astype(jnp.int32), axis=0)
        cleaned = jnp.sum(at * clean[:, None].astype(jnp.int32), axis=0)
        planted = jnp.sum(at * sow[:, None].astype(jnp.int32), axis=0)
        staked = jnp.sum(at * stake[:, None].astype(jnp.int32), axis=0) > 0
        owner = jnp.sum(at * (stake.astype(jnp.int32) * (jnp.arange(4) + 1))[:, None], axis=0) - 1
        ttl = jnp.where(staked, 6, jnp.maximum(s["claim_ttl"] - 1, 0))
        owner = jnp.where(staked, owner, jnp.where(ttl > 0, s["claim_owner"], -1))
        stock = s["stock"] - harvested
        water = s["water"] - harvested
        pollution = jnp.clip(s["pollution"] + 2 * harvested - 2 * cleaned, 0, 6)
        seedlings = jnp.minimum(2, s["seedlings"] + planted)
        invested = clean | sow | stake | pulse
        bag = s["bag"] + harvest.astype(jnp.int32) - invested.astype(jnp.int32)
        bank_event = jnp.where(a == 10, jnp.minimum(bag, 8 - s["banked"]), 0)
        bag = bag - bank_event
        banked = s["banked"] + bank_event
        t = s["t"] + 1
        # Headwater injection precedes conservative, one-reach-per-turn downstream transport.
        inflow = (jnp.mod(t, 2) == 0).astype(jnp.int32) + 2 * jnp.sum(pulse.astype(jnp.int32))
        water = water.at[0].set(jnp.minimum(3, water[0] + inflow))
        flow = (water >= 2).astype(jnp.int32)
        incoming_water = jnp.concatenate((jnp.zeros(1, dtype=jnp.int32), flow[:-1]))
        dirt_flow = jnp.minimum(pollution, 1) * flow
        incoming_dirt = jnp.concatenate((jnp.zeros(1, dtype=jnp.int32), dirt_flow[:-1]))
        water = jnp.clip(water - flow + incoming_water, 0, 3)
        pollution = jnp.clip(pollution - dirt_flow + incoming_dirt, 0, 6)
        # Silt settles every fourth turn. Only a living bed or an invested seed can regenerate.
        season = jnp.mod(t, 4) == 0
        pollution = jnp.maximum(0, pollution - season.astype(jnp.int32))
        grows = season & (pollution <= 1) & (water > 0) & (stock < 5) & ((stock > 0) | (seedlings > 0))
        stock = stock + grows.astype(jnp.int32)
        water = water - grows.astype(jnp.int32)
        seedlings = seedlings - (grows & (seedlings > 0)).astype(jnp.int32)
        # Physical movement uses the actual public bank/stream map and conservative collisions.
        pos = s["agent_pos"]
        proposed = pos + self.direction(a)
        px = jnp.clip(proposed[:, 0], 0, 10)
        py = jnp.clip(proposed[:, 1], 0, 8)
        inside = (proposed[:, 0] > 0) & (proposed[:, 0] < 10) & (proposed[:, 1] > 0) & (proposed[:, 1] < 8)
        occupied = jnp.any(jnp.all(proposed[:, None, :] == pos[None, :, :], axis=2)
                           & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        eligible = (a >= 1) & (a <= 4) & inside & (s["tiles"][py, px] != 1) & ~occupied
        duplicate = jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2)
        clash = jnp.any(duplicate & eligible[None, :] & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        pos = jnp.where((eligible & ~clash)[:, None], proposed, pos)
        success = jnp.all(banked >= 4)
        collapse = (jnp.sum(stock) + jnp.sum(seedlings) + jnp.sum(bag) == 0) & ~success
        terminated = success | collapse
        truncated = (t >= self.max_steps) & ~terminated
        ns = dict(s)
        ns["t"] = t
        ns["agent_pos"] = pos
        ns["bag"] = bag.astype(jnp.int32)
        ns["banked"] = banked.astype(jnp.int32)
        ns["stock"] = stock.astype(jnp.int32)
        ns["water"] = water.astype(jnp.int32)
        ns["pollution"] = pollution.astype(jnp.int32)
        ns["seedlings"] = seedlings.astype(jnp.int32)
        ns["claim_ttl"] = ttl.astype(jnp.int32)
        ns["claim_owner"] = owner.astype(jnp.int32)
        ns["pulse_event"] = jnp.any(pulse)
        ns["bank_event"] = bank_event.astype(jnp.int32)
        ns["total_banked"] = jnp.sum(banked, dtype=jnp.int32)
        ns["success"] = success
        ns["collapse"] = collapse
        ns["terminated"] = terminated
        ns["truncated"] = truncated
        ns["done"] = terminated | truncated
        ns = self.refresh(ns)
        reward = -0.01 + 0.2 * harvest.astype(jnp.float32) - 0.05 * invested.astype(jnp.float32)
        reward = reward + bank_event.astype(jnp.float32) + 5.0 * success.astype(jnp.float32) - 3.0 * collapse.astype(jnp.float32)
        info = {"done_all": ns["done"], "terminated": terminated, "truncated": truncated}
        return self.observe(ns), ns, reward.astype(jnp.float32), jnp.full(4, ns["done"]), info

    def absorb(self, payload):
        key, s, actions = payload
        info = {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}
        return self.observe(s), s, jnp.zeros(4, dtype=jnp.float32), jnp.ones(4, dtype=jnp.bool_), info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, actions))
