"""Lantern Intercept: asymmetric reconnaissance, shared spotlights, and contested rendezvous."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 63
    num_actions = 11
    max_steps = 100

    def direction(self, a):
        return jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                          [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32)[a]

    def refresh(self, s):
        ns = dict(s)
        distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2)
        radius = jnp.where(s["role"] == 0, 4, 2)
        lit_agent = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["beacon_pos"][None, :, :]), axis=2) <= 3
        illuminated = jnp.any(lit_agent & (s["beacon_ttl"][None, :] > 0), axis=1)
        ns["agent_visible"] = (distance <= radius[:, None]) | illuminated[None, :]
        relic_distance = jnp.sum(jnp.abs(s["agent_pos"] - s["relic_pos"][0]), axis=1)
        relic_radius = jnp.where(s["role"] == 0, 4, 1)
        lit_relic = jnp.sum(jnp.abs(s["beacon_pos"] - s["relic_pos"][0]), axis=1) <= 3
        relic_public = jnp.any(lit_relic & (s["beacon_ttl"] > 0))
        ns["relic_visible"] = ((relic_distance <= relic_radius) | relic_public)[:, None] & s["relic_active"][None, :]
        ns["beacon_active"] = s["beacon_ttl"] > 0
        return ns

    def reset(self, key):
        x = jnp.arange(11, dtype=jnp.int32)[None, :]
        y = jnp.arange(11, dtype=jnp.int32)[:, None]
        walls = (x == 0) | (x == 10) | (y == 0) | (y == 10)
        walls = walls | ((x == 5) & (y > 0) & (y < 10) & (y != 2) & (y != 5) & (y != 8))
        walls = walls | (((x == 3) | (x == 7)) & ((y == 4) | (y == 6)))
        tiles = walls.astype(jnp.int32)
        tiles = tiles.at[5, 1].set(2)
        tiles = tiles.at[5, 9].set(3)
        cache = jnp.array([[5, 2], [5, 5], [5, 8]], dtype=jnp.int32)
        cache_index = jax.random.randint(key, (), 0, 3, dtype=jnp.int32)
        s = {
            "t": jnp.array(0, dtype=jnp.int32), "tiles": tiles,
            "agent_pos": jnp.array([[2, 2], [2, 3], [8, 8], [8, 7]], dtype=jnp.int32),
            "agent_active": jnp.ones(4, dtype=jnp.bool_),
            "agent_visible": jnp.ones((4, 4), dtype=jnp.bool_),
            "team": jnp.array([0, 0, 1, 1], dtype=jnp.int32),
            "role": jnp.array([0, 1, 0, 1], dtype=jnp.int32),
            "energy": jnp.full(4, 3, dtype=jnp.int32),
            "stun": jnp.zeros(4, dtype=jnp.int32),
            "tag_cooldown": jnp.zeros(4, dtype=jnp.int32),
            "relic_pos": cache[cache_index][None, :],
            "relic_active": jnp.ones(1, dtype=jnp.bool_),
            "relic_visible": jnp.zeros((4, 1), dtype=jnp.bool_),
            "relic_carrier": jnp.array([-1], dtype=jnp.int32),
            "beacon_pos": jnp.array([[2, 2], [8, 8]], dtype=jnp.int32),
            "beacon_active": jnp.zeros(2, dtype=jnp.bool_),
            "beacon_visible": jnp.ones((4, 2), dtype=jnp.bool_),
            "beacon_ttl": jnp.zeros(2, dtype=jnp.int32),
            "base_pos": jnp.array([[1, 5], [9, 5]], dtype=jnp.int32),
            "base_active": jnp.ones(2, dtype=jnp.bool_),
            "base_visible": jnp.ones((4, 2), dtype=jnp.bool_),
            "pickup_notice": jnp.zeros(2, dtype=jnp.bool_),
            "tag_notice": jnp.zeros(2, dtype=jnp.int32),
            "claim_once": jnp.zeros(2, dtype=jnp.bool_),
            "delivery_notice": jnp.zeros(2, dtype=jnp.bool_),
            "winner": jnp.array(-1, dtype=jnp.int32),
            "done": jnp.array(False), "terminated": jnp.array(False),
            "truncated": jnp.array(False),
        }
        s = self.refresh(s)
        return self.observe(s), s

    def observe(self, s):
        agent_data = jnp.concatenate((s["agent_pos"] / 10.0, s["energy"][:, None] / 3.0,
                                      s["stun"][:, None] / 2.0, s["tag_cooldown"][:, None] / 3.0,
                                      s["role"][:, None], s["team"][:, None]), axis=1)
        agents = jnp.where(s["agent_visible"][:, :, None], agent_data[None, :, :], 0.0)
        agents = jnp.concatenate((agents, s["agent_visible"][:, :, None].astype(jnp.float32)), axis=2)
        beacon_data = jnp.concatenate((s["beacon_pos"] / 10.0, s["beacon_ttl"][:, None] / 4.0,
                                       s["beacon_active"][:, None].astype(jnp.float32)), axis=1)
        beacons = jnp.where(s["beacon_active"][:, None], beacon_data, 0.0)
        relic_data = jnp.concatenate((s["relic_pos"][0] / 10.0, (s["relic_carrier"] + 1) / 4.0))
        relic = jnp.where(s["relic_visible"], relic_data[None, :], 0.0)
        relic = jnp.concatenate((relic, s["relic_visible"].astype(jnp.float32)), axis=1)
        public = jnp.concatenate((jnp.stack((s["t"] / 100.0, s["done"], (s["winner"] + 1) / 2.0)),
                                  s["pickup_notice"], s["tag_notice"] / 2.0, s["claim_once"],
                                  s["delivery_notice"], (s["base_pos"] / 10.0).reshape(-1),
                                  beacons.reshape(-1))).astype(jnp.float32)
        return jnp.concatenate((jax.nn.one_hot(jnp.arange(4), 4),
                                jnp.broadcast_to(public, (4, 23)),
                                agents.reshape(4, 32), relic), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        carrier = s["relic_carrier"][0]
        partner = jnp.array([1, 0, 3, 2], dtype=jnp.int32)
        distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["agent_pos"][None, :, :]), axis=2)
        adjacent_partner = distance[jnp.arange(4), partner] <= 1
        adjacent_enemy = jnp.any((distance <= 1) & (s["team"][:, None] != s["team"][None, :]), axis=1)
        relic_near = jnp.sum(jnp.abs(s["agent_pos"] - s["relic_pos"][0]), axis=1) <= 1
        base_near = jnp.sum(jnp.abs(s["agent_pos"] - s["base_pos"][s["team"]]), axis=1) <= 1
        slow = (s["role"] == 0) & (carrier == jnp.arange(4)) & (jnp.mod(s["t"], 2) == 1)
        mask = jnp.ones((4, 11), dtype=jnp.bool_)
        mask = mask.at[:, 1:5].set(jnp.broadcast_to((~slow)[:, None], (4, 4)))
        mask = mask.at[:, 5].set((s["role"] == 0) & (s["energy"] > 0))
        mask = mask.at[:, 6].set((carrier < 0) & s["relic_active"][0] & relic_near)
        mask = mask.at[:, 7].set(adjacent_partner & ((carrier == jnp.arange(4)) | (carrier == partner)))
        mask = mask.at[:, 8].set(adjacent_enemy & (s["energy"] > 0) & (s["tag_cooldown"] == 0))
        mask = mask.at[:, 9].set(adjacent_partner & (s["energy"] > 0))
        mask = mask.at[:, 10].set(base_near)
        mask = mask & (s["stun"] == 0)[:, None]
        mask = mask.at[:, 0].set(True)
        return jnp.where(s["done"], jnp.broadcast_to(jnp.arange(11) == 0, (4, 11)), mask)

    def advance(self, payload):
        key, s, actions = payload
        safe = jnp.clip(actions, 0, 10)
        valid = (actions >= 0) & (actions < 11) & self.legal_actions(s)[jnp.arange(4), safe]
        a = jnp.where(valid, safe, 0)
        partner = jnp.array([1, 0, 3, 2], dtype=jnp.int32)
        ranks = jnp.mod(jnp.arange(4) - s["t"], 4)
        # All walks are simultaneous; occupied old cells and contested destinations block entry.
        pos = s["agent_pos"]
        proposed = pos + self.direction(a)
        px = jnp.clip(proposed[:, 0], 0, 10)
        py = jnp.clip(proposed[:, 1], 0, 10)
        inside = (proposed[:, 0] > 0) & (proposed[:, 0] < 10) & (proposed[:, 1] > 0) & (proposed[:, 1] < 10)
        occupied = jnp.any(jnp.all(proposed[:, None, :] == pos[None, :, :], axis=2)
                           & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        eligible = (a >= 1) & (a <= 4) & inside & (s["tiles"][py, px] != 1) & ~occupied
        duplicate = jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2)
        clash = jnp.any(duplicate & eligible[None, :] & ~jnp.eye(4, dtype=jnp.bool_), axis=1)
        pos = jnp.where((eligible & ~clash)[:, None], proposed, pos)
        # A spotlight is a public radio scan through walls, not a private message or line-of-sight model.
        scouts = jnp.array([0, 2], dtype=jnp.int32)
        flashed = a[scouts] == 5
        beacon_pos = jnp.where(flashed[:, None], pos[scouts], s["beacon_pos"])
        beacon_ttl = jnp.where(flashed, 4, jnp.maximum(0, s["beacon_ttl"] - 1))
        distance = jnp.sum(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=2)
        enemies = (distance <= 1) & (s["team"][:, None] != s["team"][None, :])
        target = jnp.argmin(jnp.where(enemies, ranks[None, :], 99), axis=1).astype(jnp.int32)
        tag_attempt = a == 8
        tag_hits = tag_attempt & jnp.any(enemies, axis=1)
        attacked = jnp.any(jax.nn.one_hot(target, 4, dtype=jnp.bool_) & tag_hits[:, None], axis=0)
        protected = (a[partner] == 9) & (distance[jnp.arange(4), partner] <= 1)
        victim = attacked & ~protected
        stun = jnp.where(victim, 2, jnp.maximum(0, s["stun"] - 1))
        tag_cooldown = jnp.where(tag_attempt, 3, jnp.maximum(0, s["tag_cooldown"] - 1))
        spent = (a == 5) | tag_attempt | (a == 9)
        energy = jnp.clip(s["energy"] + (a == 0).astype(jnp.int32) - spent.astype(jnp.int32), 0, 3)
        carrier = s["relic_carrier"][0]
        safe_carrier = jnp.clip(carrier, 0, 3)
        relic_pos = jnp.where(carrier >= 0, pos[safe_carrier], s["relic_pos"][0])
        dropped = (carrier >= 0) & victim[safe_carrier]
        carrier = jnp.where(dropped, -1, carrier)
        # Pickup is a reach of one cell; a newly dropped object cannot be picked up by an intent illegal before this step.
        pickup = (a == 6) & ~victim & (carrier < 0)
        pickup = pickup & (jnp.sum(jnp.abs(pos - relic_pos), axis=1) <= 1)
        picker = jnp.argmin(jnp.where(pickup, ranks, 99)).astype(jnp.int32)
        picked = jnp.any(pickup)
        carrier = jnp.where(picked, picker, carrier)
        # Both teammates must offer a transfer; an attack on either participant cancels the handoff.
        safe_carrier = jnp.clip(carrier, 0, 3)
        receiver = partner[safe_carrier]
        handoff = (carrier >= 0) & (a[safe_carrier] == 7) & (a[receiver] == 7)
        handoff = handoff & (distance[safe_carrier, receiver] <= 1) & ~victim[safe_carrier] & ~victim[receiver]
        carrier = jnp.where(handoff, receiver, carrier)
        safe_carrier = jnp.clip(carrier, 0, 3)
        relic_pos = jnp.where(carrier >= 0, pos[safe_carrier], relic_pos)
        carrier_team = s["team"][safe_carrier]
        held_team = jax.nn.one_hot(carrier_team, 2, dtype=jnp.bool_) & (carrier >= 0)
        new_claim = held_team & ~s["claim_once"]
        pickup_notice = jax.nn.one_hot(s["team"][picker], 2, dtype=jnp.bool_) & picked
        tag_notice = jnp.sum(jax.nn.one_hot(s["team"], 2, dtype=jnp.int32) * victim[:, None].astype(jnp.int32), axis=0)
        # The runner must reach its extraction pad while the scout co-signs from an adjacent cell.
        runners = jnp.array([1, 3], dtype=jnp.int32)
        at_base = jnp.all(pos[runners] == s["base_pos"], axis=1)
        scout_near = jnp.sum(jnp.abs(pos[scouts] - s["base_pos"]), axis=1) <= 1
        delivered = (carrier == runners) & at_base & scout_near & (a[runners] == 10) & (a[scouts] == 10)
        delivered = delivered & ~victim[runners] & ~victim[scouts]
        success = jnp.any(delivered)
        winner = jnp.where(success, jnp.argmax(delivered), -1).astype(jnp.int32)
        t = s["t"] + 1
        terminated = success
        truncated = (t >= self.max_steps) & ~terminated
        ns = dict(s)
        ns["t"] = t
        ns["agent_pos"] = pos
        ns["energy"] = energy.astype(jnp.int32)
        ns["stun"] = stun.astype(jnp.int32)
        ns["tag_cooldown"] = tag_cooldown.astype(jnp.int32)
        ns["beacon_pos"] = beacon_pos
        ns["beacon_ttl"] = beacon_ttl.astype(jnp.int32)
        ns["relic_pos"] = relic_pos[None, :]
        ns["relic_carrier"] = carrier[None].astype(jnp.int32)
        ns["relic_active"] = (~success)[None]
        ns["pickup_notice"] = pickup_notice
        ns["tag_notice"] = tag_notice.astype(jnp.int32)
        ns["claim_once"] = s["claim_once"] | new_claim
        ns["delivery_notice"] = delivered
        ns["winner"] = winner
        ns["terminated"] = terminated
        ns["truncated"] = truncated
        ns["done"] = terminated | truncated
        ns = self.refresh(ns)
        reward = -0.01 - 0.02 * spent.astype(jnp.float32) + 0.25 * new_claim[s["team"]].astype(jnp.float32)
        outcome = jnp.where(s["team"] == winner, 6.0, -6.0)
        reward = reward + jnp.where(success, outcome, 0.0)
        info = {"done_all": ns["done"], "terminated": terminated, "truncated": truncated}
        return self.observe(ns), ns, reward.astype(jnp.float32), jnp.full(4, ns["done"]), info

    def absorb(self, payload):
        key, s, actions = payload
        info = {"done_all": s["done"], "terminated": s["terminated"], "truncated": s["truncated"]}
        return self.observe(s), s, jnp.zeros(4, dtype=jnp.float32), jnp.ones(4, dtype=jnp.bool_), info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, actions))
