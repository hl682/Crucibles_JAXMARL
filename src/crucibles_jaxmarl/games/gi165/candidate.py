"""Crossing Accord: mobile one-cell bridges, two teams and physical yielding."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 4
    observation_size = 68
    num_actions = 11
    max_steps = 96

    def reset(self, key):
        terrain = jnp.zeros((7, 9), jnp.int32).at[:, 4].set(1)
        bridges = jnp.array([[4, 2], [4, 4]], jnp.int32)
        terrain = terrain.at[bridges[:, 1], bridges[:, 0]].set(2)
        s = {
            "t": jnp.array(0, jnp.int32), "done": jnp.array(False),
            "terminated": jnp.array(False), "truncated": jnp.array(False), "terrain": terrain,
            "agent_pos": jnp.array([[1, 2], [1, 4], [7, 2], [7, 4]], jnp.int32),
            "agent_active": jnp.ones(4, jnp.bool_), "agent_visible": jnp.ones((4, 4), jnp.bool_),
            "team": jnp.array([0, 0, 1, 1], jnp.int32), "energy": jnp.full(4, 6, jnp.int32),
            "delivered": jnp.zeros(4, jnp.bool_), "bridge_pos": bridges,
            "bridge_active": jnp.ones(2, jnp.bool_), "bridge_visible": jnp.ones((4, 2), jnp.bool_),
            "bridge_direction": jnp.array([0, 1], jnp.int32), "occupant": jnp.full(2, -1, jnp.int32),
            "zone_pos": jnp.array([[3, 3], [5, 3]], jnp.int32),
            "zone_active": jnp.ones(2, jnp.bool_), "zone_visible": jnp.ones((4, 2), jnp.bool_),
            "control": jnp.zeros(2, jnp.int32), "owner": jnp.full(2, -1, jnp.int32),
            "goal_pos": jnp.array([[8, 2], [8, 4], [0, 2], [0, 4]], jnp.int32),
            "goal_active": jnp.ones(4, jnp.bool_), "goal_visible": jnp.ones((4, 4), jnp.bool_),
            "team_deliveries": jnp.zeros(2, jnp.int32), "blocked": jnp.zeros(4, jnp.bool_),
            "cranked": jnp.zeros(4, jnp.bool_), "claimed": jnp.zeros(2, jnp.bool_),
            "arrived": jnp.zeros(4, jnp.bool_), "entered": jnp.zeros(4, jnp.bool_)
        }
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate((
            s["t"].reshape(1), s["agent_pos"].reshape(-1), s["team"], s["energy"], s["delivered"],
            s["bridge_pos"].reshape(-1), s["bridge_direction"], s["occupant"], s["control"], s["owner"],
            s["goal_pos"].reshape(-1), s["team_deliveries"], s["blocked"], s["cranked"], s["claimed"],
            s["arrived"], s["entered"], s["done"].reshape(1), s["terminated"].reshape(1), s["truncated"].reshape(1)
        )).astype(jnp.float32)
        return jnp.concatenate((jnp.eye(4, dtype=jnp.float32), jnp.broadcast_to(public, (4, 64))), axis=1)

    def crossing(self, s, q, direction, bridges, bridge_direction):
        on_bridge = jnp.all(q[..., None, :] == bridges, axis=-1)
        allowed = ((direction[..., None] == 2) & (bridge_direction == 0)) | ((direction[..., None] == 4) & (bridge_direction == 1))
        return (q[..., 0] != 4) | jnp.any(on_bridge & allowed, axis=-1)

    def legal_actions(self, s):
        p = s["agent_pos"]
        d = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = p[:, None, :] + d[None, :, :]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] <= 8) & (q[:, :, 1] >= 0) & (q[:, :, 1] <= 6)
        occupied = jnp.any(jnp.all(q[:, :, None, :] == p[None, None, :, :], axis=3)
                           & ~jnp.eye(4, dtype=jnp.bool_)[:, None, :], axis=2)
        move = inside & ~occupied & self.crossing(s, q, jnp.arange(1, 5)[None, :], s["bridge_pos"], s["bridge_direction"])
        zone_distance = jnp.sum(jnp.abs(p[:, None, :] - s["zone_pos"][None, :, :]), axis=2)
        near = zone_distance <= 1
        discount = jnp.any(near & (s["owner"][None, :] == s["team"][:, None]), axis=1)
        cost = jnp.where(discount, 1, 2)
        crank = jnp.any(near, axis=1) & (s["energy"] >= cost)
        rest = jnp.stack((crank & (s["occupant"][0] < 0), crank & (s["occupant"][1] < 0),
                          crank & (s["occupant"][0] < 0), crank & (s["occupant"][1] < 0),
                          jnp.any(zone_distance == 0, axis=1) & (s["energy"] > 0),
                          jnp.all(p == s["goal_pos"], axis=1) & ~s["delivered"]), axis=1)
        mask = jnp.concatenate((jnp.ones((4, 1), jnp.bool_), move, rest), axis=1)
        return mask & (~s["done"] | (jnp.arange(11)[None, :] == 0))

    def raw_step(self, key, s, actions):
        ac = jnp.clip(actions, 0, 10)
        valid = (actions >= 0) & (actions < 11) & self.legal_actions(s)[jnp.arange(4), ac]
        ac = jnp.where(valid, ac, 0)
        priority = (jnp.arange(4, dtype=jnp.int32) - s["t"]) % 4
        zone_distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["zone_pos"][None, :, :]), axis=2)
        discount = jnp.any((zone_distance <= 1) & (s["owner"][None, :] == s["team"][:, None]), axis=1)
        cost = jnp.where(discount, 1, 2)
        bridges = s["bridge_pos"]
        direction = s["bridge_direction"]
        cranked = jnp.zeros(4, jnp.bool_)
        for b in range(2):
            wants = (ac == 5 + b) | (ac == 7 + b)
            winner = jnp.argmin(jnp.where(wants, priority, 99))
            perform = jnp.any(wants)
            shift = perform & (ac[winner] == 5 + b)
            flip = perform & (ac[winner] == 7 + b)
            other_row = jnp.where(b == 0, 3 - bridges[b, 1], 9 - bridges[b, 1])
            bridges = bridges.at[b, 1].set(jnp.where(shift, other_row, bridges[b, 1]))
            direction = direction.at[b].set(jnp.where(flip, 1 - direction[b], direction[b]))
            cranked = cranked.at[winner].set(cranked[winner] | perform)

        # Current occupancy forbids following, swapping and pushing; empty-cell ties rotate.
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0],
                           [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        proposal = s["agent_pos"] + delta[ac]
        trying = (ac >= 1) & (ac <= 4)
        possible = trying & self.crossing(s, proposal, ac, bridges, direction)
        same_target = jnp.all(proposal[:, None, :] == proposal[None, :, :], axis=2)
        outranked = jnp.any(same_target & possible[None, :] & (priority[None, :] < priority[:, None]), axis=1)
        moved = possible & ~outranked
        pos = jnp.where(moved[:, None], proposal, s["agent_pos"])
        blocked = trying & ~moved
        load = jnp.all(pos[:, None, :] == bridges[None, :, :], axis=2)
        occupant = jnp.where(jnp.any(load, axis=0), jnp.argmax(load, axis=0), -1)
        entered = moved & (pos[:, 0] == 4)
        terrain = s["terrain"].at[:, 4].set(1)
        terrain = terrain.at[bridges[:, 1], bridges[:, 0]].set(2)

        claim = ac == 9
        zone = jnp.argmin(zone_distance, axis=1)
        force = jnp.sum(jax.nn.one_hot(zone, 2, dtype=jnp.int32)
                        * (claim * (1 - 2 * s["team"]))[:, None], axis=0)
        control = jnp.clip(s["control"] + force, -2, 2)
        owner = jnp.where(control == 2, 0, jnp.where(control == -2, 1, s["owner"]))
        claimed = owner != s["owner"]
        home = jnp.where(s["team"] == 0, pos[:, 0] <= 2, pos[:, 0] >= 6)
        energy = jnp.clip(s["energy"] - cranked * cost - claim + ((ac == 0) & home), 0, 6)
        arrived = ac == 10
        delivered = s["delivered"] | arrived
        team_deliveries = jnp.sum(jax.nn.one_hot(s["team"], 2, dtype=jnp.int32) * delivered[:, None], axis=0)
        allied = s["team"][:, None] == s["team"][None, :]
        team_gain = jnp.sum(allied * arrived[None, :], axis=1)
        enemy_gain = jnp.sum(~allied * arrived[None, :], axis=1)
        t = s["t"] + 1
        own_zones = jnp.sum(owner[None, :] == s["team"][:, None], axis=1)
        enemy_zones = jnp.sum((owner[None, :] >= 0) & (owner[None, :] != s["team"][:, None]), axis=1)
        territory = ((t % 8) == 0) * 0.125 * (own_zones - enemy_zones)
        rewards = (2.0 * arrived + team_gain - enemy_gain + territory - 0.0625 * cranked - 0.015625).astype(jnp.float32)
        term = jnp.any(team_deliveries >= 2)
        trunc = (t >= self.max_steps) & ~term
        ns = {**s, "t": t, "done": term | trunc, "terminated": term, "truncated": trunc,
              "terrain": terrain, "agent_pos": pos, "energy": energy, "delivered": delivered,
              "bridge_pos": bridges, "bridge_direction": direction, "occupant": occupant,
              "control": control, "owner": owner, "team_deliveries": team_deliveries,
              "blocked": blocked, "cranked": cranked, "claimed": claimed, "arrived": arrived, "entered": entered}
        ns = jax.tree.map(lambda old, new: jnp.where(s["done"], old, new), s, ns)
        rewards = jnp.where(s["done"], jnp.zeros(4, jnp.float32), rewards)
        info = {"done_all": ns["done"], "terminated": ns["terminated"], "truncated": ns["truncated"]}
        return self.observe(ns), ns, rewards, jnp.full(4, ns["done"], jnp.bool_), info
