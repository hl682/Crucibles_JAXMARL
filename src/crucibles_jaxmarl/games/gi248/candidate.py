"""Basalt Tide Salvage. Original source; no author execution or outcome evidence."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 512
    observation_size = 272
    num_actions = 16
    max_steps = 1024

    def distance(self, p, q):
        return jnp.sum(jnp.abs(p[:, None, :] - q[None, :, :]), axis=-1)

    def sight(self, p, q, active, count, radius):
        d = self.distance(p, q)
        order = jnp.argsort(jnp.where(active[None, :], d, 10000) * (q.shape[0] + 1) + jnp.arange(q.shape[0])[None, :], axis=1)
        rank = jnp.argsort(order, axis=1)
        visible = (rank < count) & (d <= radius) & active[None, :]
        return visible, order[:, :count]

    def refresh(self, s):
        s = dict(s)
        for name in ('t', 'pos', 'crew', 'tool', 'oxygen', 'blocks', 'carry', 'warning', 'knowledge', 'event', 'dock_pos', 'water', 'support', 'charge', 'delivered', 'forecast', 'flood_loss', 'brake_pos', 'cargo_pos', 'carrier', 'owner', 'rust', 'tile'):
            s[name] = s[name].astype(jnp.int32)
        s['actor_visible'], _ = self.sight(s['pos'], s['pos'], s['active'], 8, 5)
        s['actor_visible'] = s['actor_visible'] | jnp.eye(512, dtype=jnp.bool_)
        s['dock_visible'], _ = self.sight(s['pos'], s['dock_pos'], s['dock_active'], 4, 7)
        s['cargo_visible'], _ = self.sight(s['pos'], s['cargo_pos'], s['cargo_active'], 4, 5)
        s['brake_visible'], _ = self.sight(s['pos'], s['brake_pos'], s['brake_active'], 4, 5)
        return s

    def reset(self, key):
        crew = jnp.arange(512, dtype=jnp.int32) // 16
        slot = jnp.arange(512, dtype=jnp.int32) % 16
        base = jnp.stack(((crew % 8) * 8, (crew // 8) * 8), axis=1)
        starts = jnp.array([[4, 2], [4, 5], [3, 4], [6, 3], [1, 5], [1, 1], [5, 2], [5, 5], [2, 1], [2, 2], [2, 3], [2, 4], [3, 1], [3, 2], [3, 5], [1, 3]], dtype=jnp.int32)
        db = jnp.stack(((jnp.arange(32) % 8) * 8, (jnp.arange(32) // 8) * 8), axis=1).astype(jnp.int32)
        cid = jnp.arange(64, dtype=jnp.int32)
        yy, xx = jnp.arange(32)[:, None], jnp.arange(64)[None, :]
        tile = jnp.where(((xx % 8) >= 5) & ((yy % 8) >= 2) & ((yy % 8) <= 5), 1, 0)
        tile = jnp.where(((xx % 8) >= 5) & ((yy % 8) == 3), 2, tile)
        tile = jnp.where(((xx % 8) == 1) & ((yy % 8) == 1), 3, tile)
        tile = jnp.where(((xx % 8) == 1) & ((yy % 8) == 5), 4, tile)
        tile = jnp.where(((xx % 8) == 3) & ((yy % 8) == 4), 5, tile)
        tile = jnp.where(((xx % 8) == 5) & ((yy % 8) == 5), 6, tile)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False),
            'pos': base + starts[slot], 'active': jnp.ones(512, jnp.bool_), 'crew': crew,
            'tool': jnp.where(slot < 2, 2, jnp.where(slot == 2, 1, slot % 4)).astype(jnp.int32),
            'oxygen': jnp.full(512, 12, jnp.int32), 'blocks': jnp.zeros(512, jnp.int32),
            'carry': jnp.full(512, -1, jnp.int32), 'warning': jnp.zeros(512, jnp.int32),
            'knowledge': jnp.full(512, -1, jnp.int32), 'event': jnp.zeros(512, jnp.int32),
            'rescued': jnp.zeros(512, jnp.bool_), 'actor_visible': jnp.zeros((512, 512), jnp.bool_),
            'dock_pos': db + jnp.array([4, 3], jnp.int32), 'dock_active': jnp.ones(32, jnp.bool_),
            'dock_visible': jnp.zeros((512, 32), jnp.bool_), 'water': jnp.full(32, 2, jnp.int32),
            'support': jnp.zeros(32, jnp.int32), 'shelter': jnp.zeros(32, jnp.bool_),
            'valve': jnp.zeros(32, jnp.bool_), 'charge': jnp.full(32, 4, jnp.int32),
            'bridge': jnp.zeros(32, jnp.bool_), 'delivered': jnp.zeros(32, jnp.int32),
            'forecast': jax.random.randint(key, (32,), 0, 3, dtype=jnp.int32),
            'pumped': jnp.zeros(32, jnp.bool_), 'flood_loss': jnp.zeros(32, jnp.int32),
            'brake_pos': db[cid // 2] + jnp.stack((jnp.full(64, 4), jnp.where(cid % 2 == 0, 2, 5)), axis=1),
            'brake_active': jnp.ones(64, jnp.bool_), 'brake_visible': jnp.zeros((512, 64), jnp.bool_),
            'held': jnp.zeros(64, jnp.bool_), 'cargo_pos': db[cid // 2] + jnp.stack((jnp.full(64, 6), 3 + cid % 2), axis=1),
            'cargo_active': jnp.ones(64, jnp.bool_), 'cargo_visible': jnp.zeros((512, 64), jnp.bool_),
            'carrier': jnp.full(64, -1, jnp.int32), 'owner': cid // 2, 'rust': jnp.zeros(64, jnp.int32),
            'saved': jnp.zeros(64, jnp.bool_), 'lost': jnp.zeros(64, jnp.bool_), 'tile': tile.astype(jnp.int32)
        }
        s = self.refresh(s)
        return self.observe(s), s

    def context(self, s):
        p = s['pos']
        region = (p[:, 1] // 8) * 8 + p[:, 0] // 8
        local = p % 8
        return region, local

    def legal_actions(self, s):
        region, p = self.context(s)
        def at(xy):
            return jnp.sum(jnp.abs(p - jnp.array(xy)), axis=1) <= 1
        moves = s['pos'][:, None, :] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[None, :, :]
        bounded = jnp.all((moves >= 0) & (moves <= jnp.array([63, 31])), axis=-1)
        nearcargo = jnp.any((self.distance(s['pos'], s['cargo_pos']) <= 1) & s['cargo_active'][None, :] & (s['carrier'][None, :] < 0), axis=1)
        nearperson = (self.distance(s['pos'], s['pos']) <= 1) & ~jnp.eye(512, dtype=jnp.bool_)
        m = jnp.stack((jnp.ones(512, jnp.bool_), bounded[:, 0], bounded[:, 1], bounded[:, 2], bounded[:, 3],
                       at([1, 5]) & (s['tool'] == 0) & (s['blocks'] < 3),
                       ((at([6, 3]) & (s['support'][region] < 3)) | (at([1, 1]) & ~s['shelter'][region])) & (s['blocks'] > 0),
                       at([3, 4]) & (s['tool'] == 1) & (s['water'][region] > 0) & (s['charge'][region] > 0),
                       at([5, 5]), jnp.any(self.distance(s['pos'], s['brake_pos']) == 0, axis=1) & (s['tool'] == 2),
                       nearcargo & (s['carry'] < 0), s['carry'] >= 0, jnp.any(nearperson, axis=1),
                       jnp.any(nearperson & (s['oxygen'][None, :] == 0), axis=1),
                       at([1, 1]) & (s['carry'] >= 0), s['knowledge'] >= 0), axis=1)
        live = ~s['done'] & (s['oxygen'] > 0)
        return m & ((jnp.arange(16)[None, :] == 0) | live[:, None])

    def advance(self, key, s, actions):
        ids = jnp.arange(512, dtype=jnp.int32)
        safe = jnp.clip(actions, 0, 15)
        a = jnp.where((actions == safe) & self.legal_actions(s)[ids, safe], safe, 0)
        region, lp = self.context(s)
        def at(xy):
            return jnp.sum(jnp.abs(lp - jnp.array(xy)), axis=1) <= 1
        # HOLD precedes movement: both distant brakes are needed every tick.
        held = jnp.any((self.distance(s['pos'], s['brake_pos']) == 0) & (a[:, None] == 9), axis=0)
        build_bridge = (a == 6) & at([6, 3])
        add_support = jnp.minimum(3 - s['support'], jnp.sum(jax.nn.one_hot(region, 32, dtype=jnp.int32) * build_bridge[:, None], axis=0))
        support = s['support'] + add_support
        shelter_build = (a == 6) & at([1, 1])
        shelter_new = jnp.any((region[:, None] == jnp.arange(32)[None, :]) & shelter_build[:, None], axis=0) & ~s['shelter']
        bridge = (held[0::2] & held[1::2]) | (support == 3)
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[jnp.minimum(a, 4)]
        proposed = s['pos'] + jnp.where(((a >= 1) & (a <= 4))[:, None], delta, 0)
        nr = (proposed[:, 1] // 8) * 8 + proposed[:, 0] // 8
        onbridge = ((proposed[:, 0] % 8) >= 5) & ((proposed[:, 1] % 8) == 3)
        cross = ~onbridge | bridge[nr] | (s['water'][nr] <= 1)
        pos = jnp.where(cross[:, None], proposed, s['pos'])
        # A pump spends local charge and pushes actual water into the paired adjacent dock.
        pumped = jnp.any((region[:, None] == jnp.arange(32)[None, :]) & (a[:, None] == 7), axis=0)
        toggles = jnp.sum((region[:, None] == jnp.arange(32)[None, :]) & (a[:, None] == 8), axis=0) % 2
        valve = s['valve'] ^ (toggles == 1)
        neighbor = jnp.arange(32) + jnp.where(jnp.arange(32) % 2 == 0, 1, -1)
        incoming = pumped[neighbor].astype(jnp.int32)
        surge_tick = ((s['t'] + 1) % 32) == 0
        ebb_tick = ((s['t'] + 1) % 16) == 0
        water = jnp.clip(s['water'] - pumped.astype(jnp.int32) + incoming * (~valve).astype(jnp.int32) + surge_tick * s['forecast'] - ebb_tick.astype(jnp.int32), 0, 4)
        charge = jnp.clip(s['charge'] - pumped.astype(jnp.int32) + incoming * valve.astype(jnp.int32) + ebb_tick.astype(jnp.int32), 0, 8)
        # Cargo arbitration uses the smallest actor ID; one physical crate per carrier.
        eligible = (self.distance(pos, s['cargo_pos']) <= 1) & s['cargo_active'][None, :] & (s['carrier'][None, :] < 0)
        target = jnp.argmin(jnp.where(eligible, jnp.arange(64)[None, :], 1000), axis=1)
        claim = (a == 10) & eligible[ids, target]
        winner = jnp.min(jnp.where((target[:, None] == jnp.arange(64)[None, :]) & claim[:, None], ids[:, None], 512), axis=0)
        picked = winner < 512
        carrier = jnp.where(picked, winner, s['carrier'])
        owner = jnp.where(picked, s['crew'][jnp.minimum(winner, 511)], s['owner'])
        ci = jnp.maximum(carrier, 0)
        delivered = (carrier >= 0) & (a[ci] == 14)
        dropped = (carrier >= 0) & (a[ci] == 11)
        cargo_pos = jnp.where((carrier >= 0)[:, None], pos[ci], s['cargo_pos'])
        cr = (cargo_pos[:, 1] // 8) * 8 + cargo_pos[:, 0] // 8
        wet = ((cargo_pos[:, 0] % 8) >= 5) & (water[cr] >= 3)
        rust = jnp.clip(s['rust'] + (wet & (((s['t'] + 1) % 8) == 0) & s['cargo_active']).astype(jnp.int32), 0, 8)
        lost_now = (rust == 8) & s['cargo_active'] & ~delivered
        carrier = jnp.where(delivered | dropped | lost_now, -1, carrier)
        carry = jnp.max(jnp.where(carrier[None, :] == ids[:, None], jnp.arange(64)[None, :], -1), axis=1).astype(jnp.int32)
        gains = jnp.sum(jax.nn.one_hot(owner, 32, dtype=jnp.int32) * delivered[:, None], axis=0)
        losses = jnp.sum(jax.nn.one_hot(owner, 32, dtype=jnp.int32) * lost_now[:, None], axis=0)
        # Reciprocal neighboring lenders swap tools; rescue pays the helper and recipient.
        adjacent = (self.distance(pos, pos) <= 1) & ~jnp.eye(512, dtype=jnp.bool_)
        partner = jnp.argmin(jnp.where(adjacent & (a[None, :] == 12), ids[None, :], 1000), axis=1)
        swap = (a == 12) & adjacent[ids, partner] & (a[partner] == 12) & (partner[partner] == ids)
        tool = jnp.where(swap, s['tool'][partner], s['tool'])
        target_rescue = jnp.argmin(jnp.where(adjacent & (s['oxygen'][None, :] == 0), ids[None, :], 1000), axis=1)
        helpers = (a == 13) & adjacent[ids, target_rescue] & (s['oxygen'][target_rescue] == 0)
        rescuer = jnp.min(jnp.where((target_rescue[:, None] == ids[None, :]) & helpers[:, None], ids[:, None], 512), axis=0)
        rescued = rescuer < 512
        helper_paid = jnp.any((rescuer[None, :] == ids[:, None]) & rescued[None, :], axis=1)
        pr = (pos[:, 1] // 8) * 8 + pos[:, 0] // 8
        submerged = ((pos[:, 0] % 8) >= 5) & ((pos[:, 1] % 8) >= 2) & ((pos[:, 1] % 8) <= 5) & (water[pr] > 1) & ~(((pos[:, 1] % 8) == 3) & bridge[pr])
        oxygen = jnp.clip(s['oxygen'] + jnp.where(submerged, -1, 1), 0, 12)
        oxygen = jnp.where(rescued, 8, oxygen)
        warning = jnp.where(a == 15, s['knowledge'] + 1, 0).astype(jnp.int32)
        hear = adjacent & (warning[None, :] > 0)
        speaker = jnp.argmax(hear, axis=1)
        knowledge = jnp.where(jnp.any(hear, axis=1), warning[speaker] - 1, s['knowledge'])
        home_read = jnp.sum(jnp.abs((pos % 8) - jnp.array([1, 1])), axis=1) <= 1
        forecast = jnp.where(surge_tick, jax.random.randint(key, (32,), 0, 3, dtype=jnp.int32), s['forecast'])
        knowledge = jnp.where(home_read, forecast[pr], knowledge)
        blocks = jnp.clip(s['blocks'] + (a == 5).astype(jnp.int32) - (a == 6).astype(jnp.int32), 0, 3)
        terminal = jnp.all(s['saved'] | delivered | s['lost'] | lost_now)
        t = s['t'] + jnp.array(1, jnp.int32)
        done = terminal | (t >= 1024)
        reward = -0.005 * (a != 0) - 0.01 * submerged + 0.125 * gains[s['crew']] - 0.125 * losses[s['crew']] + 0.05 * shelter_new[s['crew']] + 0.2 * helper_paid + 0.1 * rescued
        reward = reward + 0.001 * jnp.sum((support == 3) & (s['support'] < 3))
        n = dict(s)
        n['t'], n['done'], n['terminated'] = t, done, terminal
        n['pos'], n['tool'], n['oxygen'], n['blocks'], n['carry'] = pos, tool, oxygen, blocks, carry
        n['warning'], n['knowledge'], n['rescued'] = warning, knowledge, rescued
        n['event'] = jnp.where(helper_paid, 17, jnp.where(swap, 16, a)).astype(jnp.int32)
        n['water'], n['support'], n['shelter'], n['valve'], n['charge'], n['bridge'] = water, support, s['shelter'] | shelter_new, valve, charge, bridge
        n['delivered'], n['forecast'], n['pumped'], n['flood_loss'], n['held'] = s['delivered'] + gains, forecast, pumped, losses, held
        n['cargo_pos'], n['carrier'], n['owner'], n['rust'] = cargo_pos, carrier, owner, rust
        n['saved'], n['lost'] = s['saved'] | delivered, s['lost'] | lost_now
        n['cargo_active'] = ~(n['saved'] | n['lost'])
        return self.refresh(n), reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(512, jnp.float32)), lambda _: self.advance(key, state, actions), operand=None)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': (n['t'] >= 1024) & ~n['terminated']}
        return self.observe(n), n, reward, jnp.full(512, n['done'], jnp.bool_), info

    def encode(self, s, posname, actname, visname, count, radius, values):
        visible, ix = self.sight(s['pos'], s[posname], s[actname], count, radius)
        mask = jnp.take_along_axis(visible, ix, axis=1)
        data = jnp.concatenate((jnp.ones((s[posname].shape[0], 1)), jnp.arange(s[posname].shape[0])[:, None], s[posname], values), axis=1)
        return jnp.where(mask[:, :, None], data[ix], -1).reshape((512, -1))

    def observe(self, s):
        selfpart = jnp.stack((jnp.arange(512), s['crew'], s['tool'], s['pos'][:, 0], s['pos'][:, 1], s['oxygen'], s['blocks'], s['carry'], s['knowledge'], s['warning'], s['event'], s['rescued']), axis=1)
        actors = self.encode(s, 'pos', 'active', 'actor_visible', 8, 5, jnp.stack((s['crew'], s['tool'], s['oxygen'], s['blocks'], s['carry'], s['warning'], s['event'], s['rescued']), axis=1))
        docks = self.encode(s, 'dock_pos', 'dock_active', 'dock_visible', 4, 7, jnp.stack((s['water'], s['support'], s['shelter'], s['valve'], s['charge'], s['bridge'], s['delivered'], s['pumped'], s['flood_loss']), axis=1))
        cargo = self.encode(s, 'cargo_pos', 'cargo_active', 'cargo_visible', 4, 5, jnp.stack((s['carrier'], s['owner'], s['rust']), axis=1))
        brakes = self.encode(s, 'brake_pos', 'brake_active', 'brake_visible', 4, 5, s['held'][:, None])
        offsets = jnp.stack(((jnp.arange(25) % 5) - 2, (jnp.arange(25) // 5) - 2), axis=1)
        q = s['pos'][:, None, :] + offsets[None, :, :]
        inside = jnp.all((q >= 0) & (q <= jnp.array([63, 31])), axis=-1)
        patch = jnp.where(inside, s['tile'][jnp.clip(q[:, :, 1], 0, 31), jnp.clip(q[:, :, 0], 0, 63)], -1)
        public = jnp.broadcast_to(jnp.stack((s['t'], s['done'], s['terminated'])), (512, 3))
        padding = jnp.zeros((512, 36))
        return jnp.concatenate((selfpart, actors, docks, cargo, brakes, patch, public, padding), axis=1).astype(jnp.float32)
