"""Orchard Lantern Commons. Original rural ecology kernel, authored without execution."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 96
    observation_size = 297
    num_actions = 20
    max_steps = 896

    def distance(self, p, q):
        return jnp.sum(jnp.abs(p[:, None, :] - q[None, :, :]), axis=-1)

    def sight(self, p, q, active, count, radius):
        d = self.distance(p, q)
        order = jnp.argsort(jnp.where(active[None, :], d, 10000) * (q.shape[0] + 1) + jnp.arange(q.shape[0])[None, :], axis=1)
        return (jnp.argsort(order, axis=1) < count) & (d <= radius) & active[None, :], order[:, :count]

    def refresh(self, s):
        s = dict(s)
        for name in ('t', 'pos', 'house', 'tool', 'wood', 'basket', 'quality', 'hive', 'knowledge', 'message', 'event', 'recipient', 'plot_pos', 'water', 'canopy', 'fruit', 'ripe', 'variety', 'pollination', 'health', 'sales', 'forecast', 'hive_pos', 'hive_carrier', 'flower_target', 'rack_pos', 'rack_load', 'rack_quality', 'breakage', 'fixture_pos', 'kind', 'fixture_health', 'tile'):
            s[name] = s[name].astype(jnp.int32)
        s['actor_visible'], _ = self.sight(s['pos'], s['pos'], s['active'], 8, 4)
        s['actor_visible'] = s['actor_visible'] | jnp.eye(96, dtype=jnp.bool_)
        s['plot_visible'], _ = self.sight(s['pos'], s['plot_pos'], s['plot_active'], 4, 7)
        s['hive_visible'], _ = self.sight(s['pos'], s['hive_pos'], s['hive_active'], 3, 5)
        s['rack_visible'], _ = self.sight(s['pos'], s['rack_pos'], s['rack_active'], 3, 5)
        s['fixture_visible'], _ = self.sight(s['pos'], s['fixture_pos'], s['fixture_active'], 5, 6)
        return s

    def reset(self, key):
        ids = jnp.arange(96, dtype=jnp.int32)
        house, slot = ids // 8, ids % 8
        h = jnp.arange(12, dtype=jnp.int32)
        base = jnp.stack(((h % 4) * 12, (h // 4) * 8), axis=1)
        starts = jnp.array([[3, 4], [3, 6], [7, 4], [3, 2], [4, 2], [5, 3], [6, 3], [9, 1]], jnp.int32)
        fixture_offsets = jnp.array([[7, 5], [9, 5], [9, 2]], jnp.int32)
        fi = jnp.arange(36, dtype=jnp.int32)
        yy, xx = jnp.arange(24)[:, None], jnp.arange(48)[None, :]
        tile = jnp.zeros((24, 48), jnp.int32)
        tile = jnp.where((xx % 12 == 4) & (yy % 8 == 3), 1, tile)
        tile = jnp.where((xx % 12 == 3) & (yy % 8 == 2), 2, tile)
        tile = jnp.where((xx % 12 == 7) & (yy % 8 == 5), 3, tile)
        tile = jnp.where((xx % 12 == 9) & (yy % 8 == 5), 4, tile)
        tile = jnp.where((xx % 12 == 9) & (yy % 8 == 2), 5, tile)
        tile = jnp.where((xx % 12 == 9) & (yy % 8 == 1), 6, tile)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False), 'terminated': jnp.array(False),
            'pos': base[house] + starts[slot], 'active': jnp.ones(96, jnp.bool_), 'house': house,
            'tool': jnp.where(slot < 3, 3, slot % 4).astype(jnp.int32), 'wood': jnp.zeros(96, jnp.int32),
            'basket': jnp.zeros(96, jnp.int32), 'quality': jnp.full(96, 3, jnp.int32), 'hive': jnp.full(96, -1, jnp.int32),
            'knowledge': jnp.full(96, -1, jnp.int32), 'message': jnp.zeros(96, jnp.int32),
            'event': jnp.zeros(96, jnp.int32), 'recipient': jnp.full(96, -1, jnp.int32), 'actor_visible': jnp.zeros((96, 96), jnp.bool_),
            'plot_pos': base + jnp.array([4, 3]), 'plot_active': jnp.ones(12, jnp.bool_), 'plot_visible': jnp.zeros((96, 12), jnp.bool_),
            'water': jnp.full(12, 2, jnp.int32), 'canopy': jnp.full(12, 2, jnp.int32),
            'fruit': jnp.full(12, 4, jnp.int32), 'ripe': jnp.full(12, 6, jnp.int32), 'variety': jnp.zeros(12, jnp.int32),
            'pollination': jnp.ones(12, jnp.int32), 'health': jnp.full(12, 3, jnp.int32), 'valve': jnp.zeros(12, jnp.bool_),
            'sales': jnp.zeros(12, jnp.int32), 'growth': jnp.zeros(12, jnp.bool_), 'pruned': jnp.zeros(12, jnp.bool_),
            'forecast': jax.random.randint(key, (12,), 0, 3, dtype=jnp.int32), 'community': jnp.array(False),
            'hive_pos': base + jnp.array([6, 3]), 'hive_active': jnp.ones(12, jnp.bool_), 'hive_visible': jnp.zeros((96, 12), jnp.bool_),
            'hive_carrier': jnp.full(12, -1, jnp.int32), 'flower_target': h, 'distracted': jnp.zeros(12, jnp.bool_),
            'rack_pos': base + jnp.array([3, 5]), 'rack_active': jnp.ones(12, jnp.bool_), 'rack_visible': jnp.zeros((96, 12), jnp.bool_),
            'rack_load': jnp.full(12, 2, jnp.int32), 'rack_quality': jnp.full(12, 3, jnp.int32),
            'rack_moved': jnp.zeros(12, jnp.bool_), 'breakage': jnp.zeros(12, jnp.int32),
            'fixture_pos': base[fi // 3] + fixture_offsets[fi % 3], 'fixture_active': jnp.ones(36, jnp.bool_),
            'fixture_visible': jnp.zeros((96, 36), jnp.bool_), 'kind': fi % 3,
            'fixture_health': jnp.where(fi % 3 == 0, 2, 3).astype(jnp.int32), 'held': jnp.zeros(36, jnp.bool_),
            'lit': jnp.zeros(36, jnp.bool_), 'tile': tile
        }
        s = self.refresh(s)
        return self.observe(s), s

    def context(self, s):
        pd = self.distance(s['pos'], s['plot_pos'])
        plot = jnp.argmin(pd, axis=1)
        rd = self.distance(s['pos'], s['rack_pos'])
        rack = jnp.argmin(rd, axis=1)
        fd = self.distance(s['pos'], s['fixture_pos'])
        fixture = jnp.argmin(fd, axis=1)
        return plot, pd[jnp.arange(96), plot] <= 1, rack, rd[jnp.arange(96), rack] <= 1, fixture, fd[jnp.arange(96), fixture] <= 1

    def legal_actions(self, s):
        plot, np, rack, nr, fixture, nf = self.context(s)
        pos = s['pos']
        local = pos % jnp.array([12, 8])
        def at(xy):
            return jnp.sum(jnp.abs(local - jnp.array(xy)), axis=1) <= 1
        moves = pos[:, None, :] + jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[None, :, :]
        bounded = jnp.all((moves >= 0) & (moves <= jnp.array([47, 23])), axis=-1)
        hd = self.distance(pos, s['hive_pos'])
        free_hive = jnp.any((hd <= 1) & (s['hive_carrier'][None, :] < 0), axis=1)
        stand = jnp.any((self.distance(pos, s['fixture_pos'] + jnp.array([0, -1])) == 0) & (s['kind'][None, :] < 2), axis=1)
        adjacent = (self.distance(pos, pos) <= 1) & ~jnp.eye(96, dtype=jnp.bool_)
        m = jnp.stack((jnp.ones(96, jnp.bool_), bounded[:, 0], bounded[:, 1], bounded[:, 2], bounded[:, 3],
                       np & (s['tool'] == 0) & (s['canopy'][plot] > 0) & (s['wood'] < 3), np & (s['health'][plot] < 3) & (s['wood'] > 0),
                       np & (s['fruit'][plot] > 0) & (s['ripe'][plot] >= 6) & (s['basket'] < 3) & (s['hive'] < 0),
                       np & (s['tool'] == 1) & (s['basket'] > 0) & (s['variety'][plot] == 0), at([3, 2]),
                       (s['hive'] >= 0) | (free_hive & (s['tool'] == 2) & (s['basket'] == 0)),
                       nr & (s['tool'] == 3), stand & (s['tool'] == 3), nr & (s['tool'] == 3),
                       nr & (s['basket'] > 0) & (s['rack_load'][rack] < 3), nf & (s['kind'][fixture] == 0) & (s['fixture_health'][fixture] < 3) & (s['wood'] > 0),
                       jnp.any(adjacent, axis=1), nf & (s['kind'][fixture] == 2),
                       (at([9, 1]) & (s['basket'] > 0)) | (nr & ((s['rack_pos'][rack, 0] % 12) >= 9) & (s['rack_load'][rack] > 0)),
                       at([9, 1]) | (s['knowledge'] >= 0)), axis=1)
        return m & ((jnp.arange(20)[None, :] == 0) | ~s['done'])

    def advance(self, key, s, actions):
        ids, h = jnp.arange(96, dtype=jnp.int32), jnp.arange(12, dtype=jnp.int32)
        safe = jnp.clip(actions, 0, 19)
        a = jnp.where((actions == safe) & self.legal_actions(s)[ids, safe], safe, 0)
        plot, np, rack, nr, fixture, nf = self.context(s)
        def winner(action, target, count):
            return jnp.min(jnp.where((target[:, None] == jnp.arange(count)[None, :]) & (a[:, None] == action), ids[:, None], 96), axis=0)
        prune = winner(5, plot, 12)
        pruned = prune < 96
        harvest = winner(7, plot, 12)
        harvested = harvest < 96
        graft = winner(8, plot, 12)
        grafted = graft < 96
        fruit = jnp.maximum(0, s['fruit'] - pruned.astype(jnp.int32) - harvested.astype(jnp.int32))
        canopy = jnp.maximum(0, s['canopy'] - pruned.astype(jnp.int32))
        variety = jnp.where(grafted, 1, s['variety'])
        basket = s['basket'] + ((a == 7) & (harvest[plot] == ids)).astype(jnp.int32) - ((a == 8) & (graft[plot] == ids)).astype(jnp.int32)
        quality = jnp.where((a == 7) & (harvest[plot] == ids), jnp.minimum(3, 2 + s['variety'][plot]), s['quality'])
        wood = s['wood'] + ((a == 5) & (prune[plot] == ids)).astype(jnp.int32) - ((a == 6) | (a == 15)).astype(jnp.int32)
        health = jnp.minimum(3, s['health'] + jnp.sum((plot[:, None] == h[None, :]) & (a[:, None] == 6), axis=0).astype(jnp.int32))
        region = (s['pos'][:, 1] // 8) * 4 + s['pos'][:, 0] // 12
        toggles = jnp.sum((region[:, None] == h[None, :]) & (a[:, None] == 9), axis=0) % 2
        valve = s['valve'] ^ (toggles == 1)
        neighbor = h + jnp.where(h % 2 == 0, 1, -1)
        flow = valve & (s['water'] > 0) & (((s['t'] + 1) % 4) == 0)
        rain = ((s['t'] + 1) % 32) == 0
        water = jnp.clip(s['water'] - flow.astype(jnp.int32) + flow[neighbor].astype(jnp.int32) + rain * s['forecast'] - (((s['t'] + 1) % 8) == 0).astype(jnp.int32), 0, 4)
        # Bracing and lamp switches are actual per-fixture state, before transport/ecology.
        held = jnp.any((self.distance(s['pos'], s['fixture_pos'] + jnp.array([0, -1])) == 0) & (a[:, None] == 12), axis=0) & (s['kind'] < 2)
        toggle_lamp = jnp.sum((fixture[:, None] == jnp.arange(36)[None, :]) & (a[:, None] == 17), axis=0) % 2
        lit = s['lit'] ^ (toggle_lamp == 1)
        repaired = jnp.sum((fixture[:, None] == jnp.arange(36)[None, :]) & (a[:, None] == 15), axis=0).astype(jnp.int32)
        fixture_health = jnp.minimum(3, s['fixture_health'] + repaired)
        damage = rain & (s['forecast'] == 2)
        fixture_health = jnp.maximum(0, fixture_health - ((s['kind'] == 0) & damage[jnp.arange(36) // 3]).astype(jnp.int32))
        health = jnp.maximum(0, health - damage.astype(jnp.int32))
        # Pruning physically opens a tree cell; movements use the new canopy.
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)[jnp.minimum(a, 4)]
        proposed = s['pos'] + jnp.where(((a >= 1) & (a <= 4))[:, None], delta, 0)
        blocked = jnp.any((self.distance(proposed, s['plot_pos']) == 0) & (canopy[None, :] > 0), axis=1)
        pos = jnp.where(blocked[:, None], s['pos'], proposed)
        hd = self.distance(pos, s['hive_pos'])
        eligible = (hd <= 1) & (s['hive_carrier'][None, :] < 0)
        hi = jnp.argmin(jnp.where(eligible, jnp.arange(12)[None, :], 100), axis=1)
        claim = (a == 10) & (s['hive'] < 0) & eligible[ids, hi]
        hw = jnp.min(jnp.where((hi[:, None] == h[None, :]) & claim[:, None], ids[:, None], 96), axis=0)
        hc = jnp.where(hw < 96, hw, s['hive_carrier'])
        dropping = (s['hive_carrier'] >= 0) & (a[jnp.maximum(s['hive_carrier'], 0)] == 10)
        hpos = jnp.where((hc >= 0)[:, None], pos[jnp.maximum(hc, 0)], s['hive_pos'])
        hc = jnp.where(dropping, -1, hc)
        hive = jnp.max(jnp.where(hc[None, :] == ids[:, None], h[None, :], -1), axis=1).astype(jnp.int32)
        distracted = jnp.any((self.distance(hpos, s['fixture_pos']) <= 6) & lit[None, :], axis=1)
        flowers = self.distance(hpos, s['plot_pos'])
        ft = jnp.argmin(flowers, axis=1)
        useful = (flowers[h, ft] <= 6) & ~distracted & (hc < 0)
        flower_target = jnp.where(useful, ft, -1)
        pollination = jnp.minimum(3, jnp.sum(jax.nn.one_hot(ft, 12, dtype=jnp.int32) * useful[:, None], axis=0))
        sunlight = 3 - canopy[neighbor]
        growth = (water > 0) & (health > 0) & (pollination > 0) & (sunlight > 0) & (((s['t'] + 1) % 8) == 0)
        ripe = jnp.where(grafted, 0, jnp.minimum(8, s['ripe'] + growth.astype(jnp.int32) * (1 + variety)))
        fruit = jnp.minimum(8, fruit + growth.astype(jnp.int32) * sunlight)
        canopy = jnp.minimum(3, canopy + (rain & (health > 0)).astype(jnp.int32))
        # A loaded rack needs a north and a south carrier agreeing on direction.
        align = s['pos'][:, None, 0] == s['rack_pos'][None, :, 0]
        north = align & (s['pos'][:, None, 1] == s['rack_pos'][None, :, 1] - 1)
        south = align & (s['pos'][:, None, 1] == s['rack_pos'][None, :, 1] + 1)
        east = jnp.any(north & (a[:, None] == 11), axis=0) & jnp.any(south & (a[:, None] == 11), axis=0)
        west = jnp.any(north & (a[:, None] == 13), axis=0) & jnp.any(south & (a[:, None] == 13), axis=0)
        direction = east.astype(jnp.int32) - west.astype(jnp.int32)
        x = s['rack_pos'][:, 0] + direction
        homebase = (h % 4) * 12
        bridge_pass = (x != homebase + 7) | held[h * 3] | (fixture_health[h * 3] == 3)
        door_pass = (x != homebase + 9) | held[h * 3 + 1]
        moved = (direction != 0) & (x >= homebase + 1) & (x <= homebase + 10) & bridge_pass & door_pass
        rpos = s['rack_pos'].at[:, 0].set(jnp.where(moved, x, s['rack_pos'][:, 0]))
        load = winner(14, rack, 12)
        loaded = load < 96
        load_actor = jnp.minimum(load, 95)
        amount = jnp.where(loaded, jnp.minimum(3 - s['rack_load'], basket[load_actor]), 0)
        basket = basket - jnp.where((a == 14) & (load[rack] == ids), amount[rack], 0)
        rload = s['rack_load'] + amount
        rq = jnp.where(loaded, jnp.minimum(s['rack_quality'], quality[load_actor]), s['rack_quality'])
        night = ((s['t'] + 1) % 32) >= 16
        dark_trip = moved & night & ~lit[h * 3 + 2] & (rload > 0)
        rq = jnp.maximum(0, rq - dark_trip.astype(jnp.int32))
        breakage = jnp.where((rq == 0) & (rload > 0), 1, 0).astype(jnp.int32)
        rload = jnp.maximum(0, rload - breakage)
        rq = jnp.where(rload == 0, 3, rq)
        # Market competition prices every sale using the same pre-step total supply.
        local = s['pos'] % jnp.array([12, 8])
        market = jnp.sum(jnp.abs(local - jnp.array([9, 1])), axis=1) <= 1
        hand_sale = (a == 18) & market & (basket > 0)
        rs = winner(18, rack, 12)
        rack_sale = (rs < 96) & ((rpos[:, 0] % 12) >= 9) & held[h * 3 + 1] & ~market[jnp.minimum(rs, 95)]
        quantities = jnp.sum(jax.nn.one_hot(s['house'], 12, dtype=jnp.int32) * jnp.where(hand_sale, basket, 0)[:, None], axis=0)
        quantities = quantities + jnp.where(rack_sale, rload, 0)
        values = jnp.sum(jax.nn.one_hot(s['house'], 12, dtype=jnp.float32) * jnp.where(hand_sale, basket * quality, 0)[:, None], axis=0)
        values = values + jnp.where(rack_sale, rload * rq, 0)
        price = 1.0 / (1.0 + jnp.sum(s['sales']).astype(jnp.float32) / 24.0)
        basket = jnp.where(hand_sale, 0, basket)
        rload = jnp.where(rack_sale, 0, rload)
        rq = jnp.where(rack_sale, 3, rq)
        sales = jnp.minimum(512, s['sales'] + quantities)
        adjacent = (self.distance(pos, pos) <= 1) & ~jnp.eye(96, dtype=jnp.bool_)
        partner = jnp.argmin(jnp.where(adjacent & (a[None, :] == 16), ids[None, :], 1000), axis=1)
        swap = (a == 16) & adjacent[ids, partner] & (a[partner] == 16) & (partner[partner] == ids)
        tool = jnp.where(swap, s['tool'][partner], s['tool'])
        message = jnp.where(a == 19, s['knowledge'] + 1, 0)
        hear = adjacent & (message[None, :] > 0)
        knowledge = jnp.where(jnp.any(hear, axis=1), message[jnp.argmax(hear, axis=1)] - 1, s['knowledge'])
        forecast = jnp.where(rain, jax.random.randint(key, (12,), 0, 3, dtype=jnp.int32), s['forecast'])
        knowledge = jnp.where((a == 19) & market, forecast[region], knowledge)
        community = s['community'] | jnp.all(pollination > 0)
        milestone = community & ~s['community']
        terminal = jnp.all(sales >= 8)
        t = s['t'] + jnp.array(1, jnp.int32)
        reward = -0.003 * (a != 0) + price * values[s['house']] / 24.0 - 0.1 * breakage[s['house']] + 0.01 * milestone
        n = dict(s)
        n['t'], n['done'], n['terminated'] = t, terminal | (t >= 896), terminal
        n['pos'], n['tool'], n['wood'], n['basket'], n['quality'], n['hive'] = pos, tool, jnp.clip(wood, 0, 3), basket, quality, hive
        n['knowledge'], n['message'], n['event'], n['recipient'] = knowledge, message, a, jnp.where(swap, partner, -1)
        n['water'], n['canopy'], n['fruit'], n['ripe'], n['variety'], n['pollination'], n['health'], n['valve'] = water, canopy, fruit, ripe, variety, pollination, health, valve
        n['sales'], n['growth'], n['pruned'], n['forecast'], n['community'] = sales, growth, pruned, forecast, community
        n['hive_pos'], n['hive_carrier'], n['flower_target'], n['distracted'] = hpos, hc, flower_target, distracted
        n['rack_pos'], n['rack_load'], n['rack_quality'], n['rack_moved'], n['breakage'] = rpos, rload, rq, moved, breakage
        n['fixture_health'], n['held'], n['lit'] = fixture_health, held, lit
        return self.refresh(n), reward.astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(96, jnp.float32)), lambda _: self.advance(key, state, actions), operand=None)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': (n['t'] >= 896) & ~n['terminated']}
        return self.observe(n), n, reward, jnp.full(96, n['done'], jnp.bool_), info

    def encode(self, s, posname, actname, count, radius, values):
        visible, ix = self.sight(s['pos'], s[posname], s[actname], count, radius)
        mask = jnp.take_along_axis(visible, ix, axis=1)
        data = jnp.concatenate((jnp.ones((s[posname].shape[0], 1)), jnp.arange(s[posname].shape[0])[:, None], s[posname], values), axis=1)
        return jnp.where(mask[:, :, None], data[ix], -1).reshape((96, -1))

    def observe(self, s):
        own = jnp.stack((jnp.arange(96), s['house'], s['tool'], s['pos'][:, 0], s['pos'][:, 1], s['wood'], s['basket'], s['quality'], s['hive'], s['knowledge'], s['message'], s['event'], s['recipient'], s['sales'][s['house']]), axis=1)
        actors = self.encode(s, 'pos', 'active', 8, 4, jnp.stack((s['house'], s['tool'], s['wood'], s['basket'], s['quality'], s['hive'], s['message'], s['event'], s['recipient']), axis=1))
        plots = self.encode(s, 'plot_pos', 'plot_active', 4, 7, jnp.stack((s['water'], s['canopy'], s['fruit'], s['ripe'], s['variety'], s['pollination'], s['health'], s['valve'], s['sales'], s['growth'], s['pruned']), axis=1))
        hives = self.encode(s, 'hive_pos', 'hive_active', 3, 5, jnp.stack((s['hive_carrier'], s['flower_target'], s['distracted']), axis=1))
        racks = self.encode(s, 'rack_pos', 'rack_active', 3, 5, jnp.stack((s['rack_load'], s['rack_quality'], s['rack_moved'], s['breakage']), axis=1))
        fixtures = self.encode(s, 'fixture_pos', 'fixture_active', 5, 6, jnp.stack((s['kind'], s['fixture_health'], s['held'], s['lit']), axis=1))
        offsets = jnp.stack((jnp.arange(25) % 5 - 2, jnp.arange(25) // 5 - 2), axis=1)
        q = s['pos'][:, None, :] + offsets[None, :, :]
        inside = jnp.all((q >= 0) & (q <= jnp.array([47, 23])), axis=-1)
        patch = jnp.where(inside, s['tile'][jnp.clip(q[:, :, 1], 0, 23), jnp.clip(q[:, :, 0], 0, 47)], -1)
        public = jnp.broadcast_to(jnp.stack((s['t'], s['done'], s['terminated'], s['community'])), (96, 4))
        padding = jnp.zeros((96, 5))
        return jnp.concatenate((own, actors, plots, hives, racks, fixtures, patch, public, padding), axis=1).astype(jnp.float32)
