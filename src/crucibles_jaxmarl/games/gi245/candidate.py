"""Lantern Ferry Supper. Original source; scripts are unexecuted hypotheses."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 12
    num_actions = 16
    observation_size = 466
    max_steps = 512

    def refresh(self, s):
        p = s['pos']
        av = jnp.max(jnp.abs(p[:, None] - p[None, :]), axis=2) <= 3
        sv = jnp.max(jnp.abs(p[:, None] - s['station_pos'][None]), axis=2) <= 3
        cv = jnp.max(jnp.abs(p[:, None] - s['counter_pos'][None]), axis=2) <= 3
        fv = jnp.max(jnp.abs(p[:, None] - s['thermal_pos'][None]), axis=2) <= 3
        x = jnp.arange(23, dtype=jnp.int32)[None, :]
        y = jnp.arange(11, dtype=jnp.int32)[:, None]
        gap = (x == 7) | (x == 15)
        tiles = jnp.where(gap, 1, 0) + jnp.zeros((11, 23), jnp.int32)
        tiles = tiles.at[s['counter_pos'][:, 1], s['counter_pos'][:, 0]].set(2)
        mv = (jnp.abs(x[None] - p[:, 0, None, None]) <= 3) & (jnp.abs(y[None] - p[:, 1, None, None]) <= 3)
        return dict(s, agent_vis=av, station_vis=sv, counter_vis=cv, thermal_vis=fv, tiles=tiles, map_vis=mv)

    def reset(self, key):
        p = jnp.array([[2, 1], [3, 2], [4, 3], [6, 2], [8, 2], [10, 5], [10, 7], [14, 8], [16, 8], [18, 2], [20, 3], [21, 8]], jnp.int32)
        ids = jnp.arange(18, dtype=jnp.int32)
        typ = ids % 6
        layout = jnp.array([[1, 1], [1, 3], [3, 1], [4, 4], [2, 7], [5, 8]], jnp.int32)
        sp = layout[typ] + jnp.stack(((ids // 6) * 8, jnp.zeros((18,), jnp.int32)), axis=1)
        pref = jax.random.randint(key, (3,), 1, 3, dtype=jnp.int32)
        order = jnp.zeros((18,), jnp.int32).at[5].set(pref[0])
        s = {'t': jnp.array(0, jnp.int32), 'episode_done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
             'pos': p, 'active': jnp.ones((12,), jnp.bool_), 'team': jnp.arange(12, dtype=jnp.int32) // 4,
             'item': jnp.zeros((12,), jnp.int32), 'allergen': jnp.zeros((12,), jnp.bool_), 'age': jnp.zeros((12,), jnp.int32),
             'ticket': jnp.zeros((12,), jnp.int32), 'station_pos': sp, 'station_active': jnp.ones((18,), jnp.bool_),
             'station_type': typ, 'stock': jnp.where(typ < 2, 16, jnp.where(typ == 4, 4, 0)).astype(jnp.int32),
             'recipe': jnp.zeros((18,), jnp.int32), 'progress': jnp.zeros((18,), jnp.float32),
             'contamination': jnp.zeros((18,), jnp.bool_), 'order': order,
             'deadline': jnp.where(order > 0, 96, 0).astype(jnp.int32), 'remaining': jnp.array([3, 4, 4], jnp.int32),
             'preference': pref, 'counter_pos': jnp.array([[7, 2], [15, 8]], jnp.int32), 'counter_active': jnp.ones((2,), jnp.bool_),
             'counter_shift': jnp.zeros((2,), jnp.int32), 'thermal_pos': jnp.array([[11, 5], [11, 7]], jnp.int32),
             'thermal_active': jnp.ones((2,), jnp.bool_), 'thermal_kind': jnp.array([0, 1], jnp.int32),
             'thermal_level': jnp.array([2.0, 0.0], jnp.float32), 'thermal_stock': jnp.array([36, 0], jnp.int32),
             'served': jnp.zeros((18,), jnp.bool_), 'missed': jnp.zeros((18,), jnp.bool_),
             'spill': jnp.zeros((12,), jnp.bool_), 'burn': jnp.zeros((12,), jnp.bool_), 'completed': jnp.zeros((3,), jnp.int32),
             'agent_vis': jnp.zeros((12, 12), jnp.bool_), 'station_vis': jnp.zeros((12, 18), jnp.bool_),
             'counter_vis': jnp.zeros((12, 2), jnp.bool_), 'thermal_vis': jnp.zeros((12, 2), jnp.bool_),
             'tiles': jnp.zeros((11, 23), jnp.int32), 'map_vis': jnp.zeros((12, 11, 23), jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def station(self, s, p):
        distances = jnp.sum(jnp.abs(p[:, None] - s['station_pos'][None]), axis=2)
        k = jnp.argmin(distances, axis=1)
        return k, jnp.min(distances, axis=1) <= 1

    def legal_actions(self, s):
        p = s['pos']
        d = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = p[:, None] + d[None]
        ok = (q[:, :, 0] >= 0) & (q[:, :, 0] < 23) & (q[:, :, 1] >= 0) & (q[:, :, 1] < 11)
        x = jnp.clip(q[:, :, 0], 0, 22)
        y = jnp.clip(q[:, :, 1], 0, 10)
        walk = ok & (s['tiles'][y, x] != 1)
        k, near = self.station(s, p)
        typ = s['station_type'][k]
        it = s['item']
        pickup = (((typ < 2) | (typ == 4) | (typ == 5)) & (it == 0) & (s['stock'][k] > 0)) | ((typ == 3) & (it == 9) & (s['recipe'][k] > 0) & (s['progress'][k] >= 4))
        chop = (typ == 2) & ((it == 1) | (it == 2))
        wash = ((typ == 2) & (it == 0)) | ((typ == 4) & (it == 7))
        load = (typ == 3) & ((it == 3) | (it == 4)) & (s['recipe'][k] == 0)
        serve = (typ == 5) & ((it == 5) | (it == 6)) & (s['order'][k] > 0)
        dist = jnp.sum(jnp.abs(p[:, None] - s['thermal_pos'][None]), axis=2)
        dryload = (dist[:, 1] <= 1) & (it == 8) & (s['thermal_stock'][1] == 0)
        drytake = (dist[:, 1] <= 1) & (it == 0) & (s['thermal_stock'][1] == 1) & (s['thermal_level'][1] >= 4)
        endpoints = (jnp.abs(p[:, None, 0] - s['counter_pos'][None, :, 0]) == 1) & (p[:, None, 1] == s['counter_pos'][None, :, 1])
        down = jnp.any(endpoints & (s['counter_pos'][None, :, 1] < 9), axis=1) & (it == 0)
        up = jnp.any(endpoints & (s['counter_pos'][None, :, 1] > 1), axis=1) & (it == 0)
        rest = jnp.stack((near & pickup | drytake, near & chop, near & wash,
             near & load | dryload, near & serve, (dist[:, 0] <= 1) & (s['thermal_stock'][0] > 0),
             dist[:, 0] <= 1, down, up, near & (typ == 5) & (s['order'][k] > 0), (it > 0) | (s['ticket'] > 0)), axis=1)
        return (jnp.concatenate((walk, rest), axis=1) & ~s['episode_done']).at[:, 0].set(True)

    def advance(self, s, actions):
        ids = jnp.arange(12, dtype=jnp.int32)
        aa = jnp.clip(actions, 0, 15)
        act = jnp.where((aa == actions) & self.legal_actions(s)[ids, aa], aa, 0)
        d = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        proposal = s['pos'] + d[jnp.minimum(act, 4)] * (act <= 4)[:, None]
        occupied = jnp.any(jnp.all(proposal[:, None] == s['pos'][None], axis=2) & (ids[:, None] != ids[None]), axis=1)
        rank = (ids - s['t']) % 12
        tied = jnp.all(proposal[:, None] == proposal[None], axis=2) & (rank[None, :] < rank[:, None])
        p = jnp.where((~occupied & ~jnp.any(tied, axis=1))[:, None], proposal, s['pos'])
        def shift_one(k, carry):
            pos, cp, shifts = carry
            left = jnp.all(pos == cp[k] + jnp.array([-1, 0], jnp.int32), axis=1)
            right = jnp.all(pos == cp[k] + jnp.array([1, 0], jnp.int32), axis=1)
            down = jnp.any(left & (act == 12)) & jnp.any(right & (act == 12))
            up = jnp.any(left & (act == 13)) & jnp.any(right & (act == 13))
            dy = down.astype(jnp.int32) - up.astype(jnp.int32)
            participants = (left | right) & ((act == 12) | (act == 13))
            dest = pos + jnp.stack((jnp.zeros((12,), jnp.int32), jnp.full((12,), dy, jnp.int32)), axis=1)
            blocked = jnp.any(jnp.all(dest[:, None] == pos[None], axis=2) & participants[:, None] & ~participants[None, :])
            dy = jnp.where(blocked, 0, dy)
            pos = pos.at[:, 1].add(participants.astype(jnp.int32) * dy)
            cp = cp.at[k, 1].add(dy)
            return pos, cp, shifts.at[k].set(dy)
        p, cp, shifts = jax.lax.fori_loop(0, 2, shift_one, (p, s['counter_pos'], jnp.zeros((2,), jnp.int32)))
        k, near = self.station(s, p)
        typ = s['station_type'][k]
        it = s['item']
        station_pick = (((typ < 2) | (typ == 4) | (typ == 5)) & (it == 0) & (s['stock'][k] > 0)) | ((typ == 3) & (it == 9) & (s['recipe'][k] > 0) & (s['progress'][k] >= 4))
        station_chop = (typ == 2) & ((it == 1) | (it == 2))
        station_wash = ((typ == 2) & (it == 0)) | ((typ == 4) & (it == 7))
        station_load = (typ == 3) & ((it == 3) | (it == 4)) & (s['recipe'][k] == 0)
        station_serve = (typ == 5) & ((it == 5) | (it == 6)) & (s['order'][k] > 0)
        candidates = near & (((act == 5) & station_pick) | ((act == 6) & station_chop) | ((act == 7) & station_wash) | ((act == 8) & station_load) | ((act == 9) & station_serve))
        costs = jnp.where(candidates[:, None] & (k[:, None] == jnp.arange(18)[None]), rank[:, None], 13)
        win = candidates & (rank == jnp.min(costs, axis=0)[k])
        typ = s['station_type'][k]
        it = s['item']
        pickup = win & (act == 5)
        chop = win & (act == 6)
        wash = win & (act == 7)
        load = win & (act == 8)
        serve = win & (act == 9)
        correct = serve & (it - 4 == s['order'][k]) & ((s['order'][k] != 1) | ~s['allergen'])
        bad = serve & ~correct
        item = jnp.where(pickup, jnp.where(typ < 2, typ + 1, jnp.where(typ == 4, 9, jnp.where(typ == 5, 7, s['recipe'][k] + 2))), it)
        item = jnp.where(chop, it + 2, item)
        item = jnp.where(wash & (typ == 4), 8, item)
        item = jnp.where(load | serve, jnp.where(bad, 7, 0), item).astype(jnp.int32)
        allergen = jnp.where(chop, s['allergen'] | s['contamination'][k] | (it == 2), s['allergen'])
        allergen = jnp.where(pickup, jnp.where(typ == 3, s['contamination'][k], typ == 1), allergen)
        allergen = allergen & (item > 0) & (item < 7)
        stockdelta = -((pickup & (typ != 3)).astype(jnp.int32)) + correct.astype(jnp.int32)
        stock = s['stock'].at[k].add(stockdelta)
        rdelta = jnp.where(load, it, jnp.where(pickup & (typ == 3), -s['recipe'][k], 0))
        recipe = s['recipe'].at[k].add(rdelta)
        change = jnp.zeros((18,), jnp.int32).at[k].add((load | (pickup & (typ == 3))).astype(jnp.int32)) > 0
        cchange = jnp.zeros((18,), jnp.int32).at[k].add((chop | (wash & (typ == 2)) | load).astype(jnp.int32)) > 0
        cvalue = jnp.zeros((18,), jnp.int32).at[k].add(((chop & ((it == 2) | s['contamination'][k])) | (load & s['allergen'])).astype(jnp.int32)) > 0
        contamination = jnp.where(cchange, cvalue, s['contamination'])
        furnace_near = jnp.sum(jnp.abs(p - s['thermal_pos'][0]), axis=1) <= 1
        stoke = jnp.any((act == 10) & furnace_near)
        vent = jnp.any((act == 11) & furnace_near)
        heat = jnp.clip(s['thermal_level'][0] - 0.05 + 2 * stoke - 2 * vent, 0, 10)
        progress = jnp.where(change, 0.0, s['progress'])
        progress = jnp.minimum(progress + (recipe > 0) * jnp.where(heat > 7, 1.5, heat / 6), 8).astype(jnp.float32)
        burntpot = (progress >= 8) & (recipe > 0)
        recipe = jnp.where(burntpot, 0, recipe).astype(jnp.int32)
        drynear = jnp.sum(jnp.abs(p - s['thermal_pos'][1]), axis=1) <= 1
        dryoffer = drynear & (act == 8) & (it == 8)
        drypick = drynear & (act == 5) & (it == 0) & ~pickup & (s['thermal_stock'][1] == 1) & (s['thermal_level'][1] >= 4)
        dw = jnp.argmin(jnp.where(dryoffer | drypick, rank, 13))
        dload = jnp.any(dryoffer) & dryoffer[dw]
        dtake = jnp.any(drypick) & drypick[dw]
        item = item.at[dw].set(jnp.where(dload, 0, jnp.where(dtake, 9, item[dw])))
        drycount = s['thermal_stock'][1] + dload.astype(jnp.int32) - dtake.astype(jnp.int32)
        dryprogress = jnp.where(dload | dtake, 0.0, s['thermal_level'][1])
        dryprogress = jnp.minimum(dryprogress + drycount * heat / 5, 4)
        served = jnp.zeros((18,), jnp.int32).at[k].add(correct.astype(jnp.int32)) > 0
        deadline = jnp.maximum(s['deadline'] - (s['order'] > 0).astype(jnp.int32), 0)
        missed = (s['order'] > 0) & (deadline == 0) & ~served
        order = jnp.where(served | missed, 0, s['order']).astype(jnp.int32)
        windows = jnp.array([5, 11, 17], jnp.int32)
        arrival = ((s['t'] + 1) % 48 == jnp.array([0, 16, 32], jnp.int32)) & (order[windows] == 0) & (s['remaining'] > 0)
        nextorder = 1 + ((s['preference'] - 1 + (4 - s['remaining'])) % 2)
        order = order.at[windows].set(jnp.where(arrival, nextorder, order[windows]))
        deadline = deadline.at[windows].set(jnp.where(arrival, 96, deadline[windows]))
        remaining = s['remaining'] - arrival.astype(jnp.int32)
        ticket = jnp.where((act == 14) & near, s['order'][k], s['ticket'])
        distance = jnp.sum(jnp.abs(p[:, None] - p[None]), axis=2)
        can = (act[:, None] == 15) & (act[None, :] == 0) & (s['item'][None, :] == 0) & (s['ticket'][None, :] == 0) & (distance <= 1) & (ids[:, None] != ids[None])
        dest = jnp.argmin(jnp.where(can, ids[None, :], 13), axis=1)
        pcost = jnp.where(jnp.any(can, axis=1)[:, None] & (dest[:, None] == ids[None]), rank[:, None], 13)
        passed = jnp.any(can, axis=1) & (rank == jnp.min(pcost, axis=0)[dest])
        recvitem = jnp.zeros((12,), jnp.int32).at[dest].add(jnp.where(passed, item, 0))
        recvticket = jnp.zeros((12,), jnp.int32).at[dest].add(jnp.where(passed, ticket, 0))
        recvallergy = jnp.zeros((12,), jnp.int32).at[dest].add((passed & allergen).astype(jnp.int32)) > 0
        received = jnp.zeros((12,), jnp.int32).at[dest].add(passed.astype(jnp.int32)) > 0
        item = jnp.where(passed, 0, jnp.where(received, recvitem, item))
        ticket = jnp.where(passed, 0, jnp.where(received, recvticket, ticket))
        allergen = jnp.where(passed, False, jnp.where(received, recvallergy, allergen))
        age = jnp.where((item == 5) | (item == 6), s['age'] + 1, 0).astype(jnp.int32)
        recvg = jnp.zeros((12,), jnp.int32).at[dest].add(jnp.where(passed, s['age'] + 1, 0))
        age = jnp.where(received & ((item == 5) | (item == 6)), recvg, age)
        spill = (age > 32) | bad
        item = jnp.where(spill, 7, item).astype(jnp.int32)
        age = jnp.where(spill, 0, age)
        burn = (heat > 7) & near & (typ == 3)
        completed = s['completed'] + served[windows].astype(jnp.int32)
        t = s['t'] + jnp.array(1, jnp.int32)
        terminated = jnp.all(remaining == 0) & ~jnp.any(order > 0)
        truncated = (t >= 512) & ~terminated
        reward = 0.8 * correct + 0.2 * served[windows][s['team']] - 0.15 * spill - 0.04 * burn - 0.002 * (act > 0) - 0.05 * missed[windows][s['team']] - 0.002 * jnp.mean((stock[windows] > 3).astype(jnp.float32))
        n = dict(s, t=t, episode_done=terminated | truncated, terminated=terminated, truncated=truncated,
                 pos=p, item=item, allergen=allergen, age=age, ticket=ticket, stock=stock, recipe=recipe,
                 progress=progress, contamination=contamination, order=order, deadline=deadline, remaining=remaining,
                 counter_pos=cp, counter_shift=shifts, thermal_level=jnp.stack((heat, dryprogress)).astype(jnp.float32),
                 thermal_stock=jnp.stack((s['thermal_stock'][0] - stoke.astype(jnp.int32), drycount)),
                 served=served, missed=missed, spill=spill, burn=burn, completed=completed)
        return self.refresh(n), jnp.clip(reward, -1, 2).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = self.advance(state, actions)
        n = jax.tree.map(lambda old, new: jnp.where(state['episode_done'], old, new), state, n)
        reward = jnp.where(state['episode_done'], jnp.zeros((12,), jnp.float32), reward)
        info = {'done_all': n['episode_done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full((12,), n['episode_done'], jnp.bool_), info

    def observe(self, s):
        p = s['pos']
        own = jnp.stack((jnp.full((12,), s['t'] / 512), jnp.arange(12) / 11, s['team'] / 2, p[:, 0] / 22, p[:, 1] / 10,
                         s['item'] / 9, s['allergen'].astype(jnp.float32), s['age'] / 32, s['ticket'] / 2, s['spill'].astype(jnp.float32), s['burn'].astype(jnp.float32), s['completed'][s['team']] / 4), axis=1)
        actor = jnp.stack((p[:, 0] / 22, p[:, 1] / 10, s['team'] / 2, s['item'] / 9, s['allergen'].astype(jnp.float32), s['age'] / 32, s['ticket'] / 2, s['spill'].astype(jnp.float32), s['burn'].astype(jnp.float32), jnp.ones((12,))), axis=1)
        station = jnp.stack((s['station_pos'][:, 0] / 22, s['station_pos'][:, 1] / 10, s['station_type'] / 5, s['stock'] / 16,
                  s['recipe'] / 4, s['progress'] / 8, s['contamination'].astype(jnp.float32), s['order'] / 2,
                  s['deadline'] / 96, s['served'].astype(jnp.float32), s['missed'].astype(jnp.float32), jnp.ones((18,))), axis=1)
        counter = jnp.stack((s['counter_pos'][:, 0] / 22, s['counter_pos'][:, 1] / 10, s['counter_shift'], jnp.ones((2,))), axis=1)
        thermal = jnp.stack((s['thermal_pos'][:, 0] / 22, s['thermal_pos'][:, 1] / 10, s['thermal_kind'], s['thermal_level'] / 10, s['thermal_stock'] / 36, jnp.ones((2,))), axis=1)
        k = jnp.arange(49, dtype=jnp.int32)
        q = p[:, None] + jnp.stack((k % 7 - 3, k // 7 - 3), axis=1)[None]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] < 23) & (q[:, :, 1] >= 0) & (q[:, :, 1] < 11)
        tiles = s['tiles'][jnp.clip(q[:, :, 1], 0, 10), jnp.clip(q[:, :, 0], 0, 22)] / 2
        patch = jnp.stack((jnp.where(inside, tiles, 0), inside.astype(jnp.float32)), axis=2)
        return jnp.concatenate((own, (actor[None] * s['agent_vis'][:, :, None]).reshape((12, 120)),
                  (station[None] * s['station_vis'][:, :, None]).reshape((12, 216)),
                  (counter[None] * s['counter_vis'][:, :, None]).reshape((12, 8)),
                  (thermal[None] * s['thermal_vis'][:, :, None]).reshape((12, 12)), patch.reshape((12, 98))), axis=1).astype(jnp.float32)
