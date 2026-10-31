"""Highland Treaty Roads. Original native strategy rules, not runtime evidence."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 192
    num_actions = 16
    observation_size = 3642
    max_steps = 896

    def refresh(self, s):
        p = s['pos']
        av = (jnp.max(jnp.abs(p[:, None] - p[None]), axis=2) <= 3) & s['alive'][:, None] & s['alive'][None]
        nv = (jnp.max(jnp.abs(p[:, None] - s['node_pos'][None]), axis=2) <= 3) & s['alive'][:, None]
        x = jnp.arange(48, dtype=jnp.int32)[None, None, :]
        y = jnp.arange(24, dtype=jnp.int32)[None, :, None]
        mv = (jnp.abs(x - p[:, 0, None, None]) <= 3) & (jnp.abs(y - p[:, 1, None, None]) <= 3) & s['alive'][:, None, None]
        city = s['node_owner']
        typ = s['node_type']
        integrity = jnp.where(typ == 6, s['brace'][city], 0).astype(jnp.int32)
        progress = jnp.where(typ == 3, s['crop'][city], 0).astype(jnp.int32)
        stress = jnp.where((typ == 1) | (typ == 6), s['stress'][city], 0).astype(jnp.int32)
        timer = jnp.where((typ == 1) | (typ == 6), s['fall_timer'][city], 0).astype(jnp.int32)
        return dict(s, agent_vis=av, node_vis=nv & s['node_active'][None], gate_vis=nv[:, 4::7], gate_pos=s['node_pos'][4::7], map_vis=mv, node_integrity=integrity,
                    node_progress=progress, node_stress=stress, node_timer=timer)

    def reset(self, key):
        a = jnp.arange(192, dtype=jnp.int32)
        c = jnp.arange(12, dtype=jnp.int32)
        base = jnp.stack(((c % 4) * 12, (c // 4) * 8), axis=1)
        pos = base[a // 16] + jnp.stack((1 + a % 4, 1 + (a % 16) // 4), axis=1)
        pos = pos.at[c * 16].set(base + jnp.array([8, 3], jnp.int32))
        pos = pos.at[c * 16 + 1].set(base + jnp.array([3, 3], jnp.int32))
        pos = pos.at[c * 16 + 4].set(base + jnp.array([8, 4], jnp.int32))
        n = jnp.arange(84, dtype=jnp.int32)
        layout = jnp.array([[3, 3], [8, 3], [2, 5], [4, 5], [10, 4], [11, 4], [8, 4]], jnp.int32)
        np = base[n // 7] + layout[n % 7]
        x = jnp.arange(48)[None, :]
        y = jnp.arange(24)[:, None]
        tiles = jnp.where((x % 12 == 11) & (y % 8 != 7), 1, 0).astype(jnp.int32)
        tiles = tiles.at[np[5::7, 1], np[5::7, 0]].set(2)
        stocks = jnp.where(n % 7 == 0, 8, jnp.where(n % 7 == 1, 40, jnp.where(n % 7 == 2, 24, 0))).astype(jnp.int32)
        s = {'t': jnp.array(0, jnp.int32), 'episode_done': jnp.array(False), 'terminated': jnp.array(False), 'truncated': jnp.array(False),
             'pos': pos, 'alive': jnp.ones((192,), jnp.bool_), 'team': a // 16, 'tool': a % 4,
             'cargo': jnp.where(a % 16 == 4, 2, 0).astype(jnp.int32), 'coins': jnp.full((192,), 6, jnp.int32),
             'health': jnp.full((192,), 100, jnp.int32), 'seal_origin': jnp.full((192,), -1, jnp.int32),
             'seal_partner': jnp.full((192,), -1, jnp.int32), 'survey_site': jnp.full((192,), -1, jnp.int32),
             'survey_stress': jnp.zeros((192,), jnp.int32), 'survey_timer': jnp.zeros((192,), jnp.int32), 'survey_time': jnp.zeros((192,), jnp.int32),
             'node_pos': np, 'node_active': n % 7 != 4, 'node_type': n % 7, 'node_owner': n // 7, 'node_stock': stocks,
             'gate_active': jnp.ones((12,), jnp.bool_), 'gate_owner': c,
             'node_integrity': jnp.zeros((84,), jnp.int32), 'node_progress': jnp.zeros((84,), jnp.int32),
             'node_stress': jnp.zeros((84,), jnp.int32), 'node_timer': jnp.zeros((84,), jnp.int32),
             'brace': jnp.zeros((12,), jnp.int32), 'stress': jax.random.randint(key, (12,), 2, 4, dtype=jnp.int32),
             'fall_timer': jnp.zeros((12,), jnp.int32), 'crop': jnp.zeros((12,), jnp.int32),
             'treasury': jnp.full((12,), 64, jnp.int32), 'received': jnp.zeros((12,), jnp.int32),
             'treaty': jnp.zeros((12, 12), jnp.int32), 'tiles': tiles,
             'collapse': jnp.zeros((12,), jnp.bool_), 'ratified': jnp.zeros((192,), jnp.bool_),
             'traded': jnp.zeros((192,), jnp.bool_), 'repaired': jnp.zeros((192,), jnp.bool_),
             'toll_paid': jnp.zeros((192,), jnp.int32), 'injury': jnp.zeros((192,), jnp.bool_),
             'agent_vis': jnp.zeros((192, 192), jnp.bool_), 'node_vis': jnp.zeros((192, 84), jnp.bool_),
             'map_vis': jnp.zeros((192, 24, 48), jnp.bool_)}
        s = self.refresh(s)
        return self.observe(s), s

    def nearest(self, s, p):
        dd = jnp.sum(jnp.abs(p[:, None] - s['node_pos'][None]), axis=2)
        k = jnp.argmin(dd, axis=1)
        return k, jnp.min(dd, axis=1) <= 1

    def legal_actions(self, s):
        p = s['pos']
        d = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        q = p[:, None] + d[None]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] < 48) & (q[:, :, 1] >= 0) & (q[:, :, 1] < 24)
        x = jnp.clip(q[:, :, 0], 0, 47)
        y = jnp.clip(q[:, :, 1], 0, 23)
        tile = s['tiles'][y, x]
        owner = (y // 8) * 4 + x // 12
        foreign = owner != s['team'][:, None]
        entering = (tile == 2) & jnp.any(q != p[:, None], axis=2)
        permitted = ~foreign | ((s['treaty'][owner, s['team'][:, None]] > 0) & (s['coins'][:, None] > 0))
        walk = inside & (tile != 1) & (tile != 3) & (~entering | permitted)
        k, near = self.nearest(s, p)
        typ = s['node_type'][k]
        city = s['node_owner'][k]
        cargo = s['cargo']
        gather = (cargo == 0) & (((typ == 0) | ((typ == 1) & (s['tool'] == 0)) | ((typ == 2) & (s['tool'] == 0))) & (s['node_stock'][k] > 0) | ((typ == 3) & (s['crop'][city] >= 12)))
        deliver = (typ == 0) & (city != s['team']) & (cargo > 0) & (s['treasury'][city] >= 2) & (s['coins'] <= 62)
        repair = (s['tool'] == 0) & (typ == 6) & ((cargo == 2) | (cargo == 1))
        issue = (typ == 0) & (city == s['team']) & (s['seal_origin'] < 0) & (s['tool'] == 1)
        present = (typ == 0) & (city != s['seal_origin']) & (s['seal_origin'] >= 0) & (s['seal_partner'] < 0)
        sign = (typ == 0) & (city == s['team']) & (s['tool'] == 3)
        ratify = (typ == 4) & (s['seal_partner'] >= 0) & ((city == s['seal_origin']) | (city == s['seal_partner']))
        plant = (typ == 3) & (s['tool'] == 2) & (cargo == 3) & (s['crop'][city] == 0)
        survey = (typ == 1) | (typ == 6)
        rest = jnp.stack((near & gather, near & deliver, near & repair, near & issue, near & present,
               near & sign, near & ratify, near & plant, near & survey, s['survey_site'] >= 0,
               (cargo > 0) | (s['seal_origin'] >= 0) | (s['survey_site'] >= 0)), axis=1)
        return (jnp.concatenate((walk, rest), axis=1) & s['alive'][:, None] & ~s['episode_done']).at[:, 0].set(True)

    def winner(self, target, wanted, count, t):
        rank = (jnp.arange(192, dtype=jnp.int32) - t) % 192
        costs = jnp.where(wanted[:, None] & (target[:, None] == jnp.arange(count)[None]), rank[:, None], 193)
        return wanted & (rank == jnp.min(costs, axis=0)[target])

    def advance(self, s, actions):
        ids = jnp.arange(192, dtype=jnp.int32)
        aa = jnp.clip(actions, 0, 15)
        act = jnp.where((aa == actions) & self.legal_actions(s)[ids, aa], aa, 0)
        d = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)
        p = s['pos'] + d[jnp.minimum(act, 4)] * (act <= 4)[:, None]
        city_at = (p[:, 1] // 8) * 4 + p[:, 0] // 12
        foreign = city_at != s['team']
        paid = (s['tiles'][p[:, 1], p[:, 0]] == 2) & foreign & jnp.any(p != s['pos'], axis=1)
        coins = s['coins'] - paid.astype(jnp.int32)
        tolls = jnp.zeros((12,), jnp.int32).at[city_at].add(paid.astype(jnp.int32))
        treasury = s['treasury'] + tolls
        k, near = self.nearest(s, p)
        typ = s['node_type'][k]
        city = s['node_owner'][k]
        win = self.winner(k, near & ((act == 5) | (act == 6) | (act == 7) | (act == 12)), 84, s['t'])
        gather = win & (act == 5)
        trade = win & (act == 6)
        repair = win & (act == 7)
        plant = win & (act == 12)
        cargo = jnp.where(gather, jnp.where(typ == 1, 1, jnp.where(typ == 2, 2, 3)), s['cargo'])
        cargo = jnp.where(trade | repair | plant, 0, cargo).astype(jnp.int32)
        stocks = s['node_stock'].at[k].add(-(gather & (typ != 3)).astype(jnp.int32))
        income = jnp.zeros((12,), jnp.int32).at[city].add(trade.astype(jnp.int32))
        treasury = treasury - income * 2
        coins = coins + trade.astype(jnp.int32) * 2
        mined = jnp.zeros((12,), jnp.int32).at[city].add((gather & (typ == 1)).astype(jnp.int32))
        braced = jnp.zeros((12,), jnp.int32).at[city].add((repair & (s['cargo'] == 2)).astype(jnp.int32))
        brace = jnp.clip(s['brace'] + braced * 12 - (s['t'] % 8 == 7).astype(jnp.int32), 0, 24)
        stress = jnp.minimum(s['stress'] + mined, 8)
        timer = jnp.maximum(s['fall_timer'] - 1, 0)
        trigger = (stress >= 4) & (s['fall_timer'] == 0)
        timer = jnp.where(trigger, 4, timer).astype(jnp.int32)
        resolved = s['fall_timer'] == 1
        collapse = resolved & (brace == 0)
        stress = jnp.where(resolved, 0, stress).astype(jnp.int32)
        brace = jnp.maximum(brace - resolved.astype(jnp.int32) * 2, 0)
        slope = s['node_pos'][6::7]
        tiles = s['tiles'].at[slope[:, 1], slope[:, 0]].set(jnp.where(collapse, 3, s['tiles'][slope[:, 1], slope[:, 0]]))
        clear = jnp.zeros((12,), jnp.int32).at[city].add((repair & (s['cargo'] == 1)).astype(jnp.int32)) > 0
        tiles = tiles.at[slope[:, 1], slope[:, 0]].set(jnp.where(clear, 0, tiles[slope[:, 1], slope[:, 0]]))
        cultivated = jnp.zeros((12,), jnp.int32).at[city].add(plant.astype(jnp.int32)) > 0
        harvested = jnp.zeros((12,), jnp.int32).at[city].add((gather & (typ == 3)).astype(jnp.int32)) > 0
        crop = jnp.where(harvested, 0, jnp.where(cultivated, 1, s['crop']))
        crop = jnp.minimum(crop + ((crop > 0) & (s['t'] % 4 == 3) & ~collapse).astype(jnp.int32), 12)
        origin = jnp.where(act == 8, city, s['seal_origin'])
        partner = s['seal_partner']
        signed_here = jnp.zeros((12,), jnp.int32).at[city].add((act == 10).astype(jnp.int32)) > 0
        signed = (act == 9) & signed_here[city]
        partner = jnp.where(signed, city, partner)
        ratified = act == 11
        orig = jnp.maximum(origin, 0)
        part = jnp.maximum(partner, 0)
        rat = jnp.zeros((12, 12), jnp.int32).at[orig, part].add(ratified.astype(jnp.int32))
        rat = rat + jnp.transpose(rat)
        treaty = jnp.where(rat > 0, 96, jnp.maximum(s['treaty'] - 1, 0)).astype(jnp.int32)
        origin = jnp.where(ratified, -1, origin)
        partner = jnp.where(ratified, -1, partner)
        survey = act == 13
        ss = jnp.where(survey, city, s['survey_site'])
        sv = jnp.where(survey, s['stress'][city], s['survey_stress'])
        st = jnp.where(survey, s['fall_timer'][city], s['survey_timer'])
        tm = jnp.where(survey, s['t'], s['survey_time'])
        dist = jnp.sum(jnp.abs(p[:, None] - p[None]), axis=2)
        nearby = (dist <= 1) & (ids[:, None] != ids[None]) & s['alive'][None]
        can = ((act[:, None] == 15) | (act[:, None] == 14)) & (act[None] == 0) & nearby
        empty = (s['cargo'] == 0) & (s['seal_origin'] < 0)
        can = can & ((act[:, None] == 14) | empty[None])
        dest = jnp.argmin(jnp.where(can, ids[None], 193), axis=1)
        send = self.winner(dest, jnp.any(can, axis=1), 192, s['t'])
        transfer = send & (act == 15)
        received = jnp.zeros((192,), jnp.int32).at[dest].add(transfer.astype(jnp.int32)) > 0
        read = jnp.zeros((192,), jnp.int32).at[dest].add((send & (ss >= 0)).astype(jnp.int32)) > 0
        def moved(v, default):
            vals = jnp.zeros((192,), jnp.int32).at[dest].add(jnp.where(transfer, v - default, 0)) + default
            return jnp.where(transfer, default, jnp.where(received, vals, v))
        cargo = moved(cargo, 0)
        origin = moved(origin, -1)
        partner = moved(partner, -1)
        def report(v, default):
            vals = jnp.zeros((192,), jnp.int32).at[dest].add(jnp.where(send & (ss >= 0), v - default, 0)) + default
            return jnp.where(read, vals, v)
        sv = report(sv, 0)
        st = report(st, 0)
        tm = report(tm, 0)
        ss = report(ss, -1)
        injury = jnp.any((jnp.sum(jnp.abs(p[:, None] - slope[None]), axis=2) <= 1) & collapse[None], axis=1) & s['alive']
        health = jnp.maximum(s['health'] - injury.astype(jnp.int32) * 5, 0)
        alive = health > 0
        received_goods = jnp.minimum(s['received'] + income, 64)
        reward = 0.6 * trade + 0.1 * income[s['team']] + 0.025 * repair - 0.002 * (act >= 5) - 0.1 * injury
        reward = reward - 0.02 * paid + 0.02 * tolls[s['team']] / 16 - 0.025 * collapse[s['team']]
        t = s['t'] + jnp.array(1, jnp.int32)
        terminated = ~jnp.any(alive) | jnp.all(received_goods >= 3)
        truncated = (t >= 896) & ~terminated
        n = dict(s, t=t, episode_done=terminated | truncated, terminated=terminated, truncated=truncated,
                 pos=p, alive=alive, cargo=cargo, coins=coins, health=health, seal_origin=origin, seal_partner=partner,
                 survey_site=ss, survey_stress=sv, survey_timer=st, survey_time=tm,
                 node_stock=stocks, brace=brace, stress=stress, fall_timer=timer, crop=crop,
                 treasury=treasury, received=received_goods, treaty=treaty, tiles=tiles, collapse=collapse,
                 ratified=ratified, traded=trade, repaired=repair, toll_paid=paid.astype(jnp.int32), injury=injury)
        return self.refresh(n), jnp.clip(reward, -2, 2).astype(jnp.float32)

    def raw_step(self, key, state, actions):
        n, reward = self.advance(state, actions)
        n = jax.tree.map(lambda old, new: jnp.where(state['episode_done'], old, new), state, n)
        reward = jnp.where(state['episode_done'], jnp.zeros((192,), jnp.float32), reward)
        info = {'done_all': n['episode_done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, reward, jnp.full((192,), n['episode_done'], jnp.bool_), info

    def observe(self, s):
        p = s['pos']
        own = jnp.stack((jnp.full((192,), s['t'] / 896), jnp.arange(192) / 191, s['team'] / 11, s['tool'] / 3,
                         p[:, 0] / 47, p[:, 1] / 23, s['cargo'] / 3, s['coins'] / 64, s['health'] / 100,
                         (s['seal_origin'] + 1) / 12, (s['seal_partner'] + 1) / 12, (s['survey_site'] + 1) / 12,
                         s['survey_stress'] / 8, s['survey_timer'] / 4, s['survey_time'] / 896, s['alive'].astype(jnp.float32)), axis=1)
        actors = jnp.stack((p[:, 0] / 47, p[:, 1] / 23, s['team'] / 11, s['tool'] / 3, s['cargo'] / 3, s['coins'] / 64,
                           s['health'] / 100, (s['seal_origin'] + 1) / 12, (s['seal_partner'] + 1) / 12,
                           (s['survey_site'] + 1) / 12, s['survey_stress'] / 8, s['survey_timer'] / 4, s['alive'].astype(jnp.float32)), axis=1)
        nodes = jnp.stack((s['node_pos'][:, 0] / 47, s['node_pos'][:, 1] / 23, s['node_type'] / 6, s['node_owner'] / 11,
                          s['node_stock'] / 64, s['node_integrity'] / 24, s['node_progress'] / 12,
                          s['node_stress'] / 8, s['node_timer'] / 4, jnp.ones((84,))), axis=1)
        gate = jnp.concatenate((s['gate_pos'] / jnp.array([47, 23], jnp.float32), s['gate_owner'][:, None] / 11,
                                jnp.ones((12, 1)), s['treaty'] / 96), axis=1)
        access = gate[None] * s['gate_vis'][:, :, None]
        k = jnp.arange(49, dtype=jnp.int32)
        q = p[:, None] + jnp.stack((k % 7 - 3, k // 7 - 3), axis=1)[None]
        inside = (q[:, :, 0] >= 0) & (q[:, :, 0] < 48) & (q[:, :, 1] >= 0) & (q[:, :, 1] < 24) & s['alive'][:, None]
        tile = s['tiles'][jnp.clip(q[:, :, 1], 0, 23), jnp.clip(q[:, :, 0], 0, 47)] / 3
        patch = jnp.stack((jnp.where(inside, tile, 0), inside.astype(jnp.float32)), axis=2)
        return jnp.concatenate((own, (actors[None] * s['agent_vis'][:, :, None]).reshape((192, 2496)),
                   (nodes[None] * s['node_vis'][:, :, None]).reshape((192, 840)), access.reshape((192, 192)),
                   patch.reshape((192, 98))), axis=1).astype(jnp.float32)
