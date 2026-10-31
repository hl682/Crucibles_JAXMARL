"""Village Loom. Original tool-loan, labor-pledge and shared-workshop game; unexecuted."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 64
    num_actions = 21
    max_steps = 96

    def reset(self, key):
        agents = jnp.array([[1, 2], [7, 2], [2, 7]], dtype=jnp.int32)
        plots = jnp.array([[1, 1], [7, 1], [1, 7]], dtype=jnp.int32)
        facilities = jnp.array([[4, 2], [4, 4]], dtype=jnp.int32)
        terrain = jnp.zeros((9, 9), dtype=jnp.int32).at[5, 6:9].set(1)
        s = {
            't': jnp.array(0, dtype=jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False), 'terrain': terrain,
            'agent_pos': agents, 'agent_active': jnp.ones((3,), dtype=jnp.bool_),
            'agent_visibility': jnp.ones((3, 3), dtype=jnp.bool_),
            'fiber': jnp.ones((3,), dtype=jnp.int32), 'yarn': jnp.zeros((3,), dtype=jnp.int32),
            'dye': jnp.zeros((3,), dtype=jnp.int32), 'cloth': jnp.zeros((3,), dtype=jnp.int32),
            'seeds': jnp.full((3,), 2, dtype=jnp.int32), 'busy': jnp.zeros((3,), dtype=jnp.int32),
            'worksite': jnp.full((3,), -1, dtype=jnp.int32),
            'delivered': jnp.zeros((3,), dtype=jnp.int32), 'score': jnp.zeros((3,), dtype=jnp.int32),
            'deadline': 32 + 8 * jax.random.randint(key, (3,), 0, 4, dtype=jnp.int32),
            'plot_pos': plots, 'plot_active': jnp.ones((3,), dtype=jnp.bool_),
            'plot_visibility': jnp.ones((3, 3), dtype=jnp.bool_),
            'plot_stage': jnp.zeros((3,), dtype=jnp.int32), 'plot_clock': jnp.zeros((3,), dtype=jnp.int32),
            'plot_yield': jnp.zeros((3,), dtype=jnp.int32),
            'facility_pos': facilities, 'facility_active': jnp.ones((2,), dtype=jnp.bool_),
            'facility_visibility': jnp.ones((3, 2), dtype=jnp.bool_),
            'health': jnp.full((2,), 2, dtype=jnp.int32), 'work_left': jnp.zeros((2,), dtype=jnp.int32),
            'job_owner': jnp.full((2,), -1, dtype=jnp.int32), 'output': jnp.zeros((2,), dtype=jnp.bool_),
            'tool_pos': agents[:2], 'tool_active': jnp.ones((2,), dtype=jnp.bool_),
            'tool_visibility': jnp.ones((3, 2), dtype=jnp.bool_),
            'tool_holder': jnp.array([0, 1], dtype=jnp.int32), 'tool_owner': jnp.array([0, 1], dtype=jnp.int32),
            'loan_left': jnp.zeros((2,), dtype=jnp.int32),
            'dye_pos': jnp.array([[7, 7]], dtype=jnp.int32), 'dye_active': jnp.ones((1,), dtype=jnp.bool_),
            'dye_visibility': jnp.ones((3, 1), dtype=jnp.bool_), 'dye_stock': jnp.array([6], dtype=jnp.int32),
            'market_pos': jnp.array([[4, 7]], dtype=jnp.int32), 'market_active': jnp.ones((1,), dtype=jnp.bool_),
            'market_visibility': jnp.ones((3, 1), dtype=jnp.bool_),
            'e_move': jnp.zeros((3,), dtype=jnp.bool_), 'e_cut': jnp.zeros((3,), dtype=jnp.int32),
            'e_gift': jnp.zeros((3,), dtype=jnp.int32), 'e_sale': jnp.zeros((3,), dtype=jnp.bool_),
            'e_pledge': jnp.zeros((3,), dtype=jnp.bool_), 'e_work': jnp.zeros((2,), dtype=jnp.int32),
            'e_complete': jnp.zeros((2,), dtype=jnp.bool_), 'e_repair': jnp.zeros((2,), dtype=jnp.bool_),
            'e_loan': jnp.zeros((2,), dtype=jnp.bool_), 'e_return': jnp.zeros((2,), dtype=jnp.bool_),
        }
        return self.observe(s), s

    def destination(self, s, actions):
        delta = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
        move = (actions >= 1) & (actions <= 4)
        p = s['agent_pos'] + delta[jnp.where(move, actions, 0)]
        valid = jnp.all((p >= 0) & (p <= 8), axis=1)
        q = jnp.clip(p, 0, 8)
        valid = valid & (s['terrain'][q[:, 1], q[:, 0]] == 0)
        return jnp.where(valid[:, None], p, s['agent_pos']), valid

    def legal_actions(self, s):
        me = jnp.arange(3)
        plot = jnp.sum(jnp.abs(s['agent_pos'] - s['plot_pos']), axis=1) <= 1
        fd = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['facility_pos'][None, :, :]), axis=2)
        near_fac = fd <= 1
        near = (jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['agent_pos'][None, :, :]), axis=2) <= 1) & ~jnp.eye(3, dtype=jnp.bool_)
        sickle = (s['tool_holder'][0] == me) & ((s['tool_owner'][0] == me) | (s['loan_left'][0] > 0))
        shuttle = (s['tool_holder'][1] == me) & ((s['tool_owner'][1] == me) | (s['loan_left'][1] > 0))
        idle = (s['work_left'] == 0) & ~s['output']
        collect_capacity = jnp.stack([s['yarn'] < 2, s['cloth'] < 2], axis=1)
        collection = near_fac & (s['job_owner'][None, :] == me[:, None]) & s['output'][None, :] & collect_capacity
        can_return = (s['tool_holder'][None, :] == me[:, None]) & (s['tool_owner'][None, :] != me[:, None])
        can_return = can_return & (jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['agent_pos'][s['tool_owner']][None, :, :]), axis=2) <= 1)
        reclaim = (s['tool_owner'][None, :] == me[:, None]) & (s['tool_holder'][None, :] != me[:, None])
        reclaim = reclaim & (jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['tool_pos'][None, :, :]), axis=2) <= 1)
        m = jnp.zeros((3, 21), dtype=jnp.bool_).at[:, 0].set(True)
        for d in range(1, 5):
            m = m.at[:, d].set(self.destination(s, jnp.full((3,), d, dtype=jnp.int32))[1])
        m = m.at[:, 5].set(plot & (s['plot_stage'] == 0) & (s['seeds'] > 0) & (s['t'] < 48))
        m = m.at[:, 6].set(plot & sickle & (s['plot_stage'] == 2) & (s['fiber'] < 2))
        m = m.at[:, 7].set((jnp.sum(jnp.abs(s['agent_pos'] - s['dye_pos'][0]), axis=1) <= 1) & (s['dye_stock'][0] > 0) & (s['dye'] < 2))
        m = m.at[:, 8].set(near_fac[:, 0] & idle[0] & (s['health'][0] > 0) & (s['fiber'] > 0))
        m = m.at[:, 9].set(near_fac[:, 1] & idle[1] & (s['health'][1] > 0) & (s['yarn'] > 0) & (s['dye'] > 0) & shuttle)
        m = m.at[:, 10].set(jnp.any(near_fac & (s['work_left'][None, :] > 0) & (s['health'][None, :] > 0), axis=1))
        m = m.at[:, 11].set(jnp.any(collection, axis=1))
        m = m.at[:, 12].set(jnp.any(near_fac & (s['health'][None, :] < 3), axis=1) & (s['fiber'] > 0))
        m = m.at[:, 13].set((s['tool_holder'][0] == me) & (s['tool_owner'][0] == me) & jnp.any(near, axis=1))
        m = m.at[:, 14].set((s['tool_holder'][1] == me) & (s['tool_owner'][1] == me) & jnp.any(near, axis=1))
        m = m.at[:, 15].set(jnp.any(can_return, axis=1)).at[:, 16].set(jnp.any(reclaim, axis=1))
        m = m.at[:, 17].set((s['fiber'] > 0) & jnp.any(near & (s['fiber'][None, :] < 2), axis=1))
        m = m.at[:, 18].set((s['yarn'] > 0) & jnp.any(near & (s['yarn'][None, :] < 2), axis=1))
        m = m.at[:, 19].set((s['dye'] > 0) & jnp.any(near & (s['dye'][None, :] < 2), axis=1))
        market = jnp.sum(jnp.abs(s['agent_pos'] - s['market_pos'][0]), axis=1) <= 1
        m = m.at[:, 20].set(market & (s['cloth'] > 0) & (s['delivered'] < 2))
        return m & ((~s['done'] & (s['busy'][:, None] == 0)) | (jnp.arange(21)[None, :] == 0))

    def raw_step(self, key, s, actions):
        safe = jnp.clip(actions, 0, 20)
        a = jnp.where((actions >= 0) & (actions < 21) & self.legal_actions(s)[jnp.arange(3), safe], safe, 0)
        pos, valid = self.destination(s, a)
        n = dict(s)
        n['t'] = s['t'] + jnp.array(1, dtype=jnp.int32)
        n['agent_pos'] = pos
        n['e_move'] = jnp.any(pos != s['agent_pos'], axis=1)
        n['loan_left'] = jnp.maximum(s['loan_left'] - 1, 0)
        n['e_cut'] = jnp.zeros((3,), dtype=jnp.int32)
        n['e_gift'] = jnp.zeros((3,), dtype=jnp.int32)
        n['e_sale'] = jnp.zeros((3,), dtype=jnp.bool_)
        n['e_pledge'] = jnp.zeros((3,), dtype=jnp.bool_)
        n['e_repair'] = jnp.zeros((2,), dtype=jnp.bool_)
        n['e_loan'] = jnp.zeros((2,), dtype=jnp.bool_)
        n['e_return'] = jnp.zeros((2,), dtype=jnp.bool_)
        payoff = jnp.zeros((3,), dtype=jnp.int32)
        # Competing uses of a workshop or finite stock use rotating initiative.
        for k in range(3):
            i = (s['t'] + k) % 3
            sow = a[i] == 5
            n['seeds'] = n['seeds'].at[i].add(-sow.astype(jnp.int32))
            n['plot_stage'] = n['plot_stage'].at[i].set(jnp.where(sow, 1, n['plot_stage'][i]))
            n['plot_clock'] = n['plot_clock'].at[i].set(jnp.where(sow, 3, n['plot_clock'][i]))
            amount = jnp.where(a[i] == 6, jnp.minimum(n['plot_yield'][i], 2 - n['fiber'][i]), 0)
            n['fiber'] = n['fiber'].at[i].add(amount)
            n['plot_yield'] = n['plot_yield'].at[i].add(-amount)
            n['plot_stage'] = n['plot_stage'].at[i].set(jnp.where((amount > 0) & (n['plot_yield'][i] == 0), 0, n['plot_stage'][i]))
            n['e_cut'] = n['e_cut'].at[i].set(amount)
            gather = (a[i] == 7) & (n['dye_stock'][0] > 0) & (n['dye'][i] < 2)
            n['dye_stock'] = n['dye_stock'].at[0].add(-gather.astype(jnp.int32))
            n['dye'] = n['dye'].at[i].add(gather.astype(jnp.int32))
            job = jnp.where(a[i] == 9, 1, 0)
            start = ((a[i] == 8) | (a[i] == 9)) & (n['work_left'][job] == 0) & ~n['output'][job] & (n['health'][job] > 0)
            spin = start & (job == 0)
            weave = start & (job == 1)
            n['fiber'] = n['fiber'].at[i].add(-spin.astype(jnp.int32))
            n['yarn'] = n['yarn'].at[i].add(-weave.astype(jnp.int32))
            n['dye'] = n['dye'].at[i].add(-weave.astype(jnp.int32))
            n['health'] = n['health'].at[job].add(-start.astype(jnp.int32))
            n['work_left'] = n['work_left'].at[job].set(jnp.where(start, 2, n['work_left'][job]))
            n['job_owner'] = n['job_owner'].at[job].set(jnp.where(start, i, n['job_owner'][job]))
            near_fac = jnp.sum(jnp.abs(s['facility_pos'] - pos[i]), axis=1) <= 1
            useful = near_fac & (n['work_left'] > 0) & (n['health'] > 0)
            site = jnp.argmax(useful).astype(jnp.int32)
            pledge = (a[i] == 10) & jnp.any(useful)
            n['busy'] = n['busy'].at[i].set(jnp.where(pledge, 2, n['busy'][i]))
            n['worksite'] = n['worksite'].at[i].set(jnp.where(pledge, site, n['worksite'][i]))
            n['e_pledge'] = n['e_pledge'].at[i].set(pledge)
            capacity = jnp.array([n['yarn'][i] < 2, n['cloth'][i] < 2])
            ready = near_fac & n['output'] & (n['job_owner'] == i) & capacity
            site = jnp.argmax(ready)
            collect = (a[i] == 11) & jnp.any(ready)
            n['yarn'] = n['yarn'].at[i].add((collect & (site == 0)).astype(jnp.int32))
            n['cloth'] = n['cloth'].at[i].add((collect & (site == 1)).astype(jnp.int32))
            n['output'] = n['output'].at[site].set(n['output'][site] & ~collect)
            n['job_owner'] = n['job_owner'].at[site].set(jnp.where(collect, -1, n['job_owner'][site]))
            repairable = near_fac & (n['health'] < 3)
            site = jnp.argmax(repairable)
            repair = (a[i] == 12) & jnp.any(repairable) & (n['fiber'][i] > 0)
            n['fiber'] = n['fiber'].at[i].add(-repair.astype(jnp.int32))
            n['health'] = n['health'].at[site].set(jnp.where(repair, jnp.minimum(3, n['health'][site] + 2), n['health'][site]))
            n['e_repair'] = n['e_repair'].at[site].set(n['e_repair'][site] | repair)
            near = (jnp.sum(jnp.abs(pos - pos[i]), axis=1) <= 1) & (jnp.arange(3) != i)
            recipient = jnp.argmax(near).astype(jnp.int32)
            for tool in range(2):
                lend = (a[i] == 13 + tool) & (n['tool_holder'][tool] == i) & (s['tool_owner'][tool] == i) & jnp.any(near)
                n['tool_holder'] = n['tool_holder'].at[tool].set(jnp.where(lend, recipient, n['tool_holder'][tool]))
                n['loan_left'] = n['loan_left'].at[tool].set(jnp.where(lend, 8, n['loan_left'][tool]))
                n['e_loan'] = n['e_loan'].at[tool].set(n['e_loan'][tool] | lend)
                owner = s['tool_owner'][tool]
                holder = n['tool_holder'][tool]
                close = jnp.sum(jnp.abs(pos[owner] - pos[holder])) <= 1
                give_back = (a[i] == 15) & (holder == i) & (owner != i) & close
                reclaim = (a[i] == 16) & (owner == i) & (holder != i) & close
                returned = give_back | reclaim
                n['tool_holder'] = n['tool_holder'].at[tool].set(jnp.where(returned, owner, n['tool_holder'][tool]))
                n['loan_left'] = n['loan_left'].at[tool].set(jnp.where(returned, 0, n['loan_left'][tool]))
                n['e_return'] = n['e_return'].at[tool].set(n['e_return'][tool] | returned)
            for field, command in [('fiber', 17), ('yarn', 18), ('dye', 19)]:
                eligible = near & (n[field] < 2)
                target = jnp.argmax(eligible)
                gift = (a[i] == command) & (n[field][i] > 0) & jnp.any(eligible)
                n[field] = n[field].at[i].add(-gift.astype(jnp.int32)).at[target].add(gift.astype(jnp.int32))
                n['e_gift'] = n['e_gift'].at[i].add(gift.astype(jnp.int32))
            sell = a[i] == 20
            n['cloth'] = n['cloth'].at[i].add(-sell.astype(jnp.int32))
            n['delivered'] = n['delivered'].at[i].add(sell.astype(jnp.int32))
            value = jnp.where(n['t'] <= s['deadline'][i], 3, 1)
            payoff = payoff.at[i].set(sell.astype(jnp.int32) * value)
            n['e_sale'] = n['e_sale'].at[i].set(sell)
        growing = n['plot_stage'] == 1
        n['plot_clock'] = jnp.where(growing, jnp.maximum(0, n['plot_clock'] - 1), n['plot_clock'])
        ripe = growing & (n['plot_clock'] == 0)
        n['plot_stage'] = jnp.where(ripe, 2, n['plot_stage'])
        n['plot_yield'] = jnp.where(ripe, 2, n['plot_yield'])
        # A labor promise lasts this step and the next, even if work ends early.
        labor = jnp.sum(((n['busy'][:, None] > 0) & (n['worksite'][:, None] == jnp.arange(2)[None, :])).astype(jnp.int32), axis=0, dtype=jnp.int32)
        work = jnp.where(n['health'] > 0, jnp.minimum(labor, n['work_left']), 0)
        n['e_work'] = work
        n['e_complete'] = (n['work_left'] > 0) & (work == n['work_left'])
        n['work_left'] = n['work_left'] - work
        n['output'] = n['output'] | n['e_complete']
        n['busy'] = jnp.maximum(n['busy'] - 1, 0)
        n['worksite'] = jnp.where(n['busy'] > 0, n['worksite'], -1)
        n['tool_pos'] = pos[n['tool_holder']]
        n['score'] = s['score'] + payoff
        n['terminated'] = jnp.all(n['delivered'] >= 2) | (n['t'] >= 80)
        n['truncated'] = (n['t'] >= self.max_steps) & ~n['terminated']
        n['done'] = n['terminated'] | n['truncated']
        n = jax.tree.map(lambda fresh, old: jnp.where(s['done'], old, fresh), n, s)
        rewards = jnp.where(s['done'], 0, payoff).astype(jnp.float32)
        info = {'done_all': n['done'], 'terminated': n['terminated'], 'truncated': n['truncated']}
        return self.observe(n), n, rewards, jnp.full((3,), n['done'], dtype=jnp.bool_), info

    def observe(self, s):
        public = jnp.concatenate([
            s['t'].reshape(1) / 96.0, (s['agent_pos'] / 8.0).reshape(-1),
            s['fiber'] / 2.0, s['yarn'] / 2.0, s['dye'] / 2.0, s['cloth'] / 2.0,
            s['seeds'] / 2.0, s['busy'] / 2.0, (s['worksite'] + 1) / 2.0, s['delivered'] / 2.0,
            s['plot_stage'] / 2.0, s['plot_clock'] / 3.0, s['plot_yield'] / 2.0,
            s['health'] / 3.0, s['work_left'] / 2.0, (s['job_owner'] + 1) / 3.0,
            s['output'].astype(jnp.float32), (s['tool_holder'] + 1) / 3.0, s['loan_left'] / 8.0,
            (s['tool_pos'] / 8.0).reshape(-1), s['dye_stock'] / 6.0, s['score'] / 6.0,
        ])
        return jnp.concatenate([jnp.broadcast_to(public, (3, 60)), jnp.eye(3), s['deadline'][:, None] / 96.0], axis=1).astype(jnp.float32)
