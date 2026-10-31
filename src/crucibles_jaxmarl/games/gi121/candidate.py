"""Original shared-pollinator orchard. Authored source only; no numeric execution."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 177
    num_actions = 13
    max_steps = 96

    def reset(self, key):
        tiles = jnp.zeros((9, 9), jnp.int32)
        tiles = tiles.at[0, :].set(1)
        tiles = tiles.at[8, :].set(1)
        tiles = tiles.at[:, 0].set(1)
        tiles = tiles.at[:, 8].set(1)
        plots = jnp.array([[2, 2], [2, 6], [4, 2], [4, 6], [6, 2], [6, 6]], jnp.int32)
        tiles = tiles.at[plots[:, 1], plots[:, 0]].set(2)
        tiles = tiles.at[4, 2].set(3)
        tiles = tiles.at[4, 6].set(3)
        tiles = tiles.at[4, 4].set(4)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False), 'map': tiles,
            'pos': jnp.array([[2, 4], [4, 4], [6, 4]], jnp.int32),
            'agent_active': jnp.ones((3,), jnp.bool_), 'agent_visible': jnp.ones((3, 3), jnp.bool_),
            'seeds': jnp.ones((3,), jnp.int32), 'water': jnp.zeros((3,), jnp.int32),
            'harvests': jnp.zeros((3,), jnp.int32),
            'plot_pos': plots, 'plot_active': jnp.ones((6,), jnp.bool_), 'plot_visible': jnp.ones((3, 6), jnp.bool_),
            'plot_owner': jnp.array([0, 0, 1, 1, 2, 2], jnp.int32),
            'stage': jnp.zeros((6,), jnp.int32), 'age': jnp.zeros((6,), jnp.int32),
            'soil_water': jnp.zeros((6,), jnp.int32), 'pollinated': jnp.zeros((6,), jnp.bool_),
            'fruit': jnp.zeros((6,), jnp.int32),
            'bee_pos': jnp.array([[2, 4], [4, 4], [6, 4]], jnp.int32),
            'bee_active': jnp.ones((3,), jnp.bool_), 'bee_visible': jnp.ones((3, 3), jnp.bool_),
            'bee_visits_left': jnp.full((3,), 3, jnp.int32), 'bee_pollen': jnp.full((3,), -1, jnp.int32),
            'bee_last_plot': jnp.full((3,), -1, jnp.int32), 'bee_target': jnp.array([6, 6, 7], jnp.int32),
            'habitat_pos': jnp.array([[2, 4], [6, 4]], jnp.int32),
            'habitat_active': jnp.ones((2,), jnp.bool_), 'habitat_visible': jnp.ones((3, 2), jnp.bool_),
            'habitat_water': jnp.full((2,), 2, jnp.int32),
            'tank_pos': jnp.array([[4, 4]], jnp.int32),
            'tank_active': jnp.ones((1,), jnp.bool_), 'tank_visible': jnp.ones((3, 1), jnp.bool_),
            'tank_water': jnp.array([18], jnp.int32), 'water_spent': jnp.array(0, jnp.int32),
            'harvest_count': jnp.array(0, jnp.int32),
            'plant_event': jnp.zeros((6,), jnp.bool_), 'water_event': jnp.zeros((6,), jnp.bool_),
            'bloom_event': jnp.zeros((6,), jnp.bool_), 'pollination_event': jnp.zeros((6,), jnp.bool_),
            'ripe_event': jnp.zeros((6,), jnp.bool_), 'wilt_event': jnp.zeros((6,), jnp.bool_),
            'visit_event': jnp.zeros((3,), jnp.bool_), 'harvest_event': jnp.zeros((3,), jnp.bool_),
            'habitat_event': jnp.zeros((2,), jnp.bool_), 'seed_transfer_event': jnp.zeros((3,), jnp.bool_),
        }
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate((s['map'].reshape((81,)) / 4.0, s['pos'].reshape((6,)) / 8.0,
            s['seeds'] / 3.0, s['water'] / 2.0, s['harvests'] / 6.0,
            s['plot_pos'].reshape((12,)) / 8.0, s['plot_owner'] / 2.0, s['stage'] / 4.0,
            s['age'] / 6.0, s['soil_water'] / 2.0, s['pollinated'], s['fruit'] / 3.0,
            s['bee_pos'].reshape((6,)) / 8.0, s['bee_visits_left'] / 3.0, s['bee_pollen'] / 2.0,
            s['bee_last_plot'] / 5.0, s['bee_target'] / 7.0, s['habitat_pos'].reshape((4,)) / 8.0,
            s['habitat_water'] / 4.0, s['tank_pos'].reshape((2,)) / 8.0, s['tank_water'] / 18.0,
            jnp.array([s['t'] / self.max_steps, s['harvest_count'] / 6.0, s['water_spent'] / 22.0])))
        return jnp.concatenate((jnp.eye(3), jnp.broadcast_to(public, (3, 174))), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        return jnp.broadcast_to(~s['done'] | (jnp.arange(13) == 0), (3, 13)).astype(jnp.bool_)

    def advance(self, key, state, actions):
        s = dict(state)
        a = jnp.where((actions >= 0) & (actions < 13), actions, 0)
        direction = jnp.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        trial = s['pos'] + direction[a]
        floor = s['map'][jnp.clip(trial[:, 1], 0, 8), jnp.clip(trial[:, 0], 0, 8)] != 1
        occupied = jnp.any(jnp.all(trial[:, None, :] == s['pos'][None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        moving = (a >= 1) & (a <= 4) & floor & ~occupied
        proposed = jnp.where(moving[:, None], trial, s['pos'])
        clash = jnp.any(jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        s['pos'] = jnp.where((moving & ~clash)[:, None], trial, s['pos'])
        s['plant_event'] = jnp.zeros((6,), jnp.bool_)
        s['water_event'] = jnp.zeros((6,), jnp.bool_)
        s['pollination_event'] = jnp.zeros((6,), jnp.bool_)
        s['visit_event'] = jnp.zeros((3,), jnp.bool_)
        s['harvest_event'] = jnp.zeros((3,), jnp.bool_)
        s['habitat_event'] = jnp.zeros((2,), jnp.bool_)
        s['seed_transfer_event'] = jnp.zeros((3,), jnp.bool_)
        rewards = jnp.zeros((3,), jnp.float32)
        for offset in range(3):
            i = (s['t'] + offset) % 3
            distance = jnp.sum(jnp.abs(s['plot_pos'] - s['pos'][i]), axis=1)
            own = (s['plot_owner'] == i) & (distance <= 1)
            empty = own & (s['stage'] == 0)
            plant = jnp.argmin(jnp.where(empty, distance * 6 + jnp.arange(6), 1000))
            sow = (a[i] == 5) & (s['seeds'][i] > 0) & jnp.any(empty)
            s['stage'] = s['stage'].at[plant].set(jnp.where(sow, 1, s['stage'][plant]))
            s['age'] = s['age'].at[plant].set(jnp.where(sow, 0, s['age'][plant]))
            s['seeds'] = s['seeds'].at[i].add(-sow.astype(jnp.int32))
            s['plant_event'] = s['plant_event'].at[plant].set(s['plant_event'][plant] | sow)
            near_tank = jnp.sum(jnp.abs(s['pos'][i] - s['tank_pos'][0])) <= 1
            draw = (a[i] == 6) & near_tank & (s['water'][i] < 2) & (s['tank_water'][0] > 0)
            s['tank_water'] = s['tank_water'] - draw.astype(jnp.int32)
            s['water'] = s['water'].at[i].add(draw.astype(jnp.int32))
            thirsty = (distance <= 1) & (s['soil_water'] < 2)
            watered = jnp.argmin(jnp.where(thirsty, distance * 6 + jnp.arange(6), 1000))
            irrigate = (a[i] == 7) & (s['water'][i] > 0) & jnp.any(thirsty)
            s['soil_water'] = s['soil_water'].at[watered].add(irrigate.astype(jnp.int32))
            s['water'] = s['water'].at[i].add(-irrigate.astype(jnp.int32))
            s['water_event'] = s['water_event'].at[watered].set(s['water_event'][watered] | irrigate)
            ripe = own & (s['stage'] == 3)
            picked = jnp.argmax(ripe.astype(jnp.int32))
            harvest = (a[i] == 8) & jnp.any(ripe) & (s['harvest_count'] < 6)
            rewards = rewards.at[i].add(harvest.astype(jnp.float32) * s['fruit'][picked].astype(jnp.float32))
            s['harvests'] = s['harvests'].at[i].add(harvest.astype(jnp.int32))
            s['harvest_count'] = s['harvest_count'] + harvest.astype(jnp.int32)
            s['harvest_event'] = s['harvest_event'].at[i].set(harvest)
            rooted = own & (s['stage'] != 0)
            uprooted = jnp.argmax(rooted.astype(jnp.int32))
            uproot = (a[i] == 12) & jnp.any(rooted)
            clear = harvest | uproot
            cleared = jnp.where(harvest, picked, uprooted)
            s['stage'] = s['stage'].at[cleared].set(jnp.where(clear, 0, s['stage'][cleared]))
            s['age'] = s['age'].at[cleared].set(jnp.where(clear, 0, s['age'][cleared]))
            s['pollinated'] = s['pollinated'].at[cleared].set(jnp.where(clear, False, s['pollinated'][cleared]))
            s['fruit'] = s['fruit'].at[cleared].set(jnp.where(clear, 0, s['fruit'][cleared]))
            s['seeds'] = s['seeds'].at[i].add(clear.astype(jnp.int32))
            habitat_distance = jnp.sum(jnp.abs(s['habitat_pos'] - s['pos'][i]), axis=1)
            habitat = jnp.argmin(habitat_distance)
            maintain = (a[i] == 9) & (habitat_distance[habitat] <= 1) & (s['water'][i] > 0) & (s['habitat_water'][habitat] < 4)
            s['habitat_water'] = s['habitat_water'].at[habitat].add(maintain.astype(jnp.int32))
            s['water'] = s['water'].at[i].add(-maintain.astype(jnp.int32))
            s['habitat_event'] = s['habitat_event'].at[habitat].set(s['habitat_event'][habitat] | maintain)
            receiver = jnp.where(a[i] == 10, (i + 2) % 3, (i + 1) % 3)
            adjacent = jnp.sum(jnp.abs(s['pos'][i] - s['pos'][receiver])) == 1
            give = ((a[i] == 10) | (a[i] == 11)) & adjacent & (s['seeds'][i] > 0)
            s['seeds'] = s['seeds'].at[i].add(-give.astype(jnp.int32))
            s['seeds'] = s['seeds'].at[receiver].add(give.astype(jnp.int32))
            s['seed_transfer_event'] = s['seed_transfer_event'].at[i].set(give)
        growing = (s['stage'] == 1) & (s['soil_water'] > 0)
        s['age'] = jnp.where(growing, jnp.minimum(2, s['age'] + 1), s['age'])
        bloom = (s['stage'] == 1) & (s['age'] >= 2) & (s['soil_water'] > 0)
        s['stage'] = jnp.where(bloom, 2, s['stage'])
        s['age'] = jnp.where(bloom, 0, s['age'])
        s['soil_water'] = s['soil_water'] - bloom.astype(jnp.int32)
        s['water_spent'] = s['water_spent'] + jnp.sum(bloom.astype(jnp.int32), dtype=jnp.int32)
        s['bloom_event'] = bloom
        phases = jax.random.randint(key, (3,), 0, 6, dtype=jnp.int32)
        for b in range(3):
            floral_distance = jnp.sum(jnp.abs(s['plot_pos'] - s['bee_pos'][b]), axis=1)
            candidates = (s['stage'] == 2) & (jnp.arange(6) != s['bee_last_plot'][b])
            rank = floral_distance * 8 + ((jnp.arange(6) + phases[b]) % 6)
            flower = jnp.argmin(jnp.where(candidates, rank, 1000))
            habitat_distance = jnp.sum(jnp.abs(s['habitat_pos'] - s['bee_pos'][b]), axis=1)
            available = s['habitat_water'] > 0
            habitat_rank = jnp.where(available | ~jnp.any(available), habitat_distance, 1000)
            habitat = jnp.argmin(habitat_rank)
            forage = (s['bee_visits_left'][b] > 0) & jnp.any(candidates)
            target = jnp.where(forage, s['plot_pos'][flower], s['habitat_pos'][habitat])
            difference = target - s['bee_pos'][b]
            x_first = ((phases[b] % 2) == 0) & (difference[0] != 0)
            x_first = x_first | (difference[1] == 0)
            step_x = jnp.where(x_first, jnp.sign(difference[0]), 0)
            step_y = jnp.where(x_first, 0, jnp.sign(difference[1]))
            new_pos = s['bee_pos'][b] + jnp.array([step_x, step_y], jnp.int32)
            s['bee_pos'] = s['bee_pos'].at[b].set(new_pos)
            s['bee_target'] = s['bee_target'].at[b].set(jnp.where(forage, flower, 6 + habitat))
            reached = jnp.all(new_pos == target)
            visit = forage & reached
            cross = visit & (s['bee_pollen'][b] >= 0) & (s['bee_pollen'][b] != s['plot_owner'][flower])
            s['pollination_event'] = s['pollination_event'].at[flower].set(s['pollination_event'][flower] | (cross & ~s['pollinated'][flower]))
            s['pollinated'] = s['pollinated'].at[flower].set(s['pollinated'][flower] | cross)
            s['bee_pollen'] = s['bee_pollen'].at[b].set(jnp.where(visit, s['plot_owner'][flower], s['bee_pollen'][b]))
            s['bee_last_plot'] = s['bee_last_plot'].at[b].set(jnp.where(visit, flower, s['bee_last_plot'][b]))
            s['bee_visits_left'] = s['bee_visits_left'].at[b].add(-visit.astype(jnp.int32))
            recharge = ~forage & reached & (s['bee_visits_left'][b] < 3) & (s['habitat_water'][habitat] > 0)
            s['habitat_water'] = s['habitat_water'].at[habitat].add(-recharge.astype(jnp.int32))
            s['water_spent'] = s['water_spent'] + recharge.astype(jnp.int32)
            s['bee_visits_left'] = s['bee_visits_left'].at[b].set(jnp.where(recharge, 3, s['bee_visits_left'][b]))
            s['bee_last_plot'] = s['bee_last_plot'].at[b].set(jnp.where(recharge, -1, s['bee_last_plot'][b]))
            s['visit_event'] = s['visit_event'].at[b].set(visit)
        flowering = s['stage'] == 2
        s['age'] = jnp.where(flowering, jnp.minimum(6, s['age'] + 1), s['age'])
        ripen = flowering & (s['age'] >= 4) & (s['soil_water'] > 0)
        wilt = flowering & (s['age'] >= 6) & ~ripen
        s['stage'] = jnp.where(ripen, 3, jnp.where(wilt, 4, s['stage']))
        s['fruit'] = jnp.where(ripen, 1 + 2 * s['pollinated'].astype(jnp.int32), s['fruit'])
        s['soil_water'] = s['soil_water'] - ripen.astype(jnp.int32)
        s['water_spent'] = s['water_spent'] + jnp.sum(ripen.astype(jnp.int32), dtype=jnp.int32)
        s['ripe_event'] = ripen
        s['wilt_event'] = wilt
        s['t'] = s['t'] + jnp.array(1, jnp.int32)
        remaining_water = s['tank_water'][0] + jnp.sum(s['water']) + jnp.sum(s['soil_water']) + jnp.sum(s['habitat_water'])
        dry_end = (remaining_water == 0) & ~jnp.any(s['stage'] == 3)
        s['terminated'] = (s['harvest_count'] >= 6) | dry_end
        s['truncated'] = (s['t'] >= self.max_steps) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        info = {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
        return self.observe(s), s, rewards, jnp.full((3,), s['done'], jnp.bool_), info

    def absorb(self, state):
        info = {'done_all': state['done'], 'terminated': state['terminated'], 'truncated': state['truncated']}
        return self.observe(state), state, jnp.zeros((3,), jnp.float32), jnp.ones((3,), jnp.bool_), info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state['done'], lambda _: self.absorb(state),
            lambda _: self.advance(key, state, actions), operand=None)
