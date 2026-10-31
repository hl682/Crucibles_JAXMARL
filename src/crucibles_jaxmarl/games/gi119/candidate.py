"""Original circuit logistics game; static author package, never a run receipt."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents = 3
    observation_size = 355
    num_actions = 12
    max_steps = 112

    def network(self, s):
        edges = jnp.arange(8, dtype=jnp.int32)
        installed = s['object_mode'] == 2
        cable = installed & (s['object_kind'] == 0)
        switch = installed & (s['object_kind'] == 1)
        present = jnp.any(cable[:, None] & (s['slot'][:, None] == edges[None, :]), axis=0)
        opened = jnp.any(switch[:, None] & ~s['on'][:, None] & (s['slot'][:, None] == edges[None, :]), axis=0)
        closed = present & ~opened
        adjacency = jnp.eye(7, dtype=jnp.bool_)
        adjacency = adjacency.at[s['edge_ends'][:, 0], s['edge_ends'][:, 1]].set(closed)
        adjacency = adjacency.at[s['edge_ends'][:, 1], s['edge_ends'][:, 0]].set(closed)
        reach = adjacency
        for k in range(7):
            reach = reach | (reach[:, k:k + 1] & reach[k:k + 1, :])
        battery = installed & (s['object_kind'] == 2) & s['on'] & (s['charge'] > 0) & (s['trip'] == 0)
        sources = jnp.sum((battery[:, None] & (s['slot'][:, None] == jnp.arange(7)[None, :])).astype(jnp.int32), axis=0, dtype=jnp.int32)
        demand = jnp.array([0, 0, 0, 0, 0, 0, 1], jnp.int32)
        demand = demand.at[4:6].set(2 * ((s['device_mode'] == 1) & (s['progress'] < 4)).astype(jnp.int32))
        supply = jnp.sum(reach.astype(jnp.int32) * sources[None, :], axis=1, dtype=jnp.int32)
        load = jnp.sum(reach.astype(jnp.int32) * demand[None, :], axis=1, dtype=jnp.int32)
        overload = (supply > 0) & (load > 3 * supply)
        powered = (supply > 0) & ~overload
        return present, closed, reach, demand, load, powered, overload, battery

    def decorate(self, state):
        s = dict(state)
        present, closed, reach, demand, load, powered, overload, battery = self.network(s)
        s['edge_present'] = present
        s['edge_closed'] = closed
        s['connectivity'] = reach
        s['node_demand'] = demand
        s['component_load'] = load
        s['node_powered'] = powered
        s['node_overload'] = overload
        s['edge_powered'] = closed & powered[s['edge_ends'][:, 0]] & powered[s['edge_ends'][:, 1]]
        s['gate_open'] = powered[6:7]
        carried = s['object_mode'] == 1
        s['object_pos'] = jnp.where(carried[:, None], s['pos'][jnp.clip(s['carrier'], 0, 2)], s['object_pos'])
        owned = carried[:, None] & (s['carrier'][:, None] == jnp.arange(3)[None, :])
        s['held'] = jnp.where(jnp.any(owned, axis=0), jnp.argmax(owned.astype(jnp.int32), axis=0), -1).astype(jnp.int32)
        proximity = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['facility_pos'][None, :, :]), axis=2) <= 1
        s['tool_power'] = jnp.any(proximity & powered[None, 4:6], axis=1)
        return s

    def reset(self, key):
        tiles = jnp.zeros((9, 9), jnp.int32)
        tiles = tiles.at[0, :].set(1)
        tiles = tiles.at[8, :].set(1)
        tiles = tiles.at[:, 0].set(1)
        tiles = tiles.at[:, 8].set(1)
        tiles = tiles.at[3:6, 5].set(1)
        tiles = tiles.at[4, 5].set(2)
        s = {
            't': jnp.array(0, jnp.int32), 'done': jnp.array(False),
            'terminated': jnp.array(False), 'truncated': jnp.array(False), 'map': tiles,
            'pos': jnp.array([[2, 4], [3, 3], [1, 4]], jnp.int32),
            'agent_active': jnp.ones((3,), jnp.bool_), 'agent_visible': jnp.ones((3, 3), jnp.bool_),
            'role': jnp.arange(3, dtype=jnp.int32), 'service_rate': jnp.array([1, 2, 1], jnp.int32),
            'held': jnp.array([0, 6, 8], jnp.int32), 'tool_power': jnp.zeros((3,), jnp.bool_),
            'object_pos': jnp.array([[2, 4], [3, 4], [3, 4], [4, 2], [4, 6], [5, 2], [3, 3], [3, 4], [1, 4], [1, 5]], jnp.int32),
            'object_active': jnp.ones((10,), jnp.bool_), 'object_visible': jnp.ones((3, 10), jnp.bool_),
            'object_kind': jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], jnp.int32),
            'object_mode': jnp.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 0], jnp.int32),
            'carrier': jnp.array([0, -1, -1, -1, -1, -1, 1, -1, 2, -1], jnp.int32),
            'slot': jnp.full((10,), -1, jnp.int32), 'on': jnp.ones((10,), jnp.bool_),
            'charge': jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 12, 12], jnp.int32),
            'trip': jnp.zeros((10,), jnp.int32),
            'energy_created': jnp.array(24, jnp.int32), 'energy_spent': jnp.array(0, jnp.int32),
            'socket_pos': jnp.array([[2, 4], [4, 3], [4, 5], [6, 2], [6, 6], [4, 4], [6, 3], [6, 5]], jnp.int32),
            'socket_active': jnp.ones((8,), jnp.bool_), 'socket_visible': jnp.ones((3, 8), jnp.bool_),
            'edge_ends': jnp.array([[0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [1, 6], [6, 4], [6, 5]], jnp.int32),
            'edge_present': jnp.zeros((8,), jnp.bool_), 'edge_closed': jnp.zeros((8,), jnp.bool_),
            'edge_powered': jnp.zeros((8,), jnp.bool_), 'connectivity': jnp.eye(7, dtype=jnp.bool_),
            'node_pos': jnp.array([[1, 4], [3, 4], [5, 2], [5, 6], [7, 2], [7, 6], [5, 4]], jnp.int32),
            'node_active': jnp.ones((7,), jnp.bool_), 'node_visible': jnp.ones((3, 7), jnp.bool_),
            'node_demand': jnp.zeros((7,), jnp.int32), 'component_load': jnp.zeros((7,), jnp.int32),
            'node_powered': jnp.zeros((7,), jnp.bool_), 'node_overload': jnp.zeros((7,), jnp.bool_),
            'power_delivered': jnp.zeros((7,), jnp.bool_),
            'facility_pos': jnp.array([[7, 2], [7, 6]], jnp.int32),
            'facility_active': jnp.ones((2,), jnp.bool_), 'facility_visible': jnp.ones((3, 2), jnp.bool_),
            'progress': jnp.zeros((2,), jnp.int32), 'device_mode': jnp.ones((2,), jnp.int32),
            'gate_pos': jnp.array([[5, 4]], jnp.int32),
            'gate_active': jnp.ones((1,), jnp.bool_), 'gate_visible': jnp.ones((3, 1), jnp.bool_),
            'gate_open': jnp.zeros((1,), jnp.bool_),
            'install_event': jnp.zeros((3,), jnp.bool_), 'handoff_event': jnp.zeros((3,), jnp.bool_),
            'power_event': jnp.zeros((7,), jnp.bool_), 'overload_event': jnp.zeros((7,), jnp.bool_),
            'service_event': jnp.zeros((2,), jnp.bool_), 'charge_event': jnp.zeros((3,), jnp.bool_),
        }
        s = self.decorate(s)
        return self.observe(s), s

    def observe(self, s):
        public = jnp.concatenate((s['map'].reshape((81,)) / 2.0, s['pos'].reshape((6,)) / 8.0,
            s['held'] / 9.0, s['tool_power'], s['object_pos'].reshape((20,)) / 8.0,
            s['object_kind'] / 2.0, s['object_mode'] / 2.0, s['carrier'] / 2.0, s['slot'] / 7.0,
            s['on'], s['charge'] / 12.0, s['trip'] / 3.0, s['edge_present'], s['edge_closed'],
            s['edge_powered'], s['connectivity'].reshape((49,)), s['node_powered'], s['node_overload'],
            s['node_demand'] / 2.0, s['component_load'] / 5.0, s['progress'] / 4.0,
            s['device_mode'], s['power_delivered'], s['gate_open'], s['socket_pos'].reshape((16,)) / 8.0,
            s['node_pos'].reshape((14,)) / 8.0, s['edge_ends'].reshape((16,)) / 6.0,
            s['facility_pos'].reshape((4,)) / 8.0, s['gate_pos'].reshape((2,)) / 8.0,
            jnp.array([s['t'] / self.max_steps, s['energy_created'] / 136.0, s['energy_spent'] / 136.0])))
        own = jnp.concatenate((jnp.eye(3), s['service_rate'][:, None] / 2.0), axis=1)
        return jnp.concatenate((own, jnp.broadcast_to(public, (3, 351))), axis=1).astype(jnp.float32)

    def legal_actions(self, s):
        return jnp.broadcast_to(~s['done'] | (jnp.arange(12) == 0), (3, 12)).astype(jnp.bool_)

    def advance(self, key, state, actions):
        s = dict(state)
        a = jnp.where((actions >= 0) & (actions < 12), actions, 0)
        direction = jnp.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32)
        trial = s['pos'] + direction[a]
        tile = s['map'][jnp.clip(trial[:, 1], 0, 8), jnp.clip(trial[:, 0], 0, 8)]
        free = (tile == 0) | ((tile == 2) & s['gate_open'][0])
        occupied = jnp.any(jnp.all(trial[:, None, :] == s['pos'][None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        moving = (a >= 1) & (a <= 4) & free & ~occupied
        proposed = jnp.where(moving[:, None], trial, s['pos'])
        clash = jnp.any(jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=2) & ~jnp.eye(3, dtype=jnp.bool_), axis=1)
        s['pos'] = jnp.where((moving & ~clash)[:, None], trial, s['pos'])
        s['trip'] = jnp.maximum(s['trip'] - 1, 0)
        s['install_event'] = jnp.zeros((3,), jnp.bool_)
        s['handoff_event'] = jnp.zeros((3,), jnp.bool_)
        s['charge_event'] = jnp.zeros((3,), jnp.bool_)
        for offset in range(3):
            i = (s['t'] + offset) % 3
            carrying = (s['object_mode'] == 1) & (s['carrier'] == i)
            has_item = jnp.any(carrying)
            held = jnp.argmax(carrying.astype(jnp.int32))
            distance = jnp.sum(jnp.abs(s['object_pos'] - s['pos'][i]), axis=1)
            loose = (s['object_mode'] == 0) & (distance <= 1)
            pick = jnp.argmax(loose.astype(jnp.int32))
            take = (a[i] == 5) & ~has_item & jnp.any(loose)
            drop = (a[i] == 5) & has_item
            s['object_mode'] = s['object_mode'].at[pick].set(jnp.where(take, 1, s['object_mode'][pick]))
            s['carrier'] = s['carrier'].at[pick].set(jnp.where(take, i, s['carrier'][pick]))
            s['object_mode'] = s['object_mode'].at[held].set(jnp.where(drop, 0, s['object_mode'][held]))
            s['carrier'] = s['carrier'].at[held].set(jnp.where(drop, -1, s['carrier'][held]))
            s['object_pos'] = s['object_pos'].at[held].set(jnp.where(drop, s['pos'][i], s['object_pos'][held]))
            kind = s['object_kind'][held]
            edge_free = ~jnp.any((s['object_mode'][:, None] == 2) & (s['object_kind'][:, None] == kind) &
                (s['slot'][:, None] == jnp.arange(8)[None, :]), axis=0)
            socket_distance = jnp.sum(jnp.abs(s['socket_pos'] - s['pos'][i]), axis=1)
            eligible = edge_free & (socket_distance <= 1)
            edge = jnp.argmin(jnp.where(eligible, socket_distance * 8 + jnp.arange(8), 1000))
            pads = jnp.array([0, 6], jnp.int32)
            pad_distance = jnp.sum(jnp.abs(s['node_pos'][pads] - s['pos'][i]), axis=1)
            pad = pads[jnp.argmin(pad_distance)]
            place_ok = jnp.where(kind == 2, jnp.min(pad_distance) <= 1, jnp.any(eligible))
            install = (a[i] == 6) & has_item & (kind == s['role'][i]) & place_ok
            target_slot = jnp.where(kind == 2, pad, edge)
            target_pos = jnp.where(kind == 2, s['node_pos'][pad], s['socket_pos'][edge])
            s['object_mode'] = s['object_mode'].at[held].set(jnp.where(install, 2, s['object_mode'][held]))
            s['carrier'] = s['carrier'].at[held].set(jnp.where(install, -1, s['carrier'][held]))
            s['slot'] = s['slot'].at[held].set(jnp.where(install, target_slot, s['slot'][held]))
            s['object_pos'] = s['object_pos'].at[held].set(jnp.where(install, target_pos, s['object_pos'][held]))
            near_installed = (s['object_mode'] == 2) & (distance <= 1) & (s['object_kind'] == s['role'][i])
            retrieve_id = jnp.argmax(near_installed.astype(jnp.int32))
            retrieve = (a[i] == 6) & ~has_item & jnp.any(near_installed)
            s['object_mode'] = s['object_mode'].at[retrieve_id].set(jnp.where(retrieve, 1, s['object_mode'][retrieve_id]))
            s['carrier'] = s['carrier'].at[retrieve_id].set(jnp.where(retrieve, i, s['carrier'][retrieve_id]))
            s['slot'] = s['slot'].at[retrieve_id].set(jnp.where(retrieve, -1, s['slot'][retrieve_id]))
            s['install_event'] = s['install_event'].at[i].set(install | retrieve)
            controllable = (s['object_mode'] == 2) & (distance <= 1) & (s['object_kind'] != 0)
            control = jnp.argmax(controllable.astype(jnp.int32))
            toggle = (a[i] == 7) & jnp.any(controllable)
            s['on'] = s['on'].at[control].set(jnp.where(toggle, ~s['on'][control], s['on'][control]))
            owner_count = jnp.sum(((s['object_mode'][:, None] == 1) & (s['carrier'][:, None] == jnp.arange(3)[None, :])).astype(jnp.int32), axis=0)
            neighbors = (jnp.sum(jnp.abs(s['pos'] - s['pos'][i]), axis=1) == 1) & (owner_count == 0)
            receiver = jnp.argmax(neighbors.astype(jnp.int32))
            transfer = (a[i] == 10) & has_item & jnp.any(neighbors)
            s['carrier'] = s['carrier'].at[held].set(jnp.where(transfer, receiver, s['carrier'][held]))
            s['handoff_event'] = s['handoff_event'].at[i].set(transfer)
            at_pedal = jnp.sum(jnp.abs(s['pos'][i] - s['node_pos'][0])) <= 1
            recharge = (a[i] == 9) & (s['role'][i] == 2) & has_item & (kind == 2) & at_pedal & (s['charge'][held] < 12)
            s['charge'] = s['charge'].at[held].add(recharge.astype(jnp.int32))
            s['energy_created'] = s['energy_created'] + recharge.astype(jnp.int32)
            s['charge_event'] = s['charge_event'].at[i].set(recharge)
            facility_distance = jnp.sum(jnp.abs(s['facility_pos'] - s['pos'][i]), axis=1)
            device = jnp.argmin(facility_distance)
            mode_toggle = (a[i] == 11) & (facility_distance[device] <= 1) & (s['progress'][device] < 4)
            s['device_mode'] = s['device_mode'].at[device].set(jnp.where(mode_toggle, 1 - s['device_mode'][device], s['device_mode'][device]))
        present, closed, reach, demand, load, powered, overload, battery = self.network(s)
        s['power_delivered'] = powered
        s['overload_event'] = overload
        source = jnp.clip(s['slot'], 0, 6)
        draw = battery & (load[source] > 0)
        used = jnp.minimum(s['charge'], jnp.where(draw, 1 + overload[source].astype(jnp.int32), 0))
        s['charge'] = s['charge'] - used
        s['energy_spent'] = s['energy_spent'] + jnp.sum(used, dtype=jnp.int32)
        s['trip'] = jnp.where(draw & overload[source], 3, s['trip'])
        old_progress = s['progress']
        for i in range(3):
            distance = jnp.sum(jnp.abs(s['facility_pos'] - s['pos'][i]), axis=1)
            facility = jnp.argmin(distance)
            work = (a[i] == 8) & (distance[facility] <= 1) & powered[facility + 4] & (s['device_mode'][facility] == 1)
            increase = work.astype(jnp.int32) * s['service_rate'][i]
            s['progress'] = s['progress'].at[facility].set(jnp.minimum(4, s['progress'][facility] + increase))
        s['service_event'] = s['progress'] > old_progress
        s['device_mode'] = jnp.where(s['progress'] == 4, 0, s['device_mode'])
        reward = jnp.sum((s['progress'] - old_progress).astype(jnp.float32)) / 8.0
        s = self.decorate(s)
        s['power_event'] = s['node_powered'] != state['node_powered']
        s['t'] = s['t'] + jnp.array(1, jnp.int32)
        s['terminated'] = jnp.all(s['progress'] == 4)
        s['truncated'] = (s['t'] >= self.max_steps) & ~s['terminated']
        s['done'] = s['terminated'] | s['truncated']
        info = {'done_all': s['done'], 'terminated': s['terminated'], 'truncated': s['truncated']}
        return self.observe(s), s, jnp.full((3,), reward, jnp.float32), jnp.full((3,), s['done'], jnp.bool_), info

    def absorb(self, state):
        info = {'done_all': state['done'], 'terminated': state['terminated'], 'truncated': state['truncated']}
        return self.observe(state), state, jnp.zeros((3,), jnp.float32), jnp.ones((3,), jnp.bool_), info

    def raw_step(self, key, state, actions):
        return jax.lax.cond(state['done'], lambda _: self.absorb(state),
            lambda _: self.advance(key, state, actions), operand=None)
