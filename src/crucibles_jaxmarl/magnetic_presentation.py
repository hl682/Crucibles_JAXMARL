"""GI214 original artwork driven only by source-pinned native observations.

Interpolation joins two committed positions for display, never advances physics.
All six roles see the full native world in this particular game. Role -1 remains
explicit debug; this adapter must not be reused for a partially observed game.
"""
import base64
import copy
import io

import numpy as np
from PIL import ImageDraw, ImageFont

from . import magnetic_art as art
from .rendering import scene_frame

SOURCE = '3a95595a4884c363195d29b2f7ca9caf00b8b92a91227d12f6fd5562005c959d'
PUBLIC = ('charge', 'weather', 'deliveries', 'builds')


def _require_env(env):
    if (env.name != 'gi214' or env.metadata['source_sha256'] != SOURCE
            or env.manifest['source_sha256'] != SOURCE
            or env.scene.get('source_sha256') != SOURCE
            or env.manifest.get('observability') != 'fully_observable'):
        raise ValueError('Magnet artwork requires the exact fully observable GI214 source')


def make_view(env, state, obs, role=0):
    _require_env(env)
    if type(role) is not int or role not in range(-1, env.num_agents):
        raise ValueError('Role must be -1 (explicit debug) or an actual GI214 crew')
    actual = np.asarray(env.observe(state))
    supplied = (np.stack([np.asarray(obs[a]) for a in env.agents])
                if isinstance(obs, dict) else np.asarray(obs))
    if supplied.shape != actual.shape or not np.array_equal(supplied, actual):
        raise ValueError('View observations must belong to the supplied committed state')
    frame = scene_frame(env, state, obs, role)
    world = env.manifest['world']['map']
    return dict(
        source_sha256=SOURCE,
        map_geometry=dict(shape=list(np.asarray(state['terrain']).shape),
                          cell_origin_xy=world['cell_origin_xy'],
                          cell_size_xy=world['cell_size_xy']),
        public_map=frame['map'], entities=frame['entities'], clock=frame['clock'],
        view='PRIVILEGED_ACTUAL_STATE' if role == -1 else 'ROLE_' + str(role),
        public_scene_fields={k: np.asarray(state[k]).tolist() for k in PUBLIC})


def changes(before, after):
    """Observed changes, not inferred causes; inputs are role-filtered views."""
    events = []
    old = {(e['group'], e['index']): e for e in before['entities']}
    for e in after['entities']:
        prior = old.get((e['group'], e['index']))
        a = e['attributes']
        if prior is None:
            if e['group'] == 'magnets':
                events.append(dict(kind='build', label='Magnet built', xy=e['xy']))
            continue
        b = prior['attributes']
        if e['group'] == 'deposits' and a['stock'] != b['stock']:
            delta = a['stock'] - b['stock']
            events.append(dict(kind='stock', label=f'Stock {delta:+d}', xy=e['xy']))
        if e['group'] == 'crews' and a['inventory'] != b['inventory']:
            events.append(dict(kind='inventory', label='Inventory changed', xy=e['xy']))
        if e['group'] == 'magnets':
            for key, labels in [('magnet_on', ('Requested OFF', 'Requested ON')),
                                ('magnet_powered', ('Supply OFF', 'Supply ON'))]:
                if a[key] != b[key]:
                    events.append(dict(kind='power', label=labels[bool(a[key])], xy=e['xy']))
            if a['polarity'] != b['polarity']:
                events.append(dict(kind='polarity', label='Pull' if a['polarity'] == 1 else 'Repel', xy=e['xy']))
        if e['group'] == 'freight':
            if e['xy'] != prior['xy']:
                events.append(dict(kind='cargo', label='Cargo moved', xy=e['xy']))
            for key, label in [('jam', 'Jammed'), ('anchored', 'Anchored')]:
                if a[key] and not b[key]:
                    events.append(dict(kind=key, label=label, xy=e['xy']))
    for y, (oldrow, row) in enumerate(zip(before['public_map'], after['public_map'])):
        for x, (a, b) in enumerate(zip(oldrow, row)):
            if a != b:
                events.append(dict(kind='terrain', label={2: 'Bridge built', 3: 'Rail laid', 4: 'Bridge rail laid'}.get(b, 'Terrain changed'), xy=[x, y]))
    return events


def draw(env, view, previous=None, phase=1., events=(), full_frame=False):
    """Render original scene geometry. Optional full frame includes English HUD."""
    _require_env(env)
    if view.get('source_sha256') != SOURCE or (previous is not None and previous.get('source_sha256') != SOURCE):
        raise ValueError('Unpinned magnetic view')
    phase = max(0., min(1., float(phase)))
    v = copy.deepcopy(view)
    if previous is not None and phase < 1:
        old = {(e['group'], e['index']): e for e in previous['entities']}
        for e in v['entities']:
            prior = old.get((e['group'], e['index']))
            if prior is not None:
                e['xy'] = [a + (b-a)*phase for a, b in zip(prior['xy'], e['xy'])]
    image = art.render(env.manifest, env.scene, v, '', v['clock'], env.max_steps+1)
    d = ImageDraw.Draw(image)
    cell, ox, oy = art.geometry(v)
    font = ImageFont.load_default(size=11)
    # A small pack icon shows carried freight without moving its native position.
    # Original freight and specialist sprites remain; this is a display offset.
    for e in v['entities']:
        if e['group'] == 'freight' and e['attributes']['holder'] >= 0:
            x = ox + (e['xy'][0]+.5)*cell + cell*.28
            y = oy + (e['xy'][1]+.5)*cell - cell*.26
            d.rectangle((x-5, y-5, x+5, y+5), fill='#d8ad62', outline='#f0d392', width=1)
    # Pulse only locations where a committed native value actually changed.
    # There are no invented field lines, forces, trajectories or extra resources.
    for e in events:
        x = ox + (e['xy'][0]+.5)*cell
        y = oy + (e['xy'][1]+.5)*cell
        r = cell*(.38+.10*phase)
        tint = '#8ed4cc' if e['kind'] in ('power', 'polarity', 'cargo') else '#f0cf8b'
        d.rectangle((x-r, y-r, x+r, y+r), outline=tint, width=2)
    if full_frame:
        return image
    return image.crop(art.SCENE)


def animation(env, state, obs, role, previous_state=None, previous_obs=None):
    view = make_view(env, state, obs, role)
    old = make_view(env, previous_state, previous_obs, role) if previous_state is not None else None
    events = changes(old, view) if old is not None else []
    frames = []
    for phase in ([.25, .5, .75, 1.] if old is not None else [1.]):
        image = draw(env, view, old, phase, events)
        output = io.BytesIO()
        image.save(output, format='PNG')
        frames.append('data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii'))
    cell, ox, oy = art.geometry(view)
    own = next((e for e in view['entities'] if e['group'] == 'crews' and e['index'] == role), None)
    return dict(frames=frames, duration_ms=130, width=730, height=586,
                geometry=dict(cell=cell, ox=ox-art.SCENE[0], oy=oy-art.SCENE[1]),
                events=[e['label'] for e in events],
                inventory=None if own is None else own['attributes']['inventory'],
                inventory_labels=['ore', 'timber', 'coil', 'brace', 'power pack', 'reserved'],
                shared=view['public_scene_fields'], information=view['view'],
                source_sha256=SOURCE)
