"""GI217 original orchard artwork; only native role-visible records enter art.

Public weather and facilities remain public. Hidden inventory, particles and
private achievements are never passed to a normal-role drawing. Interpolation
is display-only and never calls the environment transition.
"""
import base64
import copy
import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import ImageDraw
from . import orchard_art as art
from .rendering import scene_frame

SOURCE = '78a651a4ff09f8353eddeb74fcd42a854cfc1f95df4cd61cbab49e1920161c66'
PUBLIC = ('wind', 'rain', 'daylight', 'reservoir', 'roof')


def _require_env(env):
    source = Path(__file__).parent / 'games/gi217/candidate.py'
    if (env.name != 'gi217' or env.metadata['source_sha256'] != SOURCE
            or env.manifest['source_sha256'] != SOURCE
            or env.scene.get('source_sha256') != SOURCE
            or env.manifest.get('observability') != 'partially_observable'
            or hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE):
        raise ValueError('Orchard artwork requires the exact partially observable GI217 source')


def make_view(env, state, obs, role=0):
    _require_env(env)
    if type(role) is not int or role not in range(-1, env.num_agents):
        raise ValueError('Role must be -1 (explicit debug) or an actual GI217 gardener')
    actual = np.asarray(env.observe(state))
    supplied = (np.stack([np.asarray(obs[a]) for a in env.agents])
                if isinstance(obs, dict) else np.asarray(obs))
    if actual.shape != supplied.shape or not np.array_equal(actual, supplied):
        raise ValueError('View observations must belong to the supplied committed state')
    frame = scene_frame(env, state, obs, role)
    world = env.manifest['world']['map']
    # These exact five fields are native public_keys, not reconstructed secrets.
    return dict(source_sha256=SOURCE,
                map_geometry=dict(shape=list(np.asarray(state['terrain']).shape),
                                  cell_origin_xy=world['cell_origin_xy'],
                                  cell_size_xy=world['cell_size_xy']),
                public_map=frame['map'], entities=frame['entities'], clock=frame['clock'],
                view='PRIVILEGED_ACTUAL_STATE' if role == -1 else 'ROLE_' + str(role),
                public_scene_fields={k: np.asarray(state[k]).tolist() for k in PUBLIC})


def changes(before, after):
    """Describe visible committed consequences, never infer hidden events."""
    old = {(e['group'], e['index']): e for e in before['entities']}
    events = []
    labels = {'growth': 'Growth', 'trait': 'Crop changed', 'moisture': 'Soil water',
              'seedbank': 'Ripe seedbank', 'wet': 'Wet seeds', 'dry': 'Dry seeds',
              'dry_timer': 'Drying progress', 'basket_slot': 'Drying slot',
              'water': 'Carried water', 'deployed': 'Windbreak', 'break_axis': 'Windbreak rotated'}
    for e in after['entities']:
        prior = old.get((e['group'], e['index']))
        if prior is None:
            if e['group'] == 'drifting_seeds':
                events.append(dict(kind='seed', label='Seed now visible', xy=e['xy']))
            continue
        if e['group'] == 'drifting_seeds' and e['xy'] != prior['xy']:
            events.append(dict(kind='seed', label='Seed drifted', xy=e['xy']))
        for key, label in labels.items():
            if key in e['attributes'] and e['attributes'][key] != prior['attributes'][key]:
                events.append(dict(kind=key, label=label, xy=e['xy']))
    if before['public_scene_fields']['reservoir'] != after['public_scene_fields']['reservoir']:
        events.append(dict(kind='well', label='Shared well water changed', xy=[12, 12]))
    return events


def draw(env, view, previous=None, phase=1., events=(), full_frame=False):
    _require_env(env)
    if view.get('source_sha256') != SOURCE or (previous is not None and previous.get('source_sha256') != SOURCE):
        raise ValueError('Unpinned orchard view')
    if previous is not None and previous['view'] != view['view']:
        raise ValueError('Cannot interpolate across roles')
    phase = max(0., min(1., float(phase)))
    v = copy.deepcopy(view)
    if previous is not None and phase < 1:
        old = {(e['group'], e['index']): e for e in previous['entities']}
        for e in v['entities']:
            prior = old.get((e['group'], e['index']))
            if prior is not None:
                e['xy'] = [a + (b-a)*phase for a,b in zip(prior['xy'], e['xy'])]
    image = art.render(env.manifest, env.scene, v, '', v['clock'], env.max_steps+1)
    d = ImageDraw.Draw(image)
    cell, ox, oy = art.geometry(v)
    # Filter externally supplied event locations against the actual visible view.
    visible = {tuple(e['xy']) for e in view['entities']}
    for event in events:
        if tuple(event['xy']) not in visible:
            continue
        x, y = ox+(event['xy'][0]+.5)*cell, oy+(event['xy'][1]+.5)*cell
        r = cell*(.36+.12*phase)
        d.ellipse((x-r,y-r,x+r,y+r), outline='#efd689', width=2)
    # Explicit label is inside scene crop, including debug mode.
    d.rectangle((22,136,350,155), fill='#22352f')
    d.text((27,138), 'FULL STATE DEBUG' if v['view']=='PRIVILEGED_ACTUAL_STATE'
           else 'Gardener '+v['view'].split('_')[-1]+' - native local view', font=art.font(11), fill='#fff1b3')
    return image if full_frame else image.crop(art.SCENE)


def animation(env, state, obs, role, previous_state=None, previous_obs=None):
    view = make_view(env,state,obs,role)
    old = make_view(env,previous_state,previous_obs,role) if previous_state is not None else None
    events = changes(old,view) if old is not None else []
    frames = []
    for phase in ([.25,.5,.75,1.] if old is not None else [1.]):
        output = io.BytesIO()
        draw(env,view,old,phase,events).save(output,format='PNG')
        frames.append('data:image/png;base64,'+base64.b64encode(output.getvalue()).decode('ascii'))
    own = next((e for e in view['entities'] if e['group']=='gardeners' and e['index']==role),None)
    cell, ox, oy = art.geometry(view)
    return dict(frames=frames,duration_ms=130,width=730,height=586,
                geometry=dict(cell=cell,ox=ox-art.SCENE[0],oy=oy-art.SCENE[1]),
                events=[e['label'] for e in events],
                inventory=None if own is None else [own['attributes'][k] for k in ('wood','water','selected')],
                inventory_labels=['timber','water','selected trait'],
                shared=view['public_scene_fields'],information=view['view'],source_sha256=SOURCE)
