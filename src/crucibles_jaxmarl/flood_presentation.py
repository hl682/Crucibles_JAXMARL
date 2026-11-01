"""GI216 original art using exact role observation records. Debug is explicit.

Native observation layout is source-pinned. Ordinary views are decoded from
only their own observation row; state is used solely to reject stale inputs.
Animation interpolates only entities visible in both committed snapshots.
"""
import base64
import copy
import io
import numpy as np
from PIL import ImageDraw
from . import flood_art as art
from .rendering import scene_frame

SOURCE = '3820414daa8875d3a3a3285f82ae101efcbaa38e304b8cd297db77b1dbe68d2f'
GRID_FIELDS = ('terrain', 'depth', 'structure', 'durability', 'builder', 'gate_open')


def _require_env(env):
    if (env.name != 'gi216' or env.metadata['source_sha256'] != SOURCE
            or env.manifest['source_sha256'] != SOURCE
            or env.scene.get('source_sha256') != SOURCE
            or env.manifest.get('observability') != 'partially_observable'
            or env.observation_size != 3001):
        raise ValueError('Flood artwork requires exact partially observable GI216')


def make_view(env, state, obs, role=0):
    _require_env(env)
    if type(role) is not int or role not in range(-1, env.num_agents):
        raise ValueError('Role must be -1 (explicit full-state debug) or an actual envoy')
    actual = np.asarray(env.observe(state))
    supplied = (np.stack([np.asarray(obs[a]) for a in env.agents])
                if isinstance(obs, dict) else np.asarray(obs))
    if supplied.shape != actual.shape or not np.array_equal(supplied, actual):
        raise ValueError('View observations must belong to supplied committed state')
    world = env.manifest['world']['map']
    geometry = dict(shape=[20, 20], cell_origin_xy=world['cell_origin_xy'],
                    cell_size_xy=world['cell_size_xy'])
    if role == -1:
        frame = scene_frame(env, state, obs, role)
        return dict(source_sha256=SOURCE, map_geometry=geometry,
                    audit_map=dict(values=frame['map']), entities=frame['entities'],
                    visible_grid_fields={k: np.asarray(state[k]).tolist() for k in GRID_FIELDS},
                    clock=frame['clock'], view='PRIVILEGED_ACTUAL_STATE',
                    public_scene_fields={k: np.asarray(state[k]).tolist()
                                         for k in ('rain', 'alerts', 'home_pos')})
    # 18 public + 23 own + 400*7 grid + 4*8 agents + 4*8 homes + 32*3 packets.
    row = supplied[role]
    grid = row[41:2841].reshape(20, 20, 7)
    fields = {k: [[int(grid[y,x,j+1]) if grid[y,x,0] else None
                   for x in range(20)] for y in range(20)]
              for j,k in enumerate(GRID_FIELDS)}
    entities = []
    def entity(group, kind, index, xy, attrs):
        entities.append(dict(group=group, kind=kind, index=index,
                             xy=[int(v) for v in xy], attributes=attrs))
    for i, a in enumerate(row[2841:2873].reshape(4,8)):
        if a[0]:
            entity('envoys', 'agent', i, a[1:3],
                   dict(backpack=a[3:6].astype(int).tolist(), selected=int(a[6]), facing=int(a[7])))
    for i, a in enumerate(row[2873:2905].reshape(4,8)):
        if a[0]:
            entity('settlements', 'landmark', i, a[1:3],
                   dict(stores=a[3:6].astype(int).tolist(), crop=int(a[6]), home_water=int(a[7])))
    for i, a in enumerate(row[2905:3001].reshape(32,3)):
        if a[0]:
            entity('floodwater', 'water', i, a[1:3], {})
    return dict(source_sha256=SOURCE, map_geometry=geometry,
                role_map=dict(values=fields['terrain']), visible_grid_fields=fields,
                entities=entities, clock=int(row[0]), view='ROLE_'+str(role),
                public_scene_fields=dict(rain=bool(row[4]), alerts=int(row[5]),
                                         home_pos=row[6:14].reshape(4,2).astype(int).tolist()),
                own_scene_fields=dict(stores=row[23:26].astype(int).tolist(),
                                      report=row[27:31].astype(int).tolist(),
                                      report_tick=row[31:35].astype(int).tolist()))


def changes(before, after):
    """Only changes observed in both role-filtered snapshots; no hidden causes."""
    if before['view'] != after['view']:
        raise ValueError('Cannot animate across different information roles')
    events = []
    old = {(e['group'],e['index']):e for e in before['entities']}
    for e in after['entities']:
        prior = old.get((e['group'],e['index']))
        if prior is None:
            continue
        if e['group'] == 'floodwater' and e['xy'] != prior['xy']:
            events.append(dict(kind='water',label='Visible floodwater moved',xy=e['xy']))
        if e['group'] == 'envoys' and e['attributes']['backpack'] != prior['attributes']['backpack']:
            events.append(dict(kind='resource',label='Visible pack changed',xy=e['xy']))
        if e['group'] == 'settlements' and e['attributes']['stores'] != prior['attributes']['stores']:
            events.append(dict(kind='resource',label='Visible stores changed',xy=e['xy']))
    a,b=before['visible_grid_fields'],after['visible_grid_fields']
    for y in range(20):
        for x in range(20):
            if a['terrain'][y][x] is None or b['terrain'][y][x] is None:
                continue
            if a['structure'][y][x] != b['structure'][y][x]:
                label={0:'Structure removed',1:'Levee built',2:'Raised road built',3:'Culvert built'}[b['structure'][y][x]]
                events.append(dict(kind='structure',label=label,xy=[x,y]))
            elif a['gate_open'][y][x] != b['gate_open'][y][x]:
                events.append(dict(kind='gate',label='Culvert opened' if b['gate_open'][y][x] else 'Culvert closed',xy=[x,y]))
            elif a['depth'][y][x] != b['depth'][y][x]:
                events.append(dict(kind='depth',label='Visible water depth changed',xy=[x,y]))
    if (before.get('own_scene_fields',{}).get('report_tick') != after.get('own_scene_fields',{}).get('report_tick')):
        own=next((e for e in after['entities'] if e['kind']=='agent' and after['view']=='ROLE_'+str(e['index'])),None)
        if own is not None:
            events.append(dict(kind='report',label='Dated water memory updated',xy=own['xy']))
    return events


def draw(env, view, previous=None, phase=1., events=(), full_frame=False):
    _require_env(env)
    if view.get('source_sha256') != SOURCE or (previous is not None and previous.get('source_sha256') != SOURCE):
        raise ValueError('Unpinned flood view')
    if previous is not None and previous['view'] != view['view']:
        raise ValueError('Cannot interpolate different information roles')
    phase=max(0.,min(1.,float(phase)));v=copy.deepcopy(view)
    if previous is not None and phase<1:
        old={(e['group'],e['index']):e for e in previous['entities']}
        for e in v['entities']:
            prior=old.get((e['group'],e['index']))
            if prior is not None:
                e['xy']=[a+(b-a)*phase for a,b in zip(prior['xy'],e['xy'])]
    image=art.render(env.manifest,env.scene,v,'',v['clock'],env.max_steps+1)
    painter=ImageDraw.Draw(image);cell,ox,oy=art.geometry(v)
    for event in events:
        x=ox+(event['xy'][0]+.5)*cell;y=oy+(event['xy'][1]+.5)*cell
        r=cell*(.38+.1*phase)
        painter.rectangle((x-r,y-r,x+r,y+r),outline='#a7e2e5' if event['kind'] in ('water','depth') else '#f0cf8b',width=2)
    return image if full_frame else image.crop(art.SCENE)


def animation(env,state,obs,role,previous_state=None,previous_obs=None):
    view=make_view(env,state,obs,role)
    prior=make_view(env,previous_state,previous_obs,role) if previous_state is not None else None
    events=changes(prior,view) if prior is not None else []
    frames=[]
    for phase in ([.25,.5,.75,1.] if prior is not None else [1.]):
        output=io.BytesIO();draw(env,view,prior,phase,events).save(output,format='PNG')
        frames.append('data:image/png;base64,'+base64.b64encode(output.getvalue()).decode('ascii'))
    own=next((e for e in view['entities'] if e['kind']=='agent' and e['index']==role),None)
    cell,ox,oy=art.geometry(view)
    return dict(frames=frames,duration_ms=130,width=730,height=586,
                geometry=dict(cell=cell,ox=ox-art.SCENE[0],oy=oy-art.SCENE[1]),
                events=[e['label'] for e in events],
                inventory=None if own is None else own['attributes']['backpack'],
                inventory_labels=['stone','timber','grain'],shared=view['public_scene_fields'],
                information=view['view'],source_sha256=SOURCE)
