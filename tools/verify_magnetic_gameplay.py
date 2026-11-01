"""Source-pinned GI214 causal engineering checks from a genuine seeded reset.

No resource/state injection, training, service, remote access or old-GIF rewrite.
Paired branches change one command to Wait under the same native state and key.
"""
from pathlib import Path
import copy
import hashlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from crucibles_jaxmarl import make
from crucibles_jaxmarl import magnetic_presentation as presentation
from crucibles_jaxmarl import bistro_art as original_art

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'local-runs/gi214'
FIELDS = ('t', 'agent_pos', 'inventory', 'stock', 'terrain', 'magnet_on',
          'magnet_powered', 'polarity', 'cargo_pos', 'holder', 'integrity',
          'jam', 'charge', 'builds')
ACTIONS = [[5, 5, 24, 0, 0, 13], [0, 0, 12, 0, 0, 14],
           [0, 0, 0, 0, 0, 15], [0, 0, 0, 0, 0, 14]]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(state):
    return {k: np.asarray(state[k]).tolist() for k in FIELDS}


def state_hash(state):
    h = hashlib.sha256()
    for k in sorted(state):
        a = np.asarray(state[k])
        h.update(k.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def main():
    start = time.perf_counter()
    env = make('gi214')
    assert env.metadata['source_sha256'] == presentation.SOURCE
    original_paths = [ROOT / 'src/crucibles_jaxmarl/games/gi214' / f
                      for f in ('candidate.py', 'manifest.json', 'scene.json')]
    original_hashes = {p.name: digest(p) for p in original_paths}
    OUT.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(214)
    obs, state = jax.jit(env.kernel.reset)(key)
    step = jax.jit(env.kernel.raw_step)
    initial = state
    states = [state]
    observations = [obs]
    rows = []
    pairs = []
    assertions = {}

    def transition(s, commands, tick):
        legal = np.asarray(env.legal_actions(s))
        assert all(legal[i, a] for i, a in enumerate(commands)), (tick, commands)
        result = step(jax.random.fold_in(key, tick), s, jnp.asarray(commands, jnp.int32))
        jax.block_until_ready(result[2])
        return result

    # One fork per actual mechanism: each fork starts from the same real state.
    fork_specs = {1: [(0, 'mining'), (5, 'bridge')],
                  2: [(2, 'rail'), (5, 'magnet_supply')],
                  3: [(5, 'polarity')], 4: [(5, 'magnet_off')]}
    for tick, commands in enumerate(ACTIONS, 1):
        before = state
        obs, state, reward, done, info = transition(before, commands, tick)
        view = presentation.make_view(env, state, obs, role=0)
        prior = presentation.make_view(env, before, observations[-1], role=0)
        events = presentation.changes(prior, view)
        rows.append(dict(tick=tick, actions=commands, before=snapshot(before),
                         after=snapshot(state), before_hash=state_hash(before),
                         after_hash=state_hash(state), reward=np.asarray(reward).tolist(),
                         events=events))
        for role, name in fork_specs[tick]:
            alternate = list(commands); alternate[role] = 0
            _, control, _, _, _ = transition(before, alternate, tick)
            pairs.append(dict(mechanism=name, tick=tick, role=role,
                              action=commands[role], control_action=0,
                              common_before_hash=state_hash(before),
                              treatment_after=snapshot(state), control_after=snapshot(control)))
        states.append(state); observations.append(obs)
        presentation.draw(env, view, full_frame=True).save(OUT / f'state_{tick}.png')
    byname = {x['mechanism']: x for x in pairs}
    t, c = (byname['mining'][x] for x in ('treatment_after', 'control_after'))
    assertions['mining_stock_and_inventory'] = t['stock'][0] == c['stock'][0]-1 and t['inventory'][0][0] == c['inventory'][0][0]+1
    t, c = (byname['bridge'][x] for x in ('treatment_after', 'control_after'))
    assertions['bridge_gap_and_brace'] = t['terrain'][11][11] == 2 and c['terrain'][11][11] == 1 and t['inventory'][5][3] == c['inventory'][5][3]-1
    t, c = (byname['rail'][x] for x in ('treatment_after', 'control_after'))
    assertions['rail_tile_and_material'] = t['terrain'][8][9] == 3 and c['terrain'][8][9] == 0 and t['inventory'][2][:2] == [v-1 for v in c['inventory'][2][:2]]
    t, c = (byname['magnet_supply'][x] for x in ('treatment_after', 'control_after'))
    assertions['toggle_supply_charge_and_cargo'] = t['magnet_powered'][0] and not c['magnet_powered'][0] and t['charge'] == c['charge']-1 and t['cargo_pos'][2] == [13, 10] and c['cargo_pos'][2] == [14, 10]
    t, c = (byname['polarity'][x] for x in ('treatment_after', 'control_after'))
    assertions['polarity_changes_cargo_destination'] = t['polarity'][0] == -1 and c['polarity'][0] == 1 and t['cargo_pos'][2] == [14, 10] and c['cargo_pos'][2] == [12, 10]
    t, c = (byname['magnet_off'][x] for x in ('treatment_after', 'control_after'))
    assertions['off_stops_supply_and_freight'] = not t['magnet_powered'][0] and c['magnet_powered'][0] and t['cargo_pos'][2] == [14, 10] and c['cargo_pos'][2] == [15, 10]
    assertions['actions_changed_more_than_movement'] = all(x['before']['agent_pos'] == x['after']['agent_pos'] for x in rows)
    before_render = state_hash(state)
    animation = presentation.animation(env, state, obs, 0, states[-2], observations[-2])
    assertions['animation_contract'] = len(animation['frames']) == 4 and animation['width'] == 730 and animation['height'] == 586
    assertions['render_does_not_change_state'] = state_hash(state) == before_render
    assertions['native_inputs_unchanged'] = original_hashes == {p.name: digest(p) for p in original_paths}
    for role in [-1, 0, 1, 2, 3, 4, 5]:
        presentation.make_view(env, state, obs, role)
    view = presentation.make_view(env, state, obs, 0)
    assertions['original_background_and_geometry_retained'] = (
        original_art.geometry(view) == presentation.art.geometry(view)
        and original_art.background(214, json.dumps(view['public_map'])).tobytes()
        == presentation.art.background(214, json.dumps(view['public_map'])).tobytes())
    wrong = copy.copy(env)
    wrong.metadata = dict(env.metadata, source_sha256='0'*64)
    for operation, name in [(lambda: presentation.make_view(wrong, state, obs, 0), 'view_source_gate'),
                            (lambda: presentation.draw(wrong, view), 'draw_source_gate')]:
        try:
            operation()
            assertions[name] = False
        except ValueError:
            assertions[name] = True
    try:
        presentation.make_view(env, state, observations[0], 0)
        assertions['mismatched_observation_rejected'] = False
    except ValueError:
        assertions['mismatched_observation_rejected'] = True
    frames = [presentation.draw(env, presentation.make_view(env, states[0], observations[0], 0))]
    for i in range(1, len(states)):
        prior = presentation.make_view(env, states[i-1], observations[i-1], 0)
        view = presentation.make_view(env, states[i], observations[i], 0)
        events = presentation.changes(prior, view)
        frames.extend(presentation.draw(env, view, prior, phase, events) for phase in [.25, .5, .75, 1.])
    frames[0].save(OUT/'causal_walkthrough.gif', save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=False)
    presentation.draw(env, presentation.make_view(env, initial, observations[0], 0), full_frame=True).save(OUT/'state_0.png')
    result = dict(status='PASS' if all(assertions.values()) else 'FAIL', game='gi214',
                  source_sha256=presentation.SOURCE, original_hashes=original_hashes,
                  seed=214, native_reset=True, state_injection=False,
                  all_commands_legal=True, native_joint_steps=10,
                  main_steps=4, paired_single_action_forks=6,
                  assertions=assertions, sequence=rows, causal_pairs=pairs,
                  elapsed_seconds=time.perf_counter()-start,
                  jax_version=jax.__version__, backend=jax.default_backend(),
                  outputs=[str(p.relative_to(ROOT)).replace('\\', '/') for p in OUT.glob('*') if p.is_file()],
                  learning_claim=False, human_scores=None,
                  scope='Four-step native mechanism demonstration, not full-game coverage or scientific efficacy.')
    path = ROOT/'verification/gameplay/gi214_causal_gameplay.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: result[k] for k in ['status', 'game', 'assertions', 'elapsed_seconds', 'backend']}), flush=True)
    assert all(assertions.values()), assertions


if __name__ == '__main__':
    main()
