# Railguard — game-v3-00160

Status: AUTHOR_ONLY_NOT_EXECUTED. Source SHA256: `fe6b8849bb86e410ff28acff98c1ecff878dedfadbf1d240f0bc46e199c3c670`.

Native contract: A=4, O=218, U=16, H=128. Fixed-shape dictionary state, public coordinate order xy, map indexing yx.

The arena is an 11 by 7 public grid, with interior wall cells (5,1) and (5,5). West guards begin at (3,3), (4,4); east guards at (8,3), (8,5). Rail shields begin at (4,3), (7,5). The first west guard is initially screened from the east guard's westward ray by the first shield. Supply bases are at (1,3), (9,3). These positions create both a protected firing lane and alternative routes around cover; all four guards are controlled by the joint action vector.

Legal actions express affordable commands, not guaranteed motion or a guaranteed hit. Invalid or out-of-range actions become rest. A downed guard has only rest available; it stays on the map as an incapacitated body and does not recover energy until revived. A terminal state exposes only no-op and is fully absorbing.

Each pusher selects its nearest adjacent shield, choosing the first shield on an exact tie. Push votes for north/east/south/west are counted from the same joint action. A direction needs at least two votes and a unique maximum. All push attempts cost one energy, including blocked or losing votes. An eligible shield shifts one cell on its floor rails while operators remain at their current cells; this is a discrete rail-control interaction, not a continuous force simulation. Initial body positions, other shields and walls block the shift. If both proposed shield destinations coincide, neither moves. Shield motion therefore precedes and causally changes the firing geometry on the same transition.

Walking costs one energy only when it succeeds. Start-occupied cells, all destination ties, shields and walls block walking; there are no swaps or follow-through chains. Rest restores two energy up to eight. Reloading at an adjacent own base transfers one reserve round into a magazine of capacity three. Reload and donation resolve by cyclic priority, checking the current magazine and stock so no overfill or double-spend occurs. Donating one round to an adjacent living teammate costs one energy.

Fire is simultaneous after cover, movement and ammunition transfers. A shot costs one energy and one round and sets cooldown to two; two subsequent turns cannot fire. Four fixed six-cell ray arrays record actual traversed cells, including the stopping cell. Every wall, cover or body blocks the ray. An enemy body with positive starting health receives a hit; friendly and already downed bodies stop rays without damage. Each shot can hit at most the first body. All damage is collected before health changes, so a guard downed by simultaneous fire still completes its own eligible shot. The source records hit counts and actual ray cells rather than drawing an invented bullet effect.

Rescue is resolved after damage. It costs one round and two energy, requires adjacent teammates, a rescuer still alive after the volley, and a teammate who was already down at the start. Freshly downed teammates cannot be rescued in that same transition. New incapacitations award 0.5 team reward per enemy and -0.5 per own casualty. A sole surviving team additionally receives +1 and the other -1; a simultaneous wipeout gets no winner bonus. Per-agent rewards equal its team's reward. Scores count enemy incapacitations and never count ordinary hits as captures.

The intended push-versus-wait intervention is deliberately risky: shifting the initial shield north opens the westward enemy line onto guard 0, while withholding one supporting vote keeps the shield in that ray. It tests a teammate-caused exposure with opponent behavior fixed. Plausible play includes conserving ammunition for an exposed opponent, guarding a casualty to deny rescue, moving shields together before changing lanes, declining a dangerous support shove, and spending a turn on ammunition for a teammate instead of a personal shot. The opponent controls its own tactical choices; no scripted neutral enemy is embedded. No tactic has been executed or trained.

Health stays in 0..2, energy in 0..8, magazines in 0..3, stock in 0..6 and cooldown in 0..2. Ammunition is finite: reload and gift only transfer existing rounds, while firing and reviving consume them. Bodies are spatially unique even when down. Shields are distinct and never overlap bodies. The trace active masks show exactly the latest recorded ray prefix; inactive slots have no rendered projectile. Frozen terminal hit and shot flags describe the actual last transition and are not new events on repeated terminal queries.

Possible future variants could change shield mass, rail directions, ammunition scarcity, rescue cost and team starting geometry. Their policy consequences and transfer value require separate evidence. The supplied short scripts aim to expose the shield-ray dependency and cover-preserved volley, not to certify a win or an entire competitive strategy.

## Action legend

- 0: Rest and recover two energy.
- 1: North.
- 2: East.
- 3: South.
- 4: West.
- 5: Push adjacent rail shield north.
- 6: Push adjacent rail shield east.
- 7: Push adjacent rail shield south.
- 8: Push adjacent rail shield west.
- 9: Fire north.
- 10: Fire east.
- 11: Fire south.
- 12: Fire west.
- 13: Reload one round at own base.
- 14: Give one round to adjacent live teammate.
- 15: Revive adjacent previously downed teammate.

## Native observation layout

- [0, 4): Observer role one-hot in fixed agent order; visibility=self.
- [4, 81): Public arena walls, row-major yx; visibility=public.
- [81, 89): All guard positions; visibility=public.
- [89, 93): Team IDs; visibility=public.
- [93, 97): Health; zero means incapacitated; visibility=public.
- [97, 101): Available energy; visibility=public.
- [101, 105): Loaded ammunition; visibility=public.
- [105, 109): Firing cooldown ticks; visibility=public.
- [109, 113): Latest sanitized commands; visibility=public.
- [113, 117): Enemy ray hits this transition; visibility=public.
- [117, 121): Revived this transition; visibility=public.
- [121, 125): Rail-shield positions; visibility=public.
- [125, 127): Shield moved this transition; visibility=public.
- [127, 135): North east south west push votes for each shield; visibility=public.
- [135, 139): West and east supply bases; visibility=public.
- [139, 141): Remaining team supply stock; visibility=public.
- [141, 143): Cumulative enemy incapacitations by team; visibility=public.
- [143, 191): Latest four rays, six cells per shooter, xy; visibility=public.
- [191, 215): Cells actually traversed by latest rays; visibility=public.
- [215, 218): Clock, joint done flag, winning team ID or minus one; visibility=public.

## Literal source evidence

- candidate.py:70: `consensus = (votes[d] >= 2) & (jnp.sum((votes == votes[d]).astype(jnp.int32)) == 1)`
- candidate.py:79: `s['cover_pos'] = jnp.where(s['cover_moved'][:, None], proposed, s['cover_pos'])`
- candidate.py:108: `victim = bodies & (s['team'] != s['team'][i]) & (state['health'] > 0) & ray & ~wall & ~shield`
- candidate.py:112: `ray = ray & ~wall & ~shield & ~jnp.any(bodies)`
- candidate.py:113: `s['health'] = jnp.maximum(0, s['health'] - s['hit'])`
- candidate.py:117: `revive = (ac[i] == 15) & (s['health'][i] > 0) & (state['health'][mate] == 0) & (jnp.sum(jnp.abs(s['pos'][i] - s['pos'][mate])) == 1)`
- candidate.py:126: `team_reward = 0.5 * (lost[::-1] - lost).astype(jnp.float32)`
- candidate.py:134: `s = jax.tree.map(lambda new, old: jnp.where(state['episode_done'], old, new), s, state)`

Method source extents: reset lines 12–32, legal_actions lines 34–46, raw_step lines 48–136, observe lines 138–143. These are static source pointers, not executed traces.

## Reference provenance and boundary

Official community references were read only from this batch's supplied snapshots. JaxMARL informed the public multiplayer reset/step and agent-role conventions; Pgx informed explicit legal-action and terminal-state API distinctions. Craftax informed native state-backed game objects and presentation separation. PureJaxRL and the supplied Craftax PPO recurrent baseline were integration references for pure array batching and outer wrappers, not copied game libraries, not executed training code, and not replacements for the experimental learner. No upstream source code or map was copied into this game. Exact snapshot hashes and URLs are preserved in generation_record.json and author_input_freeze.json. No external network search occurred.

AUTHOR_ONLY_NOT_EXECUTED. The candidate source was never imported or evaluated, JAX was not imported by the author tooling, and reset/raw_step/observe/legal_actions were not called. Permitted checks were stdlib AST parsing and compile-only, the frozen static whitelist, data-only manifest/scene validation, literal vector structure and exact file hash bindings. These checks do not establish returned runtime shapes, numerical parity, trajectory legality, completion, strategic richness, learning, community-adapter acceptance, replay or animation. There are no experimental results, scorer outputs, learned-policy evidence or fabricated successes in this package. Three or four agent support is the frozen batch scope. Signed discrete state is represented as int32; bool flags and float32 native observations/rewards follow the contract. The supplied PRNG key is not duplicated in state. An external agent-keyed JaxMARL adapter and actual renderer playback remain separate integration work.
