# Carousel Supper

AUTHORED_NOT_EXECUTED. Original initial source attempt 00 for game-v3-00211. Requested author gpt-6-astra / max; provider measured identity, tokens and cost unknown.

The central raised 3-by-3 carousel is impassable to workers; its eight perimeter slots carry real item objects around the center. Workers use the surrounding corridor, with separate west/east preparation boards and north/south heat/serving stations. There are three vegetable objects, three protein objects and three tray objects. A cook holds at most one item. Preparing changes the ingredient flag; combining consumes the original prepared ingredient and sets its bit in the original tray's filling mask. No ingredient is created by a station.

Clockwise requests contribute +1, counterclockwise requests -1; their sum is clipped to -1..1. Rotation follows all pickup, placement and combination operations. Consequently, helping one cook receive a component can move another cook's tray away from reach or into the north heat slot. Fully assembled trays heat automatically there each tick; manual heating of a held tray beside the same station is a slower physical alternative. Heat is retained and cannot be undone. Spoilage is irreversible after heat >6 or 45 filled-tray ticks. The bin lets a crew remove waste but never restores ingredients. Floor handoffs and carrying around the corridor provide an alternative to coordinating shared rotation.

These rules give a partner-dependent timing problem: a cook can pause a rotation, oppose it, relocate a tray, carry around the ring or accept temporary delay. All observed tray preparation, filling and temperature refer to those conserved objects. Three identical orders each require one prepared vegetable, one prepared protein and a warm tray; this is a small fixed queue rather than a changing recipe generator.

## Native interface and source evidence

A/O/U/H = (3, 232, 15, 180). Humans and policies choose the same ordered native integer actions. Any subset of the three roles can be assigned to humans by the shared platform; this package adds no separate browser transition or controller restriction. Legal masks are contextual affordances, not a proof that an interaction will find an object. Out-of-range or mask-false actions become wait. Eligible interactions resolve in rotating actor order (t + k) mod 3; ties among eligible objects use lowest native object index. Movement rejects destinations occupied at the start of the tick and duplicate destinations; swaps and following into a simultaneously vacated cell therefore wait.

- candidate.py:L13-L34: reset.
- candidate.py:L36-L38: observe.
- candidate.py:L40-L50: legal_actions.
- candidate.py:L52-L127: advance.
- candidate.py:L129-L131: raw_step.

## Action legend

- 0: Wait.
- 1: Walk north.
- 2: Walk east.
- 3: Walk south.
- 4: Walk west.
- 5: Pick lowest-index eligible item within one tile; hands must be empty.
- 6: Drop held item at current floor tile.
- 7: Request one clockwise carousel step from beside ring.
- 8: Request one counterclockwise carousel step from beside ring.
- 9: Put held item onto lowest-index adjacent empty slot.
- 10: Prepare held vegetable at west board or protein at east board.
- 11: Add held prepared ingredient to adjacent compatible unattended tray.
- 12: Manually heat held full tray beside north heat station.
- 13: Serve held full unspoiled tray with heat 3..6 beside south counter.
- 14: Discard held item beside southwest bin.

## Observations and scene

Fully observable public kitchen. Every role receives the same flattened native clock/map, worker inventory and positions, station/slot positions, item existence/kind/holder/attachment/preparation/filling/heat/age/spoilage/last-owner data, and serving/discard/rotation counters. Constant entity activity and visibility masks are declared in state; inactive items are absent in the scene but their fixed-slot history remains public in observations. All state leaves and exact observation slices are in manifest.json. Every visible attribute is native state; scenes introduce no hidden policy data. Public event channels contain only intentionally public values. Debug/full-state inspection must be separate from role view. Use a warm tiled kitchen, coral workers, green food, wooden preparation stations and circular carousel markers. Keep the square native corridor and eight true slot coordinates.

## End states, reward and invariants

All actors receive new served trays - 0.15 times newly spoiled trays - 0.01 per advancing step. Natural completion at three served orders, or natural exhaustion when no active unspoiled tray remains. Otherwise truncate at tick 180. Natural termination has priority on the cap step. Already done input is absorbing with unchanged state, observation and clock; rewards become zero and terminal causes persist. No raw autoreset occurs. The declared bounds and invariants are source obligations, not observed numerical results. Nine original item slots are conserved. A tray receives at most one of each ingredient, each ingredient can be consumed once, and logical ring slots hold at most one active item. Pickup clears ring attachment; carriage follows the actor position. Serving or discarding deactivates the original item.

## Hypotheses and provenance limits

Two literal scripts and one single actor/action paired intervention are supplied in playthrough.json; suffix lengths follow the implemented timers or locations. All have UNEXECUTED_AUTHOR_HYPOTHESIS status, and the seed is literal rather than searched. They do not certify legal trajectories, solvability or mechanism success. Prospective tests could vary carousel speed, recipe mix, spoilage delay, starting supply locations or partner rotation timing under separate prospective rules; none has been run. This source does not establish compatibility, numerical parity, strategic richness, learning, generalization or transfer.

The actual allowed-input inventory, full-byte hashes and excerpt extents are frozen in author_input_freeze.json. Current scope overrides historical IDs in copied generic requirements. Four supplied official README excerpts provided API context: JaxMARL parallel dictionaries, Pgx explicit-key/terminal concepts, Craftax spatial world/interface context, PureJaxRL single-file implementation philosophy. No benchmark implementation, layout, upstream code or assets were copied; the supplied PPO implementation was not read. Static author checks parse data/AST only and import only supplied trusted stdlib validators. Candidate/JAX/NumPy import, native calls, numerical replay, rendering, training, seed search and network/model calls were not performed.

## Literal rule evidence

candidate.py:L104: turn = jnp.clip(jnp.sum((ac==7).astype(jnp.int32)-(ac==8).astype(jnp.int32),dtype=jnp.int32),-1,1)

candidate.py:L107: n['slot_pos'] = ring[(jnp.arange(8,dtype=jnp.int32)+n['phase'])%8]

candidate.py:L91: combine = (a==11)&has&(n['kind'][h]<3)&n['prepared'][h]&jnp.any(trays)

candidate.py:L111: onheat = n['item_active']&(n['kind']==3)&(n['filling']==3)&(n['slot']>=0)&(jnp.sum(jnp.abs(n['item_pos']-n['station_pos'][2]),axis=1)<=1)

candidate.py:L115: n['spoiled'] = n['spoiled']|((n['heat']>6)|(n['age']>=45))

candidate.py:L120: usable = n['item_active']&(n['kind']==3)&~n['spoiled']

candidate.py:L130: n,r = jax.lax.cond(state['done'],lambda _: (state,jnp.zeros((3,),jnp.float32)),lambda _: self.advance(state,actions),operand=None)

