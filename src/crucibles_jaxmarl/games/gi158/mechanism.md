# Pressure Relay — game-v3-00158

Status: AUTHOR_ONLY_NOT_EXECUTED. Source SHA256: `cc0e1733768baa6628b753304337966e59738a9cc364bc6738d8e2625250f592`.

Native contract: A=3, O=155, U=15, H=160. Fixed-shape dictionary state, public coordinate order xy, map indexing yx.

The three rectangular rooms occupy x ranges 1..3, 4..7 and 8..11, with door tiles at (4,3) and (8,3) assigned to their eastern room. Pump panels are at (2,2), (6,2), (10,2). The map is public. The leak at (10,5) initially drains the east room. Initial pressure is [12,6,2], the shared reserve is 32, and personal oxygen is 24 each. This is an invented discrete game, not a physical model or safety advice.

One input joint action is sanitized using the starting legal mask. Door requests resolve by cyclic agent priority t modulo 3; each requester is adjacent to at most one door. Action 5 toggles a door only if opening is pressure-safe or closing would not trap an occupant. Action 6 leaves it open and spends two personal oxygen even when already open. Action 14 enables a one-tick gentle vent. Repair precedes pressure flow. Pump/refill operations then consume the common reserve in cyclic priority. Flows across both edges are computed from the same pre-flow pressure vector, transferred conservatively, and followed by the remaining leak. Pumps add at most two pressure units; refills add at most six breathing units. Neither generates air for free.

Movement uses the resulting door states. Every start-occupied destination is blocked, including a cell another agent intends to vacate. All agents targeting one destination fail to move. Successful movement is followed by pickup, pass/drop, oxygen gift and delivery resolution in rotating priority. A carrier has capacity one. Pickup selects the lowest-index nearby floor artifact. Handoff selects an adjacent empty recipient by rotating priority, or drops the artifact if none exists. Delivery pays 0.6 to each role plus 0.4 to its delivering role. Each active tick costs 0.001 reward. A natural failure costs each role 0.5.

After interactions, a non-porter carrier compares starting pressure at its starting position with resulting pressure at its resulting position. A jump of at least three costs one integrity. This includes pressure changes caused by other crew members. The porter avoids this shock cost, but still consumes oxygen. At pressure at least six breathing costs zero, at three to five it costs one, and below three it costs two. Any exhausted tank or shattered artifact ends the expedition unless both artifacts were delivered on that transition.

Expected partner-dependent choices include waiting at a vent so a teammate can cross without spending oxygen, opening the middle room to aid an oxygen-starved teammate while risking cargo shock, reserving air for breathing instead of pumping, handing cargo to the protected porter, and using the engineer's single patch before the reservoir drains. Individual delivery credit can motivate an unprotected shortcut; the shared loss from damage and exhaustion opposes that incentive. Door opening can help one carrier while changing pressure around another. These are source-derived strategic possibilities, not observed policies or evidence of learning.

All visible attributes correspond to public observation fields or deterministic functions of them. Pressure bars bind the real panel pressure array; door openness and signed flow bind the passability and transfer arrays; cargo position follows its real carrier, and integrity and shock are recorded state. Finished cargo becomes inactive. Delivery/shock flags last one transition and remain frozen if the raw terminal state is queried again. No synthetic replay is supplied.

Declared bounds are pressure 0..12, reservoir 0..32, oxygen 0..24, patches 0..1, integrity 0..3. Flow is signed -3..3. Agent positions remain distinct and in interior walkable or open-door cells. Artifact positions may overlap agents and each other; no false cargo uniqueness invariant is imposed. A terminal input preserves every state leaf and clock and returns zero reward. Only a nonterminal call advances t. Natural causes take priority over the horizon.

Prospective variants of door placement, leak location, pump rate, pressure-safe opening threshold or which role carries the sleeve could change staging and role allocation. Such variants are not implemented, generated, measured or selected here. The included literal scripts demonstrate intended door/oxygen/cargo interactions without asserting task completion.

## Action legend

- 0: Wait.
- 1: North.
- 2: East.
- 3: South.
- 4: West.
- 5: Toggle nearest adjacent door safely.
- 6: Force nearest adjacent door open; spend two oxygen.
- 7: Operator pump current room at panel.
- 8: Engineer patch leak.
- 9: Refill breathing tank at panel.
- 10: Pick up nearby floor artifact.
- 11: Hand artifact to adjacent empty teammate; otherwise drop.
- 12: Give up to three oxygen to adjacent teammate.
- 13: Deliver carried artifact at dock.
- 14: Hold gentle equalization vent at adjacent door.

## Native observation layout

- [0, 3): Observer role one-hot in fixed agent order; visibility=self.
- [3, 94): Public room walls and door tiles, row-major yx; visibility=public.
- [94, 100): All crew positions in agent order, xy; visibility=public.
- [100, 103): Crew breathing reserves, zero to twenty-four; visibility=public.
- [103, 106): Equipment roles: engineer, operator, porter; visibility=public.
- [106, 109): Room air pressure: west, middle, east; visibility=public.
- [109, 115): Room pump-panel positions, xy; visibility=public.
- [115, 119): Door positions, xy; visibility=public.
- [119, 121): Current door passability; visibility=public.
- [121, 122): Shared air stock; visibility=public.
- [122, 123): East-room leak still open; visibility=public.
- [123, 124): Engineer patches remaining; visibility=public.
- [124, 126): Actual patch location; visibility=public.
- [126, 128): Delivery dock location; visibility=public.
- [128, 132): Artifact physical positions; visibility=public.
- [132, 134): Artifact holder ID or minus one; visibility=public.
- [134, 136): Artifact status: floor zero, carried one, delivered two, shattered three; visibility=public.
- [136, 138): Artifact integrity; visibility=public.
- [138, 141): Latest sanitized crew commands; visibility=public.
- [141, 144): Oxygen donated on latest transition; visibility=public.
- [144, 146): Door changed on latest transition; visibility=public.
- [146, 148): Latest signed air transfer across each door, positive eastward; visibility=public.
- [148, 150): Latest pressure shock to each artifact; visibility=public.
- [150, 152): Latest artifact deliveries; visibility=public.
- [152, 155): Clock, joint done flag, and air pumped on latest transition; visibility=public.

## Literal source evidence

- candidate.py:106: `magnitude = jnp.minimum(jnp.abs(delta) // 2, jnp.where(s['door_open'], 3, vent.astype(jnp.int32)))`
- candidate.py:108: `s['pressure'] = s['pressure'] + jnp.array([-s['flow'][0], s['flow'][0] - s['flow'][1], s['flow'][1]], jnp.int32)`
- candidate.py:103: `s['reservoir'] = s['reservoir'].at[0].add(-pump - fill)`
- candidate.py:153: `s['shock'] = (s['carrier'] >= 0) & (s['carrier'] != 2) & (jnp.abs(after - before) >= 3)`
- candidate.py:159: `breath = jnp.where(pressure_here >= 6, 0, jnp.where(pressure_here >= 3, 1, 2))`
- candidate.py:162: `failure = (jnp.any(s['oxygen'] == 0) | jnp.any(shattered)) & ~success`
- candidate.py:169: `s = jax.tree.map(lambda new, old: jnp.where(state['episode_done'], old, new), s, state)`

Method source extents: room lines 12–13, refresh lines 15–23, reset lines 25–53, legal_actions lines 55–67, raw_step lines 69–171, observe lines 173–180. These are static source pointers, not executed traces.

## Reference provenance and boundary

Official community references were read only from this batch's supplied snapshots. JaxMARL informed the public multiplayer reset/step and agent-role conventions; Pgx informed explicit legal-action and terminal-state API distinctions. Craftax informed native state-backed game objects and presentation separation. PureJaxRL and the supplied Craftax PPO recurrent baseline were integration references for pure array batching and outer wrappers, not copied game libraries, not executed training code, and not replacements for the experimental learner. No upstream source code or map was copied into this game. Exact snapshot hashes and URLs are preserved in generation_record.json and author_input_freeze.json. No external network search occurred.

AUTHOR_ONLY_NOT_EXECUTED. The candidate source was never imported or evaluated, JAX was not imported by the author tooling, and reset/raw_step/observe/legal_actions were not called. Permitted checks were stdlib AST parsing and compile-only, the frozen static whitelist, data-only manifest/scene validation, literal vector structure and exact file hash bindings. These checks do not establish returned runtime shapes, numerical parity, trajectory legality, completion, strategic richness, learning, community-adapter acceptance, replay or animation. There are no experimental results, scorer outputs, learned-policy evidence or fabricated successes in this package. Three or four agent support is the frozen batch scope. Signed discrete state is represented as int32; bool flags and float32 native observations/rewards follow the contract. The supplied PRNG key is not duplicated in state. An external agent-keyed JaxMARL adapter and actual renderer playback remain separate integration work.
