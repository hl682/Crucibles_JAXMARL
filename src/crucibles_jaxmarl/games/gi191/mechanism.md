# Border Loom — authored native game mechanism

Status: AUTHORED_NOT_EXECUTED. Assigned game-v3-00191; three agents; source attempt 00.

## Game and human interaction
Border Loom is a local-information territorial transport game. Western grain, northeastern copper and southeastern wool traders begin at (1,3), (5,1), (5,5). Their buyers form a cycle: 0 sells at town 1, 1 at town 2, 2 at town 0. There are four original export crates per role and eight timber per builder. Shared trunk roads leave the central (3,3) gap unbuilt. Lake inlet cells (3,2) and (3,4) are impassable. The rectangular map is actual movement geometry.

A human walks a character, sees its wagon load, timber, coin purse and facing, then paves the adjacent cell ahead or negotiates at close range. Arrow or N/E/S/W controls should map to the four native directions; contextual buttons use all other native actions and masks. Co-location is legal. Arbitrary legal command sequences are accepted; included scripts are examples only. Native human controls have not yet been implemented or run by this author.

## Rules and exact chronology
Reset samples only each sellers private per-crate value, uniformly from {2,3}. Geometry, stocks and all subsequent transitions are deterministic; raw_step does not draw randomness.

An unloaded worker may traverse any land tile. A loaded wagon may enter only an existing road. A constructed tile costs one timber, belongs to its builder and initially has zero toll. Claiming an unowned road underfoot costs two timber and opens a toll outpost charging one coin. Hometown tiles cannot be claimed. An owner can toggle its own tile toll. Foreign loaded entry is charged unless a bilateral permit is active; unladen workers and the owner are exempt. Toll payments transfer actual coins.

Masks snapshot incoming legality. Invalid/out-of-range actions become Wait. Simultaneous movement and tolls use incoming road and permit state, so a road laid this step opens movement on the next one. Facing changes only for accepted directional actions. Existing permits/offers age by one. Nonmovement intents then resolve in rotating order t mod 3, followed by the other two roles. Competing road claims are rechecked against current resources and ownership: the first successful claimant pays, others make no change. A player cannot both turn and pave in one step; the visible facing marker is part of its observation.

An offer costs one coin, names next or previous role and lasts four future acceptance opportunities. Acceptance requires incoming offer target and TTL plus incoming Manhattan adjacency at most one. If several offers are eligible, the lowest proposer index is selected. Only incoming offers can be accepted; a new offer cannot be accepted immediately. The accepting pair receives a symmetric eight-step no-toll passage commitment for the next eight movement checks. Acceptance clears the proposers offer slot when the accepting role resolves. A simultaneous replacement issued earlier in rotating order is cleared; one issued later survives. These offer-slot writes follow explicit rotating precedence, not an inferred negotiation.

Loading consumes one original hometown crate. Cargo capacity is one. Delivery at the cyclic buyer consumes cargo, increments that towns received count, pays the sellers private value and pays the receiving role one. No manufactured crates or demand replenishment exist. A player can discard a carried crate to regain unloaded movement, with no reward. Natural completion is empty hometown stocks and no carried crates, including deliberate waste. Otherwise H=112. Terminal state, cause and time absorb under the binding addendum.

## Why partners matter
An unloaded builder may invest in a route that another loaded trader can then use. Ownership can capture coin tolls, but excessive tolling can make other strategies and alternate routes preferable or stall a poor trader. An offer burns a coin before reciprocal passage is guaranteed; acceptance needs a real meeting. A trader can share access before committing timber, build its own route, pay tolls or discard goods to regain mobility. The import recipient benefits from another player completing its physical route. Finite goods and construction budgets make delay and claim order consequential.

This mechanism combines loaded mobility, contested road ownership, finite trade and temporary bilateral permits. It is not a renamed river-flow or crop-growth process. No strategic-richness, equilibrium, solvability, learning or transfer claim is established.

## Information and visual charter
O=252 comprises ten self/public fields, three nine-field local trader records, forty-nine four-field local road records, three three-field town records, a public nine-entry permit matrix and own completed exports. Trader/road visibility is Manhattan distance at most two. Own hometown stock remains visible wherever the trader walks. Other town details use the same radius. Hidden values are -1 beside explicit visibility flags. Map terrain and town coordinates are common knowledge; road ownership/build status outside view are not.

Per-role presentation must mask local agent and road attributes by their declared visibility. Global event arrays and cumulative score are a debug stream, not additional policy or human-role observations. Public treaty changes are visible through the permit matrix. Do not infer hidden ownership from an inactive/unknown road slot. A role panel can show its own private price from the self observation, but not other prices.

Use a territorial trade-map appearance with original traveling merchants, small commodity wagons, stone paving, timber frontier posts, stamped charter pennants, copper ingots, grain sacks and wool bales. Render owned road cells as contiguous native-grid surfaces, not independent dashboard markers. Fog must preserve hidden local state. Direction arrows attach to the real facing field. All artwork remains decoration of manifest positions and ownership; scene.json binds every interactive group.

## Bounds and diagnostics
Stocks, cargo, coins, timber, receipt counts and timers are nonnegative. Initial global coin supply is 15, tolls conserve it and offers reduce it; an individual coin purse is below 16. Timber cannot increase above eight. Each commodity has four crates; cargo is 0 or 1 and received/delivered at most four. Permits are 0..8; offers are 0..4. Observations are in [-1,1]. Per-step reward is 0..4 (own sale plus another traders import benefit), an algebraic bound.

The source-derived road script lets role 2 deliver wool to its actual buyer after role 0 builds the central tile. Its paired hypothesis uses only build-versus-wait and a common wait suffix, then inspects role 2 local information and legal west action. It does not force an illegal west move in the unbuilt branch. A second script illustrates paid offer and close-range acceptance. Seed 191 was chosen once without search. No replay, movement, delivery or interface execution has occurred.

## Complete native action dictionary

- 0: Wait.
- 1: Walk north.
- 2: Walk east.
- 3: Walk south.
- 4: Walk west.
- 5: Pave the cell ahead: 1 timber.
- 6: Claim neutral road underfoot as toll outpost: 2 timber.
- 7: Toggle toll on owned road underfoot.
- 8: Offer 8-step reciprocal passage to next role: 1 coin.
- 9: Offer passage to previous role: 1 coin.
- 10: Accept adjacent existing offer; lowest proposer index.
- 11: Load own commodity at own town.
- 12: Deliver commodity at next roles town.
- 13: Discard carried commodity to travel unloaded.

## Literal source-line evidence

- Actual local visibility: candidate.py line 12; literal source: def visibility(self, s):
- Loaded movement and entry fee: candidate.py line 52; literal source: def destination(self, s, actions):
- Paving and offer prerequisites: candidate.py line 76; literal source: def legal_actions(self, s):
- Rotating scarce claims: candidate.py line 121; literal source: for k in range(3):
- Bilateral accepted passage: candidate.py line 146; literal source: accept = (a[i] == 10) & jnp.any(eligible)
- Trade: candidate.py line 157; literal source: trade = (a[i] == 12) & (n['received'][target] < 4)
- Observation with local sentinels: candidate.py line 175; literal source: def observe(self, s):


## Provenance, reference use and verification limits
This is source attempt 00 of the original blind initial window, not a repair or feedback-selected revision. Current batch ordinals 186..193 were reserved inside the original cap 512; cumulative used-or-reserved count was 194 and feedback count was zero. This author owns only GI190..193 attempt_00 and does not alter other work. Exact actual input paths, byte sizes, hashes and purposes are in author_input_freeze.json, including the prospective human-playability addendum. All four assigned design/request pairs were read; no other candidate, result, selection, project training, develop/confirm, heldout, IR or final-test input was accessed.

The supplied official JaxMARL README informed a prospective dictionary-per-agent adapter boundary. Pgx informed explicit external random keys, masks and environment identity. Craftax README informed native spatial state/presentation and raw versus outer autoreset. PureJaxRL README and Craftax_Baselines_ppo_rnn.py informed only reset/step/scan integration. They are algorithm integration references, not new game libraries or replacements for the frozen learner. No upstream game, algorithm code or visual asset was copied. No network request, model/API invocation, remote session, credential inspection, download or candidate execution was performed.

Signed enums, counters and ownership fields use lossless int32; state booleans use bool. The supplied reset key is not duplicated inside state. This representation adaptation does not imply changing an executed earlier distribution. Each new private-parameter distribution is stated above. Raw terminal semantics follow the addendum: a live call advances t once, natural termination overrides horizon, and an already-done call preserves state/observation/time/cause with zero rewards.

Only stdlib AST, JSON and hash checks were authorized. No candidate or JAX import, eager call, JIT, vmap, rollout, random policy, seed search, replay, RGB render, adapter test or training was run. Shapes/dtypes, masks over trajectories, numerical parity, invariant preservation and paired effects await independent frozen-source integration. Positive, zero, negative, failed and indeterminate results remain valid reports. No reward, seed-count or success-rate acceptance threshold is introduced. Source authorship is not human-playability acceptance or evidence of learning/transfer.
