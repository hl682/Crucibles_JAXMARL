# River Charter — authored native game mechanism

Status: AUTHORED_NOT_EXECUTED. Assigned game-v3-00190; three agents; source attempt 00.

## Game and human interaction
River Charter is a square-grid territorial river strategy game for three polities. The upper, middle and lower settlements occupy (3,1), (3,4), (3,7); their docks are one cell east and their potential bridges are on river x=5. The eastern market is (8,4). The envoy is a mobile economic actor: position determines loading, construction, sluice manipulation, consumption, trade and crossing. Co-location is allowed; envoys do not block one another.

A human can select any legal joint integer action vector, walk with N/E/S/W, inspect the current mask and choose a contextual command. A player may switch controlled roles or choose all three commands before advancing one native step. The kernel contains no script lookup. Masks and observed stocks explain disabled commands. Human-session controls and native animation are subsequent integration work, not delivered or accepted here.

## Rules and exact chronology
Each polity starts with six stored grain, no cargo, a closed diversion that takes water, an unbuilt bridge and an enabled toll. Own consumption taste is sampled once from {1,2}, using only reset key. This is the only stochastic rule; raw_step is deterministic conditional on incoming state/actions and does not use its step key.

Masks are evaluated on the incoming state. Invalid or out-of-range actions become Wait. Movement resolves simultaneously from incoming bridge state; co-location is legal. Entering a foreign built bridge pays one stored grain when its toll is enabled. An owner crosses free. A bridge tile is usable only at its actual row; other river cells cannot be entered. There is no charging merely for standing on a bridge.

Nonmovement commands then build bridges, toggle diversion, purchase pledges, toggle tolls, load one grain, launch cargo, unload, eat or sell. Because each role chooses one command, it cannot both move and interact this step. Build costs two stored grain at the matching dock. A pledge costs one stored grain and forces diversion open for the next two floods; toggling is unavailable during it. Expiry leaves the setting open until changed deliberately.

All barges, including newly launched barges, then advance one y cell. Sender 0 lands at y=4 and supplies polity 1; sender 1 lands at y=7 and supplies polity 2. Sender 2 exports at y=8 for two utility per grain. Each sender has one physical barge slot. A land unload transfers cargo into the town underfoot, including own storage, and gives no reward.

Every eighth step, for at most eight floods, four water units pass upper to middle to lower. Each diverting polity takes min(remaining water,2) and adds that amount to its grain. An open gate takes none. Pledge counters decrement after flow. This is an explicit ordered resource process, with no negotiation solver. Eating yields own private taste per grain. Eastern market exports yield three per grain. Rewards are only these actual consumption/export payoffs.

Natural completion occurs once all eight floods have resolved and outstanding barges have arrived. Otherwise H=88 truncates. The actual terminal state, causes and clock are absorbing on future raw calls, with zero reward. No raw reset occurs.

## Why partners matter
An upstream envoy can retain water now or forego it for a downstream counterpart who can later return physical grain, build an accessible crossing or release another resource. A costly two-flood pledge is observable and constrains later choices. Toll income, cargo limits, walking distance and the export advantage create competing local consumption, transport investment and reciprocal-shipping strategies. A barge gift has no automatic social reward; any benefit to its sender depends on others future behavior.

The mechanism is an ordered river-allocation and transport economy. It is distinct from the road permits, orchard ecology and workshop production in the other assigned briefs. These are prospective strategy arguments, not proof of a nontrivial equilibrium, learnability, solution or transfer.

## Observation and visual charter
The 45 public columns expose normalized time, completed floods, source water, all envoy positions, grain/cargo, sluices/pledges, bridges/tolls, barge state, current flood allocation and scores. Three role bits and only the observing roles taste complete O=49. Exact slices and scales are in manifest.json. Static landmark coordinates and north-to-south ownership are public common knowledge. Tastes have no entity attribute, so a role renderer cannot display another roles taste.

Use a territorial-strategy treatment: original cloth-cloaked envoys, clay-roofed settlements, cultivated river terraces, timber bridge spans, toll pennants, sluice levers, canvas-covered grain barges and a market awning. Preserve the rectangular river topology. Material textures and character poses may decorate native geometry, but inventory, bridge availability, diversion state, motion and boat loads must come from manifest bindings. Unbuilt bridge sites should look unfinished rather than falsely passable. The scene supplies legal glyph bindings, literal rules, world, actions, attributes and events. Final artwork and interaction controls remain separate trusted integration.

## Bounds and declared invariants
All state leaves are fixed-shape int32 or bool. Positions stay in [0,10]x[0,8]. Total physical grain across storage, cargo and active barges is initially 18 and gains at most 32 from eight floods; construction, pledges, food and export remove grain. Transfers conserve it. Therefore an individual grain stock stays below 64. Cargo/boat load stay within 0..2, pledge within 0..2, season within 0..8, and diversion/toll are binary. Co-located agents are intentional, so there is no unique-position invariant. Normalized observations are in [0,1]. Returned reward lies in [0,10], a conservative algebraic bound, not a measured extremum.

## Source-derived diagnostics
The two literal scripts demonstrate intended sluice/transport affordances. The paired hypothesis changes only role 0 action at step 2, then waits until the first flood. Another politys resource state and observation are the intended comparisons. Seeds equal the assigned ordinal and were chosen once; no outcomes or seed search informed the scripts. Scripts are unexecuted author hypotheses, not successful play, an optimal policy or a restriction on human behavior.

## Complete native action dictionary

- 0: Wait.
- 1: Walk north.
- 2: Walk east.
- 3: Walk south.
- 4: Walk west.
- 5: Build own bridge: 2 grain at own dock.
- 6: Toggle own diversion at dock; unavailable during pledge.
- 7: Pledge open sluice for 2 floods: 1 grain at own dock.
- 8: Load 1 grain at own town; cargo capacity 2.
- 9: Launch all cargo downstream at own dock.
- 10: Unload all cargo into the town underfoot.
- 11: Toggle own bridge toll at own dock.
- 12: Eat 1 stored grain at own town.
- 13: Sell all carried grain at eastern market.

## Literal source-line evidence

- Movement and grain toll: candidate.py line 49; literal source: def destination(self, s, actions):
- Incoming legal context: candidate.py line 63; literal source: def legal_actions(self, s):
- Pledge and real transport: candidate.py line 100; literal source: n['pledge'] = jnp.where(pledge, 2, s['pledge'])
- Ordered shared flood: candidate.py line 137; literal source: for i in range(3):
- Natural terminal and absorption: candidate.py line 147; literal source: n['terminated'] = (n['season'] >= 8) & ~jnp.any(n['boat_active'])
- Flattened observation: candidate.py line 157; literal source: def observe(self, s):


## Provenance, reference use and verification limits
This is source attempt 00 of the original blind initial window, not a repair or feedback-selected revision. Current batch ordinals 186..193 were reserved inside the original cap 512; cumulative used-or-reserved count was 194 and feedback count was zero. This author owns only GI190..193 attempt_00 and does not alter other work. Exact actual input paths, byte sizes, hashes and purposes are in author_input_freeze.json, including the prospective human-playability addendum. All four assigned design/request pairs were read; no other candidate, result, selection, project training, develop/confirm, heldout, IR or final-test input was accessed.

The supplied official JaxMARL README informed a prospective dictionary-per-agent adapter boundary. Pgx informed explicit external random keys, masks and environment identity. Craftax README informed native spatial state/presentation and raw versus outer autoreset. PureJaxRL README and Craftax_Baselines_ppo_rnn.py informed only reset/step/scan integration. They are algorithm integration references, not new game libraries or replacements for the frozen learner. No upstream game, algorithm code or visual asset was copied. No network request, model/API invocation, remote session, credential inspection, download or candidate execution was performed.

Signed enums, counters and ownership fields use lossless int32; state booleans use bool. The supplied reset key is not duplicated inside state. This representation adaptation does not imply changing an executed earlier distribution. Each new private-parameter distribution is stated above. Raw terminal semantics follow the addendum: a live call advances t once, natural termination overrides horizon, and an already-done call preserves state/observation/time/cause with zero rewards.

Only stdlib AST, JSON and hash checks were authorized. No candidate or JAX import, eager call, JIT, vmap, rollout, random policy, seed search, replay, RGB render, adapter test or training was run. Shapes/dtypes, masks over trajectories, numerical parity, invariant preservation and paired effects await independent frozen-source integration. Positive, zero, negative, failed and indeterminate results remain valid reports. No reward, seed-count or success-rate acceptance threshold is introduced. Source authorship is not human-playability acceptance or evidence of learning/transfer.
