# Veinbound — source package game-v3-00077 / attempt_00

Status: **AUTHORED_NOT_EXECUTED**. This is the complete 4-agent, 2196-column, 15-action, H240 source proposal for existing design 77. No import, simulation, renderer, rollout or training was executed. It is not an additional novel design.

Source request SHA-256: 18f3e7720bd8ef5563c5fa5a838c0ccb461673f034c90c4d25167eaf48549559  
Design JSON SHA-256: dd04672088c1cf3fe1278ea9e3098366d405ce89ea04410b83db14c01b2abcc4  
candidate.py SHA-256: f9d8d714e8b4f58a2151a849916c2a638014ccd96a761eeb61b4f0d04bccbf74

## Spatial game and roles

Two teams occupy the actual 13×11 diggable map. Team 0 uses agents 0/1; team 1 uses 2/3. Each team has one surveyor and one excavator. Banks, protected approaches, internal boulders and walls are implemented geometry. Players move one grid cell, face an interaction direction, consume charge and carry at most one actual geode.

The reset samples four distinct mirror pairs from the specified 25 pairs, permutes two amber/two teal left-side colours with opposite right colours, and independently permutes eight public object ids. It preserves the full reset distribution. It does not choose an easier layout for playthrough.json.

Only surveyors scan; they pay 1 for Manhattan radius 2 or 2 for radius 4. Counts include only still-buried geodes at scan time. Records store origin, radius, both counts, tick and original scanner id; they are never regenerated from hidden current truth. Each holder retains six records. Physical adjacency plus one exchange action merges the two post-scan ledgers, deduplicates source/tick and retains the newest six.

Excavators add two depth units per dig and surveyors one, both for one charge. Concurrent contributions add and cap at depth two. Reaching depth two exposes the actual geode, if any; empty digging leaves a real empty crater. Persistent excavation and the depletion of buried objects change later scan signals.

Surface objects can be picked, carried, dropped, forced out of a carrier or banked. Stacks retain object identities. Same-object pickup conflicts all fail; lowest-id selection only chooses an object already at the addressed surface cell. It is not an agent-priority tie break.

## Competition and transition semantics

A legal shove requires a rival in the front cell after movement, a neutral target cell, no target brace, an empty passable destination, charge >=2 and old cooldown zero. Shared-target/shared-destination shoves cancel; a preliminary targeted actor cannot simultaneously complete its own shove. No chain pushes are simulated.

An accepted shove moves the target, drops its carried geode at the old cell, sets stun, cancels remaining operations and starts the actor's two-step cooldown. Protected bases prevent entry by enemies and protect their occupants. Bracing gives a visible one-transition shield; it does not create persistent invulnerability.

The stage order is movement/facing; simultaneous shove resolution; digging/reveal; drops then pickups; banking/recharge; scans; teammate record exchange; carried-coordinate synchronization; ending. The two teams' deposits are summed before outcome comparison.

The original design says both that the first stage applies facing changes before shoves and that a shoved target's non-movement action is canceled. The coordinator explicitly clarified this wording before any execution: an already committed turn is not rolled back; the shove cancels only operations not yet executed. This source follows that public stage order, and the turn event records the actual early change. The original design bytes remain unchanged. Later verification must use this fixed interpretation, not choose a rule after observing outcomes.

## Action masks preserve simultaneous cooperation

Actions 0..14 are wait; N/E/S/W step; clockwise turn; dig front; handle front; bank; near scan; far scan; exchange; brace; shove; recharge.

Masks intentionally allow context-valid attempts whose final condition may be created by another player's same-tick action. Empty-handed handling can be selected before a partner exposes or drops a geode; exchange can be selected before a partner moves into adjacency; a shove can be selected before its rival enters the front cell or a destination is vacated. The raw stages still require the actual post-stage conditions.

This avoids removing the design's simultaneous dig/pick and drop/pick coordination through an overly strict old-state mask. Invalid/out-of-range actions are deterministic no-ops. Stunned and ended inputs expose the wait action.

## Information and role-render boundary

observe returns exactly float32[4,2196]: local board 1296, known static terrain 715, agent roster 40, visible geodes 64, private records 48, and global/self 33.

Static geometry, assignments, public scores and banked total are public. Current depth, other agents and surface/carried objects require the returned-state radius-4 view. Buried positions/colours are engine-only, including when the surrounding soil is visible. Other holders' record arrays never enter the role observation.

The source computes actual map, agent, geode, soil and private-record visibility leaves. Historical record origins may appear outside current local view only to their holder, because they are legitimate stored information. They are not underground target coordinates.

Event masks separately restrict event existence, both endpoints, actor identity, object identity and value. Bank/end announcements are public; that does not publish an invisible depositor or geode id. A scan event carries radius only. Exchange ledger size is restricted to participants. No global reward is given for private scan/discovery/shove events.

The manifest and scene explicitly label **DEBUG FULL STATE — reveals underground geodes and every private ledger; never a policy observation**. Default role views must filter state before serialization. Static terrain outside local view may be known; current excavation depth and entities stay unknown/fogged. No omniscient ledger or hidden-truth heatmap is a valid role view.

## Rewards, endings and invariants

Team 0 receives dS0-dS1 plus 3 for a newly selected natural win or minus 3 for a natural loss; its two agents share that value. Team 1 receives the opposite. Banking preferred colour gives 2 points, other colour 1. No repeated banking or shaping for scans/digging/carrying/shoving is introduced.

After complete simultaneous deposits, any score >=8 gives a natural score-race ending. If both reach the threshold, compare the new scores; ties are allowed. Otherwise all eight banked ends the round by score comparison. At H240 without a natural ending, winner is -3, unresolved truncation. Raw ended inputs absorb without resetting, time advance or reward.

Full obligations include distinct agent positions, correct own-base access, eight conserved geodes/four per colour, reciprocal one-cargo carrier links, monotone banked status, bounded depth/charge/stun/cooldown, immutable historical measurements, six-record capacity and no hidden-state leakage. These are implementation obligations, not runtime-verified facts. Machine-readable nonnegative/unique-position invariants use the supplied known forms; all full obligations remain in declared_invariants.

## Scene bindings and representation adaptations

manifest.json enumerates 67 actual numeric state leaves. Five entity groups bind actual players, geodes, banks, 143 physical terrain cells and 24 private ledger slots. A buried geode is physically active for the engine/debug view but invisible to every role until exposed.

Actual excavation depth changes visible tile appearance. Detector rings use committed scan events and recorded radii. Visible crystals follow real pickup/carrier/drop/bank state. Shoves use their actual intermediate event endpoints. Banks change only on real points. Numbers supplement the board, not replace it.

All state numerics use int32/bool, observations/rewards float32. The reset PRNG key stays outside numeric state as allowed by addendum v2. Grid reads use yx; all world/manifest positions remain xy. Render entity arrays are deterministic reshape/projection leaves; they add no observations or randomness. Event identities retain the clock and slot so absorbing snapshots do not replay effects.

- `candidate.py:122` — `def reset(self, key):`: implements the 25-pair sampler, independent colour/id permutations, map and initial roles
- `candidate.py:49` — `def resolveMoves(self, positions, targets, moving, valid):`: implements simultaneous movement conflicts and cycles
- `candidate.py:178` — `def legal_actions(self, state):`: permits the documented concurrent-interaction attempts without hidden-state oracle masks
- `candidate.py:343` — `def activeStep(self, state, actions):`: implements shoves, exposure, possession, banking, scans, sharing and end precedence
- `candidate.py:298` — `def appendRecord(self, state, agent, active, origin, radius, counts):`: stores an actual scan result in six fixed historical slots
- `candidate.py:319` — `def mergeRecords(self, state, left, right, active):`: unions and deduplicates only the two permitted teammate ledgers
- `candidate.py:209` — `def observe(self, state):`: constructs the exact 2196-column role-specific observations
- `candidate.py:58` — `def present(self, state):`: constructs real entity and event component visibility masks
- `candidate.py:500` — `def raw_step(self, key, state, actions):`: returns the raw terminal state and suppresses rewards on absorbing calls

## Remaining feasibility and provenance limits

playthrough.json gives a fixed eight-step movement/scan/exchange/dig/recharge hypothesis whose initial positional/resource requirements are explicit and which does not depend on a particular sampled geode location. A proposed seed is not evidence of its underground outcome. No discovery, win, animation or legal-replay receipt is claimed.

The turn/shove interpretation above was explicitly fixed by the coordinator before execution and is also recorded in scene.json and generation_record.json. Execution may still reveal shape, whitelist, numerical-parity, masking, rendering, performance, balance or strategic issues. No source validator, JAX import, source call, renderer or training was used. Stdlib AST text parsing is not contract acceptance.

The 2196-column and 15-action game needs matching policy metadata/weights and new throughput/storage evidence. The earlier two-role payoff/rank implementation is not evidence of four-player support. source_split is train; no final-test data or outcomes were used.

Source execution 0; environment calls 0; rollout steps 0; training updates 0. backend_revision and token_usage remain null.
