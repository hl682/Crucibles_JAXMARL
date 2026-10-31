# 74 — Cinderline Commons — source attempt 00

Status: **AUTHORED_NOT_EXECUTED**. This is original authored source for one already-proposed game, not a new design count or an execution receipt.

## Rules and source evidence

Three specialists share the original finite-fuel containment objective. The sprayer carries six water units and refills only on a well; the cutter removes up to three fuel units from eligible adjacent brush; the steward carries two repair kits and restocks only at the depot. Spray is applied before the other tools' heat checks, allowing a genuinely simultaneous cooling-plus-cut/repair action. Fire reads one post-tool snapshot, burns only old dry fires, does not cascade new ignitions in the same step, and damages seedbeds afterward. Wind rotates after each 16 transitions. The reward remains terminal ±5 minus 0.10 per actual damage unit, 0.001 per fire-consumed fuel unit, 0.0005 per cut unit and 0.002 per live step.

- Map, roles and randomized ignition/wind: `candidate.py:43`.
- Exact 1471-column observation order: `candidate.py:85`.
- Union mask and heat-contingent tool eligibility: `candidate.py:115`.
- Simultaneous old-occupancy movement: `candidate.py:152`.
- Spray, cut and repair order: `candidate.py:172`.
- Synchronous fire and old-wind exposure: `candidate.py:190`.
- Post-fire health loss and lock: `candidate.py:202`.
- Natural precedence and exact reward terms: `candidate.py:208`.
- Actual fixed-slot transition events: `candidate.py:241`.
- Raw ended-state absorption: `candidate.py:140`.

## Interface, geometry and information

- The v2 contracts apply to ordinals 74/75 by the source-stage request despite their historical 72/73 headings.
- Design step_count is represented by scalar int32 state.t. All small signed counters, directions, codes and coordinates use int32; masks use bool; observations/rewards use float32.
- All state entity positions and event endpoints use [x=design column,y=design row]. Terrain, heat/fuel/wetness and map indexing remain [y,x]. Observation columns retain the design's row-then-column order.
- Reset PRNG key and provenance remain outside numeric state. Both original reset distributions are preserved by the source; no seed-conditioned override or canonical reset replacement is used.
- The v2 done leaf is done. Ended raw calls preserve the entire state and causes, return zero reward, and do not advance t. Natural termination takes priority over truncation.
- Native action counts [10,9,10] are padded to union 10. Cutter action 9 is always masked false and raw-canonicalized to wait. Post-spray cutter/repair eligibility remains joint-action dependent.
- Derived flattened heat/fuel/wetness views and flame/brush/wet active/visibility masks share actual cell coordinates; they add no dynamics or policy information.

The action masks omit hot-target restrictions for cutting and repair because a simultaneous spray may change eligibility. Raw execution rejects an actually ineligible tool without spending resources. Contested agent destinations all fail; swaps/follows/cycles into old occupied cells fail. Movement requires old heat <=1, while a newly hot occupied cell may lock tools without blocking escape. Role 1 padding does not appear as a native action in last_actions.

All map layouts are explicitly public in the manifest. Entity visibility equals the corresponding actual active mask for every role. Removed grazers/pucks use [-1,-1] only with inactive masks; gathers sanitize those sentinels. A ruined Cinderline bed remains a public map site even though its living entity becomes inactive. Observation row/column order is preserved despite the world position conversion to x/y. Action submissions are not visible before joint commitment.

## Termination, bounds and invariants

Any ruined seedbed causes natural failure; otherwise no heat with living beds causes natural success. Natural failure takes priority if both occur. The 144-step horizon truncates only if no natural ending occurs. Fuel is monotonically nonincreasing; nonflammable terrain holds no fuel/heat; water, kits, heat, wetness and health retain their design limits. Movement preserves unique cells. Conservative observation bounds [-64,8] include the cumulative team return: at most 144 steps, at most three damage units per step, at most 858 total initial fuel units, and at most one terminal component bound the negative return above -64. Per-step reward lies inside [-6,6]. These are analytic containment bounds, not achieved performance.

An already-ended raw state is returned unchanged, including t, last reward and last event buffers, while the returned reward is zero and dones are all true. Historical event buffers must be deduplicated using external episode identity, event_tick and slot. The kernel never auto-resets.

## Scene and event mapping

The scene uses agent, plant, hazard and water glyphs attached to actual x/y state arrays. Heat, remaining fuel and wetness are flattened deterministic views of the physical grid, not display-only mechanics. Slots 0–2 are actor moves; 3–5 tools; 6–148 changed cells; 149–151 gross seedbed repair/damage; 152–154 locks; 155 wind; 156 ending; 157–159 remain inactive. Cell event payload is [old heat,new heat,old fuel,new fuel]; seedbed payload is [old health,repair,damage,new health].

The scene is declarative and contains no HTML, script, URL or invented trajectory. It supplies actual entity positions/masks and attribute/event bindings to the later trusted renderer. No real frame or animation has yet been generated.

## Explicit unresolved limits

- No candidate import, JAX import, execution, JIT, vmap, rollout, training or trusted validator invocation occurred. Shape/dtype claims are source declarations awaiting trusted execution.
- The safe generic scene vocabulary can bind moving actors and actual flame/brush/wet entities and quantities, but cannot prescribe custom flame-height shaders, spray arcs, cutter flashes or per-leaf art. Those richer effects are not claimed implemented.
- No runtime solvability, numerical parity, balance, learning value or rendered-view acceptance is claimed. The original design uncertainties remain open.

The package includes a fixed, unexecuted action script. Its initial-state requirements are explicit. The proposed seed is not evidence that reset selects those requirements; do not override or resample the source reset to manufacture the example.

## Provenance

Author: `/root/astra_game_world_designer_001`; requested model `gpt-6-astra`, reasoning `max`. Source request SHA256: `ae7560c6c7e4dca45de71bb09283e94622a4c8d8523f08ab0acf5cf4a0db2cac`. Design JSON SHA256: `7d47b49f55fca94ca1e994d234f549b4e27bc5daeb1fd9f7d170108fb99dac40`. Source SHA256: `7c3c8e794ed83a2103d086623f9c01a837de9a6a80d2b5ec06abcf278fee9aa6`. Source split is train, final_test_used=false, source_attempt=0. Backend revision and token usage are unknown/null. One source attempt is authored for this already-existing design; repair attempts, environment calls, rollout steps and training updates are zero. No trusted validator PASS is claimed.
