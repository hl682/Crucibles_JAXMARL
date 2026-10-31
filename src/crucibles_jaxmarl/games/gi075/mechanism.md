# 75 — Scentbank Scramble — source attempt 00

Status: **AUTHORED_NOT_EXECUTED**. This is original authored source for one already-proposed game, not a new design count or an execution receipt.

## Rules and source evidence

Two teams of two indirectly guide six autonomous grazers. One indexed lure/repeller slot per actor affects all grazers by exact map shortest-path distance, regardless of ownership. Non-whistling actors repel within distance two, crowding affects candidate preferences, and public phase supplies a small directional bias. Candidates use one shared snapshot; grazers cannot follow into vacated old cells, and rotating rank settles free-target collisions. An occupied home pen needs three consecutive bench-settle transitions; valid settling holds an already-arrived grazer, while a new arrival may earn the first step. Capture removes exactly that grazer and supplies the zero-sum team difference, copied to teammates once. Puck placement is useful on its placement step, then ages; recovery starts four full cooldown states and expiration eight. No capture proxy or completion bonus is added.

- Exact map-only graph distances: `candidate.py:20`.
- Unchanged reset sampling distribution: `candidate.py:51`.
- Exact 868-column observation order: `candidate.py:92`.
- Masks and joint recovery contingency: `candidate.py:134`.
- Simultaneous agents and post-recovery placement: `candidate.py:162`.
- Shared-snapshot float32 preference computation: `candidate.py:216`.
- Exact tie order, calm hold and rotating target contention: `candidate.py:227`.
- Consecutive settling and one-time capture: `candidate.py:246`.
- Lifetime, cooldown, terminal priority and zero-sum reward: `candidate.py:257`.
- Actual movement/capture event buffers: `candidate.py:289`.
- Raw ended-state absorption: `candidate.py:157`.

## Interface, geometry and information

- The v2 contracts apply to ordinals 74/75 by the source-stage request despite their historical 72/73 headings.
- Design step_count is represented by scalar int32 state.t. All small signed counters, directions, codes and coordinates use int32; masks use bool; observations/rewards use float32.
- All state entity positions and event endpoints use [x=design column,y=design row]. Terrain, heat/fuel/wetness and map indexing remain [y,x]. Observation columns retain the design's row-then-column order.
- Reset PRNG key and provenance remain outside numeric state. Both original reset distributions are preserved by the source; no seed-conditioned override or canonical reset replacement is used.
- The v2 done leaf is done. Ended raw calls preserve the entire state and causes, return zero reward, and do not advance t. Natural termination takes priority over truncation.
- The exact graph distance geometry is held as an immutable derived int32[117,117] state leaf for static batching; it is computed from this map only and is not privileged information.
- Visibility arrays, immutable puck_owner and per-agent bank-score views expose only already-public mechanism state. Placement masks retain joint recovery opportunities; raw_step resolves actual post-recovery occupancy.

A deployment mask remains true when an occupying other-owner puck can be recovered by that owner from its current position in the same joint step. If that recovery is not actually chosen, raw deployment rejects without changing the slot. Permanently unavailable occupancy is masked. Agent target conflicts all fail; grazer conflicts choose the cyclic lowest rank and do not retry second choices. Opponent bank cells are forbidden to actors but open to grazers. A nearby opponent cannot veto successful bench settling.

All map layouts are explicitly public in the manifest. Entity visibility equals the corresponding actual active mask for every role. Removed grazers/pucks use [-1,-1] only with inactive masks; gathers sanitize those sentinels. A ruined Cinderline bed remains a public map site even though its living entity becomes inactive. Observation row/column order is preserved despite the world position conversion to x/y. Action submissions are not visible before joint commitment.

## Termination, bounds and invariants

All six captured grazers cause natural completion and a winner/tie from actual counts. At step 128 an unfinished herd truncates with winner_team=-1. Capture scores are monotone, score A + score B + active grazers = 6, and team returns equal the score difference. Active grazers and agents do not overlap; active pucks can share cells with a creature or agent but not another puck. Per-step team reward lies inside [-6,6]; observation bounds [-1,2] cover the normalized [0,1] public columns. These are declared bounds, not empirical results.

An already-ended raw state is returned unchanged, including t, last reward and last event buffers, while the returned reward is zero and dones are all true. Historical event buffers must be deduplicated using external episode identity, event_tick and slot. The kernel never auto-resets.

## Scene and event mapping

The scene binds agents, actual pucks and active grazers. Slots 0–3 are actor moves; 4–7 placement/recovery/rejection; 8–11 whistle changes; 12–17 grazer moves; 18–23 settle/capture; 24–27 expiry; 28 ending; 29–31 remain inactive. Capture payload is [old progress,new progress,team,captured flag], and the pen coordinate is recorded before the grazer is removed. Movement and capture can both occur in one transition.

The scene is declarative and contains no HTML, script, URL or invented trajectory. It supplies actual entity positions/masks and attribute/event bindings to the later trusted renderer. No real frame or animation has yet been generated.

## Explicit unresolved limits

- No candidate import, JAX import, execution, JIT, vmap, rollout, training or trusted validator invocation occurred. Shape/dtype claims are source declarations awaiting trusted execution.
- The safe generic scene vocabulary can bind actor/grazer motion, puck kinds/charge, settling progress and capture disappearance, but cannot encode the optional signed field contour overlay, curved recovery particles or score-tray flights. Those richer effects are not claimed implemented.
- No runtime solvability, numerical parity, balance, learning value or rendered-view acceptance is claimed. The original design uncertainties remain open.

The package includes a fixed, unexecuted action script. Its initial-state requirements are explicit. The proposed seed is not evidence that reset selects those requirements; do not override or resample the source reset to manufacture the example.

## Provenance

Author: `/root/astra_game_world_designer_001`; requested model `gpt-6-astra`, reasoning `max`. Source request SHA256: `ae7560c6c7e4dca45de71bb09283e94622a4c8d8523f08ab0acf5cf4a0db2cac`. Design JSON SHA256: `354a7bf0bd2148c0b0b8dd85c2f85e9c17748faa49e444341beae1cf260bb2c2`. Source SHA256: `9d6038ab31845dc1f1ce782334c3c0dee39ee4a4be409739fc21be116f39a818`. Source split is train, final_test_used=false, source_attempt=0. Backend revision and token usage are unknown/null. One source attempt is authored for this already-existing design; repair attempts, environment calls, rollout steps and training updates are zero. No trusted validator PASS is claimed.
