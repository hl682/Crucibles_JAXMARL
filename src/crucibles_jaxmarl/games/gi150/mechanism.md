# Relay Foundry

**AUTHOR_ONLY_NOT_EXECUTED.** This is initial parentless proposal 150, source attempt 0, in the frozen train-only window. Candidate runtime execution, numerical R0, replay, rendered animation, learning and scientific acceptance have not occurred in this authoring task. Python AST/syntax and declarative data-shape/schema screens are authoring checks only; they do not establish any game outcome.

Native contract: A=4, O=153, U=13, H=192. Observations are float32 `[A,O]`, actions int32 `[A]`, rewards float32 `[A]`, masks bool `[A,U]`, and state leaves are fixed-shape int32/bool arrays. Source SHA-256: `c9dedefd7cfb5f8f316d9ffabf6862034fdc8f50c90aa8a10ad10d75a1de71a9`.

## Task and rules

Complete four finite serial orders in FIFO order by relaying parts through three complementary manufacturing capabilities and one shared portable clamp.

A part cannot be completed by one autonomous worker: role 0 must cut it, role 1 must weld it with a physically present assistant and the unique clamp, and role 2 must calibrate it under the same complementary condition. Role 3 can maintain transport and assistance while specialists stay near stations. Passing is to an explicitly selected adjacent role. A single pass transfers the donor's part if the receiver is empty and also transfers the donor's clamp; each asset can transfer at most once in the phase. A full receiver can receive the clamp alone. The clamp can also be deposited for later pickup, so holding it is an actual reservation that excludes other users.

The wall splits the shop; its only gap is the central corridor. Couriers can relay at the gap, or specialists can travel with the clamp, at the expense of their worksite availability. A queue-focused strategy may rush part 0, whereas a pipeline may cut later parts while powered stages and shipping process the head. A pipeline that blocks the corridor or hoards the clamp can stop a teammate. Waiting to cool, assisting instead of moving one's own part, and deciding who carries the clamp therefore change feasible future joint actions. These are prospective strategic hypotheses, not measured policy behavior.

The four fixed part identities persist; each is either on a cell, carried by exactly one role, or shipped. A carrier has at most one part, there is exactly one clamp, stages only increase, and `queue_head` equals the number of stage-4 parts. These are source-declared invariants for later numerical checks. End-of-step local visibility is refreshed from actual positions; it does not pretend that walls occlude sight. Shipped parts become inactive, and the public queue count retains their completion.

Rule or task variants could relocate the narrow gap, reorder serial demand, change clamp cooling or remap the complementary roles. Such variants would alter transport and scheduling demands, but no variant generator, adaptation result or generalization result is claimed here. The current reset is deterministic and intentionally ignores its supplied key.

## Actions and simultaneous timing

- `0`: Wait.
- `1`: Move north.
- `2`: Move east.
- `3`: Move south.
- `4`: Move west.
- `5`: Pick up or drop a part on this cell.
- `6`: Operate matching station or ship current FIFO head.
- `7`: Assist an adjacent powered station.
- `8`: Take or release the shared clamp on this cell.
- `9`: Pass held part and clamp to adjacent agent 0.
- `10`: Pass held part and clamp to adjacent agent 1.
- `11`: Pass held part and clamp to adjacent agent 2.
- `12`: Pass held part and clamp to adjacent agent 3.

Movement rejects another agent's start-of-tick cell even if that agent moves; swaps are blocked. Among proposals to the same empty cell, the smallest (agent_id - t) mod A wins. Terrain or joint contention may make a legal movement attempt stationary.

Out-of-range or masked actions deterministically become action 0. Legal interaction attempts may still do nothing when their resolution-phase prerequisites are absent; the action labels state their intended operation. All roles remain active until joint termination, so no eliminated role is silently deleted. Terminal masks retain action 0.

## Observation and renderer contract

Map and audit counts are public. Entity records are current Manhattan-radius-3 observations without wall occlusion. Own attributes are explicit; hidden entity fields use -1 plus a zero visibility bit. The renderer must use the same entity visibility, and public events are present in flat observations.

- Columns `0:6` (self): Own role, x, y, carried part id (-1 empty), clamp-held flag, clock.
- Columns `6:11` (public): FIFO head, stage-advance count this tick, transfer count this tick, shipping bell, joint episode done.
- Columns `11:35` (local): Four agent records in id order: x,y,role,carried part id,clamp flag,visible. Hidden fields -1; final flag 0.
- Columns `35:55` (local): Four part records in serial order: x,y,assembly stage,carrier id,visible. Only active parts within Manhattan radius 3.
- Columns `55:60` (local): Clamp record: x,y,carrier id,heat,visible; hidden values -1.
- Columns `60:76` (local): Four station records in station order: x,y,type,visible; radius 3.
- Columns `76:153` (public): Static 7 by 11 public tile map flattened row-major yx; codes match world legend.

All coordinates are integer `[x,y]`; the tile map uses `[y,x]` indexing and a downward y axis. `scene.json` binds these actual map, position, active, visibility, attribute and event paths. It contains no invented motion, external image, executable markup or policy-only information. Public event channels are also present in observations. Counters and persistent levels are labeled separately from per-step events. Full-state inspection must be distinguished from role-visible playback. A real trajectory and its animation remain to be produced by the trusted implementation process.

## Termination, rewards and conservation

Terminate when queue_head equals 4, meaning all four physical parts are shipped. Otherwise truncate at 192 nonterminal steps.

A nonterminal call advances the int32 clock once. Natural completion takes precedence over the horizon. Terminal raw calls preserve every leaf, observation and clock, return zero rewards and all dones true, and preserve terminated/truncated. No episode reset occurs in raw_step.

Declared finite observation bounds are -1..192; reward bounds are -0.01..1.5. These are game data bounds, not outcome thresholds. Rule-specific reward terms and conservation claims above are literal source design claims; they remain untested numerically. `manifest.json` declares every state leaf and safe structural invariants; source-dependent accounting identities require independent execution review.

## Source-line evidence

- Spatial state and visibility: `candidate.py:12` — `def refresh(self, s):`
- Finite initial materials, map and roles: `candidate.py:29` — `def reset(self, key):`
- Attempt masks and absorbing no-op: `candidate.py:59` — `def legal_actions(self, s):`
- Conservative simultaneous collision resolution: `candidate.py:69` — `def move(self, s, actions):`
- Ownership transfers: `candidate.py:94` — `original = s["part_carrier"]`
- Complementary powered stages: `candidate.py:125` — `for role in range(3):`
- FIFO delivery and natural completion: `candidate.py:136` — `head = jnp.clip(s["queue_head"], 0, 3)`
- Raw terminal preservation: `candidate.py:150` — `s = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, s)`
- Literal flat observation construction: `candidate.py:154` — `def observe(self, s):`

## Prospective playthroughs and single-action interventions

`playthrough.json` contains literal source-derived scripts and paired branches that change exactly one role action at one decision. Each script is explicitly `UNEXECUTED_AUTHOR_HYPOTHESIS`; its action dimensions and ranges were checked as data, while runtime legality, actual trajectories, numerical parity and success were not checked. No script result, score, learning gain or browser acceptance is invented. The paired hypotheses identify state paths for later comparison and do not substitute for measured causal traces.

## Community interface and provenance

The supplied official JaxMARL README informed the separation between the homogeneous array kernel and a later agent-keyed parallel API with `__all__`; no upstream environment was copied. Pgx informed explicit legality, raw terminal state and explicit-key provenance, rather than importing a board-game implementation. Craftax supplied a reference for spatial objects, inventories and persistent consequences; this game is an original multiplayer design. PureJaxRL is an algorithm implementation reference, not a game library. The supplied Craftax recurrent PPO baseline credits PureJaxRL; its training/network/wrapper code was not copied, run or substituted for the frozen learner.

Reference snapshots are hash-frozen in `author_input_freeze.json` and the assigned request: [JaxMARL README](https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md), [Pgx README](https://github.com/sotetsuk/pgx/blob/main/README.md), [Craftax README](https://github.com/MichaelTMatthews/Craftax/blob/main/README.md), [PureJaxRL README](https://github.com/luchris429/purejaxrl/blob/main/README.md), and [Craftax recurrent PPO baseline](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py). API/design ideas only; copied upstream code: none.

The public JaxMARL adapter belongs to the trusted integration layer: it should map fixed ids `agent_0` through `agent_3` to the existing array boundary, expose masks and set done `__all__` from `done_all`. No agent dictionary adapter or new learner is smuggled into this restricted one-class kernel. Source integer categories use int32, flags use bool, and the supplied uint32 reset key is external provenance rather than a hidden mutable state object.

The source/manifest/scene hashes are bound in `generation_record.json`; `author_input_freeze.json` binds the verified frozen author inputs. No prior candidate, result, replay, rank, training feedback or final-test material was read. No numeric candidate execution, CSD3 operation, model download or background task was performed. Initial-window accounting remains four parentless proposals 150..153, 154 used or reserved of the existing 512 cap, with later windows unchanged. Authoring and visualization support alone do not establish learnability, non-isomorphism, generalization or meta-learning.
