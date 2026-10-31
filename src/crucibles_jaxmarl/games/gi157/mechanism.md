# Ghostlight Tag

**AUTHOR_ONLY_NOT_EXECUTED.** Initial parentless train proposal 157, window 0, source attempt 0. No candidate import, JAX evaluation, reset/step call, numerical R0, trajectory replay, rendered animation or learning run occurred during authoring. AST/syntax and declarative data-schema checks do not establish runtime correctness or game outcomes.

Native boundary: A=4, O=171, U=14, H=128. Observations are float32 `[A,O]`, actions int32 `[A]`, masks bool `[A,U]`, rewards float32 `[A]`, and every state leaf is fixed-shape int32 or bool. Source SHA-256: `a48d44379f9cdc85b5ee2bcda19d1c06a5e98b0e6ea29cd601f5b11d918674bd`.

## Game and cross-agent mechanisms

Coordinate beacon control, cover, pursuit, decoy placement and rescues to capture both opposing players simultaneously.

Beacon control changes both information and capture opportunities. Channeling a neutral beacon claims it with one charge; friendly channeling adds charge up to four, while enemy channeling drains it and makes it neutral when that channeling reduces charge to zero. At charge two or more it supplies team radar within radius four. A charged friendly beacon within radius two can also spend one charge to block a tag. Ward consumption can leave a zero-charge owned beacon; ownership is not silently erased by the ward phase.

Each team has one finite-lived movable decoy. Deploying places it one cell along the actor's facing, and steering changes its actual position. It jams opponent ordinary/radar sensing of targets within radius two and jams enemy wards in that area. Direct adjacency always reveals a real opponent, preventing hidden-contact ambiguity in tag resolution. Decoys are honest, labeled jammers when seen: the game does not claim their fixed observation slots are indistinguishable from real players. Their causal deception role is concealment and route baiting, not a renderer trick.

An attempted entry into an enemy decoy consumes the decoy, leaves the actor at its old cell, spends another stamina and imposes one forced-rest tick. Moving one's own decoy can open or close a pursuit route, hide a partner from a remote beacon, or prevent a ward from protecting an opponent. Own-team decoy coordinates are shared; enemy decoys require range-two sight. Beacon owner and charge are publicly visible, while hidden player/decoy positions remain masked in each role view.

Movement and rotating contention resolve before tags. Each tag action selects at most one adjacent eligible opponent, with rotating target priority. Cover protects an adjacent teammate rather than oneself. Ward charges are allocated to threatened victims in rotating order. Remaining tag decisions are simultaneous, so a captured attacker can still complete a valid simultaneous tag, and same-tick cover can still help. Rescuing is stricter: the target must have been captured before this tick and a newly captured rescuer cannot free it.

A team may invest in a safe radar/ward zone, coordinate cover while crossing a choke, jam a defended approach, or rescue its captured partner instead of pressing an attack. These are proposed strategic options, not demonstrated learning. Captured bodies remain on their real cells as obstacles and receive only the no-op mask until rescue; dones are joint because capture is reversible. Capture reward is +1 to both teammates per new opposing capture and -1 per own capture. Simultaneous events sum with bounds -2..2. Reset is deterministic and ignores its key. Future variants could change beacon placement, jamming range, cover geometry or decay; none is evaluated here.

## Actions, contention and temporal order

- `0`: Rest and recover two stamina; only action while captured or stunned.
- `1`: Move north and face north.
- `2`: Move east and face east.
- `3`: Move south and face south.
- `4`: Move west and face west.
- `5`: Attempt to tag one adjacent opponent; spend two stamina.
- `6`: Channel nearest adjacent beacon to charge own claim or drain enemy claim.
- `7`: Deploy own team decoy one cell along facing; spend two on success.
- `8`: Steer own team decoy north; spend one on success.
- `9`: Steer own team decoy east; spend one on success.
- `10`: Steer own team decoy south; spend one on success.
- `11`: Steer own team decoy west; spend one on success.
- `12`: Rescue adjacent previously captured teammate; spend two stamina.
- `13`: Cover an adjacent teammate from tags this tick; spend one stamina.

Out-of-range or masked actions deterministically become action 0. A legal attempt can be unfulfilled when its resolution-phase spatial, resource or joint-action prerequisites are absent. Ordinary movement rejects another agent's start-of-tick cell, so swaps are blocked. Empty-cell contention uses the smallest `(agent_id - t) mod A`. Rigid towing/hauling has its own explicit whole-body translation phase; its participating body can use its own vacated cells, and free agents remain blocked by the old and new body. All priorities and rule-specific ordering are literal in the source evidence below.

## Native observations and truthful rendering

Own-team telemetry, beacon state and capture/rescue identity notices are public to the relevant observation channels. Opponents require direct range or friendly beacon radar and may be hidden by enemy jammers. Adjacent opponents are always visible. Decoys are explicitly labeled when seen; no false indistinguishability claim is made.

- Columns `0:8` (self): Own id,team,x,y,stamina,stun,captured,clock.
- Columns `8:14` (public): Capture scores[2],currently captured counts[2],winner (-1 ongoing/draw),joint done.
- Columns `14:46` (local): Four player records: x,y,team,stamina,stun,captured,facing,visible. Teammates share telemetry; opponents use range/radar/jamming rules. Hidden fields -1.
- Columns `46:54` (public): Two beacon records: x,y,owner,charge.
- Columns `54:64` (local): Two decoy records: x,y,team,ttl,visible. Own decoy is known globally; enemy decoy requires radius 2.
- Columns `64:72` (public): Capture notices[4],rescue notices[4]; notices reveal identities, not hidden positions.
- Columns `72:171` (public): Static 9 by 11 public terrain map, yx row-major.

Positions are actual integer `[x,y]` coordinates with downward y; maps are indexed `[y,x]`. `scene.json` binds exact state paths for map, entities, visibility, attributes and events. Public event information is included in the flat observations. A per-step event is not relabeled as a cumulative counter or persistent level. Hidden entity fields are masked, and a full-state debug view must be labeled separately from role-visible replay. Scene descriptors contain no fabricated trajectory or executable rendering content. A real animation still requires an actual trusted replay.

## Termination, numerical bounds and intended invariants

The game terminates when either team has both players captured after the simultaneous capture/rescue phase. If both teams are fully captured on that step, winner remains -1 for a draw. A horizon of 128 otherwise truncates without assigning a winner.

Nonterminal calls advance the int32 clock exactly once. Natural completion or failure has priority over the horizon. An already terminal raw call preserves every state leaf, observation and clock, returns zero rewards and all dones true, and retains the original terminated/truncated causes. No raw auto-reset occurs.

Observation bounds are -1..256 and reward bounds -2..2. These are data/game bounds, not scientific effectiveness gates. Source-specific conservation, geometry and resource properties above are declared invariants for independent execution checks, not numerically verified claims. The manifest lists every leaf and safe structural invariants. Signed integer categories and coordinates are explicitly int32; the external uint32 reset key is not duplicated as hidden mutable state.

## Literal source evidence

- Information masks depend on real beacons and decoys: `candidate.py:12` — `def refresh(self, s):`
- Arena, teams and charge states: `candidate.py:26` — `def reset(self, key):`
- Inactive and stunned no-op semantics: `candidate.py:41` — `def legal_actions(self, s):`
- Real movable team decoys: `candidate.py:65` — `# Each team owns one real movable jammer; competing commands have rotating priority.`
- Interception-path trap: `candidate.py:81` — `# An attempted enemy-decoy entry consumes the decoy and stuns without advancing.`
- Public beacon charging: `candidate.py:94` — `# A charged beacon supplies team radar and consumable capture wards; opponents can drain it.`
- One-target attacks and ally cover: `candidate.py:106` — `attacks = (a == 5)[:,None] & (s["team"][:,None] != s["team"][None,:]) & (distance == 1) & ~state["captured"][None,:]`
- Jammable consumable wards: `candidate.py:116` — `jam = jnp.any((s["decoy_ttl"] > 0) & (s["decoy_team"] != s["team"][victim]) & (dd <= 2))`
- Simultaneous capture and rescue priority: `candidate.py:122` — `# Capture decisions are simultaneous; a newly captured rescuer cannot free a teammate.`
- Natural win/draw rule: `candidate.py:134` — `down0 = jnp.all(s["captured"][:2])`
- Role-visible native observations: `candidate.py:147` — `def observe(self, s):`

## Unexecuted playthroughs and interventions

`playthrough.json` supplies finite literal joint-action scripts and paired branches differing in exactly one role action at one decision. Every script and pair is `UNEXECUTED_AUTHOR_HYPOTHESIS`. Vector dimensions, integer action ranges and single-action differences were checked as data. Runtime legality, actual state changes, solvability, strategy quality and terminal success remain unverified. A future executor must retain failures or inconclusive traces rather than converting an intention into a success receipt.

## Reference use, interface handoff and provenance

Only the supplied official snapshots were consulted. JaxMARL supplies the agent-keyed parallel API convention and `__all__`; the restricted kernel keeps a homogeneous array boundary for later trusted integration. Pgx supplies explicit legality/terminal/key API ideas, not any board-game rules. Craftax supplies a reference for spatial inventories and persistent game state. PureJaxRL is an algorithm implementation reference, not a new environment library; the supplied Craftax recurrent PPO baseline itself credits PureJaxRL. None of their environment, model, training or wrapper code was copied, imported or run.

Frozen sources: [JaxMARL README](https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md), [Pgx README](https://github.com/sotetsuk/pgx/blob/main/README.md), [Craftax README](https://github.com/MichaelTMatthews/Craftax/blob/main/README.md), [PureJaxRL README](https://github.com/luchris429/purejaxrl/blob/main/README.md), [Craftax recurrent PPO baseline](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py). Their exact snapshot hashes are recorded in `author_input_freeze.json`.

A later trusted adapter may expose the fixed ids `agent_0` through `agent_3` as JaxMARL dictionaries, with `__all__` taken from `done_all` and unchanged observation/mask semantics. This author packet adds no learner or public adapter and does not modify the frozen experimental algorithm. Inactive-role no-op semantics, where relevant, are defined by the kernel rather than by renderer behavior.

`author_input_freeze.json` binds all twenty frozen inputs and the separately authorized administrative erratum. The erratum resolves a copied reservation description and expressly permits reading this batch authorization; it changes no design, rule or outcome. No earlier candidate files, GI/learned/CURRENT results, training/evaluation feedback or final-test material were read for this batch. Only the four assigned new output directories were written. Window accounting remains parentless ordinals 154..157, 158 used/reserved of the existing 512 cap, and unchanged later windows. This packet establishes authored assets only; runtime quality, replay, animation, learnability, transfer and scientific acceptance remain unestablished.
