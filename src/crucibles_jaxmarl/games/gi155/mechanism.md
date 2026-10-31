# Echo Tow

**AUTHOR_ONLY_NOT_EXECUTED.** Initial parentless train proposal 155, window 0, source attempt 0. No candidate import, JAX evaluation, reset/step call, numerical R0, trajectory replay, rendered animation or learning run occurred during authoring. AST/syntax and declarative data-schema checks do not establish runtime correctness or game outcomes.

Native boundary: A=4, O=165, U=14, H=144. Observations are float32 `[A,O]`, actions int32 `[A]`, masks bool `[A,U]`, rewards float32 `[A]`, and every state leaf is fixed-shape int32 or bool. Source SHA-256: `79ada6b56599ca59b250462366078ce1963c5d018bc64458658179e09f04dc79`.

## Game and cross-agent mechanisms

Recover two heavy cargo objects at matching berths by coordinating two attached tugs, scout information, current diversion and shared oxygen resupply.

A tow is a rigid three-cell body: one cargo with tugs 1 and 2 on opposite horizontal flanks. Both hooks, opposite geometry, matching directional actions and enough oxygen are required. The complete translated body must fit terrain and avoid other cargo or divers. Vertical and horizontal towing are allowed while flank orientation remains horizontal. Successful arrival at the corresponding berth accounts for the cargo, disables its active sprite and releases both hooks.

The engineer diverts a current before the tow phase. Without that action, a current cell can block cargo even when the tugs' intended path is otherwise open. A diver may traverse that cell at an extra oxygen cost, so the current is not a decorative global penalty. Waiting for diversion, taking a longer route, and splitting scout/engineer time between assistance and oxygen supply are distinct prospective choices.

The scout has longer local sonar and can spend oxygen to extend it temporarily. Broadcasting selects the nearest sensed active cargo, otherwise a sensed current. The message contains an actual position, type, object id, age and finite lifetime; all roles receive that same data in observations. The signal can therefore tell the engineer about cargo outside its own range. A moved cargo leaves the last radio position stale, with its age visible. The role renderer must never replace that marker with the object's hidden current position.

Dock oxygen stock regenerates one per step up to eighteen. At most one adjacent requester receives up to six oxygen per dock per step; rotating priority resolves contention. Each diver has capacity twenty-four. Sharing transfers three oxygen, while breathing, swimming, towing, sonar, broadcasting, diversion and current drag explicitly consume it. These are finite game resource rules, not physical diving advice. Choosing who charges first can change whether another crew survives long enough to complete a coordinated haul.

Two immutable cargo identities are conserved; a recovered one remains accounted for. The second cargo's x coordinate is sampled as seven or eight using the reset key; the first cargo and scripted opening are fixed. Every crew is retained until joint completion/abort. A final full recovery takes priority over same-step oxygen exhaustion. Future variants could vary sensor range, current placement, tow geometry or dock throughput; no such measurements are claimed.

## Actions, contention and temporal order

- `0`: Wait.
- `1`: Move north or request synchronized northward tow.
- `2`: Move east or request synchronized eastward tow.
- `3`: Move south or request synchronized southward tow.
- `4`: Move west or request synchronized westward tow.
- `5`: Tug: attach at a horizontal cargo flank, or release.
- `6`: Scout: extend sonar radius to six for three ticks.
- `7`: Scout: broadcast nearest visible active cargo, otherwise nearest visible current.
- `8`: Engineer: divert nearest adjacent current for four ticks.
- `9`: Request oxygen from nearest adjacent dock.
- `10`: Give three oxygen to adjacent agent 0.
- `11`: Give three oxygen to adjacent agent 1.
- `12`: Give three oxygen to adjacent agent 2.
- `13`: Give three oxygen to adjacent agent 3.

Out-of-range or masked actions deterministically become action 0. A legal attempt can be unfulfilled when its resolution-phase spatial, resource or joint-action prerequisites are absent. Ordinary movement rejects another agent's start-of-tick cell, so swaps are blocked. Empty-cell contention uses the smallest `(agent_id - t) mod A`. Rigid towing/hauling has its own explicit whole-body translation phase; its participating body can use its own vacated cells, and free agents remain blocked by the old and new body. All priorities and rule-specific ordering are literal in the source evidence below.

## Native observations and truthful rendering

Map, docks/stock, berths, recovery announcements, ping timer and aged radio snapshot are public. The scout sees within Manhattan radius four or six after ping; other roles see within radius two. These are range-only sensors without occlusion. The snapshot is not privileged live tracking.

- Columns `0:8` (self): Own id,x,y,oxygen,hook id,public scout ping timer,clock,joint done.
- Columns `8:13` (public): Salvaged flags[2],salvage event flags[2],joint abort flag.
- Columns `13:33` (local): Four diver records: x,y,oxygen,hook,visible. Scout radius 4 or 6; others radius 2. Hidden values -1.
- Columns `33:41` (local): Two cargo records: x,y,salvaged,visible. Only unsalvaged locally sensed cargo is shown.
- Columns `41:49` (local): Two current records: x,y,diversion timer,visible. Flowing is exactly timer==0.
- Columns `49:55` (public): Two docks: x,y,available oxygen stock.
- Columns `55:62` (public): Radio snapshot x,y,kind,object id,age,remaining ttl,active. Inactive first six values are -1. Snapshot coordinates do not track later object motion.
- Columns `62:66` (public): Two matching berth x,y coordinates.
- Columns `66:165` (public): Static 9 by 11 public terrain map, yx row-major. Hidden cargo/current locations are not encoded in terrain.

Positions are actual integer `[x,y]` coordinates with downward y; maps are indexed `[y,x]`. `scene.json` binds exact state paths for map, entities, visibility, attributes and events. Public event information is included in the flat observations. A per-step event is not relabeled as a cumulative counter or persistent level. Hidden entity fields are masked, and a full-state debug view must be labeled separately from role-visible replay. Scene descriptors contain no fabricated trajectory or executable rendering content. A real animation still requires an actual trusted replay.

## Termination, numerical bounds and intended invariants

Both cargo recovered ends naturally. Any crew oxygen reaching zero before complete recovery causes joint emergency abort. Otherwise the horizon is 144.

Nonterminal calls advance the int32 clock exactly once. Natural completion or failure has priority over the horizon. An already terminal raw call preserves every state leaf, observation and clock, returns zero rewards and all dones true, and retains the original terminated/truncated causes. No raw auto-reset occurs.

Observation bounds are -1..144 and reward bounds -1..2. These are data/game bounds, not scientific effectiveness gates. Source-specific conservation, geometry and resource properties above are declared invariants for independent execution checks, not numerically verified claims. The manifest lists every leaf and safe structural invariants. Signed integer categories and coordinates are explicitly int32; the external uint32 reset key is not duplicated as hidden mutable state.

## Literal source evidence

- Asymmetric sonar and radio visibility: `candidate.py:12` — `def refresh(self, s):`
- Explicit-key second-cargo location: `candidate.py:30` — `second_x = jax.random.randint(key,(),7,9,dtype=jnp.int32)`
- Role action masks: `candidate.py:43` — `def legal_actions(self, s):`
- Engineer diversion before tow: `candidate.py:67` — `divert = (a[3] == 8) & (distance[c] <= 1)`
- Rigid opposed tug fittings: `candidate.py:70` — `# Only the two tugs have fittings; each must occupy an opposite horizontal flank.`
- Current and body blocking: `candidate.py:89` — `current_block = jnp.any(jnp.all(body[0] == s["current_pos"],axis=1) & (s["divert_ttl"] == 0))`
- Source-faithful aged broadcast: `candidate.py:105` — `# The radio publishes a true snapshot, not a continuously privileged cargo track.`
- Oxygen-dock contention: `candidate.py:130` — `used = jnp.zeros(2,jnp.bool_)`
- Recovery and oxygen abort: `candidate.py:142` — `delivered = jnp.all(s["cargo_pos"] == s["berth_pos"],axis=1) & ~state["salvaged"]`
- Native observations: `candidate.py:158` — `def observe(self, s):`

## Unexecuted playthroughs and interventions

`playthrough.json` supplies finite literal joint-action scripts and paired branches differing in exactly one role action at one decision. Every script and pair is `UNEXECUTED_AUTHOR_HYPOTHESIS`. Vector dimensions, integer action ranges and single-action differences were checked as data. Runtime legality, actual state changes, solvability, strategy quality and terminal success remain unverified. A future executor must retain failures or inconclusive traces rather than converting an intention into a success receipt.

## Reference use, interface handoff and provenance

Only the supplied official snapshots were consulted. JaxMARL supplies the agent-keyed parallel API convention and `__all__`; the restricted kernel keeps a homogeneous array boundary for later trusted integration. Pgx supplies explicit legality/terminal/key API ideas, not any board-game rules. Craftax supplies a reference for spatial inventories and persistent game state. PureJaxRL is an algorithm implementation reference, not a new environment library; the supplied Craftax recurrent PPO baseline itself credits PureJaxRL. None of their environment, model, training or wrapper code was copied, imported or run.

Frozen sources: [JaxMARL README](https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md), [Pgx README](https://github.com/sotetsuk/pgx/blob/main/README.md), [Craftax README](https://github.com/MichaelTMatthews/Craftax/blob/main/README.md), [PureJaxRL README](https://github.com/luchris429/purejaxrl/blob/main/README.md), [Craftax recurrent PPO baseline](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py). Their exact snapshot hashes are recorded in `author_input_freeze.json`.

A later trusted adapter may expose the fixed ids `agent_0` through `agent_3` as JaxMARL dictionaries, with `__all__` taken from `done_all` and unchanged observation/mask semantics. This author packet adds no learner or public adapter and does not modify the frozen experimental algorithm. Inactive-role no-op semantics, where relevant, are defined by the kernel rather than by renderer behavior.

`author_input_freeze.json` binds all twenty frozen inputs and the separately authorized administrative erratum. The erratum resolves a copied reservation description and expressly permits reading this batch authorization; it changes no design, rule or outcome. No earlier candidate files, GI/learned/CURRENT results, training/evaluation feedback or final-test material were read for this batch. Only the four assigned new output directories were written. Window accounting remains parentless ordinals 154..157, 158 used/reserved of the existing 512 cap, and unchanged later windows. This packet establishes authored assets only; runtime quality, replay, animation, learnability, transfer and scientific acceptance remain unestablished.
