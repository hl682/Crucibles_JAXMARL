# Leeward Caravan

**AUTHOR_ONLY_NOT_EXECUTED.** Initial parentless train proposal 156, window 0, source attempt 0. No candidate import, JAX evaluation, reset/step call, numerical R0, trajectory replay, rendered animation or learning run occurred during authoring. AST/syntax and declarative data-schema checks do not establish runtime correctness or game outcomes.

Native boundary: A=3, O=159, U=15, H=128. Observations are float32 `[A,O]`, actions int32 `[A]`, masks bool `[A,U]`, rewards float32 `[A]`, and every state leaf is fixed-shape int32 or bool. Source SHA-256: `54c4ac626a7aeaf416e5b2d2ae88649299ecaa06d9b66f37102c8d076a368c75`.

## Game and cross-agent mechanisms

Deliver one heavy load across a ridge using synchronized hauling, movable wind shelter, finite water and interchangeable transport/support roles.

The shared load has north and south handles. Any pair can attach, release, rest, drink or swap duties with the remaining traveler. Hauling translates both carriers and the load as one rigid body; matching commands alone are insufficient if the handle geometry, energy or whole-body route is wrong. The central ridge blocks a straight traverse, giving upper and lower routes with different opportunities to obtain water or position shelter.

The movable windbreak occupies a real blocking cell. Its shadow extends one to three cells along the current wind vector and one cell to either side. A successful shift happens before movement cost and resting recovery, so another traveler's same-step action can become cheaper because of that shift. Only one shift succeeds per tick under rotating actor priority, and its target cannot overlap a traveler or load. A traveler occupying a tempting shaded spot may therefore prevent the support traveler from repositioning the shelter.

Hauling costs two base energy and walking one. Outside shade, movement adds the wind-strength surcharge of two; moving against wind adds one more. A move attempt that cannot pay or cannot fit does not spend movement energy. Rest restores two only inside the updated shadow, while drinking always consumes one water and restores at most four. Waiting for a wind phase, setting up a sheltered relay, spending water to rush an exposed segment, or switching the tired carrier into a support role are prospective strategies with partner-dependent effects.

The intended exact water identity is `sum(water) + sum(well_stock) + water_used = 14`: six units start in bags and eight in wells. Giving and drawing transfer units; drinking consumes them into the accounting counter. Energy remains in 0..12 and bags in 0..3. Wind changes at eight-tick boundaries after the step; costs use the wind shown at the action's input state. Post-step shadow flags describe the next state's actual wind and shelter position.

No exhaustion state is declared impossible: another traveler may bring shade or water, and an eventual wind rotation may help. Only arrival is natural completion; the horizon handles unresolved situations. Reset is deterministic and ignores its key. Future variants could vary wind phases, shadow geometry, ridge layout or handle orientation; none is claimed as evaluated.

## Actions, contention and temporal order

- `0`: Rest; gain two energy only inside current windbreak shadow.
- `1`: Move north or request synchronized northward haul.
- `2`: Move east or request synchronized eastward haul.
- `3`: Move south or request synchronized southward haul.
- `4`: Move west or request synchronized westward haul.
- `5`: Attach to a north/south load handle, or release.
- `6`: Shift adjacent windbreak north, spending two energy.
- `7`: Shift adjacent windbreak east, spending two energy.
- `8`: Shift adjacent windbreak south, spending two energy.
- `9`: Shift adjacent windbreak west, spending two energy.
- `10`: Drink one water to restore up to four energy.
- `11`: Give one water to adjacent agent 0.
- `12`: Give one water to adjacent agent 1.
- `13`: Give one water to adjacent agent 2.
- `14`: Fill one water from nearest adjacent finite well.

Out-of-range or masked actions deterministically become action 0. A legal attempt can be unfulfilled when its resolution-phase spatial, resource or joint-action prerequisites are absent. Ordinary movement rejects another agent's start-of-tick cell, so swaps are blocked. Empty-cell contention uses the smallest `(agent_id - t) mod A`. Rigid towing/hauling has its own explicit whole-body translation phase; its participating body can use its own vacated cells, and free agents remain blocked by the old and new body. All priorities and rule-specific ordering are literal in the source evidence below.

## Native observations and truthful rendering

Fully observable terrain, positions, resources, grip states, wind, shadow flags, wells and event counters. The renderer displays the same fixed state variables; own id distinguishes policy roles.

- Columns `0:1` (self): Own traveler id.
- Columns `1:7` (public): Clock,joint done,total water drunk,haul event,windbreak-move event,load delivered.
- Columns `7:25` (public): Three traveler records: x,y,energy,water,grip,sheltered.
- Columns `25:29` (public): Load x,y,number of grips,shadow flag.
- Columns `29:34` (public): Windbreak x,y,current wind dx,dy,exposure surcharge.
- Columns `34:40` (public): Two well records: x,y,remaining water.
- Columns `40:42` (public): Destination x,y.
- Columns `42:159` (public): Static 9 by 13 public map in yx row-major order.

Positions are actual integer `[x,y]` coordinates with downward y; maps are indexed `[y,x]`. `scene.json` binds exact state paths for map, entities, visibility, attributes and events. Public event information is included in the flat observations. A per-step event is not relabeled as a cumulative counter or persistent level. Hidden entity fields are masked, and a full-state debug view must be labeled separately from role-visible replay. Scene descriptors contain no fabricated trajectory or executable rendering content. A real animation still requires an actual trusted replay.

## Termination, numerical bounds and intended invariants

The load center reaching (10,4) terminates naturally. Energy or water exhaustion alone does not fabricate a failure: waiting, shade and partner support remain possible; unresolved play truncates at 128.

Nonterminal calls advance the int32 clock exactly once. Natural completion or failure has priority over the horizon. An already terminal raw call preserves every state leaf, observation and clock, returns zero rewards and all dones true, and retains the original terminated/truncated causes. No raw auto-reset occurs.

Observation bounds are -1..128 and reward bounds -0.01..1. These are data/game bounds, not scientific effectiveness gates. Source-specific conservation, geometry and resource properties above are declared invariants for independent execution checks, not numerically verified claims. The manifest lists every leaf and safe structural invariants. Signed integer categories and coordinates are explicitly int32; the external uint32 reset key is not duplicated as hidden mutable state.

## Literal source evidence

- Spatial downwind shadow: `candidate.py:12` — `def shade(self, positions, shelter, wind):`
- Deterministic weather phase: `candidate.py:18` — `def refresh(self, s):`
- Water, ridge and initial handle layout: `candidate.py:27` — `def reset(self, key):`
- Energy-based attempt masks: `candidate.py:44` — `def legal_actions(self, s):`
- Other actors costs change after a shift: `candidate.py:64` — `# One physical windbreak shift per tick; it changes other actors costs immediately.`
- Water conservation: `candidate.py:78` — `# Water transfers and drinking use real finite units, with no automatic replenishment.`
- Same-step shade-dependent costs: `candidate.py:93` — `covered = self.shade(s["agent_pos"],s["shelter_pos"][0],state["wind_dir"][0])`
- Interchangeable synchronized hauling: `candidate.py:97` — `# Any pair can haul, but must occupy opposite north/south handles and choose one direction.`
- Whole-body collision and free movement: `candidate.py:113` — `# Unattached travelers cannot walk through the load, windbreak or another traveler's old cell.`
- Natural arrival and raw terminal: `candidate.py:124` — `s["delivered"] = jnp.all(s["load_pos"][0] == s["goal_pos"][0])`
- All policy-visible state: `candidate.py:135` — `def observe(self, s):`

## Unexecuted playthroughs and interventions

`playthrough.json` supplies finite literal joint-action scripts and paired branches differing in exactly one role action at one decision. Every script and pair is `UNEXECUTED_AUTHOR_HYPOTHESIS`. Vector dimensions, integer action ranges and single-action differences were checked as data. Runtime legality, actual state changes, solvability, strategy quality and terminal success remain unverified. A future executor must retain failures or inconclusive traces rather than converting an intention into a success receipt.

## Reference use, interface handoff and provenance

Only the supplied official snapshots were consulted. JaxMARL supplies the agent-keyed parallel API convention and `__all__`; the restricted kernel keeps a homogeneous array boundary for later trusted integration. Pgx supplies explicit legality/terminal/key API ideas, not any board-game rules. Craftax supplies a reference for spatial inventories and persistent game state. PureJaxRL is an algorithm implementation reference, not a new environment library; the supplied Craftax recurrent PPO baseline itself credits PureJaxRL. None of their environment, model, training or wrapper code was copied, imported or run.

Frozen sources: [JaxMARL README](https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md), [Pgx README](https://github.com/sotetsuk/pgx/blob/main/README.md), [Craftax README](https://github.com/MichaelTMatthews/Craftax/blob/main/README.md), [PureJaxRL README](https://github.com/luchris429/purejaxrl/blob/main/README.md), [Craftax recurrent PPO baseline](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py). Their exact snapshot hashes are recorded in `author_input_freeze.json`.

A later trusted adapter may expose the fixed ids `agent_0` through `agent_2` as JaxMARL dictionaries, with `__all__` taken from `done_all` and unchanged observation/mask semantics. This author packet adds no learner or public adapter and does not modify the frozen experimental algorithm. Inactive-role no-op semantics, where relevant, are defined by the kernel rather than by renderer behavior.

`author_input_freeze.json` binds all twenty frozen inputs and the separately authorized administrative erratum. The erratum resolves a copied reservation description and expressly permits reading this batch authorization; it changes no design, rule or outcome. No earlier candidate files, GI/learned/CURRENT results, training/evaluation feedback or final-test material were read for this batch. Only the four assigned new output directories were written. Window accounting remains parentless ordinals 154..157, 158 used/reserved of the existing 512 cap, and unchanged later windows. This packet establishes authored assets only; runtime quality, replay, animation, learnability, transfer and scientific acceptance remain unestablished.
