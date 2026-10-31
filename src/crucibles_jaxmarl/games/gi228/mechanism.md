# Skybridge Quarry Cooperative

Status: AUTHORED_NOT_EXECUTED. Candidate game-v3-00228, source attempt 0, train-only prospective source stream. Assigned author identity: gpt-6-astra / max; root dispatch is the authority for actual model execution. A/O/U/H = 160/154/17/768. Source SHA-256: fd2690b11faacbb506703a3951159b8c5b6d75fd4d1ef5c7ed59f3ef3ac97575.

## Native game and dependencies

Twenty crews occupy one connected 60 by 40 stone world. Crew c has eight actor rows; no actor is an NPC or a repeated independent episode. Terraces have bypass routes at their row boundaries, while local bridges and paired lifts provide shorter routes. A crew's depot can accept ore hauled by an actor from any crew. People, inventories, tools and damaged floor are persistent state.

There are twenty ore blocks, twenty supporting blocks and twenty designated bridges. A pick extracts one ore or one stone from the block directly ahead. Removing a support immediately removes its associated bridge. An active actor on that bridge loses one health and is left without floor. It cannot walk while ungrounded; another builder can spend stone to place a scaffold under it. Repairing floor does not heal health. At zero health an actor becomes inactive and can only wait. Dead actors retain inventory; stranded ore can therefore prevent resource-exhaustion completion until the horizon.

A scaffold can also be built on any facing gap. Designated unsupported bridges have strength 3 after repair and lose one strength on each 32-tick weather boundary. Other built gap cells persist. An original support can be mined once; placing scaffold on that support cell does not turn it back into a mineable support.

A lift physically moves its rider between the two neighboring banks at its y-coordinate. Its counterweight must cover one unit, or two if the rider carries any ore or stone. Those units leave the local lift and enter its paired lift, capped at 4. Pairing is by consecutive crew IDs, 0/1 through 18/19; it is a declared cable coupling, not a distance-derived pairing. Counterweight beyond capacity is discarded. A rigger can instead spend a stone to add one local ballast unit. Thus an extraction crew's route choice changes another crew's future lift access.

Tools are actual transferable inventory codes: pick, scaffold kit, rigger and courier. Any actor can move, give, ride, deliver or signal; tool codes gate mining, building and ballast work. Giving prioritizes ore, then stone. Initial role labels describe initial equipment rather than permanently fixing a player's job.

Plausible alternatives include leaving supports intact and taking longer bypass routes, accepting the cost of mining and rescue, coordinating alternating loaded lift trips, and reserving stone for either scaffolds or counterweights. These are source-level strategic alternatives. No ranking, equilibrium, solve rate or learning result has been established.

## Actions, observations and presentation

Actions 0 through 8 are wait, four cardinal moves and four facing-only turns. Actions 9 through 16 are mine, build, unload, give, lift ride, ballast, tool swap and signal. Conditions are in the native legal mask and rechecked during resolution. The complete labels are in manifest.json and scene.json.

The 154-column row has 14 self values, a 49-cell two-channel local patch, four eight-field neighboring-actor records, five lift fields and five depot fields. The manifest specifies every column and normalizer. Detailed actor visibility is self plus the four nearest active neighbors; the patch can still show other local occupied cells without their inventories. Entity metadata exposes only attributes represented by those detailed records. Remote paired counterweights are not observed. An inventory signal reaches Chebyshev distance 6 and the maximum current code is heard; it carries no secret global state.

The block palette binds the real tiles. Mined blocks disappear through their active masks; missing bridge floor, cargo, ballast, depots and actor positions bind native leaves. Health is in the selected actor's own observation, not an extra visible neighbor attribute. No commercial assets, renderer physics or browser implementation are included.

## Rewards and termination

reward_parts has five float32 columns, summed into the native reward vector. All receipts are per actual native tick.

- Column 0 charges the acting mover 0.01 times (1 + carried ore), and charges a miner, builder or ballast operator 0.02.
- Column 1 gives the actor 0.4 for unloading one ore.
- Column 2 gives each of the destination depot's eight crew members 0.125 per ore unloaded there. The carrier need not belong to that crew.
- Column 3 gives each rescued actor 0.2 and the builder 0.4 per actor whose missing floor it restores.
- Column 4 charges an actor losing bridge support 0.5, and the actor who mined that support 0.2 per affected actor. Weather collapse charges the occupant only.

The declared reward interval is [-256, 256]. A conservative analytic bound counts no more than 160 action resolutions, at most one unique active occupant per changed bridge tile, and one weather pass. The interval is a finite interface bound, not an achieved score or acceptance threshold. There are at most twenty ore deliveries in a world, so the depot normalization by twenty is bounded.

Natural completion is no remaining ore tile and no carried ore, or loss of all active actors. H=768 is truncation only if neither natural condition holds. The native done vector additionally marks inactive actors; all actor dones become true exactly at joint termination.

## Hypotheses and limits

playthrough.json supplies two complete literal joint-action scripts and one paired hypothesis. The rescue comparison shares a support-mining prefix, changes only actor 1's build versus wait, and uses identical waiting suffixes. Intended measurements include the actual bridge tile, recipient actor 2's grounded flag and legal moves; health recovery is expressly not predicted. The alternating-lift script addresses a different cross-crew dependency.

Static author review checked syntax, declared leaf names, JSON structure and literal action dimensions. It did not execute reset, raw_step, masks, scripts, JIT, vmap or scan. Causal effects, compatibility, throughput, human animation, solvability, learning and transfer remain unverified.

## Literal source evidence

These line references identify authored implementation expressions, not measured effects.

- Tool and spatial legality: candidate.py line 30, literal source: `def legal_row(self, s, i):`.
- Initial connected world and per-actor state: candidate.py line 81, literal source: `def reset(self, key):`.
- Literal flat observation construction: candidate.py line 122, literal source: `def observe(self, s):`.
- Support-to-bridge linkage: candidate.py line 179, literal source: `hit_support = jnp.all(z["support_pos"] == q, axis=1) & rock_gain`.
- Affected actor selection: candidate.py line 184, literal source: `victim = jnp.any(jnp.all(z["pos"][:, None, :] == z["bridge_pos"][None, :, :], axis=2) & hit_support[None, :], axis=1) & z["active"]`.
- Cross-crew counterweight transfer: candidate.py line 201, literal source: `z["weight"] = z["weight"].at[pair].set(jnp.minimum(4, z["weight"][pair] + ride.astype(jnp.int32) * load))`.
- Delayed scaffold weathering: candidate.py line 229, literal source: `weather = ((s["t"] + 1) % 32) == 0`.
- Absorbing native boundary: candidate.py line 248, literal source: `def raw_step(self, key, state, actions):`.

## Read-only reference use

Only the current batch's authorized files were read. author_input_freeze.json records all 26 actual input files, bytes, hashes and use. The 24 payload files include this batch's four current designs/requests, shared contracts, current population/family control, general design and human-play requirements, and five named official reference snapshots. Three optional historical population addenda were not read. Earlier copied example numbers are noncontrolling.

JaxMARL README informed the distinction between a simultaneous native action-row boundary and a later agent-keyed dictionary adapter. Pgx README informed explicit legal masks, separate termination causes and external key provenance; no Pgx game was copied. Craftax README informed bounded persistent world state and the separation between mechanics and rendering. PureJaxRL README and Craftax_Baselines_ppo_rnn.py were read only for JAX scan/vectorization and wrapper/algorithm separation. No upstream game code, PPO code, assets or performance claims were copied into this kernel. No reference links were opened, dependencies installed, local models invoked or scientific outcomes read.

## Boundary and evidence status

The native interface is reset(key), observe(state), legal_actions(state), and raw_step(key,state,actions). All state leaves are fixed-shape int32, bool or float32 arrays declared in manifest.json. Keys are accepted but intentionally unused: these original rules and layouts are deterministic. Small integer quantities use int32 at the boundary; there is no hidden PRNG key in state.

The joint input is one int32 action per actor. Resolution uses rotating priority (k + input t) modulo A. Each actor's action is rechecked against the current intermediate state, so contention can turn a previously legal intention into a deterministic wait. Environmental processes run once afterward. This is a simultaneous input interface with explicitly sequential conflict resolution, not a claim that actions commute.

A nonterminal step advances the int32 clock once. Natural completion takes priority over H. An already-terminal raw call preserves the entire state, observation, clock and terminal causes and returns zero reward; it never auto-resets. Event flags refer to the last actual nonabsorbing step and are not new events on repeated absorbing calls. A separate trusted wrapper or public adapter remains outside this author package.

No candidate was imported or executed. No JAX/NumPy runtime, validator, trajectory script, local model, remote job, HTTP endpoint, browser, training or evaluation was run. Standard-library AST parsing and data-only JSON/static consistency checks are not native validation. Source authoring is not an R0, causal, animation, solvability, learning or transfer receipt. No additional proposal or repair attempt was created.

