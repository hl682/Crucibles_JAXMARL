# Basalt Tide Salvage

Status: AUTHORED_NOT_EXECUTED. Initial parentless train candidate; feedback_used=0; source_attempt=0; final_test_used=false.

Native A/O/U/H = 512/272/16/1024. Source SHA-256: `a3f68f170bfd0a38bc5eff6e983461f9699121fa8b529c19b65d2ad9eb008e1b`.

## Physical world and roles

The 64 by 32 world has 32 adjoining 8 by 8 dock districts. District k starts at (8*(k mod 8),8*floor(k/8)). Relative locations are shelter (1,1), quarry (1,5), pump (3,4), brake stands (4,2)/(4,5), bridge x=5..7 at y=3, and diversion (5,5). District pairs 0/1,2/3 and so on are physically adjacent. Water affects the wet strip x>=5,y=2..5. All 512 people have individual rows; their 32 crews are reward affiliations, not merged controllers. Broad working squares allow multiple people. Ground crates may share a cell, but every carried crate has exactly one carrier.

## Actions and timing

- 0 waits; dry oxygen recovers even during wait. At oxygen zero only wait remains legal until rescue or dry recovery.
- 1..4 propose north/east/south/west within bounds. Entry onto a wet bridge strip at water>1 needs both current brake holds or three supports. A legal movement intent can therefore be blocked by another actor releasing a brake this tick.
- 5 requires tool 0 within distance one of a quarry and fewer than three blocks. 6 spends a block near an unfinished bridge support site (6,3) or unbuilt shelter (1,1). Simultaneous builders may spend redundant material at the cap; no refund is invented.
- 7 requires tool 1 near a pump, positive water and charge. At most one unit is pumped per dock per tick; multiple operators each still pay effort. 8 toggles a nearby diversion; an even number of toggles cancels.
- 9 requires tool 2 exactly on a brake stand. Both separate stands must be held each tick. 10 targets the lowest-index eligible adjacent ground crate; the smallest actor ID wins a contested claim. The collecting crew becomes owner. 11 drops a carried crate at the actual actor position.
- 12 is a reciprocal adjacent tool-swap offer. 13 rescues a nearby oxygen-zero person; the smallest-ID successful helper gets credit. 14 delivers near a shelter location even before its shelter is built; construction is a separate comfort milestone.
- 15 transmits cached tide knowledge to adjacent people. Near-shelter access updates private knowledge with the district next surge, which can become stale elsewhere.

Masks read the pre-step state. Brake holds and supports resolve before movement. Pumps and diversion valves then change water, with a one-unit ebb each sixteenth tick and stored forecast surge each thirty-second tick. The next forecast is drawn only from the supplied key. Crate claims, drops and deliveries use real positions. A submerged crate gains rust each eighth tick and is lost at rust eight; delivery wins a simultaneous rust loss. Tool exchange, rescue, oxygen and local knowledge then update. A current actor event code is 0..15 for the accepted action, 16 for completed tool swap, or 17 for the credited rescuer; it is not a generic success indicator.

## Rewards and completion

Actor i receives -0.005 per accepted non-wait action and -0.01 if submerged after the water update. Every member of its 16-person crew receives +0.125 per crate delivered for that crew, -0.125 per crate lost for that crew, and +0.05 for that crew first shelter. The successful selected rescuer receives +0.2; the rescued person receives +0.1. Every one of the 512 actors receives +0.001 per bridge first reaching three supports. Conservative per-step returned bounds are [-16,16], based on at most 64 crates and 32 bridges, not an experimental success threshold.

Natural joint completion occurs when every crate is delivered or rust-lost. Oxygen-zero people remain actual actors and can recover. The horizon is 1024.

## Dependence and strategies

A coalition can hold two brakes and salvage before the next surge, or quarry permanent supports to free future operators. Pumping protects a local hauler while imposing discharge on a neighbor; that neighbor can divert it for charge, accept it, or decline. Claiming an abandoned crate changes crew ownership. Rescue and tool loans can create temporary cooperation across rival crews. These are source-level incentive hypotheses; no equilibrium, learning or human fun has been established.

## Interface, boundaries and completion

The native boundary is reset(key), raw_step(key,state,actions), observe(state), and legal_actions(state). State is a fixed-key dictionary of fixed-shape int32/bool arrays. Observation and reward are float32. Every role has an action, observation, reward and done row. All out-of-range or masked actions deterministically become wait. A legal intention may still lose a simultaneous contest. Entity caps are fixed; no hidden controllers, callbacks, file access or global cache exists.

A nonterminal raw call advances t exactly once. Natural completion has priority when it coincides with H. `truncated=(t>=H)&~terminated`, `done_all=terminated|truncated`, and every individual done equals joint done. Already-terminal input is absorbing: preserve state, observation, clock and cause; return zero rewards and all dones true. Raw auto-reset never occurs. Event flags describe the last actual transition and remain unchanged under absorbing calls; a renderer must not replay them as new events.

## Exact observation and presentation contract

The exact self fields, finite nearest-entity slots, local map patch and public clock are the only flat observations. Entity slots sort by Manhattan distance then ID; all columns of a hidden slot are -1. Actor rendering also exposes the selected actor through its self block even if crowding removes it from nearest-eight slots. The static map layout is public. Forecast/knowledge leaves are private unless explicitly observed or messaged. Masks expose local contextual affordances. Event rendering must inherit its owning entity visibility and cannot display hidden global vectors. No renderer state is an extra policy observation.

Every non-padding column has its own exact slice and meaning in `manifest.json:observation_fields`. Entity slots encode `[visible, entity_id, absolute_x, absolute_y, declared attributes...]`; a hidden slot is entirely -1. Sorting is ascending Manhattan distance, then entity ID. No value is normalized. Observations are float32 and conservatively bounded [-1,2048]. Explicit padding is zero. Each group visibility mask equals the selected local slots; actor masks additionally include the selected self, whose full attributes are in its self block.

- self: [0, 12] end-exclusive; id, crew, tool, x, y, oxygen, blocks, carry, knowledge, warning, event, rescued.
- actors: [12, 108] end-exclusive; visible, entity_id, x, y, crew, tool, oxygen, blocks, carry, warning, event, rescued.
- docks: [108, 160] end-exclusive; visible, entity_id, x, y, water, support, shelter, valve, charge, bridge, delivered, pumped, flood_loss.
- cargo: [160, 188] end-exclusive; visible, entity_id, x, y, carrier, owner, rust.
- brakes: [188, 208] end-exclusive; visible, entity_id, x, y, held.
- map_patch: [208, 233] end-exclusive; 5x5 row-major dy=-2..2, dx=-2..2.

The scene is safe declarative data with exact source and manifest hashes. Actual state coordinates, active masks, local visibility, resource levels, carried items and events determine drawing. Private forecasts are not exposed by a global scene HUD. Event vectors inherit their owning entity mask; full-state debugging is a separate labeled view. Static map layout is public. No narrative text substitutes for a transition. Every individual ID and affiliation is listed in the manifest.

## Human control and unexecuted scripts

A presentation layer should distinguish Enter game from Watch replay, offer native legal actions and pause/step, focus any actor/affiliation, and stage an arbitrary joint-action vector. A human can control one or many native rows; human participant count is not the role count. Human and policy decisions must call the same raw kernel. These files are interface metadata, not a claim that a browser UI, native animation or human session has been run.

- `held_bridge_salvage`: 8 literal joint vectors. Actors 0 and 1 hold their brakes while actor 3 claims crate 0, walks to dry shelter access and delivers it.
- `divert_flow_and_build_shelter`: 5 literal joint vectors. Actor 2 pumps twice; adjacent crew actor 23 diverts incoming flow; actor 4 quarries and builds shelter.
- Paired `release_one_distant_brake`: 1 shared prefix vectors; 2 per suffix; exactly one scalar action differs. Keeping both brakes should permit actor 3 bridge entry at initial water 2; releasing actor 0 should block the same westward attempt. Compare actual coordinates, observation and bridge state; unexecuted.

The scripts are source-reasoned hypotheses, not runtime legality receipts, success trajectories, solvability certificates or effect estimates. Identical reset and step keys are required for paired replay.

## Literal source evidence

- `candidate.py:33`: `def reset(self, key):`; definition ends at line 70.
- `candidate.py:78`: `def legal_actions(self, s):`; definition ends at line 95.
- `candidate.py:97`: `def advance(self, key, s, actions):`; definition ends at line 185.
- `candidate.py:187`: `def raw_step(self, key, state, actions):`; definition ends at line 190.
- `candidate.py:198`: `def observe(self, s):`; definition ends at line 210.
- `candidate.py:104`: `# HOLD precedes movement: both distant brakes are needed every tick.`.
- `candidate.py:118`: `# A pump spends local charge and pushes actual water into the paired adjacent dock.`.
- `candidate.py:128`: `# Cargo arbitration uses the smallest actor ID; one physical crate per carrier.`.
- `candidate.py:148`: `# Reciprocal neighboring lenders swap tools; rescue pays the helper and recipient.`.
- `candidate.py:173`: `reward = -0.005 * (a != 0) - 0.01 * submerged + 0.125 * gains[s['crew']] - 0.125 * losses[s['crew']] + 0.05 * shelter_new[s['crew']] + 0.2 * helper_paid + 0.1 * rescued`.

## Provenance, adaptations and limits

`author_input_freeze.json` records all 25 actual allowed input files with absolute path, relative name, byte length, SHA-256 and use. Tool output could be excerpted; the freeze does not claim unseen content was semantically read. Only current 248-251 request/design and population/family251 control. Removed historical194 was not searched. No earlier candidate, result, heldout, memory or other coordination input was used. The batch contains only four already reserved IDs within the user cumulative 300-game ceiling.

The supplied JaxMARL README informed parallel per-agent adapter expectations, including observation/action/reward/done dictionaries and `__all__`. This author supplies the native flat kernel; an external adapter is neither fabricated nor a change to the frozen learner. Pgx informed explicit masks and functional game-state presentation. Craftax informed the broad material/world-modification category. PureJaxRL and the supplied Craftax_Baselines PPO-RNN source were read only as integration/training context, not new game libraries, replacement learners or evidence of these games performance. No upstream code or commercial assets were copied. The four visual families remain independent.

Representation adaptations: small integers are losslessly int32; refresh explicitly normalizes integer leaves even if a later process enables JAX x64; bool leaves remain bool; observations/rewards explicitly cast float32; the supplied uint32 key is not duplicated in state. Static Python loops only traverse fixed field-name tuples. Intended JIT/vmap/scan compatibility requires later sealed execution.

No candidate/JAX/NumPy/PRNG, policy, training, throughput, browser or human-play execution occurred. Native/R0 acceptance, parity, resource use, deterministic replay, paired causal outcomes, rendered animation, solvability, strategic richness, learning and transfer remain UNKNOWN. Human-playability score is unmeasured. Negative, zero, failed and inconclusive outcomes must be preserved. This author produces no runtime/resource/release/native/PID/human receipt. After the final seven SHA-256 hashes are reported, this attempt is frozen; repairs require a separate dispatch.
