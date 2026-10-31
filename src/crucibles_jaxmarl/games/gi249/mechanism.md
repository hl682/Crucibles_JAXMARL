# Skyline Lunch Lifts

Status: AUTHORED_NOT_EXECUTED. Initial parentless train candidate; feedback_used=0; source_attempt=0; final_test_used=false.

Native A/O/U/H = 20/196/20/640. Source SHA-256: `d43097c9332aa5d61174c002bd33b649d88c7994e0dae8a761c0b93d1a038e58`.

## Physical world and roles

Twenty independently controlled cooks occupy five four-person kitchens in a 15 by 25 building. Floor f occupies y=5*f..5*f+4. The final row of each floor is a wall except the x=0 staircase. Relative stations are pantry (1,1), board (3,1), oven (4,3), freezer (5,1), counter (11,1), sink (11,3). Lift columns x=6 and x=8 have landing y=5*f+2. Each has one real cabin with item, recipe and age; cooks themselves use the staircase. Broad squares allow cooks to stand together.

## Actions and timing

- 0 waits. 1..4 move within bounds and around floor walls. The resulting lift barrier can additionally prevent entry to its cabin square.
- 5 takes the current local recipe raw ingredient near a pantry, if empty-handed and stock remains. 6 chops raw food near a board. 7 deposits chopped food in an empty adjacent oven or collects a ready meal into an empty hand.
- 8 loads an empty nearby stationary lift and opens its barrier. 9 secures only if the actor differs from its last loader/unloader. 10/11 request travel one floor higher/lower with the secured barrier shut and circuit capacity at least two; a simultaneous freezer toggle can prevent the request.
- 12 unloads a nearby stationary lift into an empty hand and reopens its barrier. Food recipe and age travel with the physical item.
- 13 toggles a nearby freezer. Floors 0/1 share one circuit, 2/3 another, and 4 its own. Capacity is 3 minus enabled freezers in that circuit. Ovens advance by min(capacity,2); lift departures need capacity>=2.
- 14 serves matching cooked food near the counter, turning it into a dirty dish in the server hand. 15 washes a carried dish or reduces local cleanup backlog near a sink. Backlog is a service-cleanup level, not another physical dish object.
- 16 exchanges item, recipe and age when adjacent consenting actors form a reciprocal pair and exactly one hand is empty. 17 signals a cached recipe. 18 reads and carries the counter ticket when empty-handed or already carrying a ticket. 19 discards an item for a personal cost.

Masks use the pre-state. Each pantry, oven, counter or lift interaction accepts the smallest actor ID. Freezer toggles set this tick power. Ovens update, then lift loading/unloading/securing/departure resolves. A departing lift takes three later raw transitions to arrive; until then its actual origin coordinate and travel countdown remain visible. Movement uses the resulting barrier. Handoffs, service, cleaning, messages and aging follow. Food spoils at age 32 into a dirty dish; rotten oven food is removed. Successful service alternates that local ticket between recipes 1 and 2. Lift direction and occupancy cannot be independently invented by a renderer.

## Rewards and completion

Each cook pays -0.005 per accepted non-wait action, -0.3 when its carried food spoils, -0.05 for discard, and -0.002 while its affiliated kitchen freezer is on. A matching delivery pays all four affiliated cooks 0.2*(32-meal_age)/32, with nonnegative quality. Every one of the twenty cooks also receives +0.01 per counter served that step. This explicitly separates personal waste, affiliated power cost, kitchen service and shared customer benefit. Conservative per-step bounds are [-2,2]. Washing provides empty hands and cleanup without a repeatable shaping bonus.

Natural completion occurs when each of the five delivery counters reaches six; otherwise the shift ends at H=640. All twenty roles remain active until joint completion.

## Dependence and strategies

One schedule can batch secured freight trips and return dirty dishes; another sends stair runners to avoid contention at additional time and aging cost. Keeping neighboring freezers on preserves ingredients but can halt departures. Kitchens may help a rival secure a load, prioritize old meals or stagger power switches. These alternatives are encoded choices, not results from human or policy evaluation.

## Interface, boundaries and completion

The native boundary is reset(key), raw_step(key,state,actions), observe(state), and legal_actions(state). State is a fixed-key dictionary of fixed-shape int32/bool arrays. Observation and reward are float32. Every role has an action, observation, reward and done row. All out-of-range or masked actions deterministically become wait. A legal intention may still lose a simultaneous contest. Entity caps are fixed; no hidden controllers, callbacks, file access or global cache exists.

A nonterminal raw call advances t exactly once. Natural completion has priority when it coincides with H. `truncated=(t>=H)&~terminated`, `done_all=terminated|truncated`, and every individual done equals joint done. Already-terminal input is absorbing: preserve state, observation, clock and cause; return zero rewards and all dones true. Raw auto-reset never occurs. Event flags describe the last actual transition and remain unchanged under absorbing calls; a renderer must not replay them as new events.

## Exact observation and presentation contract

The exact self fields, finite nearest-entity slots, local map patch and public clock are the only flat observations. Entity slots sort by Manhattan distance then ID; all columns of a hidden slot are -1. Actor rendering also exposes the selected actor through its self block even if crowding removes it from nearest-eight slots. The static map layout is public. Forecast/knowledge leaves are private unless explicitly observed or messaged. Masks expose local contextual affordances. Event rendering must inherit its owning entity visibility and cannot display hidden global vectors. No renderer state is an extra policy observation.

Every non-padding column has its own exact slice and meaning in `manifest.json:observation_fields`. Entity slots encode `[visible, entity_id, absolute_x, absolute_y, declared attributes...]`; a hidden slot is entirely -1. Sorting is ascending Manhattan distance, then entity ID. No value is normalized. Observations are float32 and conservatively bounded [-1,2048]. Explicit padding is zero. Each group visibility mask equals the selected local slots; actor masks additionally include the selected self, whose full attributes are in its self block.

- self: [0, 12] end-exclusive; id, team, x, y, item, recipe, age, knowledge, message, event, recipient, spoil.
- cooks: [12, 108] end-exclusive; visible, entity_id, x, y, team, item, recipe, age, message, event, recipient, spoil.
- kitchens: [108, 144] end-exclusive; visible, entity_id, x, y, freezer, pot_recipe, pot_time, pot_age, order, stock, delivered, dirty.
- lifts: [144, 168] end-exclusive; visible, entity_id, x, y, lift_item, lift_recipe, lift_age, barrier, secured, loader, target, travel.
- map_patch: [168, 193] end-exclusive; 5x5 row-major dy=-2..2, dx=-2..2.

The scene is safe declarative data with exact source and manifest hashes. Actual state coordinates, active masks, local visibility, resource levels, carried items and events determine drawing. Private forecasts are not exposed by a global scene HUD. Event vectors inherit their owning entity mask; full-state debugging is a separate labeled view. Static map layout is public. No narrative text substitutes for a transition. Every individual ID and affiliation is listed in the manifest.

## Human control and unexecuted scripts

A presentation layer should distinguish Enter game from Watch replay, offer native legal actions and pause/step, focus any actor/affiliation, and stage an arbitrary joint-action vector. A human can control one or many native rows; human participant count is not the role count. Human and policy decisions must call the same raw kernel. These files are interface metadata, not a claim that a browser UI, native animation or human session has been run.

- `two_cook_freight_handoff`: 9 literal joint vectors. Cook 2 loads lift 0, cook 1 secures, cook 2 dispatches to floor 1, cook 6 unloads, cook 5 secures the empty cab.
- `prepare_serve_and_wash`: 18 literal joint vectors. Cook 0 chops and deposits initial order-matched food; cook 1 collects, carries and serves it, then washes the resulting dish.
- Paired `different_cook_secures_load`: 1 shared prefix vectors; 2 per suffix; exactly one scalar action differs. Securing by cook 1 should close the barrier and enable cook 2 departure affordance. Waiting leaves the same load unsecured. Compare state and actor-2 mask without forcing an illegal departure; unexecuted.

The scripts are source-reasoned hypotheses, not runtime legality receipts, success trajectories, solvability certificates or effect estimates. Identical reset and step keys are required for paired replay.

## Literal source evidence

- `candidate.py:30`: `def reset(self, key):`; definition ends at line 68.
- `candidate.py:83`: `def legal_actions(self, s):`; definition ends at line 101.
- `candidate.py:103`: `def advance(self, key, s, actions):`; definition ends at line 212.
- `candidate.py:214`: `def raw_step(self, key, state, actions):`; definition ends at line 217.
- `candidate.py:225`: `def observe(self, s):`; definition ends at line 235.
- `candidate.py:108`: `# Each station or lift accepts at most one actor, lowest ID first.`.
- `candidate.py:135`: `# Loading opens a physical barrier. A different actor must secure the crate.`.
- `candidate.py:166`: `# A reciprocal handoff transfers an item and its age, never duplicates it.`.
- `candidate.py:194`: `spoil = (age >= 32) & (item >= 1) & (item <= 3)`.
- `candidate.py:203`: `reward = -0.005 * (a != 0) - 0.3 * spoil - 0.05 * (a == 19) + 0.2 * quality[s['team']] * served[s['team']] + 0.01 * jnp.sum(served) - 0.002 * freezer[s['team']]`.

## Provenance, adaptations and limits

`author_input_freeze.json` records all 25 actual allowed input files with absolute path, relative name, byte length, SHA-256 and use. Tool output could be excerpted; the freeze does not claim unseen content was semantically read. Only current 248-251 request/design and population/family251 control. Removed historical194 was not searched. No earlier candidate, result, heldout, memory or other coordination input was used. The batch contains only four already reserved IDs within the user cumulative 300-game ceiling.

The supplied JaxMARL README informed parallel per-agent adapter expectations, including observation/action/reward/done dictionaries and `__all__`. This author supplies the native flat kernel; an external adapter is neither fabricated nor a change to the frozen learner. Pgx informed explicit masks and functional game-state presentation. Craftax informed the broad material/world-modification category. PureJaxRL and the supplied Craftax_Baselines PPO-RNN source were read only as integration/training context, not new game libraries, replacement learners or evidence of these games performance. No upstream code or commercial assets were copied. The four visual families remain independent.

Representation adaptations: small integers are losslessly int32; refresh explicitly normalizes integer leaves even if a later process enables JAX x64; bool leaves remain bool; observations/rewards explicitly cast float32; the supplied uint32 key is not duplicated in state. Static Python loops only traverse fixed field-name tuples. Intended JIT/vmap/scan compatibility requires later sealed execution.

No candidate/JAX/NumPy/PRNG, policy, training, throughput, browser or human-play execution occurred. Native/R0 acceptance, parity, resource use, deterministic replay, paired causal outcomes, rendered animation, solvability, strategic richness, learning and transfer remain UNKNOWN. Human-playability score is unmeasured. Negative, zero, failed and inconclusive outcomes must be preserved. This author produces no runtime/resource/release/native/PID/human receipt. After the final seven SHA-256 hashes are reported, this attempt is frozen; repairs require a separate dispatch.
