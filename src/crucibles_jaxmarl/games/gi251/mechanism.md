# Orchard Lantern Commons

Status: AUTHORED_NOT_EXECUTED. Initial parentless train candidate; feedback_used=0; source_attempt=0; final_test_used=false.

Native A/O/U/H = 96/297/20/896. Source SHA-256: `0009f3127bcf22d1817749f1b520c0c3cb48e0ae37e4c1107d433518ebc2e595`.

## Physical world and roles

Twelve households occupy adjacent 12 by 8 districts in a 48 by 24 orchard. Each of the 96 villagers is a native independently controlled role. Hives are ecological objects, not additional agents. Relative orchard=(4,3), irrigation=(3,2), starting hive=(6,3), rack=(3,5), bridge=(7,5), cold-store door=(9,5), lamp=(9,2), market=(9,1). Bridge/door stands are one square north of their fixture. Plot, rack and fixture IDs encode fixed household ownership. Hives may physically relocate and serve another household. Adjacent district pairs 0/1,2/3 and so on exchange water and shade effects. People can share broad path squares.

## Actions and timing

- 0 waits; 1..4 move within bounds. A tree cell with positive canopy blocks entry; pruning to zero opens it immediately.
- 5 requires shears (tool 0), tree adjacency, canopy and timber capacity; it removes one canopy and one available fruit and gains timber. 6 spends timber to tend an adjacent damaged tree. 7 harvests ripe fruit into a basket of capacity three with no carried hive.
- 8 requires a graft knife (tool 1), an ungrafted nearby tree and one basket fruit. It changes variety and resets ripeness. 9 toggles the nearby irrigation switch; an open channel sends a water unit to the paired neighbor every fourth step if available.
- 10 with tool 2 and empty basket lifts a nearby free hive, or drops the currently carried one. A carried hive follows the real person position and cannot pollinate.
- 11/13 requests east/west rack transport with tool 3. One operator must be directly north and one directly south, aligned with rack x; completed opposite coalitions cancel. Operators must walk alongside between pulls. 12 holds a bridge brace or cold door only from its exact stand with tool 3.
- 14 loads nearby rack fruit up to capacity three. 15 spends timber to repair a nearby footbridge; health three removes its need for a brace. 16 offers a reciprocal adjacent tool exchange. 17 toggles a nearby night lamp.
- 18 sells a basket at the market, or an adjacent rack already through the cold-store entrance while its door is held. 19 reads the local market weather board or shares cached information locally.

Masks use pre-state facts. Plot operations choose smallest actor ID for each action; simultaneous pruning can remove a remaining fruit in addition to a harvest. Irrigation, rain and repairs update material state. Braces and lamps resolve before movement/ecology. Each uncarried, undistracted hive serves its nearest orchard within six grid steps; relocation reallocates that one contribution. Nearby lit lamps distract hives during both day and night. Every eighth tick, water, health, pollen and sunlight from the paired neighbor canopy jointly permit fruit growth/ripening. Every thirty-second tick canopy regrows and forecast rain may damage trees and bridges. Racks then use two carriers plus brace/door conditions. At night (step mod 32 >=16), moving a loaded unlit rack reduces quality. A loaded zero-quality rack loses one fruit each step until empty. Sales use the same pre-step total-supply price, then reciprocal tool loans/messages and visibility update.

## Rewards and completion

Every person pays -0.003 per accepted non-wait action. Each household member receives price*quality_weighted_units_sold_for_household/24, where price=1/(1+sum(previous household sales)/24). Basket sales credit the seller household; rack sales credit the rack household. Each member pays -0.1 per broken fruit from its household rack. Every one of the 96 people receives a one-time +0.01 for full current orchard pollen coverage. Initial hives already provide coverage, so this milestone may happen on the first transition; it is not a demonstrated restoration achievement. Price depends on total supply, which is not an extra privileged observation. Conservative per-step returned bounds are [-2,8], and each household ledger caps at 512.

Joint natural completion occurs when each household has sold eight fruit units, otherwise H=896 ends the episode. No household is represented by a merged agent row; ecological hives never submit actions.

## Dependence and strategies

Small baskets avoid rack staffing, while racks transport shared harvest with two carriers and a third bridge/door helper. A household can invest timber in a permanent bridge, prune for a neighbor next growth interval, relocate a hive toward a partner or away from a rival, or light a night route at a pollen cost. Weather knowledge remains local and can become stale. Grafting sacrifices immediate fruit for later ripening and quality. These are actual rule tradeoffs; human enjoyment, strategic richness and learnability remain untested.

## Interface, boundaries and completion

The native boundary is reset(key), raw_step(key,state,actions), observe(state), and legal_actions(state). State is a fixed-key dictionary of fixed-shape int32/bool arrays. Observation and reward are float32. Every role has an action, observation, reward and done row. All out-of-range or masked actions deterministically become wait. A legal intention may still lose a simultaneous contest. Entity caps are fixed; no hidden controllers, callbacks, file access or global cache exists.

A nonterminal raw call advances t exactly once. Natural completion has priority when it coincides with H. `truncated=(t>=H)&~terminated`, `done_all=terminated|truncated`, and every individual done equals joint done. Already-terminal input is absorbing: preserve state, observation, clock and cause; return zero rewards and all dones true. Raw auto-reset never occurs. Event flags describe the last actual transition and remain unchanged under absorbing calls; a renderer must not replay them as new events.

## Exact observation and presentation contract

The exact self fields, finite nearest-entity slots, local map patch and public clock are the only flat observations. Entity slots sort by Manhattan distance then ID; all columns of a hidden slot are -1. Actor rendering also exposes the selected actor through its self block even if crowding removes it from nearest-eight slots. The static map layout is public. Forecast/knowledge leaves are private unless explicitly observed or messaged. Masks expose local contextual affordances. Event rendering must inherit its owning entity visibility and cannot display hidden global vectors. No renderer state is an extra policy observation.

Every non-padding column has its own exact slice and meaning in `manifest.json:observation_fields`. Entity slots encode `[visible, entity_id, absolute_x, absolute_y, declared attributes...]`; a hidden slot is entirely -1. Sorting is ascending Manhattan distance, then entity ID. No value is normalized. Observations are float32 and conservatively bounded [-1,2048]. Explicit padding is zero. Each group visibility mask equals the selected local slots; actor masks additionally include the selected self, whose full attributes are in its self block.

- self: [0, 14] end-exclusive; id, house, tool, x, y, wood, basket, quality, hive, knowledge, message, event, recipient, own_sales.
- villagers: [14, 118] end-exclusive; visible, entity_id, x, y, house, tool, wood, basket, quality, hive, message, event, recipient.
- orchards: [118, 178] end-exclusive; visible, entity_id, x, y, water, canopy, fruit, ripe, variety, pollination, health, valve, sales, growth, pruned.
- hives: [178, 199] end-exclusive; visible, entity_id, x, y, hive_carrier, flower_target, distracted.
- racks: [199, 223] end-exclusive; visible, entity_id, x, y, rack_load, rack_quality, rack_moved, breakage.
- fixtures: [223, 263] end-exclusive; visible, entity_id, x, y, kind, fixture_health, held, lit.
- map_patch: [263, 288] end-exclusive; 5x5 row-major dy=-2..2, dx=-2..2.

The scene is safe declarative data with exact source and manifest hashes. Actual state coordinates, active masks, local visibility, resource levels, carried items and events determine drawing. Private forecasts are not exposed by a global scene HUD. Event vectors inherit their owning entity mask; full-state debugging is a separate labeled view. Static map layout is public. No narrative text substitutes for a transition. Every individual ID and affiliation is listed in the manifest.

## Human control and unexecuted scripts

A presentation layer should distinguish Enter game from Watch replay, offer native legal actions and pause/step, focus any actor/affiliation, and stage an arbitrary joint-action vector. A human can control one or many native rows; human participant count is not the role count. Human and policy decisions must call the same raw kernel. These files are interface metadata, not a claim that a browser UI, native animation or human session has been run.

- `two_carriers_bridge_cold_store`: 13 literal joint vectors. Actors 0 and 1 carry a rack, actor 2 braces the bridge, actor 7 walks to and holds the cold-store door for entry and sale.
- `prune_graft_relocate_hive`: 10 literal joint vectors. Actor 4 prunes, actor 5 harvests and grafts, actor 6 relocates a hive toward household 1, actor 7 toggles the ecological lamp tradeoff.
- Paired `third_person_braces_crossing`: 6 shared prefix vectors; 3 per suffix; exactly one scalar action differs. The same two carriers request unrepaired bridge entry. A third-person brace should allow entry and waiting should block it; common waits expose persistence. Unexecuted.

The scripts are source-reasoned hypotheses, not runtime legality receipts, success trajectories, solvability certificates or effect estimates. Identical reset and step keys are required for paired replay.

## Literal source evidence

- `candidate.py:32`: `def reset(self, key):`; definition ends at line 72.
- `candidate.py:83`: `def legal_actions(self, s):`; definition ends at line 105.
- `candidate.py:107`: `def advance(self, key, s, actions):`; definition ends at line 233.
- `candidate.py:235`: `def raw_step(self, key, state, actions):`; definition ends at line 238.
- `candidate.py:246`: `def observe(self, s):`; definition ends at line 259.
- `candidate.py:134`: `# Bracing and lamp switches are actual per-fixture state, before transport/ecology.`.
- `candidate.py:143`: `# Pruning physically opens a tree cell; movements use the new canopy.`.
- `candidate.py:158`: `distracted = jnp.any((self.distance(hpos, s['fixture_pos']) <= 6) & lit[None, :], axis=1)`.
- `candidate.py:164`: `sunlight = 3 - canopy[neighbor]`.
- `candidate.py:169`: `# A loaded rack needs a north and a south carrier agreeing on direction.`.
- `candidate.py:195`: `# Market competition prices every sale using the same pre-step total supply.`.
- `candidate.py:223`: `reward = -0.003 * (a != 0) + price * values[s['house']] / 24.0 - 0.1 * breakage[s['house']] + 0.01 * milestone`.

## Provenance, adaptations and limits

`author_input_freeze.json` records all 25 actual allowed input files with absolute path, relative name, byte length, SHA-256 and use. Tool output could be excerpted; the freeze does not claim unseen content was semantically read. Only current 248-251 request/design and population/family251 control. Removed historical194 was not searched. No earlier candidate, result, heldout, memory or other coordination input was used. The batch contains only four already reserved IDs within the user cumulative 300-game ceiling.

The supplied JaxMARL README informed parallel per-agent adapter expectations, including observation/action/reward/done dictionaries and `__all__`. This author supplies the native flat kernel; an external adapter is neither fabricated nor a change to the frozen learner. Pgx informed explicit masks and functional game-state presentation. Craftax informed the broad material/world-modification category. PureJaxRL and the supplied Craftax_Baselines PPO-RNN source were read only as integration/training context, not new game libraries, replacement learners or evidence of these games performance. No upstream code or commercial assets were copied. The four visual families remain independent.

Representation adaptations: small integers are losslessly int32; refresh explicitly normalizes integer leaves even if a later process enables JAX x64; bool leaves remain bool; observations/rewards explicitly cast float32; the supplied uint32 key is not duplicated in state. Static Python loops only traverse fixed field-name tuples. Intended JIT/vmap/scan compatibility requires later sealed execution.

No candidate/JAX/NumPy/PRNG, policy, training, throughput, browser or human-play execution occurred. Native/R0 acceptance, parity, resource use, deterministic replay, paired causal outcomes, rendered animation, solvability, strategic richness, learning and transfer remain UNKNOWN. Human-playability score is unmeasured. Negative, zero, failed and inconclusive outcomes must be preserved. This author produces no runtime/resource/release/native/PID/human receipt. After the final seven SHA-256 hashes are reported, this attempt is frozen; repairs require a separate dispatch.
