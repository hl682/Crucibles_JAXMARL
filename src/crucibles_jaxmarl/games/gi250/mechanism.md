# Canal Seal Commonwealth

Status: AUTHORED_NOT_EXECUTED. Initial parentless train candidate; feedback_used=0; source_attempt=0; final_test_used=false.

Native A/O/U/H = 160/242/20/896. Source SHA-256: `1e668db05956eb61817501a670bf4a4c126d378ad8dfb879c6d4a0dac5361472`.

## Physical world and roles

Ten adjoining eight-column guild territories share an 80 by 16 canal basin. All 160 citizens have independent observation/action/reward/done rows. Barges travel on the actual horizontal canal y=7. Citizen movement uses maintenance walkways represented by every bounded grid square, including canal-edge grating. Gate k is at (8*k+4,7); its separate inspection stand is (8*k+4,9). Relative council=(1,2), timber yard=(1,11), granary frontage=(1,5). Barge k starts at (8*k+1,7), carries three grain, and targets x=min(8*k+9,78). Citizens can share squares; an active barge cannot enter another active barge position, and conflicting target cells favor lower barge ID.

## Actions and timing

- 0 waits; 1..4 move within bounds.
- 5 gathers timber near a yard up to three. 6 requests opening near a healthy lock with tool 0; the request needs same-tick separate inspection. 7 requires tool 3 exactly on the stand and checks the next lock pre-state was closed. The final outlet counts as safely closed. 8 closes a nearby lock and wins a simultaneous open request.
- 9 issues the actor own guild seal at a council if not already carrying one. 10 posts a carried foreign seal at an empty adjacent gate. 11 withdraws a seal with an empty hand if the actor guild owns that gate or that seal. Seals are actual carried or posted permissions.
- 12/13 tows east/west with tool 2. One operator must align at barge x on y=6 and another on y=8. Complete opposite-direction coalitions cancel. Carriers stay at their actual positions and must walk alongside before pulling again.
- 14 delivers adjacent cargo only after its barge reaches the destination. 15 spends timber to repair a nearby damaged lock, otherwise a nearby active damaged hull. The helper receives a small reward; the owner gets the material benefit.
- 16 offers a reciprocal adjacent tool exchange. 17 reads the local council forecast. 18 spends timber to construct a three-unit permanent bypass near a gate. 19 broadcasts cached forecast knowledge locally.

Pre-state masks precede inspection and gate decisions. Every fourth step, open wet gates release one headwater unit to the next guild; the final outlet leaves the basin. Every thirty-second step stored forecast inflows arrive. Posting sets 24 lifetime ticks and two reciprocal uses. Both signatories must have live reciprocal seals physically posted to obtain exemption. Withdrawals, expiry or exhausted uses remove that permission. Repairs and bypass building precede two-bank towing. Entering a foreign gate cell charges one coin unless a bypass or reciprocal rights apply. Coin debit and receipt name actual guilds; free reciprocal entry consumes a use at each signatory gate. Closed gates at water 2..3 feed mills every sixteenth tick. Water four causes granary and structural/hull loss every eighth tick. Delivery has priority over simultaneous hull destruction. Gate health zero prevents a new opening request; an already open gate stays latched until closed. The next local weather forecast comes only from the supplied key.

## Rewards and completion

Each citizen pays -0.003 per accepted non-wait action. Its guild pays/receives per member: +0.25 for its barge delivery, -0.2 for barge destruction, +0.02 for mill production, -0.1 per lost granary unit, and +0.01*(toll receipts minus toll expenditures). Each legal repair helper receives +0.02. Each new completed bypass pays all 160 citizens +0.001. Toll treasury transfers explicitly debit the barge-owner guild and credit the gate-owner guild; their reward terms use those same affiliations. Simultaneous repairs/builds may spend redundant timber at the cap. Conservative per-step bounds are [-2,2].

Joint natural completion occurs when every initial barge is delivered or destroyed. Those barges become inactive; all citizens remain controllable until joint completion. Otherwise H=896 ends the episode.

## Dependence and strategies

A keeper can retain mill headwater, release it for passage, or export flood pressure to a downstream guild. Couriers can exchange short-lived reciprocal seals; another coalition may pay tolls or invest in bypasses. Towing needs continual two-person spatial alignment, and a third inspector enables a gatekeeper. Locally learned forecasts can improve timing while remaining stale and asymmetric. No optimal coalition, solvability or learning result has been measured.

## Interface, boundaries and completion

The native boundary is reset(key), raw_step(key,state,actions), observe(state), and legal_actions(state). State is a fixed-key dictionary of fixed-shape int32/bool arrays. Observation and reward are float32. Every role has an action, observation, reward and done row. All out-of-range or masked actions deterministically become wait. A legal intention may still lose a simultaneous contest. Entity caps are fixed; no hidden controllers, callbacks, file access or global cache exists.

A nonterminal raw call advances t exactly once. Natural completion has priority when it coincides with H. `truncated=(t>=H)&~terminated`, `done_all=terminated|truncated`, and every individual done equals joint done. Already-terminal input is absorbing: preserve state, observation, clock and cause; return zero rewards and all dones true. Raw auto-reset never occurs. Event flags describe the last actual transition and remain unchanged under absorbing calls; a renderer must not replay them as new events.

## Exact observation and presentation contract

The exact self fields, finite nearest-entity slots, local map patch and public clock are the only flat observations. Entity slots sort by Manhattan distance then ID; all columns of a hidden slot are -1. Actor rendering also exposes the selected actor through its self block even if crowding removes it from nearest-eight slots. The static map layout is public. Forecast/knowledge leaves are private unless explicitly observed or messaged. Masks expose local contextual affordances. Event rendering must inherit its owning entity visibility and cannot display hidden global vectors. No renderer state is an extra policy observation.

Every non-padding column has its own exact slice and meaning in `manifest.json:observation_fields`. Entity slots encode `[visible, entity_id, absolute_x, absolute_y, declared attributes...]`; a hidden slot is entirely -1. Sorting is ascending Manhattan distance, then entity ID. No value is normalized. Observations are float32 and conservatively bounded [-1,2048]. Explicit padding is zero. Each group visibility mask equals the selected local slots; actor masks additionally include the selected self, whose full attributes are in its self block.

- self: [0, 14] end-exclusive; id, guild, tool, x, y, wood, seal, knowledge, message, event, recipient, own_coins, own_grain, own_delivered.
- citizens: [14, 102] end-exclusive; visible, entity_id, x, y, guild, tool, wood, seal, message, event, recipient.
- locks: [102, 170] end-exclusive; visible, entity_id, x, y, water, health, open, verified, bypass, guest, ttl, uses, coins, grain, mill, loss, toll.
- barges: [170, 214] end-exclusive; visible, entity_id, x, y, owner, cargo, hull, destination, arrived, moved, barge_toll.
- map_patch: [214, 239] end-exclusive; 5x5 row-major dy=-2..2, dx=-2..2.

The scene is safe declarative data with exact source and manifest hashes. Actual state coordinates, active masks, local visibility, resource levels, carried items and events determine drawing. Private forecasts are not exposed by a global scene HUD. Event vectors inherit their owning entity mask; full-state debugging is a separate labeled view. Static map layout is public. No narrative text substitutes for a transition. Every individual ID and affiliation is listed in the manifest.

## Human control and unexecuted scripts

A presentation layer should distinguish Enter game from Watch replay, offer native legal actions and pause/step, focus any actor/affiliation, and stage an arbitrary joint-action vector. A human can control one or many native rows; human participant count is not the role count. Human and policy decisions must call the same raw kernel. These files are interface metadata, not a claim that a browser UI, native animation or human session has been run.

- `inspect_open_two_bank_tow`: 17 literal joint vectors. Inspector and keeper open lock 0; two rope workers alternate towing and walking alongside, then deliver their barge at x=9.
- `timber_bypass_investment`: 13 literal joint vectors. Citizen 4 gathers three timber, walks to lock 0 and makes three persistent bypass contributions.
- Paired `inspector_enables_gatekeeper`: 0 shared prefix vectors; 6 per suffix; exactly one scalar action differs. Only the inspector differs; the same keeper request and two-bank towing should encounter open versus closed lock when entering x=4. Compare actual histories; unexecuted.

The scripts are source-reasoned hypotheses, not runtime legality receipts, success trajectories, solvability certificates or effect estimates. Identical reset and step keys are required for paired replay.

## Literal source evidence

- `candidate.py:30`: `def reset(self, key):`; definition ends at line 67.
- `candidate.py:76`: `def legal_actions(self, s):`; definition ends at line 95.
- `candidate.py:97`: `def advance(self, key, s, actions):`; definition ends at line 195.
- `candidate.py:197`: `def raw_step(self, key, state, actions):`; definition ends at line 200.
- `candidate.py:208`: `def observe(self, s):`; definition ends at line 218.
- `candidate.py:102`: `# Inspection is a separate person at a separate downstream-safety stand.`.
- `candidate.py:113`: `# Seals exist in hands or on a gate, and reciprocal rights expire or are consumed.`.
- `candidate.py:132`: `# Two ropes, one on each bank, must agree on direction; opposite proposals cancel.`.
- `candidate.py:157`: `# Flooding and mill production are delayed material effects of upstream releases.`.
- `candidate.py:151`: `receipts = jnp.sum(jax.nn.one_hot(lock, 10, dtype=jnp.int32) * paid[:, None], axis=0)`.
- `candidate.py:185`: `reward = -0.003 * (a != 0) + 0.25 * delivery[s['guild']] - 0.2 * destroyed[s['guild']] + 0.02 * mill[s['guild']] - 0.1 * loss[s['guild']] + 0.01 * (receipts - expenditure)[s['guild']]`.

## Provenance, adaptations and limits

`author_input_freeze.json` records all 25 actual allowed input files with absolute path, relative name, byte length, SHA-256 and use. Tool output could be excerpted; the freeze does not claim unseen content was semantically read. Only current 248-251 request/design and population/family251 control. Removed historical194 was not searched. No earlier candidate, result, heldout, memory or other coordination input was used. The batch contains only four already reserved IDs within the user cumulative 300-game ceiling.

The supplied JaxMARL README informed parallel per-agent adapter expectations, including observation/action/reward/done dictionaries and `__all__`. This author supplies the native flat kernel; an external adapter is neither fabricated nor a change to the frozen learner. Pgx informed explicit masks and functional game-state presentation. Craftax informed the broad material/world-modification category. PureJaxRL and the supplied Craftax_Baselines PPO-RNN source were read only as integration/training context, not new game libraries, replacement learners or evidence of these games performance. No upstream code or commercial assets were copied. The four visual families remain independent.

Representation adaptations: small integers are losslessly int32; refresh explicitly normalizes integer leaves even if a later process enables JAX x64; bool leaves remain bool; observations/rewards explicitly cast float32; the supplied uint32 key is not duplicated in state. Static Python loops only traverse fixed field-name tuples. Intended JIT/vmap/scan compatibility requires later sealed execution.

No candidate/JAX/NumPy/PRNG, policy, training, throughput, browser or human-play execution occurred. Native/R0 acceptance, parity, resource use, deterministic replay, paired causal outcomes, rendered animation, solvability, strategic richness, learning and transfer remain UNKNOWN. Human-playability score is unmeasured. Negative, zero, failed and inconclusive outcomes must be preserved. This author produces no runtime/resource/release/native/PID/human receipt. After the final seven SHA-256 hashes are reported, this attempt is frozen; repairs require a separate dispatch.
