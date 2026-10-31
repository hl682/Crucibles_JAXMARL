# Breathshare Claims — authored attempt_00

Status: AUTHORED_NOT_EXECUTED. A=3, O=153, U=10, H=80. This source, declarative scene and rule-derived scripts have not been imported or run by the author.

## Physical resources and private objectives

The 11x7 asteroid has impassable rocks at (5,2) and (5,4), a shared refill/delivery station at (1,3), and three veins at (7,1), (8,3), (7,5). Vein indices are ore types 0,1,2 and each begins with two units. Each miner holds at most one ore unit and one canister.

Miners start at (2,3), (2,2), (2,4), each with 12 oxygen. Two canisters start at (2,2) and (2,4), each holding six gas units. Station stock is 24. A reset-key permutation assigns the three distinct private claim types. Actual claim values exist only in state and each role own observation.

Every living miner outside the station consumes one oxygen at the end of every nonterminal call, including Wait. The station is the only safe cell. Reaching zero kills that miner, drops its canister at its actual last position, and destroys any carried ore while incrementing lost_counts. Dead roles retain only Wait, do not block other live miners, and remain represented by a true last-position memorial. Joint dones remain false until the joint ending rule; individual inactivity is communicated by active and the mask.

## Actions, consent and deterministic conflicts

Actions 0..4 are Wait and cardinal moves. Movement masks rock/bounds; entering another live miner's old cell or a contested live destination fails. Swaps and following into a cell vacated on the same call are disallowed. Only one live miner can occupy the station, so holding it denies another miner a refill or delivery approach.

Action 5 extracts from a vein at Manhattan distance at most one if ore inventory is empty. Available lowest vein index is chosen. Agents are processed in rotating order t modulo three, then the next two indices; each extraction decrements the current shared reserve before the next request. A request losing the last unit conflict is a legal failed attempt.

Action 6 drops a held canister at the miner cell, or picks a grounded canister within Manhattan distance one. Canisters are processed in index order and competing pickup agents use lowest index. A miner who began with a canister cannot drop it and pick another in the same action. Busy-owner checks prevent one miner acquiring both canisters.

Action 7 offers one's canister. Transfer requires another live miner exactly one cell away, empty-handed, simultaneously choosing action 8. If there is more than one eligible recipient, lowest role index wins. Canisters resolve in index order with updated ownership, so a recipient cannot accept two. Without simultaneous consent, ownership is unchanged.

Action 8 also draws oxygen from the now-owned canister, including one just accepted on the same call. Draw is min(canister volume, 12 minus personal oxygen). This happens before end-of-call breathing, making a timely transfer capable of changing imminent survival. The same draw debits the real canister volume; it cannot create oxygen.

Action 9 works only on the station. At most four units are removed from finite stock per call. The miner personal deficit is filled first, then any remaining per-call budget may fill its held canister toward six. An already-full miner can refill an empty canister. The station does not manufacture new gas.

The full order is sanitize, move, Pick/Drop, Offer/Accept and canister use, station refill, extraction, automatic delivery, breathing/death and resulting canister drops, then termination. Interactions use actual resolved positions. All resource transfers have bounded integer amounts.

## Coupled choices

A portable supply can let a teammate reach a distant vein or return before depletion, at the donor's opportunity cost of losing that supply. Refilling for oneself consumes stock that someone else may need later. Occupying the only safe station blocks access. Extracting one of a vein's two units removes that delivery opportunity from every other miner; a nonmatching miner can still take it for a smaller private payment.

Possible tactics include self-provisioned short trips, a supporting canister handoff, station turn-taking, or opportunistic extraction of another miner's favored type. Common survival penalties and common delivery payments encourage support, while larger claim-matching personal payments create competing objectives. No equilibrium, optimal schedule or measured adaptation is asserted.

## Payoffs, endings and invariants

Every delivery gives all roles +0.1. Its deliverer additionally gets +1 if the ore matches its private claim, or +0.25 otherwise. Every newly dead miner subtracts 0.25 from every role, including already-dead roles that remain in the joint game. These are bounded by [-1,2] per role per call.

A natural ending occurs if no miners remain alive, or if all veins are empty and nobody carries ore. The latter includes ore accounted as lost; it is not automatically successful delivery. Otherwise H=80 truncates with natural-ending priority. Negative, deadlock, timeout and partial-delivery outcomes remain valid real outcomes for later execution.

For every ore type, remaining vein units plus carried units plus delivered_counts plus lost_counts should stay exactly two. There are always exactly two portable canisters with unique owner-or-floor status; canister gas stays 0..6, personal oxygen 0..12, and station stock 0..24. Total stored oxygen starts at 72 and falls only through breathing. Movement preserves unique live positions. Private claims stay a permutation and are not changed by play. These are declared conservation properties requiring real invariant tests.

## Observation visibility and scene

The 153 columns expose the public map, public live/dead positions, oxygen, inventories, vein depletion, canister positions/ownership/gas, station stock/occupancy, aggregate delivered/lost ore and actual event flags. Each row additionally has the observing role own private claim one-hot and own identity one-hot. Other private claim values do not enter that row.

The scene uses a separate private_claims entity group at the actual owner position and an identity-matrix visibility leaf. A role-specific renderer must display only that role claim; it must not draw the other assignments merely because full state contains them. A debug view that shows all claims must be labelled as privileged. Public ore and rewards may legitimately allow inferences during play, but direct private labels remain hidden.

Depleted veins and empty canisters stay present with actual zero quantities. A memorial uses the dead miner's retained position and true inactive status. Station occupancy is an explicit state attribute. Transfers, gas draws, pickups, drops, extraction, delivery and deaths are latest-transition flags; delivered/lost counts are cumulative. No frame or event is fabricated from the intended script.

## Literal scripts and future tasks

The long script gives miner 0 a central-vein round trip while its waiting partners pick and later use their own canisters. It targets one delivery and a station refill, not a complete or optimal episode; the private claim reward depends on the actual reset permutation. The two branch pairs isolate simultaneous consent and station departure. Each uses literal action arrays with identical branch prefixes.

Changing station location/stock, canister placement, vein supply, claim distribution or breathing cost could change support and competition strategies. Such changes are prospective new tasks requiring new source hashes and validation, not implemented variations or evidence of transfer in this attempt.

## Literal source evidence

- Key-driven private assignments: candidate.py:26 contains exactly `"claim": jax.random.permutation(key, jnp.arange(3, dtype=jnp.int32)).astype(jnp.int32),`.
- Strict owner-only visibility: candidate.py:27 contains exactly `"claim_visible": jnp.eye(3, dtype=jnp.bool_),`.
- Native observation uses own claim only: candidate.py:105 contains exactly `return jnp.concatenate((jnp.broadcast_to(common, (3,48)), jax.nn.one_hot(state["claim"], 3),`.
- Consensual transfer: candidate.py:152 contains exactly `consent = (donor >= 0) & (actions[safe_donor] == 7) & (recipient < 3)`.
- Actual portable gas debit: candidate.py:161 contains exactly `volume = volume.at[c].add(-amount)`.
- Finite shared station allocation: candidate.py:169 contains exactly `budget = jnp.where(refill, jnp.minimum(stock, 4), 0)`.
- Rotating mining priority: candidate.py:181 contains exactly `a = jnp.mod(state["t"] + order, 3)`.
- Real resource depletion: candidate.py:188 contains exactly `remaining = remaining.at[safe_vein].add(-extraction.astype(jnp.int32))`.
- Delayed oxygen survival: candidate.py:194 contains exactly `oxygen = jnp.maximum(oxygen - (alive & ~at_station).astype(jnp.int32), 0).astype(jnp.int32)`.
- Common survival and personal delivery payoffs: candidate.py:233 contains exactly `reward = jnp.full((3,), 0.1 * jnp.sum(delivery.astype(jnp.float32)) - 0.25 * jnp.sum(died.astype(jnp.float32)), jnp.float32)`.
- Raw terminal absorption: candidate.py:239 contains exactly `ns = jax.tree.map(lambda old, new: jnp.where(state["episode_done"], old, new), state, proposed)`.

The line references above are static source evidence, not executed transition evidence. Method spans are reset=12-55, legal_actions=57-81, observe=83-106, advance=108-235, raw_step=237-243.

## Interface, provenance and remaining acceptance

The native boundary is reset(key) -> (observation,state), observe(state), legal_actions(state), and raw_step(key,state,actions) -> (observation,state,reward,dones,info). All state leaves are fixed-shape int32/bool arrays; observations and rewards are float32. Signed sentinels use int32. The uint32 reset key belongs to external provenance and is not copied into state. Every valid nonterminal call increments t once. Natural termination has priority over a simultaneous horizon. Already-terminal calls preserve every state leaf and t, return the preserved observation and cause flags, zero reward, and all dones true. No raw auto-reset occurs. Invalid/out-of-range/masked actions deterministically become action 0. Action 0 remains legal for every role, including inactive or immobilized roles and terminal states.

A later public JaxMARL adapter must map the listed agent names to ordered action/observation/reward/done dictionaries and expose done["__all__"]. It must preserve these observations, masks, rewards, raw terminal states, and inactive-role semantics. The author did not implement or validate that separate adapter, replace the frozen learner, or reuse old policy-bank weights. The declared A/O/U/H require matching integration and policy metadata.

Only data schemas and source AST were inspected. Static checks do not establish JIT/vmap behavior, numerical parity, physical invariants under all actions, script legality, solvability, strategic richness, learned adaptation, transfer, or scientific acceptance. No real trajectories, animations, RGB frames, benchmark scores, training results or final-test data were read or generated. The scene is declarative binding data for a later real-state replay. Each source is attempt_00 and is frozen at its recorded SHA-256; a later repair requires separate actual feedback and dispatch.

Community references were supplied as hash-bound read-only files. JaxMARL informs the future parallel agent-keyed interface; Pgx informs explicit legal masks and terminated/truncated separation; Craftax demonstrates persistent spatial game state and explicit reset handling. PureJaxRL and Craftax_Baselines_ppo_rnn.py are algorithm/integration references, not game libraries or replacements for the experimental learner. No upstream game rules or code were copied into this kernel, and no upstream benchmark implementation was read. The original rules here are attributable to GPT-6-Astra/max.

- references\JaxMARL_README.md (https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md), SHA-256 `6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15`.
- references\Pgx_README.md (https://github.com/sotetsuk/pgx/blob/main/README.md), SHA-256 `153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df`.
- references\PureJaxRL_README.md (https://github.com/luchris429/purejaxrl/blob/main/README.md), SHA-256 `d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0`.
- references\Craftax_Baselines_ppo_rnn.py (https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py), SHA-256 `5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011`.
- references\Craftax_README.md (https://github.com/MichaelTMatthews/Craftax/blob/main/README.md), SHA-256 `7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff`.
