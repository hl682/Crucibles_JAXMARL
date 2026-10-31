# Hidden-demand exchange yard

Status: **AUTHORED_NOT_EXECUTED**. Candidate `game-v3-00117`, source attempt 0, source split train, final test unused.

## Spatial rules and cross-agent dependence

Three traders move through a 9 by 7 exchange yard with wall gaps, goods stations, three central loading bays and indexed delivery docks. Six identified goods exist, two of each of three types. Each trader initially carries one good, has only one carrying slot and privately needs two goods of a reset-assigned type. Demands are a random permutation: each role learns its own entry, which constrains but does not completely specify the other two. The map and all physical goods are public; the private order type is not painted onto a dock.

Movement uses rotating occupancy priority, blocks currently occupied destinations and checks the previous state's reservation and escrow restrictions. A nearby free bay can be reserved for six displayed ticks, excluding other traders from entering it. Reservation or renewal costs an action. A trader may hold at most one live reservation, renew its nearby existing reservation, release it or let it expire. An occupied bay cannot be reserved against its occupant. These rules make loading space and narrow routes genuine shared constraints.

At a bay, actions 10, 11 and 12 post the carried good requesting type 0, 1 or 2. The give field comes from the actual physical cargo and cannot claim a nonexistent good. The requested type is a chosen public offer, not an automatic disclosure of the hidden order. Posting commits that exact good for four displayed ticks. While committed at the beginning of a transition, the trader cannot move, pick up, drop or deliver; cancellation consumes an action and incurs 0.03. Expiry, cancellation or settlement releases the commitment.

The clearing service first chooses a reciprocal pair in rotating order. It simultaneously exchanges ownership of the two escrowed goods and then clears both offers. If no pair clears, three compatible offers can settle as a three-way cycle. A forward cycle has priority when both directions are compatible. Every transfer is a permutation of existing good owners, with no credit, duplication or half-completed transfer observable between transitions. Each good follows its actual carrier's position. Automatic settlement is authorized by the posting commitment; no extra accept action is required.

Action 13 truthfully publishes the acting trader's own demand for five ticks. It does not permit a freely chosen false type. Silence remains possible. Public observations include posted requests, deliberate signals, cargo and deliveries. A truthful reveal or the public history of a matching delivery can legitimately permit demand inference. Direct demand entries remain absent from other observation rows and all public entities. The separate private-order entity group uses identity-matrix visibility, so only the indexed owner sees its required type in a role view. A future debug view may expose all state only under an explicit omniscient label.

A matching delivery at the actor's own dock transfers the physical good to delivered ownership. Each own delivery pays four units; each delivered good also pays every trader 0.3. Active time costs 0.01 per tick. Trading itself earns nothing. Finished traders disappear from active traffic and have only wait available. Natural completion requires all six goods delivered.

## Partner-dependent strategies and limits

A trader may reveal demand to help others form useful offers, stay silent while choosing a public request, wait for a reciprocal pair or join a three-way cycle. It may transport a good to a partner, drop one to free its slot, reserve space, or leave an approach clear. A matching cycle can remove bilateral bargaining deadlock, but an early pair can also prevent a possible cycle because pair clearing has priority. Offering the type a partner needs is useful only if that partner commits compatible cargo and can reach a bay.

The seed-11701 script moves all three traders to the bays, deliberately reveals their demand signals and proposes a type cycle. The proposed cycle does not depend on observed hidden demand, and it may give a trader a good that does not satisfy its private order. It is a mechanism witness hypothesis, not an optimized policy or a delivery solution.

Three paired hypotheses separately examine the third commitment enabling a three-way settlement, a reservation excluding another trader's west move, and a truthful reveal changing public signal fields. They use fixed seeds, prefixes and integer vectors. The trade pair can produce different owners with exactly the same immediate reward; the trade can be counterproductive for private delivery. The reservation may delay a partner without improving anyone's eventual return. The reveal may enable inference but provides no guaranteed policy benefit. Missing prerequisite states, premature completion or implementation failures must be reported as such.

Changing carrying capacity, demand distribution, physical bay adjacency, reservation duration or the pair/cycle precedence would change bargaining and routing tradeoffs. These are unimplemented variation hypotheses. This source does not measure communication value, equilibrium behavior, negotiation quality or meta-learning adaptation.


## Literal source evidence

- Private entity visibility: `candidate.py:24` contains `n['order_visible'] = jnp.eye(3, dtype=jnp.bool_) & n['order_active'][None, :]`.
- Private demand construction: `candidate.py:50` contains `'demand': jax.random.permutation(key, jnp.arange(3, dtype=jnp.int32)), 'remaining': jnp.full(3, 2, jnp.int32),`.
- Own-demand observation row: `candidate.py:73` contains `return jnp.concatenate((jnp.eye(3, dtype=jnp.float32), jax.nn.one_hot(s['demand'], 3),`.
- Public availability masks: `candidate.py:78` contains `return (jnp.arange(15)[None, :] == 0) | ((~s['done'] & s['agent_active'])[:, None])`.
- Reservation exclusion: `candidate.py:91` contains `reserved = jnp.any(at_bay & (s['bay_ttl'] > 0) & (s['bay_owner'] != i))`.
- Escrow movement restriction: `candidate.py:93` contains `move = s['agent_active'][i] & ~committed[i] & ~reserved & ~occupied & (s['tiles'][target[i, 1], target[i, 0]] != 1)`.
- Goods delivery only at a matching private order: `candidate.py:112` contains `deliver = (a[i] == 7) & ~committed[i] & jnp.any(held) & (`.
- Actual escrow item at a bay: `candidate.py:138` contains `post = (a[i] >= 10) & (a[i] <= 12) & (n['offer_ttl'][i] == 0) & at_bay & jnp.any(held) & n['agent_active'][i]`.
- Truthful public signal: `candidate.py:144` contains `n['signal_type'] = n['signal_type'].at[i].set(jnp.where(signal, s['demand'][i], n['signal_type'][i]))`.
- Reciprocal atomic matching: `candidate.py:153` contains `match = valid & valid[i] & (jnp.arange(3) != i) & (n['offer_request'] == n['offer_give'][i]) & (`.
- Three-way cycle condition: `candidate.py:167` contains `cycle = ~pair_done & (forward | reverse)`.
- No trading reward: `candidate.py:184` contains `reward = (4.0 * delivered_now + 0.3 * jnp.sum(delivered_now.astype(jnp.float32)) -`.
- Absorbing terminal projection: `candidate.py:186` contains `n = {k: jnp.where(s['done'], s[k], n[k]) for k in s}`.

Method ranges: refresh at candidate.py:12-29; reset at candidate.py:31-58; observe at candidate.py:60-74; legal_actions at candidate.py:76-78; raw_step at candidate.py:80-189.

## Literal interface and execution boundary

The author supplied three-agent flat float32 observations, int32 joint actions, fixed-shape dictionary state and homogeneous discrete command attempts. The exact dimensions, every leaf, every observation column, reward bounds, actions and spatial bindings are in manifest.json. Unmet command prerequisites produce no effect; out-of-range integers are converted to wait. Each live raw transition advances t once. Natural completion takes precedence over an equal-tick timeout. Already-terminal raw calls preserve every state leaf, observation and clock, return zero reward and preserve actual terminal causes. There is no raw autoreset.

No candidate was imported or called. No JAX/NumPy import, compilation, numerical game call, training, model API, installation, SSH or remote work occurred during authoring. Standard-library AST and supplied data validators may establish only their literal static checks. They do not establish interface execution, numerical parity, conservation over actual traces, strategic richness, solvability, policy learning or transfer. AUTHORED_NOT_EXECUTED is the package status. The JaxMARL-compatible public dictionary adapter, real replay, RGB animation, interactive playback and role-view audit belong to the separately authorized implementation/validation stage and are not delivered or claimed here.

The playthrough JSON uses literal seeds and joint integer vectors. A trusted runner should use jax.random.PRNGKey(seed) for reset, then jax.random.fold_in(reset_key, step_index+1) for raw-step keys; these are prospective instructions and were not executed. All four kernels use step keys only as interface inputs, so their transition stochasticity is reset-derived. Scripts stop at joint done. They are author hypotheses, not successful trajectories or acceptance gates. Paired tests freeze the shared prefix and the non-intervened actions; they must retain zero, negative, mixed, failure and indeterminate outcomes.

The source uses int32 for small integer categories, owners and counters and bool for masks/terminal causes. The uint32 PRNG key remains trusted trajectory provenance rather than a duplicated state leaf. Render data is declarative. Map and entity positions bind real state. An omniscient debug state, where provided later, must be labelled separately and never substituted for a role observation.

## Official ecosystem inspiration and source use

Only this batch's physically supplied official README excerpts were read for broad ecosystem context: JaxMARL separates MARL environments and algorithms and exposes multi-agent interfaces; Pgx exemplifies JAX-native board states, legal action masks and explicit step randomness; Craftax illustrates persistent spatial game state, inventories and separate reset/render interfaces. These are conceptual influences. No upstream source code, environment layout, asset, baseline or algorithm was copied into this original kernel.

The supplied JaxMARL README identifies Apache-2.0, the Pgx README Apache-2.0, and the Craftax README MIT. Those are descriptions of upstream snapshots, not a new license grant for this package; an original-package release license remains the coordinator's decision. PureJaxRL describes algorithm implementations rather than an environment library. Craftax_Baselines_ppo_rnn.py was supplied and hash-accounted for but was not opened or reused. No baseline replaces the frozen experimental learner. Exact source snapshot URLs and hashes are in generation_record.json; no batch-external source or candidate feedback was consulted.

Generating or replaying these rules would not establish large-scale meta-RL adaptation. Rule/role/map variations discussed above are future research hypotheses requiring their own frozen source, cost and information controls, and genuine train/develop/confirm/final-transfer evidence. They are not implemented variants or measured generalization.
