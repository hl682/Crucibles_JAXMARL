# Village Loom — authored native game mechanism

Status: AUTHORED_NOT_EXECUTED. Assigned game-v3-00193; three agents; source attempt 00.

## Game and human interaction
Village Loom is a rural supply-chain game in a nine-by-nine village. Western and eastern homes own flax plots at (1,1), (7,1); the southern plot is (1,7). Shared spinner and loom are at (4,2), (4,4), dye hedge at (7,7), and cloth fair at (4,7). A millpond barrier at y=5,x=6..8 blocks direct northern access to the dye hedge. Actual walking, adjacency, ownership, carrying and workshop occupancy govern every operation.

Human players can move any role and choose legal sow, cut, gather, spin, weave, work, collect, repair, lend, return, reclaim, gift or sell commands. All interaction buttons correspond to the native action labels. Busy workers visibly have only Wait until their labor commitment expires. The system accepts arbitrary legal action sequences; scripts are not gameplay code. Native keyboard/buttons and animation are not yet implemented or accepted by this author.

## Rules and exact chronology
Each villager starts with one fiber, two seed packets and no other goods. Each inventory type holds two. The sickle belongs to role 0 and starts in its hand; the shuttle belongs to role 1 and starts in its hand. The hedge has six dye. Both facilities start at condition two. Reset independently samples each private preferred delivery deadline from {32,40,48,56}; this is the only stochastic state. Raw steps are deterministic and do not use their PRNG key.

Sowing within one cell of the own bare plot costs a seed and is allowed while incoming t<48. A new flax timer is three and decrements at the end of that same step, so it ripens after three growth steps including sowing. A ripe bed holds two fiber. Cutting needs the valid sickle and capacity; partial cuts leave the remainder. Dye gathering within one cell consumes one hedge berry.

Spinning reserves the spinner for one fiber; weaving reserves the loom for one yarn and one dye and requires a valid shuttle. A facility can hold exactly one unfinished job or finished output. Each start consumes one condition and creates two remaining labor units. At zero condition work is stalled, including a job started from condition one. Repair spends one fiber to restore up to two condition, capped at three.

A free villager within one cell of an active, nonbroken facility can pledge two labor steps. This means this step and the next. Its next decision is forced to Wait even if multiple helpers finish the job early. Pledges bind work to the facility, and may advance another persons job. At the end of a step, committed labor reduces remaining work, limited by actual work outstanding and positive condition. Completion puts a finished output in the same slot; only the job owner may collect it, with inventory capacity. Spinning makes yarn; weaving makes cloth. When two eligible facilities are adjacent, lowest facility index is chosen for pledge, collection or repair. An uncollected output blocks the next job.

Owners can lend the named tool to the adjacent other villager with lowest role index. The borrower may use it during the next eight action decisions. Expiry does not teleport it: its holder/position remain visible, but it no longer authorizes borrowed use. The holder may return it when beside its owner; the owner may physically reclaim it when beside the holder, even early. There is no remote recall. Material gift actions move one fiber, yarn or dye to the lowest-index adjacent other villager with room; they give no reward.

Incoming masks define permitted intents, with invalid/out-of-range commands mapped to Wait. Movement is simultaneous, and actors may co-locate. Old loan counters age by one, then interactions resolve in rotating order t mod 3. A new loan sets eight remaining decisions. Workshop races and finite stocks are rechecked during resolution; losing intents consume nothing. Tool-use permission is snapshotted by the incoming mask: a same-step return/reclaim does not retroactively revoke an already authorized cut/start. Tool transfers use postmovement adjacency and affect the next decision. Growing flax and promised labor advance after interactions. Tool positions then follow their actual holders.

Selling one cloth within one cell of the market completes one of that villagers two personal orders. A sale earns three if next-state t is at or before its private deadline, otherwise one. Natural termination is all six individual order slots filled, or the fair closing at t=80. H=96 is the raw upper bound. Natural priority and absorbing state preserve the actual causes and zero future rewards.

## Why partners matter
The initial tools are split across owners, so completing others timed chains can require a physical loan, material handoff or a different division of labor. Helping someone elses reserved spinning job consumes two work steps while postponing private delivery. Repair burns fiber that could have become yarn, but restores a shared facility. A held finished output, hoarded tool, missed loan return, finite dye stock or occupied machine can change another villagers feasible choices. Owners can offer reliable access, recall early, share material, self-service jobs, or specialize in maintaining common capacity.

This mechanism is an exclusive-workshop queue with costly labor promises, physical tool credit and a finite manufacturing chain. It differs from orchard pollination and from both territorial economies. Authoring does not establish a winning strategy, necessity of cooperation in every state, learnability or cross-environment adaptation.

## Information and visual charter
Sixty public normalized columns encode time, positions, inventories/seeds, labor/worksites, delivered orders, flax state, facility condition/jobs/output, tool holders/loan counters/positions, dye stock and earnings. Three own-role bits and only own deadline complete O=64. Tool owners are fixed public common knowledge. Deadlines have no public entity attribute.

Use a warm village-farming style: recognizable flax stalks and cut bundles, skeins, berry baskets, folded cloth, a visibly worn waterwheel spinner, timber loom, work aprons, carried sickle/shuttle and fair awning. Busy labor should appear at the actual facility, and finished output should occupy its reserved workshop location until collected. Tools must stay with their native holder after expiry; an expired badge may change, not their position. A private order panel uses only the selected roles deadline. Scene.json binds all native entities/attributes; final textured artwork and human controls are later integration, without altering rules.

## Bounds and diagnostics
All material inventories and seed counts remain 0..2. Fiber is finite: initial three plus two seed packets times two fiber per plot. Gifts conserve material; production and repairs consume it. Dye begins at six and is not replenished. Condition is 0..3; work remaining and flax yield are 0..2; flax clock is 0..3; loan counter is 0..8. Tool holders remain roles 0..2. Busy is 0..1 in returned live state, although a fresh promise is internally two before its first labor application; declaring scale 2 is conservative. Per-step reward is 0..3 and personal score at most six. All normalized observations are in [0,1]. Co-location is allowed.

The labor diagnostic has role 0 reserve a spinner job, then role 1 pledge work. The paired hypothesis differs only in that pledge and shares the following wait. It compares role 0 later collection opportunity and workshop output, along with the helpers forced wait. The second script moves the shuttle between actual adjacent characters and returns it. Seed 193 was chosen once without search. Neither sequence has been executed and neither is a learning result.

## Complete native action dictionary

- 0: Wait; committed labor advances automatically.
- 1: Walk north.
- 2: Walk east.
- 3: Walk south.
- 4: Walk west.
- 5: Sow own adjacent flax plot.
- 6: Cut own ripe flax using valid sickle.
- 7: Gather adjacent dye berries.
- 8: Reserve spinning job: 1 fiber.
- 9: Reserve weaving job: 1 yarn, 1 dye, valid shuttle.
- 10: Commit 2 work steps at adjacent active facility.
- 11: Collect own completed workshop output.
- 12: Repair adjacent facility: 1 fiber adds up to 2 condition.
- 13: Lend owned sickle to adjacent lowest-index villager.
- 14: Lend owned shuttle to adjacent lowest-index villager.
- 15: Return borrowed tools to adjacent owners.
- 16: Physically reclaim owned tools from adjacent holders.
- 17: Give 1 fiber to eligible adjacent lowest-index villager.
- 18: Give 1 yarn to eligible adjacent lowest-index villager.
- 19: Give 1 dye to eligible adjacent lowest-index villager.
- 20: Sell 1 cloth beside market for private order.

## Literal source-line evidence

- Sites, tools, private deadline: candidate.py line 12; literal source: def reset(self, key):
- Tool/adjacency/busy masks: candidate.py line 61; literal source: def legal_actions(self, s):
- Shared job reservation: candidate.py line 130; literal source: start = ((a[i] == 8) | (a[i] == 9)) & (n['work_left'][job] == 0) & ~n['output'][job] & (n['health'][job] > 0)
- Physical tool loan and return: candidate.py line 162; literal source: for tool in range(2):
- Binding labor and completion: candidate.py line 194; literal source: labor = jnp.sum(((n['busy'][:, None] > 0) & (n['worksite'][:, None] == jnp.arange(2)[None, :])).astype(jnp.int32), axis=0, dtype=jnp.int32)
- Natural fair closing and absorption: candidate.py line 204; literal source: n['terminated'] = jnp.all(n['delivered'] >= 2) | (n['t'] >= 80)
- Public state and private deadline: candidate.py line 212; literal source: def observe(self, s):


## Provenance, reference use and verification limits
This is source attempt 00 of the original blind initial window, not a repair or feedback-selected revision. Current batch ordinals 186..193 were reserved inside the original cap 512; cumulative used-or-reserved count was 194 and feedback count was zero. This author owns only GI190..193 attempt_00 and does not alter other work. Exact actual input paths, byte sizes, hashes and purposes are in author_input_freeze.json, including the prospective human-playability addendum. All four assigned design/request pairs were read; no other candidate, result, selection, project training, develop/confirm, heldout, IR or final-test input was accessed.

The supplied official JaxMARL README informed a prospective dictionary-per-agent adapter boundary. Pgx informed explicit external random keys, masks and environment identity. Craftax README informed native spatial state/presentation and raw versus outer autoreset. PureJaxRL README and Craftax_Baselines_ppo_rnn.py informed only reset/step/scan integration. They are algorithm integration references, not new game libraries or replacements for the frozen learner. No upstream game, algorithm code or visual asset was copied. No network request, model/API invocation, remote session, credential inspection, download or candidate execution was performed.

Signed enums, counters and ownership fields use lossless int32; state booleans use bool. The supplied reset key is not duplicated inside state. This representation adaptation does not imply changing an executed earlier distribution. Each new private-parameter distribution is stated above. Raw terminal semantics follow the addendum: a live call advances t once, natural termination overrides horizon, and an already-done call preserves state/observation/time/cause with zero rewards.

Only stdlib AST, JSON and hash checks were authorized. No candidate or JAX import, eager call, JIT, vmap, rollout, random policy, seed search, replay, RGB render, adapter test or training was run. Shapes/dtypes, masks over trajectories, numerical parity, invariant preservation and paired effects await independent frozen-source integration. Positive, zero, negative, failed and indeterminate results remain valid reports. No reward, seed-count or success-rate acceptance threshold is introduced. Source authorship is not human-playability acceptance or evidence of learning/transfer.
