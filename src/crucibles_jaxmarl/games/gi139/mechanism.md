# Three Bloom Exchange (game-v3-00139)

**Status: AUTHORED_NOT_EXECUTED.** This is an original initial-window train-only source package. Source, scene and literal action scripts are authored hypotheses. No candidate was imported, evaluated, replayed, trained or rendered during authoring. No final-test or earlier candidate/result input was used.

Source SHA-256: `01fe140143e8464ce2affdebe5c9aefa347536d5b309d0af03c478cf68994f08`.

Native boundary: A=3, O=120, U=12, H=96; homogeneous scalar int32 actions, float32 flat observations and rewards, fixed-key int32/bool state. Declared observation interval [0, 1], reward interval [0, 2]. These are representation/game bounds, not empirical acceptance thresholds.

## Garden, conserved resources and transition order

Three couriers move on a 9x7 open grid. Six fixed plants are at (1,1),(7,1),(1,5),(7,5),(4,1),(4,5), with species [0,1,1,0,0,1]. Each begins with two pollen packets, giving twelve finite units. Couriers start at (1,2),(4,2),(7,2). The common depot is (4,3). The reset key samples each courier's private preferred fruit species independently from {0,1}; there is no other randomness, and raw-step keys are unused.

Each courier has one shared inventory slot: one pollen packet or one fruit, never both. A pollen packet retains its original collector identity after every handoff. To graft a plant, a different courier must hold it; this explicit two-handler rule makes a partner's reception necessary. The target plant must be the opposite species and currently flowering. Species 0 opens for clock phases t=0..3 modulo 8, and species 1 for phases 4..7. Only ungrafted plants flower. Grafting therefore closes its remaining pollen stock to further collection, a real opportunity cost for future couriers.

All commands are legal intentions while the episode is live. Failed gather, give, receive, graft, harvest, delivery and discard attempts no-op. Out-of-range commands become wait. Movement obeys borders, prohibits swaps/following into currently occupied cells, and rejects all proposals sharing an otherwise empty target. Plants and the depot can share a courier's cell; they are interaction sites, not obstacles.

After movement, already-growing plants count down. A graft made on this transition starts at three; it becomes ripe after three later transitions. Then GIVE/RECEIVE rendezvous resolve. A giver carrying pollen needs an adjacent empty courier selecting RECEIVE. Ties among receivers and competing givers follow rotating rank/order (id-t) mod 3. A packet's collector identity is preserved. There is no same-step give chain because GIVE and RECEIVE are distinct actions.

Plant commands then resolve in rotating actor order. A courier considers plants within Manhattan distance one, with the lowest eligible plant id breaking site ties. GATHER removes a finite pollen unit; GRAFT consumes a compatible transferred packet and starts the growth clock; HARVEST takes a ripe fruit if the shared slot is empty. A plant can produce only one fruit. The six fruit entity slots are inactive before ripeness, located on their plant when ripe, carried at the responsible courier's actual endpoint, then inactive at the depot once delivered. A newly ripe fruit can be harvested on that same later transition.

DELIVER works within Manhattan distance one of the depot. Every delivery gives 0.5 to each courier and an additional 0.5 to the delivering courier only if the fruit matches its private preference. DISCARD destroys held pollen permanently and records that loss; fruit cannot be discarded. Delivering all six fruits naturally ends the task. Resource mistakes and failed coordination may leave the garden incomplete until the 96-step truncation; no artificial score criterion prevents those outcomes.

## Causal interaction and competing schedules

A courier cannot collect and immediately use its own packet. Its partner can enable or refuse the transfer, redirect a compatible packet to another plant, or occupy the intended rendezvous cell. Successful grafting removes a flower as a resource source now and creates a delayed ripe-fruit opportunity later, potentially for a third courier. All three agents therefore affect each other's feasible future actions through physical inventory, opening windows and locations.

A team can use short local exchanges between paired gatherers and specialized harvest runners, or circulate pollen around the garden and let couriers switch roles. Early grafting minimizes maturation delay but can seal off unused pollen. Delaying it preserves stock and may suit another courier's route, at the risk of missing the four-step flowering window. A courier may prefer to deliver its favorite species for the private bonus; allowing another courier to harvest still creates a shared reward. None of these candidate strategies has been optimized or measured.

## Information and visual fidelity

Own preference and inventory are private/self fields. Other courier inventories, all plant state and actual active fruits are observed only within Manhattan radius three. Hidden payloads are zero with an explicit visibility bit; the entity visibility matrices use exactly the same distance rule. Other couriers' private preferences are absent from scene attributes and public observations.

The depot, current flowering phase, delivery and consumed/discarded-pollen totals are public. Garden bells publicly report aggregate event counts without actor/site identities, including movement conflicts. These public bells are explicitly included in native observations; the scene does not invent extra distant-event information. Reset plant layout and the open rectangular bounds are fixed rules, but current hidden plant stock/stage remains unavailable until observed. Fruit and plant are separate physical stages; animation must not draw inactive unborn or delivered fruit.

## Declared invariants and boundedness

- sum(plant_stock) + count(pollen_kind>=0) + pollen_used + pollen_discarded = 12. A handoff preserves this total; grafting and discard only move a packet into the corresponding consumed ledger.
- Each courier carries at most one item across pollen_kind and fruit_load. An empty pollen slot has collector=-1; a held packet has a valid original collector.
- A plant stage advances only 0 -> 1 -> 2 -> 3. Its growth countdown is in 0..3; uncollected pollen stock is in 0..2.
- Each fruit id appears in at most one courier load and agrees with fruit_owner. A delivered fruit stays delivered and cannot be rewarded again.
- Active couriers remain on distinct in-bounds cells. Plant positions are constant; carried fruit positions follow the carrier.
- All preferences and species are binary; delivered is in 0..6 and consumed/discarded pollen in 0..12. Three simultaneous deliveries can reward a courier at most 2.0.
- These are declared source properties for future independent checks, not executed assertions.

Changing plant spacing, phase duration, sight radius, the two-handler rule or preference distribution could change the coordination problem. Such variants are prospective ideas requiring separately identified sources and protocols; no claim of trained adaptation or transfer follows from this authoring.

## Exact action legend

- 0: Wait.
- 1: Move north.
- 2: Move east.
- 3: Move south.
- 4: Move west.
- 5: Gather one pollen packet.
- 6: Give pollen to an adjacent receiving courier.
- 7: Receive pollen into an empty inventory.
- 8: Graft opposite-species pollen handled by another collector.
- 9: Harvest one ripe fruit into an empty inventory.
- 10: Deliver carried fruit within one cell of the depot.
- 11: Discard held pollen permanently.

## Native observation columns

- [0:5), self: Self: id/4,private preferred species,(pollen_kind+1)/2,(collector+1)/4,(fruit_load+1)/8.
- [5:10), public: Public: t/128,flowering phase,delivered/8,discarded pollen/16,used pollen/16.
- [10:28), local: Three local courier records, stride 6: visible,x/16,y/16,(pollen_kind+1)/2,(collector+1)/4,(fruit_load+1)/8; hidden payloads zero.
- [28:76), local: Six local plant records, stride 8: visible,x/16,y/16,species,flowering,stock/2,stage/4,growth/4; hidden payloads zero.
- [76:106), local: Six local active fruit records, stride 5: visible,x/16,y/16,species,(owner+2)/8; hidden/inactive payloads zero.
- [106:108), public: Public depot x/16,y/16.
- [108:115), public: Public event counts/8 in order gather,handoff,pollination,ripening,harvest,delivery,collision; no actor or site identity.
- [115:120), padding: Five constant zeros.

The renderer binds only these real state groups, positions, active masks and role visibility masks. Glyphs and inert colors carry no extra rule. A carried object shares the actual carrier endpoint; flight, growth, support changes, leases and service are real state transitions. Frame interpolation is presentation only and must not invent a catch, handoff, crossing or private observation. Debug/full-state view must remain separately labeled.

## Authored paired interventions

The complete literal integer vectors, including every prefix step, are saved in playthrough.json. These are not observed trajectories or guaranteed successful solutions.

**receiver_opens_third_courier_future_harvest**

Only courier 1 changes RECEIVE to WAIT at the rendezvous. All later actions match. A should create a fruit for courier 2; B should leave the same graft, harvest and delivery attempts without that resource. No trajectory was executed.

Prefix (2 joint steps): `[[5,0,0],[2,4,0]]`.
Branch A suffix: `[[6,7,0],[0,2,0],[0,2,0],[0,2,3],[0,1,0],[0,8,0],[0,0,1],[0,0,1],[0,0,9],[0,0,3],[0,0,3],[0,0,4],[0,0,4],[0,0,10]]`.
Branch B suffix: `[[6,0,0],[0,2,0],[0,2,0],[0,2,3],[0,1,0],[0,8,0],[0,0,1],[0,0,1],[0,0,9],[0,0,3],[0,0,3],[0,0,4],[0,0,4],[0,0,10]]`.

## Raw terminal and boundary semantics

A live raw step advances t once. Natural termination takes priority over the horizon: truncated=(next_t>=H) and not terminated; episode_done=terminated or truncated. raw_step uses an absorbing branch for an already completed state: every leaf and observation remain unchanged, reward is zero, all per-agent dones stay true, and terminal causes are preserved. No reset occurs in raw_step. A later trusted wrapper may reset outside it; an absorbing call is not a new learning transition or a new occurrence of the preserved last-transition event flags.

Invalid/out-of-range commands deterministically become action 0. Every role always has action 0. All small signed codes, clocks, indices and inventories use int32; masks and terminal causes use bool. The externally supplied uint32 reset key is not duplicated in the state. Scaling uses powers of two. This design choice is not a numeric-parity result.

The required community adapter remains a separate implementation-layer deliverable: map native array row i to agent_i in action/observation/reward/done dictionaries and map info.done_all to __all__, preserving these masks, role observations, raw terminal causes and inactive-role semantics. This six-file author package neither implements that adapter nor silently replaces a frozen learner.

## Literal source evidence

- `refresh`: candidate.py lines 15-28.
- `reset`: candidate.py lines 30-67.
- `legal_actions`: candidate.py lines 69-72.
- `advance`: candidate.py lines 74-176.
- `observe`: candidate.py lines 178-206.
- `raw_step`: candidate.py lines 208-213.

Native local visibility — candidate.py line 21:

```python
        n["plant_visibility"] = plant_distance <= 3
```

Complementary physical flowering phase — candidate.py line 17:

```python
        n["plant_flowering"] = (state["plant_stage"] == 0) & (jnp.mod(state["t"] // 4, 2) == state["plant_species"])
```

Delayed ripe-fruit creation — candidate.py line 99:

```python
        ripened = growing & (n["plant_growth"] == 0)
```

A partner must explicitly accept the packet — candidate.py line 108:

```python
            receivers = adjacent & (a == 7) & (n["pollen_kind"] < 0) & (n["fruit_load"] < 0) & (ids != i)
```

Successful two-agent rendezvous — candidate.py line 110:

```python
            give = (a[i] == 6) & (n["pollen_kind"][i] >= 0) & jnp.any(receivers)
```

Collector identity survives the transfer — candidate.py line 112:

```python
            old_collector = n["pollen_collector"][i]
```

Grafting cannot use a self-collected packet — candidate.py line 136:

```python
            graft = (a[i] == 8) & (n["pollen_kind"][i] >= 0) & (n["pollen_collector"][i] != i) & jnp.any(compatible)
```

Harvest creates another courier transport opportunity — candidate.py line 147:

```python
            harvest = (a[i] == 9) & empty & jnp.any(harvestable)
```

Shared delivery and private preference reward — candidate.py line 175:

```python
        reward = (0.5 * jnp.sum(delivered_by.astype(jnp.float32)) + 0.5 * preferred_by.astype(jnp.float32)).astype(jnp.float32)
```

Public bells have native observation columns — candidate.py line 206:

```python
                                jnp.broadcast_to(bells, (3, 7)), jnp.zeros((3, 5), dtype=jnp.float32)), axis=1).astype(jnp.float32)
```

Actual natural completion — candidate.py line 171:

```python
        n["terminated"] = n["delivered"][0] == 6
```

## Reference use and provenance

All game rules and kernel expressions were authored for this assigned design; no upstream game, board implementation, model weights or learner code was copied. The allowed reference snapshots were read as inspiration and interface context. Their exact identities are also recorded in generation_record.json.

- JaxMARL supplies the community parallel, agent-keyed API model and the idea of distinct environment/algorithm components. Its example game rules were not copied.
- Pgx supplies explicit-key, legal-mask and state/visualization API context for discrete games. It is a board-game environment library, not this spatial game's implementation.
- Craftax supplies inspiration for stateful spatial inventory and persistent world consequences. The assigned rules are independent of Craftax's game mechanics.
- PureJaxRL is an end-to-end algorithm implementation reference. Craftax_Baselines/ppo_rnn.py is recurrent PPO integration code and credits Chris Lu/PureJaxRL. Neither is treated as a new game library or substituted for the experiment's learner.

- `references\JaxMARL_README.md`; SHA-256 `6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15`; source: https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md.
- `references\Pgx_README.md`; SHA-256 `153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df`; source: https://github.com/sotetsuk/pgx/blob/main/README.md.
- `references\PureJaxRL_README.md`; SHA-256 `d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0`; source: https://github.com/luchris429/purejaxrl/blob/main/README.md.
- `references\Craftax_Baselines_ppo_rnn.py`; SHA-256 `5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011`; source: https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py.
- `references\Craftax_README.md`; SHA-256 `7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff`; source: https://github.com/MichaelTMatthews/Craftax/blob/main/README.md.

Only this batch's sanitized requests/designs/contracts and the five supplied snapshots were inputs. Shared input bytes were not modified. Local AST/data checks do not establish successful JAX execution, eager/JIT/vmap parity, legal-script replay, balance, learnability, strategic richness, non-isomorphism, adaptation or meta-RL transfer. Actual replay, animation, community adapter behavior and scientific evidence require the separately authorized execution and review stages.
