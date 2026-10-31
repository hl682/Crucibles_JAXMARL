# Three Quays — game-v3-00100

Three merchants transport six individually tracked goods through a 9 by 9 yard with narrow market approaches. Each merchant starts with two goods of its own kind and needs two of the preceding kind: role 0 wants kind 2, role 1 wants kind 0, role 2 wants kind 1. Own depots and seller-specific escrow pads are separate physical locations. Nobody starts with the goods required for independent completion.

## Physical goods and commitments

Actions 0–4 wait or move. A merchant carries at most one active good. Action 5 picks a resting, nonescrow good on its cell only if it owns that good or the good was publicly gifted. While carrying, the same action installs a required-kind good at its own unfinished depot; otherwise it drops the good without changing ownership. Cargo coordinates follow carriers; installation consumes the good and increments that depot's counter.

Action 6 deposits carried cargo on the merchant's own escrow pad, provided it has no outstanding offer. The good stores the seller, target recipient and a three-tick nonwithdrawable timer. Goods already in escrow cannot be picked up. Action 7 toggles the next target between the other two merchants, but cannot revise an outstanding commitment. Action 8 withdraws a mature offer only when its seller returns to its own pad. The good remains there under its previous owner. Action 9 releases carried goods as public gifts; the next claimant becomes owner without having to give anything back.

Timers decline before actions. New commitments receive timer three. After movement and actions, settlement considers complete compatible bilateral cycles, then a compatible three-party cycle. Every leg must be present and mature, every intended recipient must receive its required kind, and recipients in a three-party cycle must be a permutation with no self-transfer. No partial leg transfers. A permitted withdrawal in that call removes its offer before matching. Title then changes, escrow clears, and goods remain on their actual seller pads rather than teleporting to the recipient.

The default recipient quotes form a three-way ring. Bilateral settlement is a supported rule, not a claim that it is immediately feasible from the initial ownership configuration. Gifts and previously moved ownership can change what compatible offers are available.

## Returns and strategic dependence

Every active transition costs every merchant one. A required-kind depot delivery gives four additional reward to its recipient and one to every merchant, including that recipient. Conditional settlement gives one to each new owner. Finishing all six deliveries gives three to everyone and ends naturally; otherwise the horizon is 120.

An absent third offer can hold two partners' already deposited goods in place. After maturity, a merchant can continue waiting, return to withdraw, or redirect its next offer; physical return travel has a cost. An unsecured gift removes the reciprocal condition and exposes the sender to nonreciprocation or another claimant. Goods occupy actual transport routes even after ownership changes. Merchants can prioritize their own deliveries, wait for a shared cycle, or physically obstruct a market approach. The relative value depends on partners' choices; no negotiated text or synthetic payoff dashboard substitutes for those rules.

All changing positions, title, carrier, commitment, timer, recipient quote and counters are public in the 104-column observation. The scene reads the same state. The paired author witness changes only merchant 2's deposit while holding the other merchants' commitments fixed; the intended contrast is complete settlement versus stalled offers, not an executed result.

## Declared invariants and prospective variations

Six original good identities are conserved: active goods plus installed-unit counts equal six. Each merchant carries at most one good and has at most one outstanding offer. A carried good is outside escrow. Active escrow cargo remains uncarried on the corresponding seller pad. Counters stay in [0,2], commitment timers in [0,3], and positions stay within the board. Ordinary movement preserves distinct merchant positions; goods may stack.

Different future depot needs, ownership allocations, transport corridors, commitment durations or allowed counterparties would alter which cycles and gift strategies matter. Such variants are prospective; this source is one fixed deterministic instance.

## Literal source evidence

**Ownership governs feasible pickup**

```python
L116:             available = (q["goods_active"] & (q["goods_carrier"] < 0) & (q["goods_escrow"] < 0)
L117:                          & ((q["goods_owner"] == i) | (q["goods_owner"] == -1))
L118:                          & jnp.all(q["goods_pos"] == q["agent_pos"][i], axis=1))
L119:             lift = (a[i] == 5) & ~has & jnp.any(available)
L120:             loose = jnp.argmax(available)
```

**Physical locked commitment**

```python
L124:             commit = (a[i] == 6) & has & at_pad & ~has_offer
L125:             q["goods_carrier"] = q["goods_carrier"].at[good].set(jnp.where(commit, -1, q["goods_carrier"][good]))
L126:             q["goods_escrow"] = q["goods_escrow"].at[good].set(jnp.where(commit, i, q["goods_escrow"][good]))
L127:             q["goods_timer"] = q["goods_timer"].at[good].set(jnp.where(commit, 3, q["goods_timer"][good]))
L128:             q["goods_target"] = q["goods_target"].at[good].set(jnp.where(commit, q["quote"][i], q["goods_target"][good]))
L129:             q["offer_event"] = q["offer_event"].at[i].set(commit)
```

**Withdrawal before matching**

```python
L133:             withdraw = (a[i] == 8) & at_pad & has_offer & (q["goods_timer"][offered_good] == 0)
L134:             q["goods_escrow"] = q["goods_escrow"].at[offered_good].set(jnp.where(withdraw, -1, q["goods_escrow"][offered_good]))
L135:             q["goods_target"] = q["goods_target"].at[offered_good].set(jnp.where(withdraw, -1, q["goods_target"][offered_good]))
L136:             q["withdraw_event"] = q["withdraw_event"].at[i].set(withdraw)
```

**Unsecured gift transfers control**

```python
L137:             gift = (a[i] == 9) & has
L138:             q["goods_carrier"] = q["goods_carrier"].at[good].set(jnp.where(gift, -1, q["goods_carrier"][good]))
L139:             q["goods_owner"] = q["goods_owner"].at[good].set(jnp.where(gift, -1, q["goods_owner"][good]))
L140:             q["gift_event"] = q["gift_event"].at[i].set(gift)
```

**Whole-cycle matching**

```python
L165:         ring = (jnp.all(ready) & ~jnp.any(settled) & jnp.all(target_counts == 1)
L166:                 & jnp.all(recipient != jnp.arange(3)) & jnp.all(kind == n["demand"][target_safe]))
L167:         settled = settled | ring
L168:         seller = jnp.maximum(n["goods_escrow"], 0)
L169:         transferred = (n["goods_escrow"] >= 0) & settled[seller]
```

**Title changes without teleportation**

```python
L172:         n["goods_owner"] = jnp.where(transferred, n["goods_target"], n["goods_owner"])
L173:         n["goods_escrow"] = jnp.where(transferred, -1, n["goods_escrow"])
L174:         n["goods_target"] = jnp.where(transferred, -1, n["goods_target"])
L175:         n["goods_timer"] = jnp.where(transferred, 0, n["goods_timer"])
L176:         n["goods_pos"] = jnp.where((n["goods_carrier"] >= 0)[:, None],
L177:                                    n["agent_pos"][jnp.maximum(n["goods_carrier"], 0)], n["goods_pos"])
```

**Private and common returns**

```python
L184:         rewards = (-jnp.ones((3,), dtype=jnp.int32) + 4 * n["delivery_event"].astype(jnp.int32)
L185:                    + deliveries + n["exchange_event"].astype(jnp.int32) + 3 * success.astype(jnp.int32))
L186:         return n, rewards.astype(jnp.float32)
```

## Author boundary and provenance

This package is authored, frozen and **not executed**. Only the assigned batch's eight design/request files, delivery requirements, six contract/interface files and five supplied official reference files were inputs. There was no candidate import, JAX/NumPy import, compiler, contract validator, simulator, training, SSH or network operation. The only local source check is Python AST parsing; JSON parsing and SHA-256 establish file syntax and identity, not runtime validity.

All dynamics and integer design fields use fixed-shape int32/bool arrays; observations and rewards are float32. The supplied reset and step keys are unused because this original instance is deterministic. The key is not duplicated into state. Natural completion has priority at the horizon. A terminal raw call preserves every state leaf and observation, keeps the clock unchanged, returns zero reward and preserves its cause flags. The outer training wrapper remains responsible for resetting.

Unavailable contextual intents and blocked movement are legal attempts with no effect. Out-of-range or masked actions become wait. Normal manual movement uses rotating role priority (clock modulo the role count), refuses occupied target cells and has no implicit swap. This is a public rule. Every script and paired branch in playthrough.json is UNEXECUTED_AUTHOR_HYPOTHESIS; it contains actions and intended differences only, with no invented rollout states, returns or successful-completion claims.

## Supplied reference use

The supplied official JaxMARL README informed the distinction between this fixed-array kernel and an external agent-keyed parallel API adapter with an __all__ done entry. The supplied Pgx README informed explicit legal-action and terminal-state metadata and the separation of caller keys from hidden state. The supplied Craftax README informed the goal of a persistent spatial world with real objects, resource use and visible consequences. No existing game map, benchmark implementation, game rules or upstream code was copied.

PureJaxRL and the supplied Craftax_Baselines PPO-RNN file were read as algorithm/integration references only. They are not game libraries and are not incorporated here. This package does not replace the frozen experimental learner or implement the coordinator's separate community adapter and replay runtime. All five input snapshots and their actual hashes are recorded in generation_record.json.

Reference attribution: JaxMARL (Rutherford et al., NeurIPS Datasets and Benchmarks 2024), Pgx (Koyamada et al., NeurIPS 2023), Craftax (Matthews et al., ICML 2024), and PureJaxRL/Discovered Policy Optimisation (Lu et al., NeurIPS 2022). The Craftax PPO-RNN integration source credits Chris Lu's PureJaxRL implementation. No license from those references is asserted to cover this new author's own rules or source.

## Scientific limits

The intended strategies and interventions are hypotheses from source reasoning. Static syntax and real future causal replay do not establish learnability, optimality, solvability, strategic richness, non-isomorphism, generalization or meta-learning transfer. Negative, failed, zero-effect and indeterminate validation outcomes must remain visible. No reward, success-count, percentage or seed-count acceptance gate is introduced. Rule, role and layout variations mentioned below are prospective design possibilities, not implemented sampled task distributions or measured results.
