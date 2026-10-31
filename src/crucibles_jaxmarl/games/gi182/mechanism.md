# Rhythm Foundry — original source-only mechanism

Status: AUTHORED_NOT_EXECUTED. This document describes source semantics and prospective hypotheses. No candidate import, reset, observe, mask, step, rollout, numerical library, rendering, training or seed search was performed. Static syntax/schema/whitelist checks, when recorded, are a different evidence class from R0 or game acceptance.

Three workers inhabit the actual 7 × 7 map. Outer walls and the center wall block movement. Workers begin at (2,1), (3,2), (2,5); the first holds one ore atom and the second one flux atom. The ore bin at (1,1) and flux bin at (1,5) each hold two more atoms. Furnaces at (3,1) and (3,5), shipping at (5,1), and waste at (5,5) are physical interaction locations. Workers may occupy station tiles; only workers block other workers.

One carried lot per worker is encoded as 0 empty, 1 ore, 2 flux, 3 north good, 4 south good, 5 ore-only slag, or 6 ore-and-flux slag. Each idle furnace has one ore slot and one flux slot. Deposit is possible within Manhattan distance one. Fire requires ore and locks the current composition; it is permitted without flux, with a persistent material consequence. A new fire sets stage 1 and timer 2 without ticking that new timer. Two subsequent transitions produce either a stage-2 cooling lot with timer 2, or ore-only slag if flux was missing. Quenching within distance one spends one of that furnace's two finite water charges and produces a good. An unquenched cooling lot expires after two subsequent transitions into two-atom slag. Quenching resolves before expiry on the final available transition. No operation repairs slag or adds flux after firing.

Action 5 collects the lowest-index eligible source: a bin must be directly underfoot, while finished goods/slag require distance at most one from the furnace. Taking a furnace lot clears its ingredient slots and resets its stage/timer. Action 9 transfers the unchanged physical lot to an adjacent empty worker whose effective action is wait; lowest cyclic priority selects a recipient. Shipping converts good into an immutable delivery count. The waste dock accepts any carried nonempty lot and increments discarded ore/flux atoms. Delivery and waste never replenish stock.

Movement uses the input state's occupied cells, forbids swaps and following into vacated cells within the same transition, and then resolves competing empty destinations by cyclic actor priority (actor minus clock modulo three). Illegal or out-of-range actions become wait. After movement, worker actions resolve in that same cyclic order with current stock/slot checks. An action admitted by the old mask can therefore lose a contested resource without spending it. Timer aging then uses the input stage and the still-current stage, which prevents a newly fired lot from aging or a quenched lot from expiring. Terminal handling finally applies.

The material identities are retained through processing rather than created by an event counter. Let c_ore map carried codes to [0,1,0,1,1,1,1], and c_flux to [0,0,1,1,1,0,1]. The source implies:
- 3 = ore-bin stock + sum(c_ore of all carries) + sum(furnace ore slots) + total shipped goods + discarded ore.
- 3 = flux-bin stock + sum(c_flux of all carries) + sum(furnace flux slots) + total shipped goods + discarded flux.
These are source-derived conservation claims awaiting numerical validation. No slag is silently deleted; water only decreases on an accepted quench. Workers remain at unique walkable positions; furnace slots remain binary, stages 0–4, timers 0–2, and carry codes 0–6 under source transitions.

Every shipment gives its worker 2 if the good's furnace matches that worker's private binary order, otherwise 1. Every worker also receives 0.25 per shipment in that joint transition. Thus the declared per-step reward bound [0,3] contains the source-derived maximum 2.75. Private orders are sampled only from the supplied reset key and remain fixed. Other workers observe physical lots and public shipment counts, not a recipient's preference or realized private utility.

Possible strategies depend on partners: divide ore and flux collection; establish a timed handoff from a busy quencher to a shipper; reserve scarce water for a preferred furnace; or start an ore-only charge that blocks a partner's desired complete lot. Rushing shipment versus retaining inputs for another furnace changes the next partner's feasible actions. These are plausible alternatives, not proven equilibria, optimal policies, learning results or measures of strategic richness.

The literal paired hypothesis changes only worker 1's first flux deposit. After the common subsequent firing and two heating transitions, worker 0 is hypothesized to face a quench opportunity in one branch and slag in the other. The scripts and all seed choices are original literals, not search products. The identical downstream quench probe can be invalid in one branch and is explicitly normalized to wait. Its magnitude, actual legality and causal effect remain unmeasured.

The flat observation is 76 public columns followed by own order and own index, O=78. All render entities bind actual positions/active/visibility and state attributes. All physical entities are public. No private order field is an entity attribute or event; renderers must show only the selected role's observation row. Stages and timers represent actual processing, and carry codes represent real conserved lots. No rendered movement is invented.

Natural completion means all bins, carried lots and furnace contents are empty, including a run that discarded all material. Stranded flux or an unfinished lot may instead reach H=96. Natural completion takes priority on the horizon transition. A terminal raw call is absorbing, with unchanged clock/state/observation, zero reward, all dones true and preserved terminal cause; raw_step never resets.

Small integer fields use int32; observations/rewards use float32 and flags bool. The external uint32 key is not duplicated in state. There is no adapter, training replacement or upstream copied code in this package. The five official supplied references were byte/hash checked only, not semantically consulted. This batch's controlled design/contracts are the semantic inputs. Originality here means authorship of this implementation, not proven novelty or non-isomorphism. Runtime shape agreement, eager/JIT/vmap parity, solvability, causal interventions, replay, animation, learning and transfer all remain unproven.

## Literal source evidence

- candidate.py:23: `"carry": jnp.array([1,2,0], jnp.int32),`
- candidate.py:123: `loading = (a[i] == 6) & jnp.any(deposit[i])`
- candidate.py:132: `cooling = (a[i] == 8) & jnp.any(quench[i])`
- candidate.py:151: `heating = (state["stage"] == 1) & (s["stage"] == 1)`
- candidate.py:154: `hot_end = heating & (s["timer"] == 0)`
- candidate.py:158: `s["stage"] = jnp.where(cool_end,4,s["stage"])`
- candidate.py:162: `s["natural"] = (jnp.sum(s["stock"]) == 0) & jnp.all(s["carry"] == 0) & (`
- candidate.py:165: `s = jax.tree.map(lambda new, old: jnp.where(state["done"],old,new),s,state)`

Interface anchors from static AST: reset at candidate.py:12–44; legal_actions at candidate.py:72–77; observe at candidate.py:79–85; raw_step at candidate.py:87–171.

The literal anchors and declared shapes were inspected statically. They are not execution traces or a proof that the implementation satisfies these claims under JAX.
