# Dune Caravan Commons

Status: AUTHORED_NOT_EXECUTED. All gameplay, scripts and causal predictions below are source-derived author hypotheses, not measured results.

## Rules and physical dependence

The 96 by 60 block world contains 48 crews of sixteen people, divided into six connected dry-channel rows. Four starting tools enable engineering, intake maintenance, carrying and gate work; fixed profession labels do not prevent trading tools. Engineers start with two building blocks. Actions 5/8/9 collect two blocks, lay pipe underfoot and build a solid wall one cell east. Construction consumes one carried block. A wall is canceled without material loss if another actor moves onto its target during that same step. All people may travel through the connected corridors; long paths around a settlement wall remain available.

A pump supplies its actual cistern two cells east only if both intermediate pipe cells exist, an engineer operates (7), and a different tool-2 actor holds its intake clear (6) in that same tick. One actor cannot select both actions. Cistern capacity is twelve; each of six row aquifers holds at most 32 water. Simultaneous requests consume shared aquifer units in west-to-east geographical order. Aquifers recharge eight units every 48 ticks. Pump pressure displays the actual shared reserve.

Every sixteen ticks the wind side flips and unshaded intakes acquire sand. Three or more sand units bury the actual intermediate pipe cell, blocking travel and supply. Holding clear removes two sand and restores a buried pipe. A wall immediately north or on the windward side shades a pump. The wall remains a real movement obstacle and may interrupt signals. Gates on settlement walls open only when a nearby tool-4 actor staffs them (12); bypass travel remains possible.

A beacon transmits the actor's carried code (13). Only a corridor clear of walls and closed gates relays the old signal to the next crew east one step later. Reading the local received signal (17) stores it in the actor. Intake work records the encountered sand code. Messages do not grant remote state access: the actual carried code is all a recipient receives.

Cistern fill (10) withdraws one water into empty hands. Deliver (11) transfers it to a camel whose real x position advances among three cells every sixteen ticks; hydration changes over time. Water can instead rescue (16) an adjacent actor at stamina five or less, restoring six stamina and consuming the unit. Cargo/message handoff (15) and tool exchange (14) require adjacent consenting partners. Contested pickup, delivery and rescue use a deterministic lowest-index winner.

## Strategies, reward and scope

Crews can share scarce aquifer draw schedules or take upstream priority; invest in shade or clear frequently; staff gates or use detours; reserve water for caravan deliveries or worker rescue. Information relays require keeping their physical corridor open. None of these alternatives has measured strategic or learning evidence.

Every nonrest action costs 0.01; building or laying costs an additional 0.02. Delivery pays the actor 1 and all sixteen home members of the camel's crew 0.03. Rescue pays the rescuer 0.25 and recipient 0.1. A successful pump pays its crew members 0.005 each. Cumulative 96 caravan deliveries ends naturally; otherwise H=1024 truncates. Stamina models exertion; separate tool durability, camel agency and fluid simulation are not implemented. The camels and wind are environmental entities/processes, not extra controllable actors.

## Literal diagnostics

The first script pairs engineer 0 and tender 1, then courier 2 fills the cistern. The second moves engineer 0 west, builds a north shade wall, then rests. The paired hypothesis changes only tender 1 from clear to rest while engineer 0 operates, followed by identical rest ticks. It predicts cistern, reserve and later opportunity differences without claiming any measured effect.


## Native boundary and local information

Every listed actor is an independently selectable row of the same int32 action vector. Human or AI control uses that unchanged row; a frontend, keyboard binding, animation, camera, agent-keyed adapter and human usability have not been implemented or verified by this author. All actors remain active. Rest restores one stamina unit; exertion consumes one, with documented game-specific additions. Movement is cardinal; destinations occupied at the start of a step cannot be entered. Simultaneous claims for an empty destination favor the lowest actor index. Spatial interactions use start-of-step positions and Manhattan distance at most one to the nearest relevant object, unless a specific rule says otherwise. Tool exchange requires both adjacent actors to choose exchange and mutually select each other. Invalid or out-of-range actions become rest deterministically.

Each role receives the own-state prefix plus a 5x5 radius-two patch. The manifest lists every channel and normalization; the renderer's entity masks and map mask are derived from the same radius. Out-of-world patch cells are all -1. There is no omniscient policy feed. Agent IDs, fixed role and home crew are public metadata; current tools, cargo, stamina, messages and world attributes are visible only in the patch. Entity positions are used by movement, proximity, ecology or construction; none are layout-only proxies.

A nonterminal call advances the int32 clock once. Natural completion has priority over horizon truncation. A terminal raw call preserves every state leaf, observation and terminal cause, returns zero reward and all dones true, and does not reset. All boundary/state integers are int32; the supplied uint32 key is used only through JAX randomness and is not copied into state. State keys/shapes remain fixed. Rewards declare conservative [-8,8] bounds and observations [-1,1]; these are interface ranges, not scientific performance gates. Concurrency, bounds, terminal absorption, JIT/vmap, causal effects, throughput and numerical parity remain unverified.

## Source-line evidence

- candidate.py:103 'request =': pump needs simultaneous tender and engineering work
- candidate.py:105 'allocation =': upstream competing withdrawals share row reserves
- candidate.py:127 'buried =': wind, shade and physical pipe availability
- candidate.py:140 'transmitted =': wall and gate obstruction of delayed messages
- candidate.py:153 'rescued =': water rescues a different nearby actor

## Provenance and limits

The source is original for this batch and train split. Only current batch authority, four assigned design/request pairs, population/family direction, human-playability addendum and copied interface contracts/templates were read. No prior candidate, IR, result, training, heldout or memory was read; no remote query was made. Official repository references were allowed but not read or copied. The family inspiration comes solely from the supplied design direction: no commercial assets or upstream game implementation is included. PureJaxRL/Craftax baseline code is not used as a game or learner replacement.

The two literal scripts and the paired single-action hypothesis in playthrough.json are unexecuted diagnostics, not a controller restriction or proof of task success. Source-authored invariants and bounds await the independent sealed validator. No candidate import, Python candidate execution, JAX/NumPy computation, validator, training, replay, browser session, human play, accepted R0/M0, solvability, learning, strategic richness, non-isomorphism or transfer result is claimed. Zero, negative, failed, missing and indeterminate outcomes remain admissible.
