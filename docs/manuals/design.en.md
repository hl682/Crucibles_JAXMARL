# Design philosophy and classification rationale

## A world that remembers what people do

Crucibles is a collection of persistent multi-agent transition systems. An action can change a resource, route, machine, message, promise, or another actor's future options. The central design question is what agents learn when their opportunities depend on other agents' actions over time. Cooperation is a possible behavior, not an assumed outcome or a reward label attached to every action.

The library packages existing source identities. It does not claim that these mechanisms are unprecedented, pairwise strategically distinct, ecologically calibrated, or already validated as reinforcement-learning benchmarks. A different source hash establishes a distinct registered implementation, not a theorem of game-theoretic non-equivalence. Software execution checks and learning evidence are separate.

## Four design commitments

First, consequences persist. A valve stays set, a shipment remains in flight, equipment wears, a crop matures, a promise expires, or an injured partner loses options. An episode is more than a repeated static payoff matrix. This does not prove that every proposed mechanism creates interesting long-term behavior; that requires controlled experiments.

Second, interaction has an operational substrate. Cargo is actually carried, a tool has a location, a receiver has capacity, a bridge has occupancy, or two inputs must align. Helpful behavior matters because it changes reachable states or rewards. The rendering cannot invent motion, access, material or success that the kernel does not contain.

Third, information is part of the task. A private target, noisy scan, historical report and public statement are different things. A message may be stale, false, delayed, costly or lost in contention. Role observations define the policy's information contract. A full-state picture is an explanatory debugging view, never a substitute observation for decentralized evaluation.

Fourth, incentives can differ. Some tasks share delivery rewards; others include private preferences, fees, first-arrival bonuses, team opposition or local public-good costs. The words “commons” and “cooperative” do not imply identical returns. Read each original reward expression before summing, averaging, normalizing or shaping it.

## What the collection contains

The registry contains 256 active source identities: 180 spatial games GI072–GI251, 72 discovery-v2 abstract games GI000–GI071, and four distinct window-v1 abstract games V1_000–V1_003. The latter are not aliases for GI000–GI003. Two alternate historical sources are archived without increasing the active count.

The interface distinction describes representation. Spatial games expose native maps, entities, physical positions and visibility. Abstract games expose resource, queue, register, message or other symbolic state through their native observations. Some abstract mechanisms contain a ring or positions internally; the “abstract” label does not assert the absence of every positional variable. Their current display is a state schematic, not an invented terrain world. Population, horizon, map and action dimensions remain the original ones.

## Why a multi-axis classification

A kitchen may combine finite water, private orders, shared tools and a delivery bottleneck. A canal may combine physical navigation, a toll contract and a downstream ecological cost. Assigning either game to one exclusive narrative family would erase part of the task. The catalog therefore offers an interface filter and overlapping mechanism filters.

The seven mechanism labels are resource commons and externalities; information, sensing and communication; commitment, exchange and governance; complementary production and service; joint transport and shared access; team opposition and interception; and survival, protection and rescue. They describe rule ingredients. They do not rank difficulty, quality, novelty or learning success. A label is not a claim that a game belongs to a formally established equivalence class.

## How labels are assigned and audited

The assignment procedure is in tools/manual_content.py. It reads the unchanged manifest objective, declared action labels, scene rules and, for abstract games, a causal paragraph of mechanism.md. The deterministic cue lists produce candidate reading labels; each game's manual shows the exact field and source text supporting every assignment. These are transparent editorial retrieval rules, not an automatic semantic proof. A word such as “private” supports inspecting information structure, but does not by itself prove a communication bottleneck. A delivery objective supports a transport label, but not a need for joint force.

To revise a questionable label, inspect the cited field, legal_actions, observe and raw_step. Ask which state variable persists, which other role can affect it, what is observable, and how reward or feasibility changes. Amend the editorial classifier or add a documented exception; do not modify the original kernel to make the label fit. The source hashes and evidence excerpts make this review reproducible.

## The seven mechanism lenses

Resource commons and externalities concern shared stocks or transported burdens: water, pasture, pollution, heat, pollinators and maintenance. A useful question is whether an action shifts future opportunities or costs onto another role. Plot resource and burden trajectories alongside private returns. Resource-themed artwork alone is insufficient evidence.

Information, sensing and communication concern asymmetric facts and their movement: private demands, scans, cross-targets, decoders and aged reports. Record measurement time, sender, recipient, delivery and validity. Distinguish access to information from use of information, and prediction from causal influence. State access available only to the critic must not reach the actor.

Commitment, exchange and governance concern persistent obligations and rights: escrow, bonds, signatures, auctions, permits, leases, votes and treaties. Log posting, acceptance, activation, cancellation, expiry and settlement separately. A displayed promise is not a settled transfer. These are stylized mechanisms, not financial or legal advice or calibrated institutional models.

Complementary production and service concern ordered transformations and circulation: unfinished goods, manufacturing stages, tools, kitchens, trays and repair work. Measure valid finished goods and actual service, as well as starvation, contamination, waste and idle equipment. More intermediate activity need not mean higher task completion.

Joint transport and shared access concern physical or symbolic flow: loads, routes, gates, canals, queues, packets, lifts and shared control. Some games require simultaneous force, others sequential access or handoff. Read the specific rule before treating them as interchangeable coordination problems. Preserve geometry and capacity in every comparison.

Team opposition and interception concern opposed goals, captures, contested possession and delivery. Team cooperation is compatible with between-team competition. Keep opponent policies, team assignments and role information matched. A local word cue can also identify an interception option in an otherwise mixed-incentive game; inspect the reward contract before calling the entire game zero-sum.

Survival, protection and rescue concern exposure, injury, collapse, fire, warmth, flooding and evacuation. Protective work may create indirect value by keeping a partner able to act. Record deaths, losses and unresolved episodes rather than reporting only successful survivors. A hazard label does not imply physical realism.

## Time, observability and incentives are independent descriptors

Every manual reports the original agent count, discrete action count, observation width, horizon, observation regime and declared per-step reward bounds. These describe a task's contract rather than a difficulty score. Read the task's natural termination and raw done semantics as well as its time limit. A long horizon can hide sparse credit, while a short abstract episode can still require memory.

Similarly, a private objective does not necessarily imply opposed rewards; a shared score does not guarantee identical capabilities; and team competition does not remove within-team coordination. A planned study should explicitly state reward alignment, role heterogeneity, information scope and temporal dependencies rather than infer them from a theme label.

## Presentation and language policy

Interactive game controls, state labels and active GIF text are English. Each manual and the site have separate English and Chinese pages with a corresponding-page switch. Chinese prose explains the same game; literal English button labels, code identifiers and source evidence remain identifiable as technical references. Changing language never changes a game rule.

Every catalog card includes its own animated GIF. These new previews replay saved verification actions through the original JAX kernel and sample actual endpoints across the original episode. Spatial images show the complete native world; abstract images show native-state fields. The camera is explicitly full-state debugging. Pictures are explanatory previews of untrained legal-action trajectories, not demonstrations of a solved task. No interpolation invents a transition between captured endpoints. Timing is presentation speed, not a claim about simulated real time.

The earlier bespoke saved recordings and their Chinese instructional panels remain archival evidence. They are not active catalog illustrations. The complete original GI226/227 and GI229 endpoint/player package remains linked from the relevant manuals, with its failure and uncertainty history intact. Visual cleanup does not revise those conclusions.

## A reinforcement-learning study that can be interpreted

Begin with the per-game source identity, observation contract, legal masks, natural termination and original reward. Establish that reset, transition, masks, logging and terminal handling work for that exact interface. Choose a learner suited to the intended question; fixed-learner studies of environment mechanisms and studies of meta-learning update rules are distinct scientific questions. Required gradient or update-semantics checks remain necessary when studying those updates.

Use independent initialization keys and record the source hash, dependency versions, device, actual runtime and sampling cost. Use step_env when terminal observations must be retained without reset; step provides explicit outer autoreset and exposes raw terminal data in info. Do not feed browser projections, private partner state, test identities or future outcomes into an actor by accident. Batch equal-shaped game identities or use an explicitly specified padding scheme; do not shrink a population to fit a learner silently.

Match information, reward, compute and selection rules across baselines. Parameter sharing should account for distinct role capabilities; a centralized critic can be a declared training condition while actors remain role-filtered. Recurrence, communication, partner adaptation, reward shaping and centralized access are experimental factors, not default improvements. Change one justified mechanism at a time or use a stated factorial design. If changing a source is necessary, save it as an experimental variant with its own identity.

Freeze training, development and final-test partitions and checkpoint selection before reporting generalization. Group estimates by the independent training or source-pairing units actually randomized; episodes from one policy are not independent training replications. Report returns together with task-specific valid deliveries, resource health, failed commitments, losses, deadlocks and unresolved episodes. Include negative, zero, mixed, failed and indeterminate outcomes and the actual cost of producing them. No fixed score, percentage gain or seed count is an acceptance gate in this collection.

## Evidence limits and future changes

The v0.1.0 engineering archive records 256 native full-episode checks and 256 browser-control checks. Eager/JIT parity was exact for 179 identities; 77 had measured nonzero floating differences and remain numerically undetermined. The presentation replay receipts separately report differences against those saved reward traces without adding a new numerical tolerance. Neither set of checks establishes strategic depth, learning gain, transfer, human performance or scientific acceptance.

New mechanics, polished art, translated prose and classifier revisions should be versioned separately from source preservation. A future benchmark claim needs its own protocol and observed learning evidence. Existing Alien reproduction is an external case reference, not evidence for these games or a requirement to reproduce other Atari titles. No cluster deployment, training run, model download or new game generation is implied by publishing these manuals.
