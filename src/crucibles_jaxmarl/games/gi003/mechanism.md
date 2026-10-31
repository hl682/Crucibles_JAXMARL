# Owned gates on a shared circular route
Status: authored proposal, not executed or validated.

Two couriers move on an eight-location ring. Agent zero controls the edge between locations one and two; agent one controls the edge between five and six. Either owner can toggle its own gate from anywhere, and the resulting open/closed state persists. Moves use old gate states, so a same-step toggle changes passage opportunities only on a later step.

Each courier chooses wait, clockwise move, anticlockwise move, toggle, or delivery. Delivery succeeds only at its currently designated target, then shifts that target to the opposite ring location. Positions may coincide. Co-location reduces successful delivery payment, introducing a congestion incentive without order-dependent collision resolution. Gate control can facilitate or obstruct either courier; everyone can also take the longer ring path.

Positions, targets, gates, clock, role identity, own previous action, own recent delivery and blocked-move indicators are observed. There is no hidden physical state. Each legal action remains available everywhere, with ineffective deliveries returning zero and blocked moves staying in place. Successful deliveries pay 0.5 or 0.8, toggles cost 0.03, and blocked movement costs 0.02; rewards are bounded within [-1,1].

This resembles generic navigation with persistent switches and congestion, rather than any asserted unprecedented mechanism. Structural distinctness, mechanism qualification, compiler behavior, and learning effectiveness remain unassessed. Initial positions use the explicit random key; both agents affect persistent positions and gate states. Terminal observations retain the final state after 32 transitions.
