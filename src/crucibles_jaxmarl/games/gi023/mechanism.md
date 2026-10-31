# Bidirectional messages under persistent private decoding rules

Candidate: astra-d2-00023. This is a parentless source proposal with no supplied predecessor, score, or feedback.

Each role is both a sender and a receiver. It observes its own private task bit and private decoder bit. Action 0 waits. Actions 1 and 2 transmit symbol zero or one to the other role, attaching the sender's current task bit as an initially hidden payload. Action 3 flips the agent's persistent decoder bit. Action 4 accepts its current incoming message when that message has aged for at least one transition. Reception applies the decoder to the old incoming symbol by addition modulo two, then compares the result with the hidden payload.

An agent chooses only one action per transition, so sending, accepting and changing its own decoder compete for time. The sender selects the symbol but cannot directly select or observe the receiver's decoder. The receiver controls the timing and interpretation of a stored message. Decoders persist and reinterpret even messages sent before a decoder change. Correct acceptance gives both participants a positive payoff; incorrect acceptance penalizes both, at different magnitudes.

A newly sent message has age zero. It must remain stored for a further transition before acceptance is legal. Sending another message can replace an unaccepted message, incurring an additional sender cost. If an old message is accepted while a fresh one arrives, the old one settles and the fresh one remains pending. Payloads are captured at transmission: later private task refreshes do not retroactively change a stored message. Private tasks refresh on outgoing acceptance or every fourth transition, using only the supplied step key.

A receiver has persistent fluency, which rises on correct acceptance, falls on mistakes, and also falls after changing its decoder. Fluency and message age scale correct-payment quality. This creates a cost for repeatedly changing convention and a loss from leaving messages unaccepted. Incorrect acceptance reveals the old payload after the decision, so feedback can inform later communication without providing the answer before acceptance.

The sixteen observation columns are time fraction, role sign, own task, own decoder, own fluency, visible incoming symbol or -1, incoming pending flag, incoming age fraction, outgoing pending flag, outgoing age fraction, last incoming success/failure feedback, last outgoing feedback, last accepted decoded bit or -1, last accepted true payload or -1, visible outgoing symbol or -1, and cumulative transmissions divided by 32. The other role's current task, decoder and fluency are private.

Correct recipient and sender payments have coefficients 0.65 and 0.35 times bounded quality; incorrect payments are -0.45 and -0.25. Sending, decoder changes and replacement add small costs. The final reward clips this finite sum to [-1, 1]. Both roles terminate at transition 32 and retain their terminal observations.

This is a generic sender-receiver coordination and signaling mechanism with asynchronous two-way messages, persistent private codebooks, replacement and action-dependent proficiency. It is not claimed to be unrelated to familiar communication games, structurally unique, eligible for the pool, successfully compiled or effective for learning. The author has not executed candidate code.
