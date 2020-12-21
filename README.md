# Noughts and Crosses Q-learning

A simple Q-learning Algorithm for the game "Noughts and Crosses".

The machine will learn how to play noughts and crosses by playing many games against itself and keeping track of 
"Q-values" for each "state". A state is characterised by the state of the board. Each state keeps track of the actions
that can be played from this state, and q-values associated with each action. The higher the q-value, the better the 
move is. Q-values are adjusted according to rewards: -1 if the move causes an immediate loss, 1 if it causes an 
immediate win, and 0 if it causes a draw. Q-values are also adjusted according to the q-values of the next states
to allow for long-term strategy.

This Q-learning algorithm can be applied to other games, simply be re-writing the "game-dependents" and the parameters. 
All other files are not specific to the game. 

