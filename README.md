# Noughts and Crosses Q-learning

A simple Q-learning Algorithm for the game "Noughts and Crosses".

The machine will learn how to play noughts and crosses by playing many games against itself and keeping track of 
"Q-values" for each "state". A state is characterised by the state of the board. Each state keeps track of the actions
that can be played and attaches "q-values" to each action. The higher the q-value, the better the 
move. Q-values are adjusted according to rewards: -1 if the move causes an immediate loss, 1 if it causes an 
immediate win, and 0 if it causes a draw. Q-values are also adjusted according to the q-values of the next states
to allow for long-term strategy. 

At each state, the agent can either "explore" (choose an action at random) or "exploit" (choose the action with the 
highest q-value). Exploration is preferred at first because the agent hasn't come across the state very much. Once the 
agent has explored the state a lot, exploitation is preferred. The probability of exploration is given by epsilon, which 
decays as the agent comes across the state more times. See params.py for further explanation. 

This Q-learning algorithm can be applied to other games by adjusting the "game-dependents". 
All other files are not specific to Noughts and Crosses. 

For Noughts and Crosses, effective parameters are given in n_and_c_params.py. 
A strategy can be tested by putting it against a strategy that chooses actions at random. 
Over 10,000 such games, the q-learning strategy is shown to be very effective: it loses 96 times when playing as noughts 
and 0 times when playing as crosses. Crosses always start.

## Future improvements:

Exploit symmetry between states.
