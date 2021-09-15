# deep_learning_briscola

Briscola is a popular italian card game. In this repository I train two neural network to play briscola (specifically, to beat two hard-coded briscola players).

In the module briscola_players, there are four player classes: random player, deterministic player, human player, and deep player.

**Random player**: plays randomly.

**Deterministic player**: This is a _greedy_ player. If it plays first, it plays the card with less points. If it plays second and it can win an hand, it plays the card with less points among those which make it win, if it cannot win it plays the card with less points.

**Human player**: to play on the command line (or in the notebook)

**Deep player**: class that plays with a nn.

The (ideal) goal of this project is to have an ai that masters briscola. The actual goal is to have an ai that beats the deterministic player, the random player, and some previous versions of itself as often as possible.

In this project I trained a few models, below are the two most remarkable.

## Gru model.

This is a model 


![Alt text](https://github.com/Inc-G/deep_learning_briscola/blob/main/Bellman_eq.png?raw=true "Optional Title")

