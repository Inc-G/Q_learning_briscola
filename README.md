# deep_learning_briscola

Briscola is a classic italian card game. In this repository I train two neural network to play briscola.

In the module briscola_players, there are four player classes: random player, deterministic player, human player, and deep player.

**Random player**: plays randomly.

**Deterministic player**: This is a _greedy_ player. If it plays first, it plays the card with less points. If it plays second and it can win an hand, it plays the card with less points among those which make it win, if it cannot win it plays the card with less points.

**Human player**: to play on the command line

**Deep player**: class that plays with a nn.

The (ideal) goal of this project is to have an ai that masters briscola. The actual goal is to have an ai that beats the deterministic player, the random player, and some previous versions of itself as often as possible.

In this project I trained a few models, below are the two most interesting.

## Estimate the probability of winning.

The function this model tries to approximate is the function that sends a state-action pair (s,a) to the probability of winning if at state s we perform action a (i.e. if we play card a).




The advantage is that the goal is very intuitive (i.e. the formula is easier to process than the one of the Bellman equation). The drawback is that... it is worse than the model which estimates the number of points. It might be because, while the random variable we use to approximate the q-function is unbaised, it _might_ have bigger variance than the one used to approximate the weighted sum of the points we will make.


![Alt text](https://github.com/Inc-G/deep_learning_briscola/blob/main/Estimate_probability_of_winning.png?raw=true "Optional Title")

