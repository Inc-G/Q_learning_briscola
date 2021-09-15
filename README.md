# deep_learning_briscola

Briscola is a classic italian card game. If you haven't heard of it, you can think of it as a card game where each card is worth some points, and at each round both you and your opponent play a card. If you win a round, you win the sum of the number of points of the cards played at that round. The goal is to have more points than your opponent by the end of the game. In this repository I train two neural networks to play briscola.

The (ideal) goal of this project is to have an ai that masters briscola. The actual goal is to have an ai that beats the deterministic player, the random player, and some previous versions of itself as often as possible.

In this project I trained a few models, below are the two most interesting.

## Estimate the probability of winning.

The function this model tries to approximate is the function that sends a state-action pair (s,a) to the probability of winning if at state s we perform action a (i.e. if we play card a). Below there is what math happens under the hood.

**TL;DR**: I use the Bellman equation with gamma = 1 and the reward being 1 if I win, 0 if I lose, 1/2 if it's a draw.


![Alt text](https://github.com/Inc-G/deep_learning_briscola/blob/main/Estimate_probability_of_winning.png?raw=true "Optional Title")

I trained both the GRU models and the dense model, and the deepest GRU model (4 GRU layers) outperforms all of them.

## Estimate the number of points.

The function this model tries to approximate is the function that sends a state-action pair (s,a) to the expectation of the discounted sum of the number of points I make. The math under the hood is very similar to the one above, I report below the salient steps. 

**TL;DR**:I use the Bellman equation with gamma = .8 and .85, and the reward being the number of points I win or lose at each hand.

![Alt text](https://github.com/Inc-G/deep_learning_briscola/blob/main/Bellman_eq.png?raw=true "Optional Title")


I trained both the GRU models and the dense model, and the dense model outperforms all of them. It also outperforms the best GRU model trained to estimate the probability of winning.

## Organization of the repository.

There are three notebooks and four moules.

### Modules:
#### briscola_players: 
There are four player classes: random player, deterministic player, human player, and deep player.

**Random player**: plays randomly.

**Deterministic player**: This is a _greedy_ player. If it plays first, it plays the card with less points. If it plays second and it can win an hand, it plays the card with less points among those which make it win, if it cannot win it plays the card with less points.

**Human player**: to play on the command line

**Deep player**: class that plays with a nn.

There are two neural networks, MyModel and MyModel_dense. The first one (with compute_prob_winning = True, simplified = False) is the architecture of the best model I trained to estimate the probability of winning. The second one (with compute_prob_winning = False) is the architecture of the best model I trained to estimate the number of points.

#### environment:

Contains the environment (i.e. briscola).

#### model_nest_states:
Gets next states from a batch of games, and encodes a game for the nn. For encoding a game, each card (including when I have no card) is encoded as a OH vector.

#### simulate_games:

Contains simulate_games_and_record_data which simulates a number of games and records the data in a pd.df. It also contains simulate_games which simulates the games without returning a pd.df but the ratio player_2_wins/number_of_simulations.




