# deep_learning_briscola

[Briscola](https://en.wikipedia.org/wiki/Briscola) is a classic italian card game. You play with 40 cards and each card is worth some points. At each round both you and your opponent play a card. If you win a round, you win the sum of the points of the cards played at that round. The goal is to have more points than your opponent by the end of the game. In this repository I train two neural networks to play briscola.

The (ideal) goal is to have an ai that masters briscola. The actual goal is to have an ai that beats two hard-coded players (the deterministic player and the random player), and some previous versions of itself as often as possible.

In this project I trained a few models, below are the two most interesting.

## Estimate the probability of winning:

The function the neural network tries to approximate sends a state-action pair (s,a) to the probability of winning if at state s we perform action a (i.e. if we play card a). Below there is what math happens under the hood.

**TL;DR**: I use the Bellman equation with gamma = 1 and the reward 0 unless I won the game (in which case it is 1), or it is the last hand and it's a draw (in which case it is 1/2).


![Alt text](https://github.com/Inc-G/deep_learning_briscola/blob/main/Estimate_probability_of_winning.png?raw=true "Optional Title")

I trained both a GRU models and a MLP model, and the deepest GRU model (4 GRU layers + dense layer) outperforms all of them.

## Estimate the number of points:

The function the neural network tries to approximate sends a state-action pair (s,a) to the expectation of the discounted sum of the number of points I make. The math under the hood is very similar to the one above, I report below the salient steps. 

**TL;DR**:I use the Bellman equation with gamma = .8 and .9, and the reward being the number of points I win or lose at each hand.

![Alt text](https://github.com/Inc-G/deep_learning_briscola/blob/main/Bellman_eq.png?raw=true "Optional Title")

I trained both a GRU models and a MLP model, and the dense model (3 dense layers with activation tanh) outperforms all of them. 

## Results:
All the models I trained have an average winning rate vs a random player between 80 and 90%, and vs the deterministic "greedy" player between 70 and 80%.

The first model I trained was the GRU model, with the neural network approximating the probability of winning. After I trained a simpler MLP model with the neural network approximating the (discounted) sum of the expected number of points, and this model beated the previously trained GRU model around 60% of the times. Then I fine-tuned better the simpler MLP model (gamma = .9 seems the best choice), to get the MLP model in the folder MLP_best_model.

Below there is the window = 10 rolling average of the fraction of games losts for the GRU best player, when playing vs another model previously trained, vs the deterministic "greedy" player and vs the random player. The results below are after 58K steps of gradient descent.
![Alt text](https://github.com/Inc-G/deep_learning_briscola/blob/main/metrics/Final%20GRU%20-%20rolling%20lost%20games.png?raw=true "Optional Title")
Below there is the window = 10 rolling average of the fraction of games losts for the MLP best player, when playing vs another model previously trained (in this case, the first MLP that beated the GRU player), vs the deterministic "greedy" player and vs the random player. The results below are after 75K steps of gradient descent.
![Alt text](https://github.com/Inc-G/deep_learning_briscola/blob/main/metrics/Final%20MLP%20-%20rolling%20lost%20games.png?raw=true "Optional Title")

In the folder metrics you can find similar graphs for when the first MLP player beated the GRU player.

### Further (possible) improvements:

Probably longer training will lead to better results (in the graph above, it looks like the MLP model is still _slowly_ improving). I suspect that a fine-tuning of the GRU model would outperform the MLP model (however, training and consequently hyperparameters tuning  takes much longer for the GRU model).

One could try different reinforcement learning algorithms (actor-critic?).

## Organization of the repository.

There are three notebooks and four moules.

### Modules:
#### briscola_players: 
There are four player classes: random player, deterministic player, human player, and deep player.

_Random player_: plays randomly.

_Deterministic player_: This is a _greedy_ player. If it plays first, it plays the card with less points. If it plays second and it can win an hand, it plays the card with less points among those which make it win, if it cannot win it plays the card with less points.

_Human player_: to play on the command line

_Deep player_: class that plays with a nn.

There are two neural networks, MyModel and MyModel_dense. The first one (with compute_prob_winning = True, simplified = False) is the architecture of the best model I trained to estimate the probability of winning. The second one (with compute_prob_winning = False) is the architecture of the best model I trained to estimate the number of points.

#### environment:

Contains the environment (i.e. briscola).

#### model_next_states:
Gets next states from a batch of games, and encodes a game for the nn. For encoding a game, each card (including when I have no card) is encoded as a OH vector.

#### simulate_games:

Contains simulate_games_and_record_data which simulates a number of games and records the data in a pd.df. It also contains simulate_games which simulates the games without returning a pd.df but the ratio player_2_wins/number_of_simulations.

### MLP_best_model:
The best model's weights.

### notebooks:
The notebooks with a sample training loop.

### metrics:
Some pictures.

### Briscola gui:

In this folder there is a _very basic_ flask application + html page to play on your browser vs the deterministic model, vs the random model, and vs the best MLP model I trained.

To play, you need Flask 2.0.1, tensorflow >= 2.4.0, numpy, pandas, scikit-learn. The server-side is in Briscola_app.py,
the client-side is the html page Briscola.html in the folder "templates". You might want to (1) have the model weights and the modules briscola_players.py and environment.py in the same folder where Briscola_app.py is and (2) if you use Windows you might need to change the localhost address from 0.0.0.0 to 127.0.0.1 in Briscola.html. A sample of the html page is below (the ui can be improved :)) 




https://user-images.githubusercontent.com/55004390/133670961-268abb62-799f-4081-a4a4-7ba87d941f18.mp4


