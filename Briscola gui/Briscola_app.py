import os
import posix

from flask import request
from flask import Flask, render_template
from flask import jsonify

import tensorflow as tf
import briscola_players as players
import environment as env

import numpy as np

app = Flask(__name__)

partita = env.Briscola_env()

player_1 = None
client_player = None
card_player_1 = 'Not played yet'

client_previous_card = 'Not played yet'
pl_1_previous_card = 'Not played yet'

_ = tf.keras.models.load_model('MLP_best_model')
best_model = players.MyModel_dense(compute_prob_winning = False)
best_model.compile()
best_model(np.linspace(1,100,250)[np.newaxis][np.newaxis])
best_model.set_weights(_.get_weights())

def string_to_card(card):
	res = [3, 3]
	if card[0] == '8':
		res[0] = 'Fante'
	elif card[0] == '9':
		res[0] = 'Cavallo'
	elif card[0] == '0':
		res[0] = 'Re'
	elif card[0] == '1':
		res[0] = 'Asso'
	else:
		res[0] = card[0]
	res[1] = card[1:]
	if card == 'Not played yet':
		return 'Not played yet'
	else:
		return res[0] + ' ' + res[1]

def card_to_string(string):
	card = string.split()
	res = [3, 3]
	if card[0] == 'Fante':
		res[0] = '8'
	elif card[0] == 'Cavallo':
		res[0] = '9'
	elif card[0] == 'Re':
		res[0] = '0'
	elif card[0] == 'Asso':
		res[0] = '1'
	else:
		res[0] = card[0]
	res[1] = card[1]

	return res[0] + res[1]


@app.route('/new_game_vs_det', methods = ['POST'])
def new_game_det():
	global partita
	partita.reset()

	global player_1
	global client_player
	global card_player_1
	global client_previous_card 
	global pl_1_previous_card

	card_player_1 = 'Not played yet'

	client_previous_card = 'Not played yet'
	pl_1_previous_card = 'Not played yet'

	first_hand, briscola = partita.start_game()
	
	if 0 == np.random.randint(2):
		coin_flip_player_1_is_first = True
	else:
		coin_flip_player_1_is_first = False

	player_1 = players.DeterministicPlayer(first_hand[0], briscola, coin_flip_player_1_is_first)
	client_player = players.HumanPlayer(first_hand[1], briscola, not coin_flip_player_1_is_first)

	if client_player.is_first_player:
		first_pl_message = 'I will be playing first'
	else:
		first_pl_message = 'I will be playing second'
		card_player_1 = player_1.policy(0, None, False)

	response = {
	'first_player':  first_pl_message,
	'my_hand':  [string_to_card(card) for card in client_player.hand],
	'briscola': string_to_card(client_player.briscola),
	'card_player_1': string_to_card(card_player_1),
	'client_previous_card' : 'Not played yet',
	'pl_1_previous_card' : 'Not played yet'
	}
	return jsonify(response)

@app.route('/New_game_vs_random', methods = ['POST'])
def new_game_rand():
	global partita
	partita.reset()

	global player_1
	global client_player
	global card_player_1
	global client_previous_card 
	global pl_1_previous_card

	card_player_1 = 'Not played yet'

	client_previous_card = 'Not played yet'
	pl_1_previous_card = 'Not played yet'

	first_hand, briscola = partita.start_game()
	
	if 0 == np.random.randint(2):
		coin_flip_player_1_is_first = True
	else:
		coin_flip_player_1_is_first = False

	player_1 = players.RandomPlayer(first_hand[0], briscola, coin_flip_player_1_is_first)
	client_player = players.HumanPlayer(first_hand[1], briscola, not coin_flip_player_1_is_first)

	if client_player.is_first_player:
		first_pl_message = 'I will be playing first'
	else:
		first_pl_message = 'I will be playing second'
		card_player_1 = player_1.policy(0, None, False)

	response = {
	'first_player':  first_pl_message,
	'my_hand':  [string_to_card(card) for card in client_player.hand],
	'briscola': string_to_card(client_player.briscola),
	'card_player_1': string_to_card(card_player_1),
	'client_previous_card' : 'Not played yet',
	'pl_1_previous_card' : 'Not played yet'
	}
	return jsonify(response)

@app.route('/New_game_vs_best', methods = ['POST'])
def new_game_best():
	global partita
	partita.reset()

	global player_1
	global client_player
	global card_player_1
	global client_previous_card 
	global pl_1_previous_card

	card_player_1 = 'Not played yet'

	client_previous_card = 'Not played yet'
	pl_1_previous_card = 'Not played yet'

	first_hand, briscola = partita.start_game()
	
	if 0 == np.random.randint(2):
		coin_flip_player_1_is_first = True
	else:
		coin_flip_player_1_is_first = False

	player_1 = players.DeepPlayer(first_hand[0], briscola, coin_flip_player_1_is_first)
	client_player = players.HumanPlayer(first_hand[1], briscola, not coin_flip_player_1_is_first)
	player_1.model = best_model

	#print('beginning of the game: ', player_1.hand)
	if client_player.is_first_player:
		first_pl_message = 'I will be playing first'
	else:
		first_pl_message = 'I will be playing second'
		card_player_1 = player_1.policy(0, None, False)

	response = {
	'first_player':  first_pl_message,
	'my_hand':  [string_to_card(card) for card in client_player.hand],
	'briscola': string_to_card(client_player.briscola),
	'card_player_1': string_to_card(card_player_1),
	'client_previous_card' : 'Not played yet',
	'pl_1_previous_card' : 'Not played yet'
	}
	return jsonify(response)


@app.route('/play_hand', methods = ['POST'])
def my_play_hand():
	global partita
	global player_1
	global client_player
	global card_player_1
	global client_previous_card 
	global pl_1_previous_card

	message = request.get_json(force = True)
	card_player_2 = card_to_string(message['my_card'])

	if card_player_1 == 'Not played yet':
		card_player_1 = player_1.policy(0., card_player_2, False)

	pl_1_previous_card = card_player_1
	client_previous_card =  card_player_2
		

	new_card_player_1, new_card_player_2, played_cards, is_end_game, player_1_wins, player_1_pts, player_2_pts = partita.step(card_player_1, card_player_2, player_1.is_first_player)

	player_1.gain_info_after_a_hand(new_card_player_1, card_player_1, played_cards, player_1_wins, player_1_pts, player_2_pts)

	client_player.gain_info_after_a_hand(new_card_player_2, card_player_2, played_cards, not player_1_wins, player_1_pts, player_2_pts)
	
	if client_player.is_first_player:
		first_pl_message = 'I will be playing first'
		card_player_1 = 'Not played yet'
	else:
		first_pl_message = 'I will be playing second'
		if len(player_1.hand) > 0:
			card_player_1 = player_1.policy(0, None, False)
		else:
			card_player_1 = '  The game is over'

	response = {
	'first_player':  first_pl_message,
	'my_hand':  [string_to_card(card) for card in client_player.hand],
	'briscola': string_to_card(client_player.briscola),
	'card_player_1': string_to_card(card_player_1),
	'client_previous_card' : string_to_card(client_previous_card),
	'pl_1_previous_card': string_to_card(pl_1_previous_card),
	'points_pl_1':  str(player_1.player_1_pts),
	'points_pl_2':  str(player_1.player_2_pts)
	}
	return jsonify(response)

@app.route("/")
def index():
	return render_template("Briscola.html")

if __name__ == '__main__':
	app.run(debug = True, host = '0.0.0.0', port = int(os.environ.get("PORT", '8080')))

