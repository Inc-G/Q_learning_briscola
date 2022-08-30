"""Gets new states given current state. The main function is get_new_single_state_from_a_batch_of_games.
Encodes a game for nn. Main function is encode_a_game
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import time

semi = ['bastoni', 'coppe', 'denari', 'spade']
numeri = ['2', '3', '4', '5', '6', '7', '8', '9', '0', '1']

all_cards_in_strings = [i + j for i in numeri for j in semi]
all_cards_in_strings.append('None')
sorted_cards_in_string = sorted(all_cards_in_strings)
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes = sorted_cards_in_string)


def get_new_single_state_from_a_batch_of_games(batch_of_games):
    '''
    returns a np.ndarray of shape (batch size, 20, 1, #features for next states)
    '''    
    index = [0,1,2,3] + [7,8,9]+list(range(11,52))
    next_state = batch_of_games[:,1:,index]
    
    reshaped = np.reshape(next_state, (next_state.shape[0], next_state.shape[1], 1, next_state.shape[2]))
    return reshaped


def player_1_wins_or_player_2_wins(row):
    '''
    Auxiliary
    '''
    [pts_player_1, pts_player_2] = row
    if pts_player_1 > 0.5:
        return 1.
    elif pts_player_2 > 0.5:
        return 0.
    else:
        return 0.5


def encode_a_game(game):
    '''
    It takes a np.ndarray [(numb of games) * (numb of rounds), features]
    where we dropped the columns with the opponent's hand and the action from a single game.
        
    Returns the game encoded, of shape [batch_size * #rounds, encoded features].
    '''   
    is_first_player = game[:,0][...,np.newaxis].astype('float32')
    pl_1_hand_1 = game[:,1][...,np.newaxis]
    pl_1_hand_2 = game[:,2][...,np.newaxis]
    pl_1_hand_3 = game[:,3][...,np.newaxis]
    briscola = game[:,4][...,np.newaxis]
    points = game[:,5:7].astype('float32')
    opponent_card_ = game[:,7][...,np.newaxis]
    cards_seen = game[:,8:]
    
    hand_1_oh = mlb.fit_transform(pl_1_hand_1)
    hand_2_oh = mlb.fit_transform(pl_1_hand_2)
    hand_3_oh = mlb.fit_transform(pl_1_hand_3)
    briscola_oh = mlb.fit_transform(briscola)
    points_encoded = points / 120
    opponent_card_oh = mlb.fit_transform(opponent_card_)
    cards_seen_oh = mlb.fit_transform(cards_seen)
    who_wins = np.apply_along_axis(player_1_wins_or_player_2_wins, 1, points_encoded)[...,np.newaxis]
    
    return np.concatenate([is_first_player, hand_1_oh, hand_2_oh, hand_3_oh, briscola_oh,
                           points_encoded, opponent_card_oh, cards_seen_oh, who_wins], axis = 1)
    


def before_encoding_a_game(game):
    '''
    Reshape a single game before encoding it, for nn.
    
    Gets rid of the 'played card', ['pl 2 hand 1', 'pl 2 hand 2', 'pl 2 hand 3', 'played card'], which have
    indices [4,5,6,10].
    '''
    return np.delete(game, [4, 5, 6, 10], axis = -1)



