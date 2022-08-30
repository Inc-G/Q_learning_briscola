"""Briscola environment and players. Contains nn for deep player"""
### Briscola environment and players
# 
import sys #for human player

import numpy as np
import pandas as pd
import tensorflow as tf
import environment as env

semi = ['bastoni', 'coppe', 'denari', 'spade']
numeri = ['2', '3', '4', '5', '6', '7', '8', '9', '0', '1']
my_points = [ 0, 10, 0, 0, 0, 0, 2, 3, 4, 11]
my_cards = pd.DataFrame({i:[1 for j in numeri] for i in semi}, index = numeri)
my_cards['points'] = my_points
sorted_cards = my_cards.sort_values(by = ['points'], ascending = False)


all_cards_in_strings = [i + j for i in numeri for j in semi]
all_cards_in_strings.append('None')
sorted_cards_in_string = sorted(all_cards_in_strings)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes = sorted_cards_in_string)

cards_in_string = [numero+seme for numero in numeri for seme in semi]

###General Player class          
class Player(env.Briscola_env):
    def __init__(self, cards_in_my_hand, briscola, is_first_player):
        super().__init__()
        self.is_first_player = is_first_player
        self.briscola = briscola
        self.hand = cards_in_my_hand
        
        #get cards seen
        self.cards_seen = set([card for card in cards_in_my_hand]+[briscola])

        self.player_1_pts = 0
        self.player_2_pts = 0
                
        self.initial_states = [None]*4 #needed for rnn
        
        
    def gain_info_after_a_hand(self, cards_received, card_played, cards_seen, player_1_wins,
                               player_1_pts, player_2_pts):
        '''
        Updates: cards_seen, hand, is_first_player
        '''
        self.update_cards_seen(cards_seen)
        self.update_my_hand(cards_received, card_played)
        self.is_first_player = player_1_wins
        self.update_points(player_1_pts, player_2_pts)
        
        
    def update_points(self, player_1_pts, player_2_pts):
        '''
        Updates the points after each round
        '''
        self.player_1_pts += player_1_pts
        self.player_2_pts += player_2_pts
        
        
    def update_cards_seen(self, cards):
        '''
        Returns: None
        Updates: self.cards_seen
        '''
        self.cards_seen.add(cards[0])
        self.cards_seen.add(cards[1])
        
        
    def update_my_hand(self, card_received, card_played):
        hand_withot_dropped_card = [my_card for my_card in self.hand if my_card != card_played]
        if not card_received == None:
            self.hand = hand_withot_dropped_card + [card_received]
            self.cards_seen.add(card_received)
        else:
            self.hand = hand_withot_dropped_card
                
                
    def new_game(self, cards_in_my_hand, briscola, is_first_player):
        '''
        Re-initializes the player 
        '''
        self.is_first_player = is_first_player
        self.briscola = briscola
        self.hand = cards_in_my_hand
        
        #get cards seen
        self.cards_seen = set([card for card in cards_in_my_hand]+[briscola]) 
        self.player_1_pts = 0
        self.player_2_pts = 0      
        self.initial_states = [None]*4 #needed for rnn


###Human player 
class HumanPlayer(Player):
    def __init__(self, cards_in_my_hand, briscola, is_first_player):
        super().__init__(cards_in_my_hand, briscola, is_first_player)
        
    def policy(self, epsilon, first_played_card = None, vs_human = False):
        if self.is_first_player:
            print('I am the first player')
        else:
            print('I am the second player and my opponent played', first_played_card)
        print('briscola:', self.briscola)
        my_hand = [card for card in self.hand]
        if len(my_hand) == 1:
            print('My points:', self.player_1_pts)
            print('Points of my opponent:', self.player_2_pts)
        print('hand:', my_hand)
        played_card = input("I play the following:")
        return played_card


###Deterministic "greedy" player
class DeterministicPlayer(Player):
    def __init__(self, cards_in_my_hand, briscola, is_first_player):
        super().__init__(cards_in_my_hand, briscola, is_first_player)
        
    def deterministic_policy_first_player(self):
        '''
        Plays the card with less points
        '''
        numbers_of_cards = [card[0] for card in self.hand]
        points = sorted_cards.loc[numbers_of_cards, 'points'].values
        min_points = min(points)
        
        cards_with_min_pts =[]
        for card in self.hand:
            if sorted_cards.loc[card[0], 'points'] == min_points:
                cards_with_min_pts.append(card)
        if len(cards_with_min_pts) > 0:
            
            cards_with_min_pts_without_briscola = []#if i can win without a briscola i do that
            for card in cards_with_min_pts:
                if card[1:] != self.briscola:
                    cards_with_min_pts_without_briscola.append(card)
            if len(cards_with_min_pts_without_briscola) > 0:
                card_with_min_points =  cards_with_min_pts_without_briscola                  
        return cards_with_min_pts[np.random.randint(len(cards_with_min_pts))]
        
    
    def deterministic_policy_second_player(self, first_played_card):
        '''
        First determines if we can win. If we can win the round, plays the card
        with less points that makes us win. Otherwise, plays the card with less points.
        '''
        my_victories = []
        for card in self.hand:
            if 'second_played_card' == self.which_card_wins_a_round(first_played_card, card):
                my_victories.append(card)
                
        if len(my_victories) == 0:
            return self.deterministic_policy_first_player()
        else:
            numbers_of_cards = [card[0] for card in my_victories]
            points = sorted_cards.loc[numbers_of_cards, 'points'].values
            min_points = min(points)
        
            cards_with_min_pts =[] 
            for card in my_victories:
                if sorted_cards.loc[card[0], 'points'] == min_points:
                    cards_with_min_pts.append(card)
            return cards_with_min_pts[np.random.randint(len(cards_with_min_pts))]
                
    
    def policy(self, epsilon, first_played_card = None, vs_human = False):
        if self.is_first_player:
            card = self.deterministic_policy_first_player()
            if vs_human:
                print('opponent plays ', card)
            return card
        else:
            card = self.deterministic_policy_second_player(first_played_card)
            if vs_human:
                print('opponent plays ', card)
            return card
        
###Random player

class RandomPlayer(Player):
    def __init__(self, cards_in_my_hand, briscola, is_first_player):
        super().__init__(cards_in_my_hand, briscola, is_first_player)
        
    def policy(self, epsilon, first_played_card = None, vs_human = False):
        card = self.hand[np.random.randint(len(self.hand))]
        if vs_human:
                print('opponent plays ', card)
        return card

###Deep player

##Model
class FinalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FinalLayer, self).__init__()
        
    def call(self, my_input, final_output):
        '''
        Returns a tensor of shape [batch_size, 3].
        Changes final_output assigning [1.,1.,1.] (resp.[0.,0.,0.]) if I have already won (resp. lost).

        final_output.shape == [batch_size, 3] and my_input.shape == [batch_size, number_of_rounds, features] 
        '''
        results = my_input[:,-1,-1] #1d array with 1. if pl_1 wins, 0. if it loses, 0.5 otherwise
        #1d array with 1. if pl_1 wins, 0. otherwise
        winning = tf.reshape(tf.floor(results), (results.shape[0],1))
        
        #1d array with 0. if pl_1 loses, 1. otherwise
        losing = tf.reshape(1- tf.floor(1-results), (results.shape[0],1))
        
        #needed for taking elementwise max-min.
        winning_tensor = tf.concat([winning, winning, winning], axis = 1)
        losing_tensor = tf.concat([losing, losing, losing], axis = 1)
        
        #Changes final_output assigning [1.,1.,1.] if I have already won
        first = tf.math.maximum(winning_tensor, final_output)
        
        #Changes final_output assigning [0.,0.,0.] lost
        second = tf.math.minimum(losing_tensor,first)
        return second


class MyModel(tf.keras.Model):
    '''
    To initialize set compute_prob_winning=False if the model estimates the number of points, set it to True if it estimates
    the probability of winning the game.
    
    The call has two parameters, initial_states=[None]*4 and my_return_sequences=False.
    - initial_states are the initial states for the gru layers.
    - my_return_sequences returns a list of the time-distributed outputs of the gru layers,
    and the time-distributed outputs of the dense layer + final layer.
    '''
    def __init__(self, compute_prob_winning=True, simplified=False):
        super(MyModel, self).__init__()
        self.compute_prob_winning = compute_prob_winning
        self.simplified = simplified
        
        self.gru1 = tf.keras.layers.GRU(200, return_sequences = True)
        self.gru2 = tf.keras.layers.GRU(200, return_sequences = True)
        if not simplified:
            self.gru3 = tf.keras.layers.GRU(200, return_sequences = True)
            self.gru4 = tf.keras.layers.GRU(200, return_sequences = True)
        
        if self.compute_prob_winning:
            self.dense = tf.keras.layers.Dense(3, activation = "sigmoid")
            self.final = FinalLayer()
            self.td1 = tf.keras.layers.TimeDistributed(self.dense)
            self.td2 = tf.keras.layers.TimeDistributed(self.final)
        else:
            self.dense = tf.keras.layers.Dense(3, activation = "tanh")
            self.td1 = tf.keras.layers.TimeDistributed(self.dense)        

    def call(self, state, initial_states=[None]*4, my_return_sequences=False):
        h_1 = self.gru1(state, initial_state=initial_states[0])#initial states has shape (batch_size, 200) 
        h_2 = self.gru2(h_1, initial_state=initial_states[1])
        if not my_return_sequences:
            if not self.simplified:
                h_3 = self.gru3(h_2, initial_state = initial_states[2])
                h_4 = self.gru4(h_3, initial_state = initial_states[3])
                final_output = self.dense(h_4[:,-1,:])

                if self.compute_prob_winning:
                    return [h_1[:,-1,:], h_2[:,-1,:], h_3[:,-1,:], h_4[:,-1,:]], self.final(state, final_output)
                else:
                    return [h_1[:,-1,:], h_2[:,-1,:], h_3[:,-1,:], h_4[:,-1,:]], final_output 
            else:
                final_output = self.dense(h_2[:,-1,:])

                if self.compute_prob_winning:
                    #the last 2 h-states are never used whene simplified == True
                    return [h_1[:,-1,:], h_2[:,-1,:], h_2[:,-1,:], h_2[:,-1,:]], self.final(state, final_output)
                else:
                    #the last 2 h-states are never used whene simplified == True
                    return [h_1[:,-1,:], h_2[:,-1,:], h_2[:,-1,:], h_2[:,-1,:]], final_output       
        else:
            if not self.simplified:
                h_3 = self.gru3(h_2, initial_state = initial_states[2])
                h_4 = self.gru4(h_3, initial_state = initial_states[3])
                tdense = self.td1(h_4)

                if self.compute_prob_winning:
                    #Apply FinalLayer (if I already won or lost replace the prob with 1. or 0 respectively)
                    #to each output, by reshaping the outputs.
                
                    reshaped_tdense = tf.reshape(tdense, (tdense.shape[1] * tdense.shape[0], tdense.shape[2]))
                    reshaped_input = tf.reshape(state, ( state.shape[1] * state.shape[0],1, state.shape[2]))
                    final_result = self.final(reshaped_input, reshaped_tdense)
                    return [h_1, h_2, h_3, h_4], tf.reshape(final_result, tdense.shape)
                else:
                    return [h_1, h_2, h_3, h_4], tdense

            else:
                tdense = self.td1(h_2)

                if self.compute_prob_winning:
                    #Apply FinalLayer (if I already won or lost replace the prob with 1. or 0 respectively)
                    #to each output, by reshaping the outputs.
                
                    reshaped_tdense = tf.reshape(tdense, (tdense.shape[1] * tdense.shape[0], tdense.shape[2]))
                    reshaped_input = tf.reshape(state, ( state.shape[1] * state.shape[0],1, state.shape[2]))
                    final_result = self.final(reshaped_input, reshaped_tdense)
                
                    #the last 2 h-states are never used whene simplified == True
                    return [h_1, h_2, h_2, h_2], tf.reshape(final_result, tdense.shape)

                else:
                    #the last 2 h-states are never used whene simplified == True
                    return [h_1, h_2, h_2, h_2], tdense
        
class MyModel_dense(tf.keras.Model):
    '''
    If the nn is approximating the probability of winning, initialize with compute_prob_winning = True,
    if it is approximating the (weighted) average of the number of points we make per round initialize with
    compute_prob_winning = False.
    
    The call has the parameter initial_states = [None]*4, just to make it compatible with MyModel.
    '''
    def __init__(self, compute_prob_winning = True):
        super(MyModel_dense, self).__init__()
        
        self.my_layers = [tf.keras.layers.Dense(200, activation = "tanh"),
                       tf.keras.layers.Dense(200, activation = "tanh")]
        
        if compute_prob_winning:
            self.my_layers.append(tf.keras.layers.Dense(3, activation = "sigmoid"))
            self.final = FinalLayer()
        else:
            self.my_layers.append(tf.keras.layers.Dense(3, activation = "tanh"))
        
        self.compute_prob_winning = compute_prob_winning

    def call(self, state, initial_states = [None]*4):
        current_state = tf.identity(state)
        state = state[:,-1,:] #to make it compatible with the input of MyModel
        
        for layer in self.my_layers:
            state = layer(state)
        
        if self.compute_prob_winning:
            return [None]*4, self.final(current_state, state)
        else:
            return [None]*4, state 


## Deep player
class DeepPlayer(Player):
    def __init__(self, cards_in_my_hand, briscola, is_first_player):
        super().__init__(cards_in_my_hand, briscola, is_first_player)
        self.model = MyModel()
        self.cards_not_seen = set(all_cards_in_strings)
        self.cards_not_seen.remove('None')
        self.cards_not_seen = self.cards_not_seen.difference(self.cards_seen)
        self.as_opponent = False
        
    def one_hot(self, is_first_player, hand , briscola,
                player_1_pts, player_2_pts, cards_seen, card_opponent = None):
        '''
        Gets one hot state for nn.
        '''
        if card_opponent == None:
            card_opponent = 'None'
        result1 = [1. if is_first_player else 0.]
        position = []

        for card in hand:
            position.append(sorted_cards_in_string.index(card))
        if len(position)<3:
            position = position + [40]*(3-len(position))
        
        card_1 = np.eye(41)[position[0]]
        card_2 = np.eye(41)[position[1]]
        card_3 = np.eye(41)[position[2]]
        briscola = np.eye(41)[sorted_cards_in_string.index(briscola)]

        pts = np.array([player_1_pts, player_2_pts])/120 if not self.as_opponent else np.array([player_2_pts, player_1_pts])/120
        if pts[0] > 0.5:
            winning = [1.]
        elif pts[1] > 0.5:
            winning = [0.]
        else:
            winning = [.5]
            
        card_opponent_oh = np.eye(41)[sorted_cards_in_string.index(card_opponent)]
        
        encode_seen_cards = np.zeros(41)
        for i in range(len(sorted_cards_in_string)):
            if sorted_cards_in_string[i] in cards_seen:
                encode_seen_cards[i] = 1
                        
        result = np.concatenate([result1, card_1, card_2, card_3, briscola, pts, card_opponent_oh, encode_seen_cards, winning ])
        return result[np.newaxis].astype('float32')
        
               
    #Auxuliary 1 -- see policy last rounds    
    def last_two_rounds(self, my_hand, opponent_hand, played_card):
        if played_card != None:
            best_option = ('s', - 200, 200) #(card, my pts, opp points)
            for card in my_hand:
                remaining_cards = [_ for _ in my_hand if _ != card]
                remaining_opp_cards = [_ for _ in opponent_hand if _ != played_card]
                _, _, _, _, player_1_wins, player_1_pts, player_2_pts = self.step(card, played_card, False, update_global_var = False)
                my_points = player_1_pts
                opp_points = player_2_pts
                _, _, _, _, _, player_1_pts, player_2_pts = self.step(remaining_cards[0], remaining_opp_cards[0], player_1_wins, update_global_var = False)
                my_points += player_1_pts
                opp_points += player_2_pts
                if my_points > best_option[1]:
                    best_option = [card, my_points, opp_points]
            return best_option
        else:
            best_option = ('s', -200, 200)
            for card in my_hand:
                local_option = self.last_two_rounds(opponent_hand, my_hand, card)
                if local_option[2] > best_option[1]:
                    best_option = (card, local_option[2],local_option[1])
            return best_option
        
    #Auxuliary 2 -- see policy last rounds
    def last_three_rounds(self, my_hand, opponent_hand, played_card):
        if played_card != None:
            best_option = ('s', - 200, 200) #(card, my pts, opp points)
            for card in my_hand:
                remaining_cards = [_ for _ in my_hand if _ != card]
                remaining_opp_cards = [_ for _ in opponent_hand if _ != played_card]
            
                _, _, _, _, player_1_wins, player_1_pts, player_2_pts = self.step(card, played_card, False, update_global_var = False)            
                my_points = player_1_pts
                opp_points = player_2_pts            
                if not player_1_wins:
                    opponent_best_option = self.last_two_rounds(remaining_opp_cards, remaining_cards, None)
                
                    if my_points + opponent_best_option[2] > best_option[1]:
                        #I make more points with this card so I switch
                        best_option = (card, my_points + opponent_best_option[2], opp_points + opponent_best_option[1])
                else:
                    next_best_option = self.last_two_rounds(remaining_cards, remaining_opp_cards, None)
                    if my_points + next_best_option[1] > best_option[1]:
                        #I make more points with this card so I switch
                        best_option = (card, my_points + next_best_option[1], opp_points + next_best_option[2])
            return best_option
        else:
            best_option = ('s', -200, 200)
            for card in my_hand:
                local_option = self.last_three_rounds(opponent_hand, my_hand, card)
                if local_option[2] > best_option[1]:
                    #I make more points with this card so I switch
                    best_option = (card, local_option[2],local_option[1])
            return best_option
    
    #top play for last 3 rounds (when we know opponent's hand) assuming we are playing vs top player.
    def policy_last_rounds(self, cards_seen, my_hand, pl_pts, briscola, played_card = None):
        cards_not_seen = set(all_cards_in_strings)
        cards_not_seen.remove('None')        
        opponent_hand = list(cards_not_seen.difference(cards_seen))
    
        if len(opponent_hand) < len(my_hand):
            opponent_hand.append(briscola)
        
        if len(my_hand) == 3:
            best_option = self.last_three_rounds(my_hand, opponent_hand, played_card)
        elif len(my_hand) == 2:
            best_option = self.last_two_rounds(my_hand, opponent_hand, played_card)
        else:
            best_option = (my_hand[0], - 200, 200)
        
        return best_option[0]
        
               
        
    def policy(self, epsilon, first_player_card = None, vs_human = False):
        
        oh_state = self.one_hot(is_first_player = self.is_first_player, hand = self.hand,
                                briscola = self.briscola, player_1_pts = self.player_1_pts,
                                player_2_pts = self.player_2_pts, cards_seen = self.cards_seen, 
                               card_opponent = first_player_card)
        initial_states, total_q_vals = self.model(oh_state[np.newaxis], self.initial_states)
        self.initial_states = initial_states
        q_vals = total_q_vals[0][:len(self.hand)]
        
        if np.random.rand() < epsilon:
            return self.hand[np.random.randint(len(self.hand))] 
        else:
            if len(self.cards_seen) < 37:
                card = self.hand[tf.argmax(q_vals).numpy()] 
                if vs_human:
                    print('opponent plays ', card)
                return card
            else:
                #pl knows opponent's hand
                card = self.policy_last_rounds(self.cards_seen, self.hand, self.player_1_pts, self.briscola, played_card = first_player_card)
                if vs_human:
                    print('opponent plays ', card)
                return card
      
