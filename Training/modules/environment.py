"""Briscola environment and players. Contains nn for deep player"""

### Briscola environment and players
import numpy as np
import pandas as pd
import tensorflow as tf
import time

semi = ['bastoni', 'coppe', 'denari', 'spade']
numeri = ['2', '3', '4', '5', '6', '7', '8', '9', '0', '1']
my_points = [ 0, 10, 0, 0, 0, 0, 2, 3, 4, 11]
my_cards = pd.DataFrame({i:[1 for j in numeri] for i in semi}, index = numeri)
my_cards['points'] = my_points
sorted_cards = my_cards.sort_values(by=['points'], ascending=False)


class Briscola_env:
    def __init__(self):
        self.points_player_1 = 0 #player 1 is the first player at round 1
        self.points_player_2 = 0
        self.cards_in_the_deck = np.array([numero+seme for numero in numeri for seme in semi])
        np.random.shuffle(self.cards_in_the_deck)
        self.round = 20
        self.index_top_card = 0
        
        #choose a briscola
        self.briscola = self.cards_in_the_deck[0] #pick a briscola
        self.index_top_card = 1
        self.is_briscola_on_the_table = True
    
    
    def draw_cards(self, number_cards):
        '''
        Draws number_cards from the deck.
        
        Returns: a list of length number_cards
        Updades: self.cards_in_the_deck
        '''
        if len(self.cards_in_the_deck)- self.index_top_card > 0:
            drawn = [ _ for _ in self.cards_in_the_deck[self.index_top_card : self.index_top_card + number_cards]]
            self.index_top_card = self.index_top_card + number_cards
            return drawn
        else:
            if self.is_briscola_on_the_table:
                self.is_briscola_on_the_table = False
                return [self.briscola]
            else:
                raise ValueError('Tried to draw from an empty deck..')
    
    
    def start_game(self):
        '''
        Deals 3 cards for each player
        
        Returns: (first_three_cards, second_three_cards), briscola
        Updates: self.cards_in_the_deck by removing the 6 cards sampled
        '''
        first_three_cards = self.draw_cards(3)
        second_three_cards = self.draw_cards(3)
        return (first_three_cards, second_three_cards), self.briscola
        
        
    def which_card_wins_a_round(self, first_played_card, second_played_card, passed_briscola = None):
        '''
        Determines which card wins a round
        
        Returns: either 'first_played_card' or 'second_played_card'
        Updates: nothing
        '''
        if passed_briscola == None:
            briscola = self.briscola[1:]
        else:
            briscola = passed_briscola[1:]
            
        if first_played_card[1:] == briscola and second_played_card[1:]!= briscola:
            return 'first_played_card'
        elif first_played_card[1:] != briscola and second_played_card[1:] == briscola:
            return 'second_played_card'
        elif first_played_card[1:] != second_played_card[1:]:
            return 'first_played_card'
        else:
            if sorted_cards.loc[first_played_card[0], 'points'] > 0 or sorted_cards.loc[second_played_card[0], 'points'] > 0:
                return 'first_played_card' if sorted_cards.loc[first_played_card[0], 'points'] > sorted_cards.loc[second_played_card[0], 'points'] else 'second_played_card'
            else:
                return 'first_played_card' if int(first_played_card[0]) > int(second_played_card[0]) else 'second_played_card'
    
    
    def pre_step(self, player_1, player_2, epsilon, vs_human = False):
        '''
        The player who plays second decides his action based on the other player's action
        
        Returns: card_player_1, card_player_2
        Updates: nothing
        '''
        if player_1.is_first_player:
            card_player_1 = player_1.policy(epsilon, None, vs_human)
            card_player_2 = player_2.policy(epsilon, card_player_1, vs_human)
        else:
            card_player_2 = player_2.policy(epsilon, None, vs_human)
            card_player_1 = player_1.policy(epsilon, card_player_2, vs_human)
        return card_player_1, card_player_2
    
    
    def step(self, card_player_1, card_player_2, order, update_global_var=True):
        '''
        order == True -> card_first_player is played first
        - Determines if player_1 wins a round
        - Updates the points and the cards in the deck

        - Draws 2 cards after the round if there are cards in the deck
        
        Returns: new_card_player_1, new_card_player_2, played_cards, is_end_game, player_1_wins
        Updates: - self.cards_in_the_deck if it deals from the deck
                 - self.points_player_1
                 - self.points_player_2
                 - self.round
        '''
        #Determines who wins and how many points
        [first_card, second_card] = [card_player_1, card_player_2] if order else [card_player_2, card_player_1]
        winner = self.which_card_wins_a_round(first_card, second_card)
        points = my_cards.loc[first_card[0], 'points'] + my_cards.loc[second_card[0], 'points']
        played_cards = (first_card, second_card)
        
        player_1_wins = winner == 'first_played_card' if order else not winner == 'first_played_card'
        
        is_briscola_on_the_table = self.is_briscola_on_the_table
        
        if is_briscola_on_the_table:
            #draws the new cards
            card_winner = self.draw_cards(1)
            card_loser = self.draw_cards(1)
        
        if player_1_wins:
            if update_global_var:
                self.points_player_1 += points
            if is_briscola_on_the_table: #deals the new cards
                new_card_player_1 = card_winner
                new_card_player_2 = card_loser
        else:
            if update_global_var:
                self.points_player_2 += points
            if is_briscola_on_the_table: #deals the new cards
                new_card_player_2 = card_winner
                new_card_player_1 = card_loser
                
        if update_global_var:
            self.round = self.round - 1
        is_end_game = self.round == 0
        
        player_1_pts = points if player_1_wins else 0
        player_2_pts = 0 if player_1_wins else points
                
        
        if is_briscola_on_the_table:
            return new_card_player_1[0], new_card_player_2[0], played_cards, is_end_game, player_1_wins, player_1_pts, player_2_pts
        else:
            return None, None, played_cards, is_end_game, player_1_wins, player_1_pts, player_2_pts
        
        
    def reset(self):
        self.points_player_1 = 0 #player 1 is the first player at round 1
        self.points_player_2 = 0
        np.random.shuffle(self.cards_in_the_deck) 
        self.round = 20
        self.index_top_card = 0
        
        #choose a briscola
        self.briscola = self.cards_in_the_deck[0] #pick a briscola
        self.index_top_card = 1
        self.is_briscola_on_the_table = True
        