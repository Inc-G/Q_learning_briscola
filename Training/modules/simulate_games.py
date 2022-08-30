"""Module which contains simulate_games_and_record_data which simulates a number of games and records the data in a pd.df. It also contains the function simulate_games which simulates the games without returning a pd.df but the ratio player_2_wins/number_of_simulations"""

import numpy as np
import pandas as pd

semi = ['bastoni', 'coppe', 'denari', 'spade']
numeri = ['2', '3', '4', '5', '6', '7', '8', '9', '0', '1']
my_points = [ 0, 10, 0, 0, 0, 0, 2, 3, 4, 11]
my_cards = pd.DataFrame({i:[1 for j in numeri] for i in semi}, index = numeri)
my_cards['points'] = my_points
sorted_cards = my_cards.sort_values(by = ['points'], ascending = False)
sorted_cards

cards_in_string = [numero+seme for numero in numeri for seme in semi]


### Play and record matches

def play_first_round_for_prob(player_1, player_2, partita, epsilon = 0.1):
    '''
    Plays the first round of a series of matches. 
    
    Returns: a np.ndarray with column 0 and the following indices:
    who plays first, which card each player plays, each player's hands, each player's points.
    
    new_matches_df is a np.ndarray of previous matches.
    number_of_previous_games is the numb of prev games in the df new_matches_df.
    '''
    #each player plays a card
    card_player_1, card_player_2 = partita.pre_step(player_1, player_2, epsilon)
    
    is_first_player = [1. if player_1.is_first_player else 0.]
    
    opponent_card = 'None' if player_1.is_first_player else card_player_2
    
    current_state = is_first_player + list(player_1.hand) + list(player_2.hand) + [player_1.briscola] + [player_1.player_1_pts] + [player_1.player_2_pts]  + [card_player_1] + [opponent_card] + list(player_1.cards_seen) + ['None']*(40 - len(player_1.cards_seen))
    
    new_matches = np.array(current_state)[np.newaxis,...]
    
    #determine who wins, which cards they get back, is it the end of the game, how many points
    new_card_player_1, new_card_player_2, played_cards, is_end_game, player_1_wins, player_1_pts, player_2_pts = partita.step(card_player_1, card_player_2, player_1.is_first_player)
    
    #updates info after the hand is played
    player_1.gain_info_after_a_hand(new_card_player_1, card_player_1, played_cards, player_1_wins, player_1_pts, player_2_pts)
    player_2.gain_info_after_a_hand(new_card_player_2, card_player_2, played_cards, not player_1_wins, player_1_pts, player_2_pts)
    
    return new_matches
    


def play_a_round_for_prob(player_1, player_2, partita, new_matches, number_of_previous_games = 0, epsilon = 0.1):
    """
    Plays a round of a match, and saves: who plays first, which card each player plays, each player's hands,
    each player's points, cards seend my player_1.
    
    Returns the np.ndarray new_matches_df.
    
    new_matches_df is a np.ndarray of previous matches.
    number_of_previous_games is the numb of prev games in the df new_matches_df.
    """
    card_player_1, card_player_2 = partita.pre_step(player_1, player_2, epsilon)
    
    is_first_player = [1. if player_1.is_first_player else 0.]
    
    opponent_card = 'None' if player_1.is_first_player else card_player_2
    
    if len(player_1.hand ) == 3:
        current_state = is_first_player + list(player_1.hand)+ list(player_2.hand) + [player_1.briscola] + [player_1.player_1_pts] + [player_1.player_2_pts]  + [card_player_1] + [opponent_card] + list(player_1.cards_seen) + ['None']*(40 - len(player_1.cards_seen))
    elif len(player_1.hand ) == 2:
        current_state = is_first_player + list(player_1.hand) + ['None'] + list(player_2.hand) + ['None'] + [player_1.briscola] + [player_1.player_1_pts] + [player_1.player_2_pts]  + [card_player_1] + [opponent_card] + list(player_1.cards_seen) + ['None']*(40 - len(player_1.cards_seen))
    else:
        current_state = is_first_player + list(player_1.hand) + ['None', 'None'] + list(player_2.hand) + ['None', 'None'] + [player_1.briscola] + [player_1.player_1_pts] + [player_1.player_2_pts] + [card_player_1] + [opponent_card] + list(player_1.cards_seen) + ['None']*(40 - len(player_1.cards_seen))
    
    current_state = np.array(current_state)[np.newaxis,...]
    new_matches = np.concatenate([new_matches, current_state], axis=0)
    
    #determine who wins, which cards they get back, is it the end of the game, how many points
    new_card_player_1, new_card_player_2, played_cards, is_end_game, player_1_wins, player_1_pts, player_2_pts = partita.step(card_player_1, card_player_2, player_1.is_first_player)
    
    #updates info after the hand is played
    player_1.gain_info_after_a_hand(new_card_player_1, card_player_1, played_cards, player_1_wins, player_1_pts, player_2_pts)
    player_2.gain_info_after_a_hand(new_card_player_2, card_player_2, played_cards, not player_1_wins, player_1_pts, player_2_pts)
        
    if partita.round == 0:
        is_first_player_end = [1. if player_1.is_first_player else 0.]
        final_state = is_first_player_end + ['None']*6 + [player_1.briscola] + [player_1.player_1_pts] + [player_1.player_2_pts] + [None] + ['None'] + list(player_1.cards_seen) + ['None']*(40 - len(player_1.cards_seen))
        final_state = np.array(final_state)[np.newaxis,...]
        new_matches = np.concatenate([new_matches, final_state], axis=0) 
    return new_matches
        
        
def coin_flip():
    return 0 == np.random.randint(2)
        
    
def simulate_games_and_record_data(number_of_simulations, player1, player2, partita, epsilon = 0.1, pl_1_model = None, pl_2_model = None):
    '''
    Simulates number_of_simulations matches between player1 and player2.
    
    Returns: a pd.DataFrame with data from each hand in each game. The data is:
    who plays first, which card each player plays, each player's hands, each player's points.
    '''

    ## Plays first game to initialize players

    first_hand, briscola = partita.start_game()
    coin_flip_player_1_is_first = coin_flip()
    player_1 = player1(first_hand[0], briscola, coin_flip_player_1_is_first)
    player_2 = player2(first_hand[1], briscola, not coin_flip_player_1_is_first)
    
    if pl_1_model != None:
        player_1.model = pl_1_model
    if pl_2_model != None:
        player_2.model = pl_2_model
        player_2.as_opponent = True
        
    new_matches_df = play_first_round_for_prob(player_1, player_2, partita, epsilon)

    while partita.round > 0:
        new_matches_df = play_a_round_for_prob(player_1, player_2, partita, new_matches_df, 0, epsilon)
    partita.reset()
    
    ## Play the rest of the games
    for _ in range(1, number_of_simulations):
        start_a_game(partita,player_1,player_2)

        while partita.round > 0:
            new_matches_df = play_a_round_for_prob(player_1, player_2, partita, new_matches_df, number_of_previous_games =  _, epsilon = epsilon)
            
        partita.reset()
    return new_matches_df


def start_a_game(partita,player_1,player_2):
    """
    Auxiliary function
    """
    first_hand, briscola = partita.start_game()
    coin_flip_player_1_is_first = coin_flip()
    player_1.new_game(first_hand[0], briscola, coin_flip_player_1_is_first)
    player_2.new_game(first_hand[1], briscola, not coin_flip_player_1_is_first)



def put_in_df(matches):
    '''
    puts a game into df to make it more readable
    '''
    result = pd.DataFrame(matches, columns = ['is first to play', 'pl 1 hand 1', 'pl 1 hand 2', 'pl 1 hand 3', 'pl 2 hand 1', 'pl 2 hand 2', 'pl 2 hand 3', 'briscola', 'pt player 1', 'pt player 2',  'played card', 'opponent card'] + list(range(40)))
    return result


def get_points(my_games):
    '''
    Returns the points gained or lost at each hand of a match.
    
    Input: the output of simulate_games_and_record_data.
    Output: np ndarray of shape(# games, 20), where at entry (i, j) there is the number of points pl_1 gained or lost
    at hand j of game i
    '''
    my_points = my_games[:,[8,9]]
    reshaped_pts = np.reshape(my_points, (int(my_points.shape[0]/21), 21, 2)).astype(float)
    pl_1_points_per_match = reshaped_pts[:,:,0]
    points_gained_pl_1 = np.diff(pl_1_points_per_match)
    
    pl_2_points_per_match = reshaped_pts[:,:,1]
    points_gained_pl_2 = np.diff(pl_2_points_per_match)
    return points_gained_pl_1 - points_gained_pl_2

#---

def play_a_round(player_1, player_2, partita, epsilon = 0.01, vs_human = False):
    card_player_1, card_player_2 = partita.pre_step(player_1, player_2, epsilon, vs_human)
    new_card_player_1, new_card_player_2, played_cards, is_end_game, player_1_wins, player_1_pts, player_2_pts = partita.step(card_player_1, card_player_2, player_1.is_first_player)
    player_1.gain_info_after_a_hand(new_card_player_1, card_player_1, played_cards, player_1_wins, player_1_pts, player_2_pts)
    player_2.gain_info_after_a_hand(new_card_player_2, card_player_2, played_cards, not player_1_wins, player_1_pts, player_2_pts)
    
    


def simulate_games(number_of_simulations, player1, player2, partita, 
    pl_1_model=None, pl_2_model=None, vs_human=False):
    '''
    simulates number_of_simulations matches between player1 and player2
    
    Returns: player_2_wins/number_of_simulations
    '''
    opponent_wins = 0
    first_hand, briscola = partita.start_game()
    coin_flip_player_1_is_first = coin_flip()
    player_1 = player1(first_hand[0], briscola, coin_flip_player_1_is_first)
    player_2 = player2(first_hand[1], briscola, not coin_flip_player_1_is_first)
    
    if pl_1_model != None:
        player_1.model = pl_1_model
    if pl_2_model != None:
        player_2.model = pl_2_model
        player_2.as_opponent = True

    while partita.round > 0:
        play_a_round(player_1, player_2, partita, epsilon = 0., vs_human = vs_human)
       
    if player_2.player_2_pts >=  player_2.player_1_pts:
        opponent_wins = opponent_wins + 1
        
    partita.reset()
    for _ in range(1,number_of_simulations):
        start_a_game(partita,player_1,player_2)

        while partita.round > 0:
            play_a_round(player_1, player_2, partita, epsilon = 0., vs_human = vs_human)
            
        if player_2.player_2_pts >=  player_2.player_1_pts:
            opponent_wins = opponent_wins + 1
        partita.reset()
    return opponent_wins/number_of_simulations


def how_many_lost_games(games):
    '''
    Returns the number of games pl 1 lost
    
    Input: the output of simulate_games_and_record_data
    '''
    number_of_games = games.shape[0]/21
    a = np.linspace(0,games.shape[0],int(number_of_games) +1) + 20.
    b = a[:-1].astype(int)
    finals = games[b,:]
    final_points = finals[:,[8,9]]
        
    return (number_of_games - final_points[final_points[:,0] >= final_points[:,1]].shape[0])/ number_of_games
    
    