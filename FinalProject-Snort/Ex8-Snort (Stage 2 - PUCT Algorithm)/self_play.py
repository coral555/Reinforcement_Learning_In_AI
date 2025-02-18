import random  # Importing the random module for random move selection
from SnortGame import SnortGame  # Importing the SnortGame class to simulate games

def generate_self_play_data(num_games=10000):
    """
    Generates training data by simulating self-play games.
    The function generates self-play data by simulating games of Snort with random moves and recording game states, visit counts, and winners.
    
    Parameters:
    num_games (int): Number of self-play games to generate data from.

    Returns:
    tuple: (states, visit_counts, winners)
        - states: List of encoded game states from all self-play games.
        - visit_counts: List of visit count distributions for each move.
        - winners: List of game outcomes (1 for Red win, -1 for Blue win, 0 for draw).
    """

    # Initialize lists to store game data
    states, visit_counts, winners = [], [], []
    
    # Loop to generate the specified number of games
    for _ in range(num_games):
        game = SnortGame()  # Create a new instance of the game
        state_history = []  # Stores the sequence of game states during the game
        visit_count_history = []  # Stores visit counts for moves taken

        # Play the game until it reaches a terminal state (win, lose, or draw)
        while game.status() == 'ongoing':
            state_history.append(game.encode())  # Encode and store the current game state
            
            legal_moves = game.legal_moves()  # Get all legal moves
            visit_count_vector = [0] * (game.size ** 2)  # Initialize a visit count vector for all board positions
            
            move = random.choice(legal_moves)  # Select a move randomly from the list of legal moves
            move_index = move[0] * game.size + move[1]  # Convert move coordinates to a linear index
            visit_count_vector[move_index] = 1  # Assign a visit count of 1 to the selected move
            
            visit_count_history.append(visit_count_vector)  # Store the visit count vector
            game.make_move(move)  # Apply the selected move to the game

        # Determine the final game outcome:
        # 1 if Red wins, -1 if Blue wins, 0 if the game is a draw
        final_winner = 1 if game.status() == 'R_wins' else -1 if game.status() == 'B_wins' else 0

        # Store the collected data
        states.extend(state_history)  # Store the recorded game states
        visit_counts.extend(visit_count_history)  # Store the visit counts for each state
        winners.extend([final_winner] * len(state_history))  # Assign the final winner to all recorded states

    # Return the collected data for training
    return states, visit_counts, winners
