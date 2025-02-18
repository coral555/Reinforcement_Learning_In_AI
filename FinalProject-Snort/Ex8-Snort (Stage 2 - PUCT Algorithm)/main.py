# Import required classes and functions
from SnortGame import SnortGame  # Class that defines the rules and mechanics of the Snort game
from PUCTPlayer import PUCTPlayer  # Class representing a player using the PUCT algorithm
from GameNetwork import GameNetwork  # Class representing a neural network for learning the game
from self_play import generate_self_play_data  # Function to generate self-play game data for training the network

def main():
    """
    Main function that manages the training and gameplay process.
    The code trains a neural network for the Snort game using the PUCT algorithm and then plays the game.
    """

    # Step 1: Generate training data through self-play
    states, visit_counts, winners = generate_self_play_data(num_games=1000)
    # `states` - List of game states that occurred during self-play
    # `visit_counts` - Visit count for each move at the root node
    # `winners` - Game results used to train the value head of the neural network

    # Step 2: Create a neural network for the game
    network = GameNetwork(input_shape=(102,), num_actions=100)
    # `input_shape=(102,)` - Defines the input size (encoded game state)
    # `num_actions=100` - Defines the number of possible actions in the game

    # Step 3: Train the neural network using the generated self-play data
    network.train(states, visit_counts, winners, epochs=10)
    # `epochs=10` - Number of training iterations to improve the network's performance

    # Step 4: Initialize a new Snort game instance
    game = SnortGame()

    # Step 5: Create a player that uses the trained PUCT algorithm with the neural network
    player = PUCTPlayer(network)

    # Step 6: Start the game with the trained player
    game.play(player)

# Check if the script is being run directly
if __name__ == "__main__":
    main()
    # Calls the `main` function if the script is executed directly (not imported as a module)
