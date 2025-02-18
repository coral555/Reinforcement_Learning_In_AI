import numpy as np  # Import NumPy for handling numerical operations
import tensorflow as tf  # Import TensorFlow for deep learning operations
from keras import layers, models  # Import Keras layers and models for building the neural network

class GameNetwork:
    """
    Represents a neural network for learning and evaluating the Snort game.
    This class defines the neural network used for policy (move selection) and value (win probability) estimation in the PUCT algorithm.
    
    The network consists of:
    - A policy head: Predicts the probability of each move.
    - A value head: Estimates the probability of winning.
    
    Attributes:
    - model: The TensorFlow/Keras model used for training and inference.
    """

    def __init__(self, input_shape, num_actions):
        """
        Initializes the GameNetwork.

        Parameters:
        - input_shape (tuple): The shape of the input data (game state encoding).
        - num_actions (int): The number of possible actions (moves) in the game.
        """
        self.model = self.build_network(input_shape, num_actions)  # Build the neural network

    def build_network(self, input_shape, num_actions):
        """
        Builds the neural network model with two heads: policy and value.

        The network consists of:
        - An input layer.
        - Two hidden layers with ReLU activation.
        - A policy head (softmax output).
        - A value head (tanh output).

        Parameters:
        - input_shape (tuple): The shape of the input data.
        - num_actions (int): The number of possible actions in the game.

        Returns:
        - model: A compiled Keras model.
        """
        inputs = layers.Input(shape=input_shape)  # Define input layer

        # Hidden layers with ReLU activation
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)

        # Policy head: Predicts action probabilities
        policy_head = layers.Dense(num_actions, activation='softmax', name='policy_head')(x)

        # Value head: Predicts the probability of winning
        value_head = layers.Dense(1, activation='tanh', name='value_head')(x)

        # Define and compile the model
        model = models.Model(inputs=inputs, outputs=[policy_head, value_head])
        model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])

        return model

    def predict(self, state):
        """
        Predicts the policy and value for a given game state.

        Parameters:
        - state (array-like): The encoded game state.

        Returns:
        - policy (array): The predicted probabilities for each possible move.
        - value (float): The estimated probability of winning.
        """
        state = np.array(state)  # Convert input to NumPy array
        policy, value = self.model.predict(state.reshape(1, -1), verbose=0)  # Reshape and predict
        return policy[0], value[0]  # Return the first (and only) prediction result

    def train(self, states, visit_counts, outcomes, epochs=10, batch_size=64):
        """
        Trains the neural network using self-play data.

        The training process:
        - Normalizes visit counts to create a probability distribution.
        - Trains the policy head with visit counts.
        - Trains the value head with game outcomes.

        Parameters:
        - states (list): List of game states (input data).
        - visit_counts (list): Visit counts from MCTS (used for policy training).
        - outcomes (list): Final results of games (used for value training).
        - epochs (int): Number of training epochs (default: 10).
        - batch_size (int): Number of samples per training batch (default: 64).
        """
        # Debugging prints to verify data format
        print(f"Type of visit_counts: {type(visit_counts)}")
        print(f"First element type: {type(visit_counts[0]) if visit_counts else 'Empty'}")
        print(f"Length of visit_counts: {len(visit_counts)}")

        # Print shapes of first few visit count vectors
        for i, vc in enumerate(visit_counts[:5]): 
            print(f"visit_counts[{i}] shape: {np.shape(vc)}")
        
        # Convert inputs to NumPy arrays
        states = np.array(states, dtype=np.float32)
        visit_counts = np.array(visit_counts, dtype=np.float32)  
        outcomes = np.array(outcomes, dtype=np.float32)

        print(f"Converted visit_counts shape: {visit_counts.shape}")  # Debugging output

        # Normalize visit counts to form a probability distribution
        visit_counts = visit_counts / np.sum(visit_counts, axis=1, keepdims=True)

        # Train the neural network on self-play data
        self.model.fit(
            states,
            [visit_counts, outcomes],  # Train policy and value heads
            epochs=epochs,
            batch_size=batch_size
        )

    def save_weights(self, filepath):
        """
        Saves the trained weights of the neural network.

        Parameters:
        - filepath (str): The file path to save the weights.
        """
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        """
        Loads weights into the neural network.

        Parameters:
        - filepath (str): The file path to load the weights from.
        """
        self.model.load_weights(filepath)
