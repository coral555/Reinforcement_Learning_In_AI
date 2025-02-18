import random  # Importing the random module for selecting random moves
from PUCTNode import PUCTNode  # Importing the PUCTNode class for tree search nodes
import numpy as np  # Importing NumPy for mathematical operations like normalization

class PUCTPlayer:
    """
    Represents a player using the Predictor Upper Confidence Tree (PUCT) algorithm.
    The player uses a neural network for move selection and evaluation.

    Attributes:
    - network: A trained neural network for evaluating game states.
    - iterations (int): Number of search iterations performed per move.
    """

    def __init__(self, network, iterations=10):
        """
        Initializes the PUCT player.

        Parameters:
        - network: The neural network used for policy and value estimation.
        - iterations (int): Number of simulations to perform before making a move.
        """
        self.network = network  # Neural network for move evaluation
        self.iterations = iterations  # Number of PUCT iterations per move

    def select_move(self, game):
        """
        Selects the best move using the PUCT algorithm.

        Steps:
        1. Create a root node representing the current game state.
        2. Perform multiple simulations to build a search tree.
        3. Return the best move based on the search tree.

        Parameters:
        - game: The current game state.

        Returns:
        - tuple: The best move selected based on the search tree.
        """
        root = PUCTNode(game)  # Create the root node with the current game state

        # Perform multiple search iterations to build the tree
        for _ in range(self.iterations):
            leaf = self.traverse(root)  # Traverse the tree to find an expandable node
            outcome = self.evaluate(leaf)  # Evaluate the node using the neural network
            self.backpropagate(leaf, outcome)  # Update the tree with the evaluation result
        
        # Choose the best move from the root node
        best_child = root.best_child()
        if best_child.move is None:
            raise ValueError("PUCTPlayer: No valid moves found!")  # Ensure a valid move is selected
        return best_child.move  # Return the selected move

    def traverse(self, node):
        """
        Traverses the search tree to find an expandable node.

        The traversal follows the best UCT score until a leaf node is reached.

        Parameters:
        - node (PUCTNode): The current node in the tree.

        Returns:
        - PUCTNode: The leaf node to be expanded.
        """
        while node.children:  # While there are children, keep selecting the best one
            node = node.best_child()

        # If the game has ended, return the terminal node
        if node.game.status() != 'ongoing':
            return node

        # Otherwise, expand the tree by adding new children
        return self.expand(node)

    def expand(self, node):
        """
        Expands the tree by adding new child nodes.

        Steps:
        1. Use the neural network to get policy predictions.
        2. Normalize the probabilities.
        3. Create new child nodes for legal moves.

        Parameters:
        - node (PUCTNode): The node to expand.

        Returns:
        - PUCTNode: A randomly selected child node.
        """
        policy, _ = self.network.predict(node.game.encode())  # Get policy probabilities from the neural network
        legal_moves = node.game.legal_moves()  # Get the list of legal moves

        # Compute the total probability of legal moves
        total_prob = sum(policy[move[0] * node.game.size + move[1]] for move in legal_moves)
        
        if total_prob == 0:
            policy /= np.sum(policy)  # Normalize the policy if all values are zero

        # Add child nodes for each legal move
        for move in legal_moves:
            move_index = move[0] * node.game.size + move[1]  # Convert move to a linear index
            prior_probability = policy[move_index]  # Get the probability of this move
            node.add_child(move, prior_probability)  # Add the move as a child node

        return random.choice(node.children)  # Return a randomly chosen child node

    def evaluate(self, node):
        """
        Evaluates a node using the neural network.

        The neural network returns a value estimate indicating the expected game outcome.

        Parameters:
        - node (PUCTNode): The node to evaluate.

        Returns:
        - float: The estimated value of the node.
        """
        _, value = self.network.predict(node.game.encode())  # Get value estimate from the neural network
        return value  # Return the evaluation result

    def backpropagate(self, node, outcome):
        """
        Backpropagates the result of a simulation up the search tree.

        Steps:
        1. Update the visit count (N).
        2. Update the value estimate (Q).
        3. Flip the outcome sign to account for alternating turns.

        Parameters:
        - node (PUCTNode): The node where backpropagation starts.
        - outcome (float): The evaluation result to propagate.
        """
        while node:
            node.visits += 1  # Increase the visit count
            node.Q += (outcome - node.Q) / node.visits  # Update the average value estimate
            
            outcome = -outcome  # Flip the sign to reflect opponent's perspective
            node = node.parent  # Move up the tree to update the parent node