import math  # Importing the math module for mathematical operations like square root

class PUCTNode:
    """
    Represents a node in the PUCT (Predictor Upper Confidence Tree) search tree.
    This class implements PUCT (Predictor Upper Confidence Tree) search nodes, which are used in game-tree search with neural network evaluation.
    
    Attributes:
    - game: The game state at this node.
    - parent: The parent node in the search tree.
    - move: The move that led to this node.
    - children: A list of child nodes representing possible future states.
    - visits (N): The number of times this node has been visited.
    - Q: The average estimated value of this node.
    - P: The prior probability of selecting this move, provided by the neural network.
    - C: The exploration constant that controls the balance between exploration and exploitation.
    """

    def __init__(self, game, parent=None, move=None, prior_probability=1.0, C=1.0):
        """
        Initializes a new PUCT node.

        Parameters:
        - game: The current game state.
        - parent (PUCTNode, optional): The parent node in the search tree.
        - move (tuple, optional): The move that led to this node.
        - prior_probability (float): The prior probability of this move, given by the policy network.
        - C (float): The exploration constant used in the UCT formula.
        """
        self.game = game.clone()  # Create a deep copy of the game state
        self.parent = parent  # Store reference to the parent node
        self.move = move  # The move taken to reach this node
        self.children = []  # List to store child nodes
        self.visits = 0  # The number of times this node has been visited (N)
        self.Q = 0.0  # The estimated average value of this node (Q)
        self.P = prior_probability  # The prior probability from the policy network
        self.C = C  # Exploration constant used in the PUCT formula

    def uct_score(self, cpuct=1.5):
        """
        Calculates the PUCT score for this node.

        The score consists of two terms:
        - Q: The exploitation term (average value of the node).
        - Exploration term: Encourages selecting nodes with higher prior probability and less visits.

        U(s, a) = Q(s, a) + cpuct * P(s, a) * sqrt(N_parent) / (1 + N(s, a))

        Parameters:
        - cpuct (float): The exploration-exploitation balance parameter.

        Returns:
        - float: The computed UCT score for this node.
        """
        if self.visits == 0:
            return float('inf')  # Encourage visiting unexplored nodes

        # Compute the exploration term
        exploration = cpuct * self.P * (math.sqrt(self.parent.visits) / (1 + self.visits))

        # Return the final UCT score
        return self.Q + exploration

    def add_child(self, move, prior_probability):
        """
        Expands the search tree by adding a new child node.

        Parameters:
        - move (tuple): The move to apply to the current game state.
        - prior_probability (float): The prior probability from the policy network.

        Returns:
        - PUCTNode: The newly created child node.
        """
        new_state = self.game.clone()  # Clone the current game state
        new_state.make_move(move)  # Apply the move to generate the new state
        child = PUCTNode(new_state, self, move, prior_probability, self.C)  # Create a child node
        self.children.append(child)  # Add the child node to the list of children
        return child  # Return the newly created node

    def best_child(self):
        """
        Selects the best child node based on the highest UCT score.

        Returns:
        - PUCTNode: The best child node to explore.
        """
        return max(self.children, key=lambda child: child.uct_score())  # Select the child with the highest UCT score
