import random
from MCTSNode import MCTSNode

class MCTSPlayer:
    def __init__(self, iters):
        self.iters = iters

    def select_move(self, game):
        root = MCTSNode(game)  
        for _ in range(self.iters):
            node = self.select(root)         
            resultOfMoves = self.simulate(node)  
            self.backpropagate(node, resultOfMoves) 

        if not root.children:
            raise ValueError("No valid moves available.")
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move

    def select(self, node):
        while not node.game.status in (node.game.RED_WIN, node.game.YELLOW_WIN, node.game.DRAW):
            if not node.explored():
                return self.expand_New(node)
            else:
                node = node.best_child_score()
        return node

    def expand_New(self, node):
        # Prioritize winning moves if possible
        winning_moves = [move for move in node.game.legal_moves() if self.is_winning_move(node.game, move)]
        if winning_moves:
            move = random.choice(winning_moves)  # Choose a winning move
        else:
            tried_moves = set(child.move for child in node.children)
            null_moves = [move for move in node.game.legal_moves() if move not in tried_moves]
            if not null_moves:
                raise ValueError("No moves left to expand.")
            move = random.choice(null_moves)  # If no winning moves, expand randomly
        return node.add_child(move)

    def simulate(self, node):
        simulate_game = node.game.clone()  
        while simulate_game.status == simulate_game.ONGOING:
            # Bias the simulation toward winning moves
            winning_moves = [move for move in simulate_game.legal_moves() if self.is_winning_move(simulate_game, move)]
            if winning_moves:
                move = random.choice(winning_moves)
            else:
                move = random.choice(simulate_game.legal_moves())  # Random move if no immediate winning move
            simulate_game.make(move)  
        return simulate_game.status

    def backpropagate(self, node, resultOfMoves):
        while node is not None:
            node.visits += 1 
            # Determine if the current node represents the winning player
            red_win = node.game.RED_WIN
            yellow_win = node.game.YELLOW_WIN
            current_player = node.game.player
            if (resultOfMoves == red_win and current_player == node.game.RED) or \
               (resultOfMoves == yellow_win and current_player == node.game.YELLOW):
                node.wins += 1  # Increment wins for the current player

            node = node.parent

    def is_winning_move(self, game, move):
        """
        Check if making the move leads to a win.
        """
        game_clone = game.clone()
        game_clone.make(move)
        return game_clone.status == game_clone.RED_WIN or game_clone.status == game_clone.YELLOW_WIN
