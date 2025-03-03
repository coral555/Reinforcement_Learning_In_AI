import numpy as np
import random
import torch
from MCTSNode import MCTSNode
from SnortGame import SnortGame

class MCTSPlayer:
    def __init__(self, iters=10000, collect_data=False):
        self.iters = iters
        self.collect_data = collect_data
        self.training_data = []  

    def select_move(self, game):
        root = MCTSNode(game)

        for _ in range(self.iters):
            node = self.select(root)
            result = self.simulate(node)  
            self.backpropagate(node, result, node.game.current_player)

        move_counts = {child.move: child.visits for child in root.children} if root.children else {}
        visit_counts = torch.tensor(list(move_counts.values()), dtype=torch.float32) if move_counts else torch.zeros(len(game.legal_moves()))

        if root.children:
            best_move = max(root.children, key=lambda child: child.visits).move
        else:
            best_move = random.choice(game.legal_moves())  

        return best_move, visit_counts  

    def select(self, node):
        while node.game.status() == SnortGame.ONGOING:
            if not node.explored():
                return self.expand(node)
            node = node.best_child_score()
        return node

    def expand(self, node):
        unexplored_moves = [move for move in node.game.legal_moves() if move not in [child.move for child in node.children]]
        if not unexplored_moves:
            return node  
        move = random.choice(unexplored_moves)
        return node.add_child(move)

    def simulate(self, node):
        sim_game = node.game.clone()
        current_player = sim_game.current_player  

        while sim_game.status() == SnortGame.ONGOING:
            legal_moves = sim_game.legal_moves()
            if not legal_moves:
                break  

            best_move = min(legal_moves, key=lambda move: len(self.simulate_after_move(sim_game, move)))
            sim_game.make_move(best_move)  

        if sim_game.status() == SnortGame.R_WINS:
            return 1 if current_player == 'R' else -1  
        elif sim_game.status() == SnortGame.B_WINS:
            return 1 if current_player == 'B' else -1  
        else:
            return 0  

    def simulate_after_move(self, game, move):
        sim_game = game.clone()
        sim_game.make_move(move)
        return sim_game.legal_moves()

    def backpropagate(self, node, result, current_player):
        while node:
            node.visits += 1
            if node.parent and node.parent.game.current_player != node.game.current_player:
                result = -result  
            node.q_value = ((node.visits - 1) * node.q_value + result) / node.visits  
            node = node.parent
