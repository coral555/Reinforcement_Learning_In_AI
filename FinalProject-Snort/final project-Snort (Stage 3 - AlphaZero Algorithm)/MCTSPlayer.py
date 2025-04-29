import random
import numpy as np
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

        if root.children:
            best_move = max(root.children, key=lambda child: child.visits).move
        else:
            best_move = random.choice(game.legal_moves())  

        return best_move

    def select(self, node):
        while node.game.status() == SnortGame.ONGOING:
            if not node.explored():
                return self.expand(node)
            node = node.best_child_score()
        return node

    def expand(self, node):
        legal_moves = node.game.legal_moves()
        random.shuffle(legal_moves)  # ערבוב למניעת בחירה חדגונית

        for move in legal_moves:
            if move not in [child.move for child in node.children]:
                return node.add_child(move)
        return node  # במקרה שכל המהלכים כבר נוספו

    def simulate(self, node):
        sim_game = node.game.clone()
        current_player = sim_game.current_player  
        legal_moves = sim_game.legal_moves()  # שמירה על רשימת המהלכים החוקיים
        
        while sim_game.status() == SnortGame.ONGOING:
            if not legal_moves:
                break  
            move_index = random.randint(0, len(legal_moves) - 1)
            move = legal_moves[move_index]
            
            sim_game.make_move(move)
            
            # Swap-and-Pop: החלפה עם האחרון והסרה בזמן O(1)
            legal_moves[move_index] = legal_moves[-1]
            legal_moves.pop()

        if sim_game.status() == SnortGame.R_WINS:
            return 1 if current_player == 'R' else -1  
        elif sim_game.status() == SnortGame.B_WINS:
            return 1 if current_player == 'B' else -1  
        else:
            return 0  

    def backpropagate(self, node, result, current_player):
        while node:
            node.visits += 1
            if node.parent and node.parent.game.current_player != node.game.current_player:
                result = -result  
            node.q_value = ((node.visits - 1) * node.q_value + result) / node.visits  
            node = node.parent
