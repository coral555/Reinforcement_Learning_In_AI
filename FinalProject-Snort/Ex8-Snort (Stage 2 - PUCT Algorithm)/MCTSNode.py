from SnortGame import SnortGame
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MCTSNode:
    def __init__(self, game, parent=None, move=None, C=1.0):
        self.game = game.clone()
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.q_value = 0 
        self.C = C  

    def explored(self):
        return len(self.children) > 0 
    def best_child_score(self):
        scores = [
            (child.wins / (child.visits + 1e-8)) + self.C * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-8))
            for child in self.children
        ]
        return self.children[np.argmax(scores)]

    def add_child(self, move):
        new_game = self.game.clone()
        new_game.make_move(move)
        new_child = MCTSNode(new_game, self, move, self.C)
        self.children.append(new_child)
        return new_child
