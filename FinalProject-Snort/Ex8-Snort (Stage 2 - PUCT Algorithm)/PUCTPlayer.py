import random
from PUCTNode import PUCTNode
import numpy as np
from SnortGame import SnortGame
import torch
import torch.nn as nn

class PUCTPlayer:
    def __init__(self, network, iters=10000, cpuct=1.0):
        self.network = network
        self.iters = iters
        self.cpuct = cpuct  

    def select_move(self, game):
        root = PUCTNode(game)
        game_state = torch.tensor(game.encode(), dtype=torch.float32).unsqueeze(0)  
        policy, value = self.network.forward(game_state)
        policy = policy.squeeze().detach().numpy()  
        value = value.item()

        if game.current_player == 'B': 
            value = -value  

        legal_moves = game.legal_moves()
        policy_logits = np.array([policy[game.size * move[0] + move[1]] for move in legal_moves])
        policy_probs = np.exp(policy_logits - np.max(policy_logits)) 
        policy_probs /= np.sum(policy_probs) + 1e-8  
        move_probs = {move: prob for move, prob in zip(legal_moves, policy_probs)}

        for move, prob in move_probs.items():
            root.add_child(move, prior=prob)
        # print("iters: ",self.iters)
        for _ in range(self.iters):
            node = self.select(root)
            node_value = self.evaluate(node)
            self.backpropagate(node, node_value)

        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move

    def select(self, node):
        while node.game.status() == SnortGame.ONGOING:
            if not node.children:
                return node
            node = node.best_child(self.cpuct)  
        return node

    def evaluate(self, node):
        game_state = torch.tensor(node.game.encode(), dtype=torch.float32).unsqueeze(0)  
        _, value = self.network.forward(game_state)
        value = value.item()
        if node.game.current_player == 'B':  
            value = -value  
        return value  
    
    def backpropagate(self, node, value):
        while node:
            node.visits += 1

            if node.game.current_player == 'B':
                value = -value

            node.q_value = ((node.visits - 1) * node.q_value + value) / node.visits  
            node = node.parent