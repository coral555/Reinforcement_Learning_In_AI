import math
import random

class MCTSNode:

    def __init__(self, game, parent=None, move=None, C=0.5):
        self.game = game.clone() 
        self.parent = parent   
        self.move = move    
        self.children = []     
        self.visits = 0   
        self.wins = 0         
        self.C = C               

    def explored(self):
        return len(self.children) == len(self.game.legal_moves())

    def best_child_score(self):
        result = [
            -1 * (child.wins / child.visits) + self.C * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        bestIndex = result.index(max(result))
        bestNode = self.children[bestIndex]
        return bestNode

    def add_child(self, move):
        AddNewChildToGame = self.game.clone() 
        AddNewChildToGame.make(move)     
        newChild = MCTSNode(AddNewChildToGame, self, move, self.C)
        self.children.append(newChild)   
        return newChild