import math
import numpy as np

class PUCTNode:
    def __init__(self, game,parent=None,move=None,prior=1.0,c_puct=1.5):
        self.game =game.clone()
        self.parent=parent
        self.move=move
        self.children=[]
        self.visits=0
        self.q_value=0  
        self.prior=prior  
        self.c_puct=c_puct 

    def best_child(self,cpuct):  
        scores = [child.q_value + cpuct * child.prior * math.sqrt(self.visits + 1) / (1 + child.visits)
            for child in self.children]
        return self.children[np.argmax(scores)]

    def add_child(self,move,prior):
        child_game =self.game.clone()
        child_game.make_move(move)
        child= PUCTNode(child_game, parent=self, move=move, prior=prior, c_puct=self.c_puct)
        self.children.append(child)
        return child
