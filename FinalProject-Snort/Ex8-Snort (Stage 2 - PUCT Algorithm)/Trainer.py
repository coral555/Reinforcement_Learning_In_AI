import torch
import torch.nn as nn
import torch.optim as optim
from MCTSPlayer import MCTSPlayer
from SnortGame import SnortGame
from torch.nn import functional as F

class Trainer:
    def __init__(self,network,game,iters=10000, lr=0.001):
        self.network = network
        self.game =game
        self.iters = iters  
        self.optimizer =optim.Adam(network.parameters(), lr=lr)
        self.loss_fn_value = nn.MSELoss()

    def generate_self_play_data(self, update_callback=None):
        data = []
        for i in range(self.iters):
            game_instance = self.game.clone()
            player =MCTSPlayer(iters=20)
            move_history =[]
            visit_counts = []

            while game_instance.status() ==SnortGame.ONGOING:
                move, visit_count =player.select_move(game_instance)  
                move_history.append((game_instance.encode(), move))
                visit_counts.append(visit_count.clone().detach()) 

                game_instance.make_move(move)

            result = 1 if game_instance.status() ==SnortGame.R_WINS else -1 if game_instance.status() == SnortGame.B_WINS else 0
            data.append((move_history, visit_counts, result))

            if i % 1000 == 0:
                print(f"Generated {i} self-play games...")  

            if update_callback:  
                update_callback()

        return data

    def train(self,update_callback=None):
        data = self.generate_self_play_data(update_callback)
        for move_history, visit_counts, result in data:
            for (state, move), visit_count in zip(move_history, visit_counts):
                state_tensor =torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                visit_count_tensor = visit_count.clone().detach() + 1e-8  

                value_target =torch.tensor(result, dtype=torch.float32)

                policy_pred, value_pred = self.network(state_tensor)
                visit_probs =visit_count_tensor / visit_count_tensor.sum()
                log_policy_pred = F.log_softmax(policy_pred, dim=-1)

                loss_policy = F.kl_div(log_policy_pred, visit_probs, reduction='batchmean')  
                loss_value = self.loss_fn_value(value_pred.squeeze(), value_target)

                self.optimizer.zero_grad()
                (loss_policy + loss_value).backward()
                self.optimizer.step()

        print("Training complete!")
