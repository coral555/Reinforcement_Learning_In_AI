import torch
import torch.nn as nn
import torch.optim as optim
from PUCTPlayer import PUCTPlayer
from SnortGame import SnortGame
from torch.nn import functional as F

class Trainer:
    def __init__(self, network, game, iters=10000, lr=0.001, batch_size=32, noise=True):
        self.network = network
        self.game = game
        self.iters = iters  
        self.batch_size = batch_size
        self.noise = noise  # שליטה בהוספת רעש לאימון
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        self.loss_fn_value = nn.MSELoss()
        self.training_data = []

    def generate_self_play_data(self, update_callback=None, num_games=100):
        data = []
        for i in range(num_games):
            game_instance = self.game.clone()
            player = PUCTPlayer(self.network, iters=100, cpuct=1.0)
            move_history = []
            visit_counts = []

            while game_instance.status() == SnortGame.ONGOING:
                move = player.select_move(game_instance)
                move_history.append((game_instance.encode(), move))

                game_instance.make_move(move)

            result = 1 if game_instance.status() == SnortGame.R_WINS else -1 if game_instance.status() == SnortGame.B_WINS else 0
            data.append((move_history, result))

            if i % 10 == 0:
                print(f"Generated {i} self-play games...")  

            if update_callback:  
                update_callback()

        return data

    def train(self, update_callback=None):
        data = self.generate_self_play_data(update_callback, num_games=self.iters)
        for move_history, result in data:
            batch_loss = 0
            for state, move in move_history:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                value_target = torch.tensor(result, dtype=torch.float32)
                
                policy_pred, value_pred = self.network(state_tensor)
                visit_probs = torch.ones(policy_pred.shape) / policy_pred.shape[-1]  # קירוב פשוט
                log_policy_pred = F.log_softmax(policy_pred, dim=-1)

                loss_policy = F.kl_div(log_policy_pred, visit_probs, reduction='batchmean')  
                loss_value = self.loss_fn_value(value_pred.squeeze(), value_target)

                self.optimizer.zero_grad()
                (loss_policy + loss_value).backward()
                self.optimizer.step()
                batch_loss += (loss_policy + loss_value).item()

            print(f"Batch loss: {batch_loss / len(move_history):.4f}")

        print("Training complete!")
