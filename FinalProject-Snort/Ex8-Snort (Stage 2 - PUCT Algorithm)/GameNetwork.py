import torch
import torch.nn as nn

class GameNetwork(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=256):
        super(GameNetwork, self).__init__()
        self.fc1=nn.Linear(input_size, hidden_size)
        self.fc2=nn.Linear(hidden_size, hidden_size)
        self.policy_head=nn.Linear(hidden_size, action_size)
        self.value_head=nn.Linear(hidden_size, 1)

    def forward(self,x,return_policy=True):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = torch.tanh(self.value_head(x)) 
        
        if return_policy:
            policy =torch.softmax(self.policy_head(x), dim=-1) 
            return policy, value
        return value  

    def predict(self,game):
        game_state =torch.tensor(game.encode(), dtype=torch.float32).unsqueeze(0)  
        with torch.no_grad():
            policy, value = self.forward(game_state) 
        
        return policy.cpu().numpy(), value.cpu().numpy()

    def save_model(self,path):
        torch.save(self.state_dict(), path)

    def load_model(self,path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.eval()
