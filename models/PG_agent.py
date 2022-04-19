import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np

import pathlib
ff = pathlib.Path(__file__).parent.resolve()
ff = str(ff)
#print(ff)

##torch.manual_seed(123)
##np.random.seed(123)
##random.seed(123)

# ------------------------------------------------------------------

batch_size = 1  #1 for all fct with bach size param
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, embed_size=30, lr=0.0005):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
                 # Layers:
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x, hidden):
        out = self.embed(x)
        out, hidden = self.gru(out.unsqueeze(1), hidden)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, hidden
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden

class PolicyGradientAgent():
    def __init__(self, n_chars, hidden_size, num_layers, output_size, batch_size=1,
                 lr=0.0005, policy_file=None):
        
        self.gamma = 0.99
        self.batch_size = batch_size
        self.reward_memory = []
        self.action_memory = []
        
        if policy_file == None:
            self.policy = RNN(n_chars, hidden_size, num_layers, output_size,  lr=lr) 
        else:
            filename = ff + '\\pretrained\\' + policy_file + '.pth'
            self.policy = RNN(n_chars, hidden_size, num_layers, output_size, lr=lr) 
            self.policy.load_state_dict(torch.load(filename))
        
        # batch_size is length to input and to output
        self.hidden = self.policy.init_hidden(batch_size= batch_size) 
        
    def choose_action(self, observation): #observation is prev_char's idx (i.e., int)
        if self.batch_size > 1:
            state = torch.tensor(observation).to(device) 
        else:
            state = torch.tensor([observation]).to(device) 
        output, self.hidden = self.policy.forward(state, self.hidden)
        probabilities = F.softmax(output, dim=1) # gives values of actions
        action_probs = torch.distributions.Categorical(probabilities) # gives prob distribution according to these values 
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()   # idx of next_char

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.hidden.detach_()
        self.hidden = self.hidden.detach()
        self.policy.optimizer.zero_grad()
                      # G_t = R_t+1 + gamma * R_t+2 + gamma**2 * R_t+3
                      # G_t = sum from k=0 to k=T {gamma**k * R_t+k+1}
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = torch.tensor(G, dtype=torch.float).to(device)
        
        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob
        
        if self.batch_size > 1:
            loss.sum().backward()
        else:
            loss.backward()
        
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
        
    def save_policy(self, filename):
        torch.save(self.policy.state_dict(), filename)












