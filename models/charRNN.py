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

all_chars_list = ['<', 'a','b','c','e','g','i','l','n',
      'o','p','r','s','t','A','B','C','F','H','I','K','L','M','N',
      'O','P','R','S','T','V','X','Z','0','1','2','3','4','5','6','7',
       '8','9', '=','#','+','-','[',']','(',')','/','\\', '@','.','%', '>']

all_chars = ''.join(all_chars_list)
n_chars = len(all_chars)

# ------------------------------------------------------------------

batch_size = 1  #1 for all fct with bach size param
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 embed_size=30, batch_first=True):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
                 # Layers:
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, hidden):
        out = self.embed(x)
        out, hidden = self.gru(out.unsqueeze(1), hidden)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, hidden
    def init_hidden(self, batch_size=1):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden

def make_charRNN(n_chars=n_chars, hidden_size=512, num_layers=3,
                 lr=0.0005, pretrained_file=None):
    rnn = RNN(n_chars, hidden_size, num_layers, n_chars).to(device)
    if pretrained_file == None:
        pass
    else:
        filename = ff + '\\pretrained\\' + pretrained_file + '.pth'
        rnn.load_state_dict(torch.load(filename))

    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return rnn, optimizer, criterion


def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_chars.index(string[c])
    return tensor

def get_random_batch(file, chunk_len):    
    start_idx = random.randint(0, len(file) - chunk_len)
    end_idx = start_idx + chunk_len + 1
    text_str = file[start_idx:end_idx]
    text_input = torch.zeros(batch_size, chunk_len)
    text_target = torch.zeros(batch_size, chunk_len)

    for i in range(batch_size):
        text_input[i,:] = char_tensor(text_str[:-1])
        text_target[i,:] = char_tensor(text_str[1:])
    return text_input.long(), text_target.long()


def generate(rnn, batch_size=1, initial_str='<', predict_len=100, temperature=0.85):
    rnn.eval()

    hidden = rnn.init_hidden(batch_size=batch_size)
    initial_input = char_tensor(initial_str)
    predicted = ''

    for p in range(len(initial_str) - 1):
        _, hidden = rnn(initial_input[p].view(1).to(device),hidden)

    last_char = initial_input[-1]

    # each step, take the char with the highest softmax value
    for p in range(predict_len):
        output, hidden = rnn(last_char.view(1).to(device), hidden)
        # next line is the softmax (with temperature)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_char = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_chars[top_char]
        if predicted_char=='>' or p==predict_len:
            break
        else:
            predicted += predicted_char
            last_char = char_tensor(predicted_char)

    return predicted








