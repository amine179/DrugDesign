import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np

import pathlib
ff = pathlib.Path(__file__).parent.resolve()  #path where this file exists
ff = str(ff)
#print(ff)

##torch.manual_seed(123)
##np.random.seed(123)
##random.seed(123)

# ------------------------------------------------------------------
# list of all characters used for molecules in the Chembl21 dataset
all_chars_list = ['<', 'a','b','c','e','g','i','l','n',
      'o','p','r','s','t','A','B','C','F','H','I','K','L','M','N',
      'O','P','R','S','T','V','X','Z','0','1','2','3','4','5','6','7',
       '8','9', '=','#','+','-','[',']','(',')','/','\\', '@','.','%', '>']

all_chars = ''.join(all_chars_list)
n_chars = len(all_chars)

# ------------------------------------------------------------------
#1 for all fct with bach size param
batch_size = 1

# setting the device (CPU or GPU) for Pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# defining the architecture of the generative model G
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 embed_size=30):
        """
        Parameters
        ----------
        input_size: input for the embedding layer, that is, the vocabular size/the number of characters
        hidden_size: the number of units for the GRU layers
        num_layers: the number of GRU layers
        output_size: it should be the vocabular size as well, to output a probability over all characters
        embed_size: how many features should each character be embedded with, we set it to 30
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
                 # Layers:
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        """
        x: the input character
        hidden: the hidden state of the RNN
        """
        out = self.embed(x)
        out, hidden = self.gru(out.unsqueeze(1), hidden)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, hidden
    
    def init_hidden(self, batch_size=1):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden

# function to make an RNN from an existing file or a new one
def make_charRNN(n_chars=n_chars, hidden_size=512, num_layers=3,
                 lr=0.0005, pretrained_file=None, model_path='pretrained'):
    rnn = RNN(n_chars, hidden_size, num_layers, n_chars).to(device)
    if pretrained_file == None:
        pass
    else:
        filename = ff + '\\' + model_path + '\\' + pretrained_file + '.pth'
        rnn.load_state_dict(torch.load(filename))

    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return rnn, optimizer, criterion

# function to convert character to its index in the all_chars_list then to Pytorch tensor
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_chars.index(string[c])
    return tensor

# function to sample random batch for training
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

# function to generate a smiles using a given generator (rnn)
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








