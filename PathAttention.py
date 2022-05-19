import torch
import torch.nn.functional as torch_function
from torch import nn


class PathAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PathAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.__init_weight()

    def forward(self, batch_size_plus_pool_size_plus_embedding_size):
        input = batch_size_plus_pool_size_plus_embedding_size
        output = self.linear1(batch_size_plus_pool_size_plus_embedding_size)
        #output[:,:,64:]=0
        output = self.tanh(output)

        output = self.linear2(output)
        output = output.squeeze(-1)  # batch_size * pool_size
        weight = self.softmax(output)
        weight = weight.unsqueeze(1)  # batch_size * 1 * pool_size
        attention_pool = torch.bmm(weight, batch_size_plus_pool_size_plus_embedding_size)
        attention_pool = attention_pool.squeeze(1)
        return attention_pool
#        pool_size=batch_size_plus_pool_size_plus_embedding_size.size(1)
#
#        path_vector = torch_function.max_pool1d(batch_size_plus_pool_size_plus_embedding_size.transpose(1, 2), pool_size).squeeze(2)
#        print(path_vector.shape)
#        return path_vector


    def __init_weight(self):
        self.linear1.weight.data.uniform_(-0.1, 0.1)
        self.linear2.weight.data.uniform_(-0.1, 0.1)
