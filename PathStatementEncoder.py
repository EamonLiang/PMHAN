from torch import nn


class PathStatementEncoder(nn.Module):
    def __init__(self, embedding_size, lstm_hidden_size, layer_num=1):
        super(PathStatementEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.layer_num = layer_num

        self.lstm = nn.LSTM(embedding_size, lstm_hidden_size, batch_first=True, bidirectional=True,
                            num_layers=layer_num)
        self.__init_weight()

    def forward(self, batch_size_plus_padded_sequence, real_sequence_len):
        if len(real_sequence_len.size()) > 1:
            real_sequence_len = real_sequence_len.squeeze()

        real_sequence_lens_sorted, indices = real_sequence_len.sort(descending=True)
        inputs_sorted = batch_size_plus_padded_sequence.index_select(0, indices)
        batch_size_plus_padded_sequence = nn.utils.rnn.pack_padded_sequence(inputs_sorted,
                                                                            real_sequence_lens_sorted.data.tolist(),
                                                                            batch_first=True)
        # batch, seq_len, num_directions * hidden_size
        output, (_, _) = self.lstm(batch_size_plus_padded_sequence)  # output, (h_n, c_n)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        _, invert_indices = indices.sort()
        output = output.index_select(0, invert_indices)
        return output

    def __init_weight(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)
