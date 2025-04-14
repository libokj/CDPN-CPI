from torch import nn, zeros, cat


class LSTM(nn.Module):
    def __init__(
            self,
            n_samples: int,
            hidden_layers: int = 64):
        super().__init__()
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)
        self.n_samples = n_samples

    def forward(self, y, future_preds=0):
        outputs, num_samples = [], y.size(0)
        h_t = zeros(self.n_samples, self.hidden_layers)
        c_t = zeros(self.n_samples, self.hidden_layers)
        h_t2 = zeros(self.n_samples, self.hidden_layers)
        c_t2 = zeros(self.n_samples, self.hidden_layers)

        for time_step in y.split(1, dim=1):
            # N, 1
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))  # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))  # new hidden and cell states
            output = self.linear(h_t2)  # output from the last FC layer
            outputs.append(output)

        for i in range(future_preds):
            # this only generates future predictions if we pass in future_preds>0
            # mirrors the code above, using last output/prediction as input
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        # transform list to tensor
        outputs = cat(outputs, dim=1)
        return outputs
