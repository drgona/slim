# machine learning/data science imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ecosystem imports
import slim


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False, nonlin=F.gelu,
                 hidden_map=slim.Linear, input_map=slim.Linear, linargs=dict()):
        """

        :param input_size:
        :param hidden_size:
        :param bias:
        :param nonlinearity:
        :param linear_map
        :param linargs:
        """
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.in_features, self.out_features = input_size, hidden_size
        self.nonlin = nonlin
        self.lin_in = input_map(input_size, hidden_size, bias=bias, **linargs)
        self.lin_hidden = hidden_map(hidden_size, hidden_size, bias=bias, **linargs)
        if type(input_map) is slim.Linear:
            torch.nn.init.orthogonal_(self.lin_hidden.linear.weight)

    def reg_error(self):
        return (self.lin_in.reg_error() + self.lin_hidden.reg_error())/2.0

    def forward(self, input, hidden):
        return self.nonlin(self.lin_hidden(hidden) + self.lin_in(input))


class RNN(nn.Module):
    def __init__(self, insize, outsize=0, hsizes=(16,),
                 cell_args=dict(), bias=False):
        """

        :param input_size: (int) Dimension of inputs
        :param output_size: (int) If outsize > 0, a final linear map will be applied to allow for outputs outside
                                  the range of the cell's nonlinear activation function.
        :param hsizes: (list of int) Hidden sizes for stacked RNNs.
        :param bias: If true bias will be added to final linear map.
        :param nonlinearity: (callable which acts on pytorch tensors) Activation for output. A common choice here might be
                             log_softmax for classification problems, identity for regression problems.
        :param stable:
        """
        super().__init__()
        assert len(set(hsizes)) == 1, 'All hiddens sizes should be equal for the RNN implementation'
        hidden_size = hsizes[0]
        num_layers = len(hsizes)
        self.in_features = insize
        if outsize > 0:
            self.out_features = outsize
            self.out_map = nn.Linear(hsizes[-1], outsize, bias=bias)
        else:
            self.out_features = hsizes[-1]
            self.out_map = nn.Identity()
        rnn_cells = [RNNCell(insize, hidden_size, *cell_args)]
        rnn_cells += [RNNCell(hidden_size, hidden_size, *cell_args)
                      for k in range(num_layers-1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)
        self.num_layers = len(rnn_cells)
        self.init_states = nn.ParameterList([nn.Parameter(torch.zeros(1, cell.hidden_size))
                                             for cell in self.rnn_cells])

    def reg_error(self):
        return torch.mean(torch.stack([cell.reg_error() for cell in self.rnn_cells]))

    def forward(self, sequence, init_states=None):
        """
        :param sequence: a tensor(s) of shape (seq_len, batch, input_size)
        :param init_state: h_0 (num_layers, batch, hidden_size)
        :returns:
        - output: (seq_len, batch, hidden_size)
        - h_n: (num_layers, batch, hidden_size)
        """
        assert len(sequence.shape) == 3, 'RNN takes order 3 tensor with shape=(seq_len, nsamples, dim)'
        if init_states is None:
            init_states = self.init_states
        final_hiddens = []
        for h, cell in zip(init_states, self.rnn_cells):
            states = []
            for seq_idx, cell_input in enumerate(sequence):
                h = cell(cell_input, h)
                states.append(h.unsqueeze(0))
            sequence = torch.cat(states, 0)
            final_hiddens.append(h)
        final_hiddens = final_hiddens
        assert torch.equal(sequence[-1, :, :], final_hiddens[-1])
        return sequence, self.out_map(torch.stack(final_hiddens))


if __name__ == '__main__':
    x = torch.rand(20, 5, 8)
    for bias in [True, False]:
        for name, map in slim.maps.items():
            print(name)
            rnn = RNN(8, hsizes=[8, 8], bias=bias, Linear=map)
            out = rnn(x)
            print(out[0].shape, out[1].shape)

        for map in set(slim.maps.values()) - slim.square_maps:
            rnn = RNN(8, hsizes=[16, 16], bias=bias, Linear=map)
            out = rnn(x)
            print(out[0].shape, out[1].shape)

        for name, map in slim.maps.items():
            print(name)
            rnn = RNN(8, hsizes=[8, 8], bias=bias, Linear=map)
            out = rnn(x)
            print(out[0].shape, out[1].shape)

        for map in set(slim.maps.values()) - slim.square_maps:
            rnn = RNN(8, hsizes=[16, 16], bias=bias, Linear=map)
            out = rnn(x)
            print(out[0].shape, out[1].shape)
