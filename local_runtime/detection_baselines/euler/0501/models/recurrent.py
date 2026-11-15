from torch import nn

class GRU(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(
            x_dim, h_dim, num_layers=hidden_units
        )

        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(h_dim, z_dim)

        self.z_dim = z_dim

    def forward(self, xs, h0, include_h=False):
        xs = self.drop(xs)

        if isinstance(h0, type(None)):
            xs, h = self.rnn(xs)
        else:
            xs, h = self.rnn(xs, h0)

        if not include_h:
            return self.lin(xs)

        return self.lin(xs), h

class LSTM(GRU):

    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(LSTM, self).__init__(x_dim, h_dim, z_dim, hidden_units=hidden_units)

        self.rnn = nn.LSTM(
            x_dim, h_dim, num_layers=hidden_units
        )

class Lin(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(Lin, self).__init__()

        self.layers = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(h_dim, z_dim)
        )

    def forward(self, xs, h0, include_h=False):
        if not include_h:
            return self.layers(xs)

        return self.layers(xs), None

class EmptyModel(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(EmptyModel, self).__init__()
        self.id = nn.Identity()

    def forward(self, xs, h0, include_h=False):
        if not include_h:
            return self.id(xs)

        return self.id(xs), None
