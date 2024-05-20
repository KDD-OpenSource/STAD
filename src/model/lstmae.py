import torch
import torch.nn as nn


def device(gpu):
    return torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')


class EncDecAD(nn.Module):
    """
    The encoder-decoder anomaly detection model follows the design of
    Malhotra et al. LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection
    """

    def __init__(self, input_size, hidden_dim, batch_size, num_layers, dropout_rate):
        """
        @param hidden_dim: The number of features in the hidden state
        @param input_size: The number of expected features in the input

        @return: None
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # Identical encoder and decoder structures
        self.encoder = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout_rate,
                               batch_first=True
                               )
        self.decoder = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout_rate,
                               batch_first=True
                               )

        # A linear layer maps the output of decoder to symmetrical form of encoder inputs
        self.output = nn.Linear(self.hidden_dim, self.input_size)

    def forward(self, x, gpu=None):
        """
        @param x: input dataset <batch_size, seq_len, input_size>
        @return:
        """

        if torch.cuda.is_available():
            x.to(device(gpu))
        encoder_hidden = self.init_hidden(gpu)
        encoder_output, encoder_hidden = self.encoder(x, encoder_hidden)
        """
        As stated in https://arxiv.org/pdf/1706.08838, the hidden state from encoder 
        contains sufficient information for the decoder to reconstruct the input. Rather 
        than using real time series value or decoder predicted value, they use constants as 
        decoder input achieves better reconstruction results.
        """
        decoder_input = torch.flip(x, [1])

        """
        Pass the last hidden state of encoder to decoder as initial hidden state.
        decoder_output shape: <batch_size, seq_len, hidden_size>.
        """
        decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_hidden)
        """
        reconstruction shape: <batch_size, seq_len, input_size>
        """
        reconstruction = self.output(decoder_output)
        return reconstruction, encoder_hidden

    def init_hidden(self, gpu):
        """
        Returns: two zero matrix for initial hidden state and cell state
        """
        h_s, c_s = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device=f'cuda:{gpu}'), \
                   torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device=f'cuda:{gpu}')

        return h_s, c_s
