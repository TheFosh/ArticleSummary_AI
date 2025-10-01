import random

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


class Encoder(nn.Module):
    """
       GRU Encoder for seq2seq model.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        """
        Initializes the encoder.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of embedding vectors.
            hidden_dim (int): Number of hidden units in LSTM.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input sequence tensor of shape (batch, seq_len)

        Returns:
            Tuple[Tensor, Tensor]: Final hidden and cell states
        """
        x = self.embedding(x)
        outputs, h_n = self.gru(x)
        return outputs, h_n

class Decoder(nn.Module):
    """
    LSTM Decoder for seq2seq model.
    """

    def __init__(self, vocab_size, sosidx, eosidx, embedding_dim, hidden_dim, num_layers):
        """
        Initializes the decoder.

        Args:
            vocab_size (int): Size of the vocabulary.
            sosidx (int): Index of start-of-sequence token.
            embedding_dim (int): Dimension of embedding vectors.
            hidden_dim (int): Number of hidden units in LSTM.
            forcing (float): Probability of teacher forcing.
        """

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.project_h = nn.Linear(2 * hidden_dim, hidden_dim)  # To combine bidirectional encoder h_n
        self.attn_proj = nn.Linear(2 * hidden_dim, hidden_dim)  # For projecting encoder outputs before attention
        self.combine_context = nn.Linear(3 * hidden_dim, hidden_dim)  # Combine context + decoder output
        self.hidden_to_vocab = nn.Linear(hidden_dim, vocab_size)  # Final projection to vocab size

        self.num_layers = num_layers
        self.hid_dim = hidden_dim

        self.sos_idx = sosidx
        self.eos_idx = eosidx

    def forward(self, enc_outputs, enc_h_n, targets):
        """
        Decoder forward pass

        Args:
            enc_outputs: (batch_size, src_len, 2*hid_dim)
            enc_h_n: (2*num_layers, batch_size, hid_dim)
            targets: (batch_size, target_len) - ground truth sequences

        Returns:
            outputs: (batch_size, target_len, vocab_size)
        """
        batch_size, target_len = targets.size()
        device = targets.device

        # Initialize decoder input with <sos> token
        decoder_input = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)

        # Reshape encoder hidden state to initialize decoder hidden state
        enc_h_n = enc_h_n.view(self.num_layers, 2, batch_size, self.hid_dim)
        decoder_h_n = self.project_h(torch.cat((enc_h_n[:, 0], enc_h_n[:, 1]), dim=2))

        # Project encoder outputs for attention scoring
        projected_enc_outputs = self.attn_proj(enc_outputs)

        outputs = []

        for t in range(target_len):
            embedded = self.embedding(decoder_input)  # (batch_size, 1, emb_dim)
            output, decoder_h_n = self.gru(embedded, decoder_h_n)  # output: (batch_size, 1, hid_dim)

            # Attention mechanism
            decoder_output = output[:, -1, :]  # (batch_size, hid_dim)

            attn_scores = torch.bmm(projected_enc_outputs, decoder_output.unsqueeze(2)).squeeze(
                2)  # (batch_size, src_len)
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, src_len)

            context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)  # (batch_size, 2*hid_dim)

            combined = torch.cat((decoder_output, context), dim=1)  # (batch_size, 3*hid_dim)
            combined = torch.tanh(self.combine_context(combined))  # (batch_size, hid_dim)

            logits = self.hidden_to_vocab(combined)  # (batch_size, vocab_size)
            outputs.append(logits)


        return torch.stack(outputs, dim=1)  # (batch_size, target_len, vocab_size)

    def inference(self, enc_outputs, enc_h_n, max_len=50):
        """
        Greedy decoding (inference mode).

        Args:
            enc_outputs: (batch_size, src_len, 2*hid_dim)
            enc_h_n: (2*num_layers, batch_size, hid_dim)
            max_len: maximum length of decoded sequence

        Returns:
            decoded_ids: (batch_size, decoded_len)
        """
        batch_size = enc_outputs.size(0)
        device = enc_outputs.device

        decoder_input = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)

        enc_h_n = enc_h_n.view(self.num_layers, 2, batch_size, self.hid_dim)
        decoder_h_n = self.project_h(torch.cat((enc_h_n[:, 0], enc_h_n[:, 1]), dim=2))

        projected_enc_outputs = self.attn_proj(enc_outputs)

        outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            embedded = self.embedding(decoder_input)
            output, decoder_h_n = self.gru(embedded, decoder_h_n)
            decoder_output = output[:, -1, :]

            attn_scores = torch.bmm(projected_enc_outputs, decoder_output.unsqueeze(2)).squeeze(2)
            attn_weights = F.softmax(attn_scores, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)

            combined = torch.cat((decoder_output, context), dim=1)
            combined = torch.tanh(self.combine_context(combined))

            logits = self.hidden_to_vocab(combined)
            next_token = torch.argmax(logits, dim=1)  # (batch_size,)

            outputs.append(next_token.unsqueeze(1))  # (batch_size, 1)
            decoder_input = next_token.unsqueeze(1)

            # Stop decoding if all sequences predicted <eos>
            finished = finished | (next_token == self.eos_idx)
            if finished.all():
                break

        return torch.cat(outputs, dim=1)  # (batch_size, decoded_len)


class Seq2Seq(nn.Module):
    """
    Full Seq2Seq model combining encoder and decoder.
    """

    def __init__(self, vocab_size, sosidx, embedding_dim=32, hidden_dim=96, num_layers = 2):
        """
        Initializes the Seq2Seq model.

        Args:
            vocab_size (int): Vocabulary size.
            sosidx (int): Index of start-of-sequence token.
            embedding_dim (int): Embedding size.
            hidden_dim (int): LSTM hidden size.
            forcing (float): Teacher forcing probability.
        """
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers)
        self.decoder = Decoder(vocab_size, sosidx, embedding_dim, hidden_dim, num_layers)

    def forward(self, input_dates, target_dates):
        """
        Forward pass through the model.

        Args:
            input_dates (Tensor): Input sequence batch.
            target_dates (Tensor): Target sequence batch.

        Returns:
            Tensor: Decoder output logits.
        """
        h_n, c_n = self.encoder(input_dates)
        return self.decoder(h_n, c_n, target_dates)

def train_nn(epochs=5, batch_size=64, lr=0.001):
    """
    Trains the Seq2Seq model.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DateData() # TODO
    vocab_size = len(dataset.idx2token) ## DONT TOUCH ##
    sosidx = dataset.token2idx['^'] ## DONT TOUCH ##
    model = Seq2Seq(vocab_size, sosidx).to(device) # Step into ##

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Maybe touch #
    loss_fn = nn.CrossEntropyLoss() # Maybe touch #

    for epoch in range(epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Can touch
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}") # Can touch

        for _, data in enumerate(progress_bar): ## DONT TOUCH ##
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Actual output of the model
            output = model(inputs, targets)

            # Resizing of the model to be able to fit in the loss function
            output = output.view(-1, output.size(-1)) # Dim: [N,C]
            target = targets.view(-1) # Dim: [N]

            loss = loss_fn(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) ## Preventing parameters from blowing up.
            optimizer.step()

            progress_bar.set_postfix({'loss': loss.item()})
        torch.save(model.state_dict(), "seq2seq.pt")
