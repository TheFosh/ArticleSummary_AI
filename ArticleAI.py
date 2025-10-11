import json
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from CleanArticle import ArticleData


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers):
        """
        Bidirectional GRU Encoder.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(input_size=emb_dim,
                          hidden_size=hid_dim,
                          num_layers=num_layers,
                          bidirectional=True,
                          batch_first=True)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) - Input token indices

        Returns:
            output: (batch_size, seq_len, 2*hid_dim) - GRU outputs
            h_n: (2*num_layers, batch_size, hid_dim) - Hidden states from both directions
        """
        embedded = self.embedding(x)
        output, h_n = self.gru(embedded)
        return output, h_n


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, sos_idx, eos_idx):
        """
        GRU Decoder with Attention.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(input_size=emb_dim,
                          hidden_size=hid_dim,
                          num_layers=num_layers,
                          batch_first=True)

        self.project_h = nn.Linear(2 * hid_dim, hid_dim)  # To combine bidirectional encoder h_n
        self.attn_proj = nn.Linear(2 * hid_dim, hid_dim)  # For projecting encoder outputs before attention
        self.combine_context = nn.Linear(3 * hid_dim, hid_dim)  # Combine context + decoder output
        self.hidden_to_vocab = nn.Linear(hid_dim, vocab_size)  # Final projection to vocab size

        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, enc_outputs, enc_h_n, targets, forcing=0.5):
        """
        Decoder forward pass with teacher forcing.

        Args:
            enc_outputs: (batch_size, src_len, 2*hid_dim)
            enc_h_n: (2*num_layers, batch_size, hid_dim)
            targets: (batch_size, target_len) - ground truth sequences
            forcing: float in [0, 1] - teacher forcing ratio

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

            # Teacher forcing
            if random.random() < forcing:
                decoder_input = targets[:, t].unsqueeze(1)  # (batch_size, 1)
            else:
                decoder_input = torch.argmax(logits, dim=1, keepdim=True)  # (batch_size, 1)

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
    def __init__(self, enc_voc_size, dec_voc_size, enc_emb_dim, dec_emb_dim, hid_dim, num_layers, sos_idx, eos_idx):
        """
        Full Seq2Seq model with encoder and decoder.
        """
        super().__init__()
        self.encoder = Encoder(enc_voc_size, enc_emb_dim, hid_dim, num_layers)
        self.decoder = Decoder(dec_voc_size, dec_emb_dim, hid_dim, num_layers, sos_idx, eos_idx)

    def forward(self, input, targets, forcing=0.5):
        """
        Training forward pass.

        Args:
            input: (batch_size, src_len)
            targets: (batch_size, tgt_len)
            forcing: teacher forcing ratio

        Returns:
            logits: (batch_size, tgt_len, vocab_size)
        """
        enc_outputs, enc_h_n = self.encoder(input)
        return self.decoder(enc_outputs, enc_h_n, targets, forcing)

    def convert_to_string(self, inputs):
        with open("t_idx2token.json", 'r', encoding="utf-8") as f:
            idx2token = {int(idx): token for idx, token in json.load(f).items()}

        line = "\n"
        for token in inputs:
            line += idx2token[token.item()]

        return line

    def predict(self, input_tensor, max_len=50):
        """
        Inference forward pass.

        Args:
            input_tensor: (batch_size, src_len)

        Returns:
            predicted_sequences: (batch_size, decoded_len)
        """
        self.eval()
        with torch.no_grad():
            enc_outputs, enc_h_n = self.encoder(input_tensor)
            return (self.decoder.inference(enc_outputs, enc_h_n, max_len=max_len))

def train_nn(epochs=5, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    article_data = ArticleData()
    print("Test 1")
    vocab_size = 2500
    MEMORY_LIMIT = 10000

    model = Seq2Seq( vocab_size, vocab_size, 32, 32, 128, 2, vocab_size -2, vocab_size-1).to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Test 2")
    for epoch in range(epochs):
        article_loader = DataLoader(article_data, batch_size=1, shuffle=True)
        progress_bar = tqdm(article_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, data in enumerate(progress_bar):
            model.train()
            inputs, targets = data

            if len(inputs) > MEMORY_LIMIT:
                continue

            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs, targets, 0.0)
            output = output.view(-1, output.size(-1)).to(device)
            target = targets.view(-1).to(device)

            loss = loss_fn(output, target).to(device)

            if i % batch_size == batch_size -1:
                optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            predication_tensor = model.predict(inputs)[0]
            print(model.convert_to_string(predication_tensor))

        torch.save(model.state_dict(), "article.pt")

def summarize(article_filename = "testFile.txt") -> None:
    vocab_size = 2500
    model = Seq2Seq(vocab_size, 9, 32, 32, 128, 2, vocab_size-2, vocab_size-1)
    model.load_state_dict(torch.load("article.pt", map_location="cpu"))
    model.eval()
    with open(article_filename, 'r', encoding="utf-8") as f:
        article = json.load(f)
    input_tensor = torch.tensor(article)
    tensor_prediction = model.predict(input_tensor)[0]
    print(model.convert_to_string(tensor_prediction, ))

train_nn()
summarize()