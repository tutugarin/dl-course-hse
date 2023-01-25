import torch
from torch import nn
from dataset import TextDataset

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: nn.RNNBase = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        self.embedding: nn.Embedding = nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim=embed_size, 
            padding_idx=dataset.pad_id
        )
        self.rnn: nn.RNNBase = rnn_type(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            batch_first=True,
            num_layers=rnn_layers
        )
        self.linear: nn.Linear = nn.Linear(hidden_size, self.vocab_size)
        self.device = None

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        # https://github.com/isadrtdinov/intro-to-dl-hse/blob/2022-2023/seminars/202/seminar-07-rnn.ipynb

        embeds = self.embedding(indices)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.rnn(packed_embeds)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        logits = self.linear(outputs)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        # https://github.com/isadrtdinov/intro-to-dl-hse/blob/2022-2023/seminars/202/seminar-07-rnn.ipynb

        tokens = []
        tokens.append(self.dataset.bos_id)
        tokens.extend(self.dataset.text2ids(prefix))
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)

        embeds = self.embedding(tokens)
        output, hidden = self.rnn(embeds)
        logits = self.linear(output) / temp

        new_tokens = Categorical(logits=logits[:, -1:]).sample()
        tokens = torch.cat([tokens, new_tokens], dim=1)

        while tokens.shape[1] < self.max_length:
            if new_tokens.item() == self.dataset.eos_id:
                break

            embeds = self.embedding(new_tokens)
            output, hidden = self.rnn(embeds, hidden)
            logits = self.linear(output) / temp

            new_tokens = Categorical(logits=logits[:, -1:]).sample()
            tokens = torch.cat([tokens, new_tokens], dim=1)

        generated = self.dataset.ids2text(tokens.squeeze())
        return generated

    def to(self, device, **kwargs):
      self.device = device
      return super().to(device, **kwargs)
