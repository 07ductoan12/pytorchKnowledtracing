import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout


class DKTPlus(Module):
    def __init__(
        self,
        num_c,
        emb_size,
        lambda_r,
        lambda_w1,
        lambda_w2,
        dropout=0.1,
        emb_type="qid",
    ) -> None:
        super().__init__()
        self.model_name = "dkt+"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.lambda_r = lambda_r
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)

    def forward(self, q, r):
        emb_type = self.emb_type
        if emb_type == "qid":
            inp = q + self.num_c * r
            xemb = self.interaction_emb(inp)

        out, _ = self.lstm_layer(xemb)
        out = self.dropout_layer(out)
        out = self.out_layer(out)
        out = torch.sigmoid(out)
        return out
