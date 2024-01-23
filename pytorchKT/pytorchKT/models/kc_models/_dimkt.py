import torch
import torch.nn as nn
from torch.autograd import Variable


class DIMKT_CC(nn.Module):
    def __init__(
        self,
        num_c,
        dropout,
        emb_size,
        batch_size,
        num_steps,
        difficult_levels,
        emb_type,
    ) -> None:
        super().__init__()
        self.model_name = "dimkt_cc"
        self.num_c = num_c
        self.emb_type = emb_type
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.difficult_levels = difficult_levels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        if emb_type.startswith("qid"):
            self.interaction_emb = nn.Embedding(self.num_c * 2, self.emb_size)

        self.knowledge = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(1, self.emb_size)), requires_grad=True
        )

        self.c_emb = nn.Embedding(self.num_c + 1, self.emb_size, padding_idx=0)
        self.sd_emb = nn.Embedding(
            self.difficult_levels + 2, self.emb_size, padding_idx=0
        )
        self.a_emb = nn.Embedding(2, self.emb_size)

        self.linear_1 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.linear_2 = nn.Linear(1 * self.emb_size, self.emb_size)
        self.linear_3 = nn.Linear(1 * self.emb_size, self.emb_size)
        self.linear_4 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.linear_5 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.linear_6 = nn.Linear(3 * self.emb_size, self.emb_size)

    def forward(self, c, sd, a, cshft, sdshft):
        if self.batch_size != len(c):
            self.batch_size = len(c)

        c_emb = self.c_emb(Variable(c))
        sd_emb = self.sd_emb(Variable(sd))
        a_emb = self.a_emb(Variable(a))

        target_c = self.c_emb(Variable(cshft))
        target_sd = self.sd_emb(Variable(sdshft))

        input_data = torch.cat((c_emb, sd_emb), -1)
        input_data = self.linear_1(input_data)

        target_data = torch.cat((target_c, target_sd), -1)
        target_data = self.linear_1(target_data)

        shape = list(sd_emb.shape)
        padd = torch.zeros(shape[0], 1, shape[2], device=self.device)
        sd_emb = torch.cat((padd, sd_emb), 1)
        slice_sd_embedding = sd_emb.split(1, dim=1)

        shape = list(a_emb.shape)
        padd = torch.zeros(shape[0], 1, shape[2], device=self.device)
        a_emb = torch.cat((padd, a_emb), 1)
        slice_a_embedding = a_emb.split(1, dim=1)

        shape = list(input_data.shape)
        padd = torch.zeros(shape[0], 1, shape[2], device=self.device)
        input_data = torch.cat((padd, input_data), 1)
        slice_input_data = input_data.split(1, dim=1)

        k = self.knowledge.repeat(self.batch_size, 1).cuda()

        h = list()
        seqlen = c.size(1)
        for i in range(1, seqlen + 1):
            sd_1 = torch.squeeze(slice_sd_embedding[i], 1)
            a_1 = torch.squeeze(slice_a_embedding[i], 1)
            input_data_1 = torch.squeeze(slice_input_data[i], 1)

            qq = k - input_data_1

            gates_SDF = self.linear_2(qq)
            gates_SDF = self.sigmoid(gates_SDF)
            SDFt = self.linear_3(qq)
            SDFt = self.tanh(SDFt)
            SDFt = self.dropout(SDFt)

            SDFt = gates_SDF * SDFt

            x = torch.cat((SDFt, a_1), -1)

            gates_PKA = self.linear_4(x)
            gates_PKA = self.sigmoid(gates_PKA)

            PKAt = self.linear_5(x)
            PKAt = self.tanh(PKAt)

            PKAt = gates_PKA * PKAt

            ins = torch.cat((k, a_1, sd_1), -1)
            gates_KSU = self.linear_6(ins)
            gates_KSU = self.sigmoid(gates_KSU)

            k = gates_KSU * k + (1 - gates_KSU) * PKAt

            h_i = torch.unsqueeze(k, dim=1)
            h.append(h_i)

        output = torch.cat(h, axis=1)
        logits = torch.sum(target_data * output, dim=-1)
        y = self.sigmoid(logits)

        return y
