import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Parameter,
    Embedding,
    Linear,
    MaxPool1d,
    AvgPool1d,
    Dropout,
    LSTM,
)
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
import numpy as np
import datetime


device = "cpu" if not torch.cuda.is_available() else "cuda"


class DKVMNHeadGroup(nn.Module):
    def forward(self, input_):
        pass

    def __init__(self, memory_size, memory_state_dim, is_write):
        super(DKVMNHeadGroup, self).__init__()
        """"
        Parameters
            memory_size:        scalar
            memory_state_dim:   scalar
            is_write:           boolean
        """
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        if self.is_write:
            self.erase = torch.nn.Linear(
                self.memory_state_dim, self.memory_state_dim, bias=True
            )
            self.add = torch.nn.Linear(
                self.memory_state_dim, self.memory_state_dim, bias=True
            )
            nn.init.kaiming_normal_(self.erase.weight)
            nn.init.kaiming_normal_(self.add.weight)
            nn.init.constant_(self.erase.bias, 0)
            nn.init.constant_(self.add.bias, 0)

    @staticmethod
    def addressing(control_input, memory):
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim)
            memory:                 Shape (memory_size, memory_state_dim)
        Returns
            correlation_weight:     Shape (batch_size, memory_size)
        """
        similarity_score = torch.matmul(control_input, torch.t(memory))  # torch.t做转置
        correlation_weight = F.softmax(
            similarity_score, dim=1
        )  # Shape: (batch_size, memory_size)
        return correlation_weight

    ## Read Process By Sum Memory By Read_Weight
    def read(self, memory, control_input=None, read_weight=None):
        """
        Parameters
            control_input:  Shape (batch_size, control_state_dim)
            memory:         Shape (batch_size, memory_size, memory_state_dim)
            read_weight:    Shape (batch_size, memory_size)
        Returns
            read_content:   Shape (batch_size,  memory_state_dim)
        """
        if read_weight is None:
            read_weight = self.addressing(control_input=control_input, memory=memory)
        read_weight = read_weight.view(-1, 1)  # 列tensor
        memory = memory.view(-1, self.memory_state_dim)
        rc = torch.mul(read_weight, memory)  # 矩阵对应位相乘，两者维度要相等
        read_content = rc.view(-1, self.memory_size, self.memory_state_dim)
        read_content = torch.sum(read_content, dim=1)
        return read_content

    def write(self, control_input, memory, write_weight=None):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """
        assert self.is_write
        if write_weight is None:
            write_weight = self.addressing(control_input=control_input, memory=memory)
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        erase_reshape = erase_signal.view(-1, 1, self.memory_state_dim)
        add_reshape = add_signal.view(-1, 1, self.memory_state_dim)
        write_weight_reshape = write_weight.view(-1, self.memory_size, 1)
        erase_mul = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        memory = memory.to(device)
        if add_mul.shape[0] < memory.shape[0]:
            sub_memory = memory[: add_mul.shape[0], :, :]
            new_memory = torch.cat(
                [
                    sub_memory * (1 - erase_mul) + add_mul,
                    memory[add_mul.shape[0] :, :, :],
                ],
                dim=0,
            )
        else:
            new_memory = memory * (1 - erase_mul) + add_mul
        return new_memory


class DKVMN(nn.Module):
    def forward(self, input_):
        pass

    def __init__(
        self,
        memory_size,
        memory_key_state_dim,
        memory_value_state_dim,
        init_memory_key,
        memory_value=None,
    ):
        """
        :param memory_size:             scalar
        :param memory_key_state_dim:    scalar
        :param memory_value_state_dim:  scalar
        :param init_memory_key:         Shape (memory_size, memory_value_state_dim)
        """
        super(DKVMN, self).__init__()
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.key_head = DKVMNHeadGroup(
            memory_size=self.memory_size,
            memory_state_dim=self.memory_key_state_dim,
            is_write=False,
        )
        self.value_head = DKVMNHeadGroup(
            memory_size=self.memory_size,
            memory_state_dim=self.memory_value_state_dim,
            is_write=True,
        )

        self.memory_key = init_memory_key

    def attention(self, control_input):
        correlation_weight = self.key_head.addressing(
            control_input=control_input, memory=self.memory_key
        )
        return correlation_weight

    def read(self, read_weight, memory_value):
        read_content = self.value_head.read(
            memory=memory_value, read_weight=read_weight
        )

        return read_content

    def write(self, write_weight, control_input, memory_value):
        memory_value = self.value_head.write(
            control_input=control_input, memory=memory_value, write_weight=write_weight
        )

        return memory_value


class SKVMN(Module):
    def __init__(
        self,
        num_c,
        dim_s,
        size_m,
        dropout=0.2,
        emb_type="qid",
        emb_path="",
        use_onehot=False,
    ):
        super().__init__()
        self.model_name = "skvmn"
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type
        self.use_onehot = use_onehot
        print(f"self.use_onehot: {self.use_onehot}")

        if emb_type.startswith("qid"):
            self.k_emb_layer = Embedding(self.num_c, self.dim_s)
            self.x_emb_layer = Embedding(2 * self.num_c + 1, self.dim_s)
            self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
            self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.mem = DKVMN(
            memory_size=size_m,
            memory_key_state_dim=dim_s,
            memory_value_state_dim=dim_s,
            init_memory_key=self.Mk,
        )

        if self.use_onehot:
            self.a_embed = nn.Linear(self.num_c + self.dim_s, self.dim_s, bias=True)
        else:
            self.a_embed = nn.Linear(self.dim_s * 2, self.dim_s, bias=True)
        self.v_emb_layer = Embedding(self.dim_s * 2, self.dim_s)
        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.hx = Parameter(torch.Tensor(1, self.dim_s))
        self.cx = Parameter(torch.Tensor(1, self.dim_s))
        kaiming_normal_(self.hx)
        kaiming_normal_(self.cx)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)
        self.lstm_cell = nn.LSTMCell(self.dim_s, self.dim_s)

    def ut_mask(self, seq_len):
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=0).to(dtype=torch.bool)

    def triangular_layer(
        self, correlation_weight, batch_size=64, a=0.075, b=0.088, c=1.00
    ):
        batch_identity_indices = []

        # w'= max((w-a)/(b-a), (c-w)/(c-b))
        # min(w', 0)
        correlation_weight = correlation_weight.view(
            batch_size * self.seqlen, -1
        )  # (seqlen * bz) * |K|
        correlation_weight = torch.cat(
            [correlation_weight[i] for i in range(correlation_weight.shape[0])], 0
        ).unsqueeze(
            0
        )  # 1*(seqlen*bz*|K|)
        correlation_weight = torch.cat(
            [(correlation_weight - a) / (b - a), (c - correlation_weight) / (c - b)], 0
        )
        correlation_weight, _ = torch.min(correlation_weight, 0)
        w0 = torch.zeros(correlation_weight.shape[0]).to(device)
        correlation_weight = torch.cat(
            [correlation_weight.unsqueeze(0), w0.unsqueeze(0)], 0
        )
        correlation_weight, _ = torch.max(correlation_weight, 0)

        identity_vector_batch = torch.zeros(correlation_weight.shape[0]).to(device)

        identity_vector_batch = identity_vector_batch.masked_fill(
            correlation_weight.lt(0.1), 0
        )
        identity_vector_batch = identity_vector_batch.masked_fill(
            correlation_weight.ge(0.1), 1
        )
        _identity_vector_batch = identity_vector_batch.masked_fill(
            correlation_weight.ge(0.6), 2
        )

        identity_vector_batch = _identity_vector_batch.view(
            batch_size * self.seqlen, -1
        )
        identity_vector_batch = torch.reshape(
            identity_vector_batch, [batch_size, self.seqlen, -1]
        )

        # A^2
        iv_square_norm = torch.sum(
            torch.pow(identity_vector_batch, 2), dim=2, keepdim=True
        )
        iv_square_norm = iv_square_norm.repeat((1, 1, iv_square_norm.shape[1]))
        # B^2.T
        unique_iv_square_norm = torch.sum(
            torch.pow(identity_vector_batch, 2), dim=2, keepdim=True
        )
        unique_iv_square_norm = unique_iv_square_norm.repeat(
            (1, 1, self.seqlen)
        ).transpose(2, 1)
        # A * B.T
        iv_matrix_product = torch.bmm(
            identity_vector_batch, identity_vector_batch.transpose(2, 1)
        )  # A * A.T
        # A^2 + B^2 - 2A*B.T
        iv_distances = iv_square_norm + unique_iv_square_norm - 2 * iv_matrix_product
        iv_distances = torch.where(
            iv_distances > 0.0, torch.tensor(-1e32).to(device), iv_distances
        )
        masks = self.ut_mask(iv_distances.shape[1]).to(device)
        mask_iv_distances = iv_distances.masked_fill(
            masks, value=torch.tensor(-1e32).to(device)
        )
        idx_matrix = (
            torch.arange(0, self.seqlen * self.seqlen, 1)
            .reshape(self.seqlen, -1)
            .repeat(batch_size, 1, 1)
            .to(device)
        )
        final_iv_distance = mask_iv_distances + idx_matrix
        values, indices = torch.topk(final_iv_distance, 1, dim=2, largest=True)

        _values = values.permute(1, 0, 2)
        _indices = indices.permute(1, 0, 2)
        batch_identity_indices = (_values >= 0).nonzero()  # 找到t
        identity_idx = []
        for identity_indices in batch_identity_indices:
            pre_idx = _indices[identity_indices[0], identity_indices[1]]  # 找到t-lamda
            idx = torch.cat([identity_indices[:-1], pre_idx], dim=-1)
            identity_idx.append(idx)
        if len(identity_idx) > 0:
            identity_idx = torch.stack(identity_idx, dim=0)
        else:
            identity_idx = torch.tensor([]).to(device)

        return identity_idx

    def forward(self, q, r, qtest=False):
        emb_type = self.emb_type
        bs = q.shape[0]
        self.seqlen = q.shape[1]

        if emb_type == "qid":
            x = q + self.num_c * r
            k = self.k_emb_layer(q)

        if self.use_onehot:
            q_data = q.reshape(bs * self.seqlen, 1)
            r_onehot = torch.zeros(bs * self.seqlen, self.num_c).long().to(device)
            r_data = (
                r.unsqueeze(2)
                .expand(-1, -1, self.num_c)
                .reshape(bs * self.seqlen, self.num_c)
            )
            r_onehot_content = r_onehot.scatter(1, q_data, r_data).reshape(
                bs, self.seqlen, -1
            )

        value_read_content_l = []
        input_embed_l = []
        correlation_weight_list = []
        ft = []

        mem_value = self.Mv0.unsqueeze(0).repeat(bs, 1, 1).to(device)
        for i in range(self.seqlen):
            q = k.permute(1, 0, 2)[i]
            correlation_weight = self.mem.attention(q).to(device)

            read_content = self.mem.read(correlation_weight, mem_value)  # [bs, dim_s]

            # modify
            correlation_weight_list.append(correlation_weight)  # [bs, size_m]

            ## save intermedium data
            value_read_content_l.append(read_content)
            input_embed_l.append(q)

            # modify
            batch_predict_input = torch.cat([read_content, q], 1)  ###q: 是r emb后的qemb
            f = torch.tanh(self.f_layer(batch_predict_input))
            ft.append(f)

            if self.use_onehot:
                y = r_onehot_content[:, i, :]
            else:
                y = self.x_emb_layer(x[:, i])

            write_embed = torch.cat([f, y], 1)  # bz * 2dim_s
            write_embed = self.a_embed(write_embed).to(device)  # [bs, dim_s]
            new_memory_value = self.mem.write(
                correlation_weight, write_embed, mem_value
            )
            mem_value = new_memory_value

        w = torch.cat(
            [correlation_weight_list[i].unsqueeze(1) for i in range(self.seqlen)], 1
        )
        ft = torch.stack(ft, dim=0)

        idx_values = self.triangular_layer(w, bs)

        # Hop-LSTM
        # original
        hidden_state, cell_state = [], []
        hx, cx = self.hx.repeat(bs, 1), self.cx.repeat(bs, 1)
        for i in range(self.seqlen):  # 逐个ex进行计算
            for j in range(bs):
                if (
                    idx_values.shape[0] != 0
                    and i == idx_values[0][0]
                    and j == idx_values[0][1]
                ):
                    hx[j, :] = hidden_state[idx_values[0][2]][j]
                    cx = cx.clone()
                    cx[j, :] = cell_state[idx_values[0][2]][j]
                    idx_values = idx_values[1:]
            hx, cx = self.lstm_cell(ft[i], (hx, cx))
            hidden_state.append(hx)
            cell_state.append(cx)
        hidden_state = torch.stack(hidden_state, dim=0).permute(1, 0, 2)
        cell_state = torch.stack(cell_state, dim=0).permute(1, 0, 2)

        p = self.p_layer(self.dropout_layer(hidden_state))
        p = torch.sigmoid(p)
        p = p.squeeze(-1)
        if not qtest:
            return p
        else:
            return p, hidden_state
