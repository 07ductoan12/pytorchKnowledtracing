# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HawkesKT(nn.Module):
    # def __init__(self, args, corpus):
    def __init__(self, n_skills, n_problems, emb_size, time_log, emb_type="qid"):
        super().__init__()
        self.model_name = "hawkes"
        self.emb_type = emb_type
        self.problem_num = n_problems
        self.skill_num = n_skills
        self.emb_size = emb_size
        self.time_log = time_log
        self.gpu = device

        self.problem_base = torch.nn.Embedding(self.problem_num, 1)
        self.skill_base = torch.nn.Embedding(self.skill_num, 1)

        self.alpha_inter_embeddings = torch.nn.Embedding(
            self.skill_num * 2, self.emb_size
        )
        self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = torch.nn.Embedding(
            self.skill_num * 2, self.emb_size
        )
        self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)

        self.count = 0

    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def printparams(self):
        print("=" * 20)
        for m in list(self.named_parameters()):
            print(m[0], m[1])
        self.count += 1
        print(f"count: {self.count}")

    def forward(self, skills, problems, times, labels, qtest=False):
        # assert False
        mask_labels = labels  # * (sm == 1).long()#labels * (labels > -1).long()

        # # assert labels == mask_labels
        inters = skills + mask_labels * self.skill_num
        # print(f"inters: {inters}")

        alpha_src_emb = self.alpha_inter_embeddings(inters)  # [bs, seq_len, emb]
        # print(f"alpha_src_emb:{alpha_src_emb}")
        alpha_target_emb = self.alpha_skill_embeddings(skills)
        # print(f"alpha_target_emb:{alpha_target_emb}")
        alphas = torch.matmul(
            alpha_src_emb, alpha_target_emb.transpose(-2, -1)
        )  # [bs, seq_len, seq_len]
        # print(f"alphas:{alphas}")
        beta_src_emb = self.beta_inter_embeddings(inters)  # [bs, seq_len, emb]
        # print(f"beta_src_emb:{beta_src_emb}")
        beta_target_emb = self.beta_skill_embeddings(skills)
        # print(f"beta_target_emb:{beta_target_emb}")
        betas = torch.matmul(
            beta_src_emb, beta_target_emb.transpose(-2, -1)
        )  # [bs, seq_len, seq_len]
        betas = torch.clamp(betas + 1, min=0, max=10)

        if times.shape[1] > 0:
            times = times.double() / 1000
            delta_t = (times[:, :, None] - times[:, None, :]).abs().double()
            # print(times.shape, delta_t)
            # assert False
        else:
            # 1 if no timestamps
            delta_t = (
                torch.ones(skills.shape[0], skills.shape[1], skills.shape[1])
                .double()
                .to(device)
            )
        delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

        cross_effects = alphas * torch.exp(-betas * delta_t)

        seq_len = skills.shape[1]
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        mask = torch.from_numpy(valid_mask) == 0
        mask = mask.cuda() if self.gpu != "" else mask
        sum_t = cross_effects.masked_fill(mask, 0).sum(-2)

        problem_bias = self.problem_base(problems).squeeze(dim=-1)
        skill_bias = self.skill_base(skills).squeeze(dim=-1)
        prediction = (problem_bias + skill_bias + sum_t).sigmoid()

        h = problem_bias + skill_bias + sum_t
        if not qtest:
            return prediction
        else:
            return prediction, h
