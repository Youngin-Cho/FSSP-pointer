import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, load_pkl, pkl_parser
from env import PanelBlockShop


class Greedy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_p):
        return torch.argmax(log_p, dim=1).long()


class Categorical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_p):
        return torch.multinomial(log_p.exp(), 1).long().squeeze(1)


class PtrNet1(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Embedding = nn.Linear(6, params["n_embedding"], bias=False)
        self.Encoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)
        self.Decoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)
        if torch.cuda.is_available():
            self.Vec = nn.Parameter(torch.cuda.FloatTensor(params["n_embedding"]))
            self.Vec2 = nn.Parameter(torch.cuda.FloatTensor(params["n_embedding"]))
        else:
            self.Vec = nn.Parameter(torch.FloatTensor(params["n_embedding"]))
            self.Vec2 = nn.Parameter(torch.FloatTensor(params["n_embedding"]))
        self.W_q = nn.Linear(params["n_hidden"], params["n_hidden"], bias=True)
        self.W_ref = nn.Conv1d(params["n_hidden"], params["n_hidden"], 1, 1)
        self.W_q2 = nn.Linear(params["n_hidden"], params["n_hidden"], bias=True)
        self.W_ref2 = nn.Conv1d(params["n_hidden"], params["n_hidden"], 1, 1)
        self.dec_input = nn.Parameter(torch.FloatTensor(params["n_embedding"]))
        self._initialize_weights(params["init_min"], params["init_max"])
        self.clip_logits = params["clip_logits"]
        self.softmax_T = params["softmax_T"]
        self.n_glimpse = params["n_glimpse"]
        self.block_selecter = {'greedy': Greedy(), 'sampling': Categorical()}.get(params["decode_type"], None)

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x, device):
        '''	x: (batch, block_num, process_num)
            enc_h: (batch, block_num, embed)
            dec_input: (batch, 1, embed)
            h: (1, batch, embed)
            return: pi: (batch, block_num), ll: (batch)
        '''
        x = x.to(device)
        batch, block_num, _ = x.size()
        embed_enc_inputs = self.Embedding(x)
        embed = embed_enc_inputs.size(2)
        mask = torch.zeros((batch, block_num), device=device)
        enc_h, (h, c) = self.Encoder(embed_enc_inputs, None)
        ref = enc_h
        pi_list, log_ps = [], []
        dec_input = self.dec_input.unsqueeze(0).repeat(batch, 1).unsqueeze(1).to(device)
        for i in range(block_num):
            _, (h, c) = self.Decoder(dec_input, (h, c))
            query = h.squeeze(0)
            for i in range(self.n_glimpse):
                query = self.glimpse(query, ref, mask)
            logits = self.pointer(query, ref, mask)
            log_p = torch.log_softmax(logits, dim=-1)
            next_block = self.block_selecter(log_p)
            dec_input = torch.gather(input=embed_enc_inputs, dim=1,
                                     index=next_block.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed))

            pi_list.append(next_block)
            log_ps.append(log_p)
            mask += torch.zeros((batch, block_num), device=device).scatter_(dim=1, index=next_block.unsqueeze(1), value=1)

        pi = torch.stack(pi_list, dim=1)
        ll = self.get_log_likelihood(torch.stack(log_ps, 1), pi)
        return pi, ll

    def glimpse(self, query, ref, mask, inf=1e8):
        """	-ref about torch.bmm, torch.matmul and so on
            https://qiita.com/tand826/items/9e1b6a4de785097fe6a5
            https://qiita.com/shinochin/items/aa420e50d847453cc296

                Args:
            query: the hidden state of the decoder at the current
            (batch, 128)
            ref: the set of hidden states from the encoder.
            (batch, block_num, 128)
            mask: model only points at cities that have yet to be visited, so prevent them from being reselected
            (batch, block_num)
        """
        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, block_num)
        u2 = self.W_ref(ref.permute(0, 2, 1))  # u2: (batch, 128, block_num)
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, block_num) => u: (batch, 1, block_num) => (batch, block_num)
        u = u - inf * mask
        a = F.softmax(u / self.softmax_T, dim=1)
        d = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
        # u2: (batch, 128, block_num) * a: (batch, block_num, 1) => d: (batch, 128)
        return d

    def pointer(self, query, ref, mask, inf=1e8):
        """	Args:
            query: the hidden state of the decoder at the current
            (batch, 128)
            ref: the set of hidden states from the encoder.
            (batch, block_num, 128)
            mask: model only points at cities that have yet to be visited, so prevent them from being reselected
            (batch, block_num)
        """
        u1 = self.W_q2(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, block_num)
        u2 = self.W_ref2(ref.permute(0, 2, 1))  # u2: (batch, 128, block_num)
        V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, self.clip_logits * torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, block_num) => u: (batch, 1, block_num) => (batch, block_num)
        u = u - inf * mask
        return u

    def get_log_likelihood(self, _log_p, pi):
        """	args:
            _log_p: (batch, block_num, block_num)
            pi: (batch, block_num), predicted tour
            return: (batch)
        """
        log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None])
        return torch.sum(log_p.squeeze(-1), 1)