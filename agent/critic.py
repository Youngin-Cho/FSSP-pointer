import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config, load_pkl, pkl_parser


class PtrNet2(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Embedding = nn.Linear(params["num_of_process"], params["n_embedding"], bias=False)
        self.Encoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)
        if torch.cuda.is_available():
            self.Vec = nn.Parameter(torch.cuda.FloatTensor(params["n_hidden"]))
        else:
            self.Vec = nn.Parameter(torch.FloatTensor(params["n_hidden"]))
        self.W_q = nn.Linear(params["n_hidden"], params["n_hidden"], bias=True)
        self.W_ref = nn.Conv1d(params["n_hidden"], params["n_hidden"], 1, 1)
        self.final2FC = nn.Sequential(
            nn.Linear(params["n_hidden"], params["n_hidden"], bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(params["n_hidden"], 1, bias=False))
        self._initialize_weights(params["init_min"], params["init_max"])
        self.n_glimpse = params["n_glimpse"]
        self.n_process = params["n_process"]

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x, device):
        '''	x: (batch, block_num, process_num)
            enc_h: (batch, block_num, embed)
            query(Decoder input): (batch, 1, embed)
            h: (1, batch, embed)
            return: pred_l: (batch)
        '''
        x = x.to(device)
        embed_enc_inputs = self.Embedding(x)
        enc_h, (h, c) = self.Encoder(embed_enc_inputs, None)
        ref = enc_h
        query = h[-1]

        for i in range(self.n_process):
            query = self.glimpse(query, ref)

        pred_l = self.final2FC(query).squeeze(-1)
        return pred_l

    def glimpse(self, query, ref, infinity=1e8):
        """	Args:
            query: the hidden state of the decoder at the current
            (batch, 128)
            ref: the set of hidden states from the encoder.
            (batch, city_t, 128)
        """
        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, city_t)
        u2 = self.W_ref(ref.permute(0, 2, 1))  # u2: (batch, 128, city_t)
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128,block_num) => u: (batch, 1, block_num) => (batch, block_num)
        a = F.softmax(u, dim=1)
        g = torch.bmm(a.unsqueeze(1), ref).squeeze(1)
        # d = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
        # u2: (batch, 128, block_num) * a: (batch, block_num, 1) => d: (batch, 128)
        return g


if __name__ == '__main__':
    cfg = load_pkl(pkl_parser().path)
    model = PtrNet2(cfg)
    inputs = torch.randn(3, 20, 2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    pred_l = model(inputs, device)
    print('pred_length:', pred_l.size(), pred_l)

    cnt = 0
    for i, k in model.state_dict().items():
        print(i, k.size(), torch.numel(k))
        cnt += torch.numel(k)
    print('total parameters:', cnt)

# pred_l.mean().backward()
# print(model.W_q.weight.grad)