import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from apex import amp
import copy

EPS = 1e-6


class dpaNet(nn.Module):
    def __init__(self, N, L, B, H, K, R, C):
       
        super(dpaNet, self).__init__()
        self.N, self.L, self.B, self.H, self.D, self.R, self.C = N, L, B, H, K, R, C
        
        self.encoder = Encoder(L, N)
        self.separator = dpa(N, B, H, K, R, C)
        self.decoder = Decoder(N, L)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source

class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [M, C, N, K]
        source_w = torch.transpose(source_w, 2, 3) # [M, C, K, N]
        est_source = self.basis_signals(source_w)  # [M, C, K, L]
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T
        return est_source


class Dual_DPA_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels,
                 dropout=0.1,num_spks=2):
        super(Dual_DPA_Block, self).__init__()
        self.intra_a = cmEncoderLayer(out_channels,nhead = 1,dim_feedforward = hidden_channels, dropout=dropout)
        self.inter_a = cmEncoderLayer(out_channels,nhead = 1,dim_feedforward = hidden_channels, dropout=dropout)
        # Norm
        self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.inter_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [K, BS, N]
        intra_rnn = x.permute(2, 0, 3, 1).contiguous().view(K, B*S, N)
        # [K, BS, N]
        intra_rnn = self.intra_a(intra_rnn)
        # [K, B, S, N]
        intra_rnn = intra_rnn.view(K, B, S, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(1, 3, 0, 2).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)
        
        # [B, N, K, S]
        intra_rnn = intra_rnn + x
        
        # inter RNN
        # [S, BK, N]
        inter_rnn = intra_rnn.permute(3, 0, 2, 1).contiguous().view(S, B*K, N)
        # [S, BK, H]
        inter_rnn = self.inter_a(inter_rnn)
        # [S, B, K, N]
        inter_rnn = inter_rnn.view(S, B, K, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(1, 3, 2, 0).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn
        return out

class dpa(nn.Module):
    def __init__(self, N, B, H, K, R, C):
        super(dpa, self).__init__()
        self.C ,self.K , self.R = C, K, R
        # [M, N, K] -> [M, N, K]
        self.layer_norm = nn.GroupNorm(1, N, eps=1e-8)
        # [M, N, K] -> [M, B, K]
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)

        self.dual_dpa = nn.ModuleList([])
        for i in range(R):
            self.dual_dpa.append(Dual_DPA_Block(B, H,dropout=0.1))

        self.prelu = nn.PReLU()
        self.mask_conv1x1 = nn.Conv1d(B, C*N, 1, bias=False)

    def forward(self, x):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, D = x.size()

        x = self.layer_norm(x) # [M, N, K]
        x = self.bottleneck_conv1x1(x) # [M, B, K]
        x, gap = self._Segmentation(x, self.K) # [M, B, k, S]

        for i in range(self.R):
            x = self.dual_dpa[i](x)

        x = self._over_add(x, gap)

        x = self.prelu(x)
        x = self.mask_conv1x1(x)

        x = x.view(M, self.C, N, D) # [M, C*N, K] -> [M, C, N, K]
        x = F.relu(x)
        return x

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap


    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input



# class cmEncoder(nn.Module):
#     def __init__(self, encoder_layer, num_layers):
#         super(cmEncoder,self).__init__()
#         self.layers = _clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
    
#     def forward(self, a_src):
#         for mod in self.layers:
#             a_src = mod(a_src)
#         return a_src

def _clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class cmEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(cmEncoderLayer,self).__init__()
        self.multihead_ca = multihead_ca(d_model, nhead, att_dropout = dropout)

        self.a_linear1 = nn.Linear(d_model, dim_feedforward)
        self.a_dropout = nn.Dropout(dropout)
        self.a_linear2 = nn.Linear(dim_feedforward, d_model)
        self.a_norm1 = nn.LayerNorm(d_model)
        self.a_norm2 = nn.LayerNorm(d_model)
        self.a_dropout1 = nn.Dropout(dropout)
        self.a_dropout2 = nn.Dropout(dropout)


    def forward(self, a_src):
        a_src2 = self.multihead_ca(a_src)

        a_src = a_src + self.a_dropout1(a_src2)
        a_src = self.a_norm1(a_src)
        a_src2 = self.a_linear2(self.a_dropout(F.relu(self.a_linear1(a_src))))
        a_src = a_src + self.a_dropout2(a_src2)
        a_src = self.a_norm2(a_src)

        return a_src

class multihead_ca(nn.Module):
    def __init__(self, d_model, nhead, att_dropout):
        super(multihead_ca, self).__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_qkv = d_model // nhead
        self.a_linears = _clones(nn.Linear(d_model, d_model), 4)
        self.a_att_dropout = nn.Dropout(att_dropout)

    def forward(self, a_src):
        # [seq, batch, dim]
        nbatches = a_src.size(1)

        a_src = a_src.transpose(0,1) # [Batch, Seq, Dim]

        a_query, a_key, a_value = \
            [l(a).view(nbatches, -1, self.nhead, self.d_qkv).transpose(1, 2)
            for l, a in zip(self.a_linears, (a_src, a_src, a_src))]   # [batch, seq, head, dim] -> [batch, head, seq, dim]


        a_scores = torch.matmul(a_query, a_key.transpose(-1, -2)) / math.sqrt(self.d_qkv) # [batch, head, seq_q, seq_av_key]
        a_p_attn = F.softmax(a_scores, dim = -1)
        a_p_attn = self.a_att_dropout(a_p_attn)
        a = torch.matmul(a_p_attn, a_value).transpose(1,2).transpose(0,1) # [batch, head, seq, dim] -> [seq, batch, head, dim]
        a = a.contiguous().view(-1, nbatches, self.nhead * self.d_qkv)

        return self.a_linears[-1](a)


@amp.float_function
def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result
