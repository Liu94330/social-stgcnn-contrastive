import torch
import torch.nn as nn
import torch.nn.functional as Func

class GraphContrastiveLearning(nn.Module):
    def __init__(self, feature_dim, temperature=0.5):
        super(GraphContrastiveLearning, self).__init__()
        self.temperature = temperature
        self.feature_dim = feature_dim

    def forward(self, features):
        # 计算相似度矩阵
        sim_matrix = Func.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1) / self.temperature
        labels = torch.arange(features.size(0)).cuda()
        
        # 计算对比损失
        loss = Func.cross_entropy(sim_matrix, labels)
        return loss


class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A
    

class st_gcn_with_contrastive(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 feature_dim,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn_with_contrastive, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()
        self.contrastive_loss = GraphContrastiveLearning(feature_dim)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        # 特征提取
        features = x.mean(dim=[2, 3])  # 平均池化，提取图特征

        # 计算对比学习损失
        contrastive_loss = self.contrastive_loss(features)

        x = self.prelu(x)
        return x, A, contrastive_loss


class social_stgcnn_with_contrastive(nn.Module):
    def __init__(self, n_stgcnn=1, n_txpcnn=1, input_feat=2, output_feat=5, 
                 seq_len=8, pred_seq_len=12, kernel_size=3, feature_dim=128):
        super(social_stgcnn_with_contrastive, self).__init__()

        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn_with_contrastive(input_feat, output_feat, (kernel_size, seq_len), feature_dim))
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn_with_contrastive(output_feat, output_feat, (kernel_size, seq_len), feature_dim))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_output = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

    def forward(self, v, a):
        total_contrastive_loss = 0
        for k in range(self.n_stgcnn):
            v, a, contrastive_loss = self.st_gcns[k](v, a)
            total_contrastive_loss += contrastive_loss

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_output(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        return v, a, total_contrastive_loss