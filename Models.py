import torch
import torch.nn as nn
import math
import numpy as np
from entmax import sparsemax, entmax15, entmax_bisect

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x



class STKernelAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_regularization, num_subset, num_point, num_frame,
                 kernel_size, stride, num_point_0, att_s=True,use_spatial_att=True, attentiondrop=0.2, use_pes=True,sparsity_alpha=2 ):
        super(STKernelAttentionBlock, self).__init__()
        self.is_regularization = is_regularization
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset  # num-head
        self.att_s = att_s
        self.use_pes=use_pes
        self.sparsity_alpha = sparsity_alpha
        self.kernel_size = kernel_size
        self.num_point = num_point
        self.num_point_0 = num_point_0

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att


        if use_spatial_att:
            self.theta = torch.nn.Parameter(torch.zeros((1, num_subset, 1, 1)))
            self.atts = nn.Parameter(torch.zeros(size=(1, num_subset, num_point, num_point)))

            self.V_mapping = nn.Conv2d(in_channels, in_channels, 1, bias=True)
            self.pes = PositionalEncoding(in_channels, num_point, num_frame, 'spatial')
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),  # 不改，
                nn.BatchNorm2d(out_channels),
            )

            self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)

            self.learable_sparsity_alpha = nn.Parameter(torch.zeros(1))


            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                          bias=True, stride=(stride, 1)),

                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=(pad, 0), bias=True, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, (1, 1), padding=(0, 0), stride=(stride, 1), bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), padding=(0, 0), stride=(stride, 1), bias=True),
                nn.BatchNorm2d(out_channels),
            )

        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x

        # self.tan = nn.Tanh()

        self.soft = nn.Softmax(dim=-2)
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)



    def forward(self, x):

        N, C, T, V = x.size()  # batch_size,Channel,Frame,Nodes

        if self.use_spatial_att:
            if self.is_regularization == True:
                attention = self.atts
            else:
                attention = 0

            if self.use_pes:
                y = self.pes(x)
            else:
                y = x
            if self.att_s:

                y = y.permute(0, 3, 1, 2).contiguous()  # N,V,C,T
                dism = torch.cdist(y.view(y.shape[0], y.shape[1], y.shape[2] * y.shape[3]),
                                   # (N，V，CT ),(N,V,CT) --->N,V,V
                                   y.view(y.shape[0], y.shape[1], y.shape[2] * y.shape[3]), p=2)  # L2范数

                dism = torch.pow(dism / (C * T), 2)
                dism = dism / dism.max()  # 归一化

                dsexp = torch.exp(self.theta)  # e**(θ)

                dism = torch.unsqueeze(dism, 1).repeat(1, self.num_subset, 1, 1)  # N,S,V,V

                kernel_atten = torch.exp(-1 * dsexp * dism)

                '''sparsity'''
                if self.sparsity_alpha == 0:
                    kernel_atten = self.soft(kernel_atten)
                    attention = attention + kernel_atten * self.alphas

                elif self.sparsity_alpha == 2:
                    kernel_atten = sparsemax(kernel_atten, dim=-2)
                    attention = attention + kernel_atten * self.alphas

                elif self.sparsity_alpha == 1.5:
                    kernel_atten = entmax15(kernel_atten, dim=-2)
                    attention = attention + kernel_atten * self.alphas

                else:
                    learable_sparsity_alpha = torch.exp(torch.exp(self.learable_sparsity_alpha))
                    learable_sparsity_alpha = (1 / learable_sparsity_alpha) + 1     #learable_sparsity_alpha belongs to 1~2
                    kernel_atten = entmax_bisect(kernel_atten, alpha=learable_sparsity_alpha, dim=-2)
                    attention = attention + kernel_atten  # * self.alphas

            # io.savemat('./attention.mat',{'k0':kernel_atten.cpu().detach().numpy(),'k1':attention_df2.cpu().detach().numpy(),'k2':attention_df3.cpu().detach().numpy(),'k3':attention_df4.cpu().detach().numpy(),'global':self.attention0s.cpu().detach().numpy()})


            attention = self.drop(attention)

            if self.num_point > self.num_point_0:

                x_average = x.mean(dim=-2, keepdim=True)  # N,C,1,V
                y_average = torch.einsum('nctu,nsuv->nsctv', [x_average, attention]).contiguous() \
                    .view(N, self.num_subset * self.in_channels, -1, V)
                y = y_average.repeat(1, 1, T, 1) + x.repeat(1, self.num_subset, 1, 1)

            else:
                y = torch.einsum('nctu,nsuv->nsctv', [self.V_mapping(x), attention]).contiguous() \
                    .view(N, self.num_subset * self.in_channels, T, V)


            y = self.out_nets(y)  # nctv
            y = self.relu(self.downs1(x) + y)
            y = self.ff_nets(y)
            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)

        return y



class CrossKTnet(nn.Module):
    def __init__(self, sparsity_alpha, num_subset, num_frame, num_point, kernel_size, use_pes,config=None,
                 num_class=2,raw_channel=1, dropout=0.1, att_s=True, mean_pool=True,
                 use_spatial_att=True, attentiondrop=0.2,):

        super(CrossKTnet, self).__init__()

        self.out_channels = config[-1][1]
        self.config = config
        self.mean_pool = mean_pool
        in_channels = config[0][0]
        self.input_mapping = nn.Sequential(
            nn.Conv2d(raw_channel, in_channels, kernel_size=(3, 1), padding=(1, 0), stride=(1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        param = {
            'num_frame': num_frame,
            'num_point': num_point,
            'num_point_0':num_point,
            'num_subset': num_subset,
            'att_s': att_s,
            'use_spatial_att': use_spatial_att,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop,
            'sparsity_alpha': sparsity_alpha,
            'kernel_size': kernel_size,
        }

        self.graph_layers = nn.ModuleList()

        for index, (in_channels, out_channels, is_regularization, Conv_stride, chunk_kernel_size, total_chunk, num_subset) in enumerate(self.config):
            param['num_frame'] = num_frame
            param['num_point'] = num_point

            if total_chunk > 2:
                param['num_frame'] = int(param['num_frame'] / ((total_chunk + 1) / 2))
                param['num_point'] = num_point * total_chunk

            self.graph_layers.append(
                    STKernelAttentionBlock(in_channels, out_channels, is_regularization, stride=Conv_stride,  **param))
            num_frame = int(num_frame / Conv_stride)

        self.drop_out = nn.Dropout(dropout)
        self.fc = nn.Linear(self.config[-1][1], num_class)

        for m in self.modules():  # Initialize the weight of the model
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)


    def segment_frame_overlapping(self, x, kernel_size, total_chunk):
        """
       :param x: b,c,t,v   kernel_size:window-length, total_chunk:the number of channel
        :return: x1: b,c,t1,v1
        """
        chunk = int((total_chunk + 1) / 2)
        length = x.shape[-2]
        if chunk * kernel_size != length:
            raise ValueError("frames don't match the chunk and kernel_size")

        y1 = torch.chunk(x, chunk, dim=-2)
        y1 = torch.cat(y1, dim=-1)

        stride = int(kernel_size / 2)
        y2 = torch.chunk(x[:, :, stride:-stride, :], (chunk - 1), dim=-2)
        y2 = torch.cat(y2, dim=-1)

        y = torch.cat([y1, y2], dim=-1)
        return y

    def restore_frame_overlapping(self, x, kernel_size, total_chunk):
        """
        :param x: b,c,t,v   kernel_size:window-length, total_chunk:the number of channel
         :return: x1: b,c,t2,v2
        """
        chunk = int((total_chunk + 1) / 2)
        N = int(x.shape[-1] / total_chunk)
        x1 = x[:, :, :, 0:N * chunk]
        x2 = x[:, :, :, N * chunk:]
        y1 = torch.chunk(x1, chunk, dim=-1)
        y1 = torch.cat(y1, dim=-2)
        # print(y1.shape)
        y2 = torch.chunk(x2, chunk - 1, dim=-1)
        y2 = torch.cat(y2, dim=-2)
        # print(y2.shape)
        stride = int(kernel_size / 2)
        y1[:, :, stride:-stride, :] = y1[:, :, stride:-stride, :] / 2 + y2[:, :, :, :] / 2
        return y1

    def forward(self, x):
        """
        :param x: N M C T V
        :return: classes scores
        """
        N, M, T, V, C = x.shape

        x = x.permute(0, 1, 4, 2, 3).contiguous().view(N * M, C, T, V)  # b,c,t,v    batch_size,channel,time_point,ROI
        x = self.input_mapping(x)

        for i, m in enumerate(self.graph_layers):
            if self.config[i][-2] > 1:
                x = self.segment_frame_overlapping(x, kernel_size=self.config[i][-3], total_chunk=self.config[i][-2])
                x = m(x)
                x = self.restore_frame_overlapping(x, kernel_size=self.config[i][-3],
                                                   total_chunk=self.config[i][-2])
            else:
                x=m(x)

        return self.fc(self.drop_out(x.mean(3).mean(2)))


if __name__ == '__main__':

    config = [[6, 12, False, 1, 32, 7, 2],
              # in_channels, out_channels, is_regularization, Conv_stride,chunk_kernel_size,total_chunk,num_head
              [12, 12, True, 1, 32, 7, 2],
              [12, 12, False, 2, 1, 1, 2],
              [12, 12, True, 2, 1, 1, 2],
              [12, 12, True, 1, 1, 1, 2], ]

    # config = [[6, 12, False, 1, 32,7, 2],   # in_channels, out_channels, is_regularization, Conv_stride,chunk_kernel_size,total_chunk,num_head
    #          [12, 12, True, 2, 1, 1, 2],
    #          [12, 12, False, 1, 32, 3, 2],
    #          [12, 12, True, 2, 1, 1, 2],
    #          [12, 12, True, 1, 1, 1, 2],]

    net = CrossKTnet(sparsity_alpha=2, num_subset=2, num_frame=128, num_point=90, kernel_size=1, use_pes=True, config=config)  # .cuda()
    net.eval()
    data=torch.zeros(16, 1, 128, 90, 1)  #batch_size,num_person,time_point,num_ROI,feature_dimension   # .cuda()  # N, M, T, V, C
    y=net(data)
    prediction=torch.argmax(y, dim=-1)
    print(prediction)
