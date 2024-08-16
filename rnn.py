import torch
import torch.nn as nn
import math

def get_para_bias_attr(l2_decay, k):
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = torch.nn.init.uniform_  # PyTorch의 Uniform 초기화 함수
    weight_attr = torch.empty(k)  # 가중치 텐서 생성
    bias_attr = torch.empty(k)  # 바이어스 텐서 생성
    initializer(weight_attr, -stdv, stdv)  # 가중치 초기화
    initializer(bias_attr, -stdv, stdv)  # 바이어스 초기화

    # L2 정규화를 위한 파라미터 설정
    weight_attr = torch.nn.Parameter(weight_attr)
    bias_attr = torch.nn.Parameter(bias_attr)
    weight_attr.regularizer = torch.optim.Adam([weight_attr], weight_decay=l2_decay)
    bias_attr.regularizer = torch.optim.Adam([bias_attr], weight_decay=l2_decay)

    return [weight_attr, bias_attr]

class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Im2Seq, self).__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(2)
        x = x.transpose(1, 2)  # (NTC)(batch, width, channels)
        return x

class EncoderWithFC(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        weight_attr, bias_attr = get_para_bias_attr(l2_decay=0.00001, k=in_channels)
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name="reduce_encoder_fea",
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == "reshape":
            self.only_reshape = True
        else:
            support_encoder_dict = {
                "reshape": Im2Seq,
                "fc": EncoderWithFC,
                "rnn": EncoderWithRNN,
                "svtr": EncoderWithSVTR,
                "cascadernn": EncoderWithCascadeRNN,
            }
            assert encoder_type in support_encoder_dict, "{} must be in {}".format(
                encoder_type, support_encoder_dict.keys()
            )
            if encoder_type == "svtr":
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, **kwargs
                )
            elif encoder_type == "cascadernn":
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size, **kwargs
                )
            else:
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size
                )
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != "svtr":
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x