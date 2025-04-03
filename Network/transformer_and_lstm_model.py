import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用编码器和自注意力机制的transformer模型
class TransformerEMG(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout,seq_len):
        super(TransformerEMG, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout)
        self.fc_out = nn.Linear(d_model, 1)
        self.d_model = d_model
        self.reset = nn.Linear(seq_len, 1)

    def forward(self, src):
        src = self.input_linear(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=src.device))
        src = src.permute(1, 0, 2)
        output = self.transformer.encoder(src)
        output = self.fc_out(output)
        output = output.permute(1, 2, 0)
        output = self.reset(output)
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_directions = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputseq):
        h_0 = torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(inputseq, (h_0, c_0))
        pred = self.linear(output[:, -1, :])
        return pred

# 未使用自回归的transformer_and_lstm模型
class MyModule(nn.Module):
    def __init__(self, transformer_args, lstm_args):
        super(MyModule, self).__init__()

        # 定义 Transformer 模块
        self.transformer = TransformerEMG(**transformer_args)

        # 定义 LSTM 模块
        self.lstm = LSTM(**lstm_args)

    def forward(self, x):
        # 假设输入 x 的形状为 (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        num_segments = seq_len // 24  # 计算分割份数

        # 存储每次 Transformer 输出的列表
        transformer_outputs = []

        # 将输入 x 分割为 num_segments 份，并传入 Transformer
        for i in range(num_segments):
            segment = x[:, i * 24:(i + 1) * 24, :]  # 获取每个 segment (batch_size, 24, input_dim)
            transformer_output = self.transformer(segment)  # 传入 Transformer
            transformer_outputs.append(transformer_output)  # 收集输出

        # 将所有 Transformer 输出拼接为一个张量 (batch_size, num_segments * transformer_output_dim)
        transformer_concat = torch.cat(transformer_outputs, dim=1)
        # print("transformer输出形状:", transformer_concat.shape)

        # 将拼接的 Transformer 输出传递给 LSTM
        lstm_output = self.lstm(transformer_concat)

        return lstm_output

    def generate(self, x):
        # 假设输入 x 的形状为 (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        num_segments = seq_len // 24  # 计算分割份数
        # 存储每次 Transformer 输出的列表
        transformer_outputs = []
        for i in range(num_segments):
            segment = x[:, i * 24:(i + 1) * 24, :]  # 获取每个 segment (batch_size, 24, input_dim)
            transformer_output = self.transformer(segment)  # 传入 Transformer
            transformer_outputs.append(transformer_output)  # 收集输出
        # 将所有 Transformer 输出拼接为一个张量 (batch_size, num_segments * transformer_output_dim)
        transformer_concat = torch.cat(transformer_outputs, dim=1)
        # 获取transformer输出形状
        batch_size, seq_len, input_dim = transformer_concat.shape
        # 存储每次 lstm 输出的列表
        lstm_outputs = []
        for i in range(seq_len):
            lstm_segment = transformer_concat[:, i:i + 24, :]
            lstm_output = self.lstm(lstm_segment)
            lstm_outputs.append(lstm_output)  # 收集输出
        lstm_concat = torch.cat(lstm_outputs, dim=1)
        return lstm_concat



# 使用自回归的transformer_and_lstm模型
class Transformer_and_Lstm_autoregresson_model(nn.Module):
    def __init__(self, transformer_args, lstm_args):
        super(Transformer_and_Lstm_autoregresson_model, self).__init__()

        # 定义 Transformer 模块
        self.transformer = TransformerEMG(**transformer_args)

        # 定义 LSTM 模块
        self.lstm = LSTM(**lstm_args)

    def forward(self, data, label):
        # 假设输入 data 的形状为 (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = data.shape  # 获取输入数据形状
        num_segments = seq_len // 24  # 计算分割份数，用于将输入分割后送入transformer

        # 存储每次 Transformer 输出的列表
        transformer_outputs = []

        # 将输入 x 分割为 num_segments 份，并传入 Transformer
        for i in range(num_segments):
            segment = data[:, i * 24:(i + 1) * 24, :]  # 获取每个 segment (batch_size, 24, input_dim)
            transformer_output = self.transformer(segment)  # 传入 Transformer
            transformer_outputs.append(transformer_output)  # 收集输出

        # 将所有 Transformer 输出拼接为一个张量 (batch_size, num_segments * transformer_output_dim)
        transformer_concat = torch.cat(transformer_outputs, dim=1)
        # print("transformer输出形状:", transformer_concat.shape)
        # 将transformer输出和过去时刻角度合并成为一个张量作为lstm输入，实现自回归
        lstm_input = torch.cat((transformer_concat, label), dim=2)

        # 将拼接的 Transformer 输出传递给 LSTM
        lstm_output = self.lstm(lstm_input)

        return lstm_output

    def generate(self, data):  # 模型推理
        batch_size, seq_len, input_dim = data.shape  # 获取形状
        num_segments = seq_len // 24
        transformer_outputs = []

        # Transformer部分
        for i in range(num_segments):
            segment = data[:, i * 24:(i + 1) * 24, :]
            transformer_output = self.transformer(segment)
            transformer_outputs.append(transformer_output)

        # 拼接所有 Transformer 输出
        transformer_concat = torch.cat(transformer_outputs, dim=1)

        # 自回归生成过程
        lstm_outputs = [torch.zeros(1, 1) for _ in range(24)]

        for i in range(transformer_concat.size(1)):
            # 取出 transformer 的每个时间步的输出作为输入
            transformer_segment = transformer_concat[:, i:i + 24, :]  # (batch_size, 1, 1)
            # print("transformer_segment.shape",transformer_segment.shape)
            transformer_output_len = transformer_segment.size(1)

            last_24_tensors = lstm_outputs[-transformer_output_len:]  # 从过去预测角度列表中截取上次推理结果用于自回归生成
            last_24_tensors = [tensor.to(device) for tensor in last_24_tensors]  # 张量移动到显卡
            lstm_last_output = torch.cat(last_24_tensors, dim=1).to(device)  # 张量移动到显卡
            # print("lstm_last_output.shape",lstm_last_output.shape)
            lstm_last_output = lstm_last_output.unsqueeze(-1)  # 在最后增加一个维度，用于适配模型输入形状要求

            # 合并 transformer 输出和上一个时刻的 lstm 输出
            lstm_input = torch.cat((transformer_segment, lstm_last_output), dim=2)  # (batch_size, 1, 2)

            # 通过 LSTM 推理
            lstm_output = self.lstm(lstm_input)
            # print("lstm_output.shape",lstm_output.shape)
            lstm_outputs.append(lstm_output)

        # 拼接所有 LSTM 输出
        # 确保所有 LSTM 输出都在同一个设备上
        lstm_outputs = [output.to(device) for output in lstm_outputs]
        lstm_outputs = lstm_outputs[24:]
        lstm_concat = torch.cat(lstm_outputs, dim=1)
        return lstm_concat



"""# 模型定义和使用测试
transformer_args = {
    "input_size": 6,
    "d_model": 512,
    "nhead": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "seq_len":24
}

lstm_args = {
    "input_size": 1,  # 假设 Transformer 输出维度为 512
    "hidden_size": 64,
    "num_layers": 3,
    "output_size": 1,
    "batch_size": 32
}

mymodule=MyModule(transformer_args,lstm_args).to(device)
# 设置随机输入的参数
batch_size = 32  # 批量大小
src_seq_len = 576  # 编码器输入序列长度
input_size = 6    # 输入维度

# 生成随机输入数据
src = torch.rand(batch_size, src_seq_len, input_size).to(device)  # 编码器输入


# 模型推理输出
output = mymodule.generate(src)
print("模型输出形状:", output.shape)
print("模型输出:", output)
"""