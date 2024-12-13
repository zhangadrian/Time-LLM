from math import sqrt  # 导入平方根函数，用于后续计算

import torch  # 导入 PyTorch 库

import torch.nn as nn  # 导入 PyTorch 的神经网络模块

# 导入 transformers 库中的模型配置和模型类
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, \
    GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer

from layers.Embed import PatchEmbedding  # 从自定义层导入 PatchEmbedding 类

import transformers  # 导入 transformers 库以使用其功能

from layers.StandardNorm import Normalize  # 从自定义层导 入 Normalize 类

transformers.logging.set_verbosity_error()  # 设置 transformers 日志级别为错误，仅显示错误信息


# 定义 FlattenHead 类，用于将输入展平并通过线性层进行映射
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()  # 调用父类构造函数
        self.n_vars = n_vars  # 保存变量数量
        self.flatten = nn.Flatten(start_dim=-2)  # 初始化展平层，从倒数第二维开始展平
        self.linear = nn.Linear(nf, target_window)  # 初始化线性层，将特征维度从 nf 映射到目标窗口大小
        self.dropout = nn.Dropout(head_dropout)  # 初始化 dropout 层，用于防止过拟合

    def forward(self, x):  # 定义前向传播方法
        x = self.flatten(x)  # 展平输入张量，shape: [batch_size, seq_len * n_vars]
        x = self.linear(x)  # 应用线性变换，shape: [batch_size, target_window]
        x = self.dropout(x)  # 应用 dropout，shape: [batch_size, target_window]
        return x  # 返回处理后的张量


# 定义主模型类 Model
class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()  # 调用父类构造函数

        # 从配置中获取模型参数
        self.task_name = configs.task_name  # 任务名称，例如预测类型
        self.pred_len = configs.pred_len  # 预测长度
        self.seq_len = configs.seq_len  # 输入序列长度
        self.d_ff = configs.d_ff  # 前馈网络维度
        self.top_k = 5  # 用于计算滞后时的前 K 个值
        self.d_llm = configs.llm_dim  # LLM 模型的维度
        self.patch_len = configs.patch_len  # 每个 patch 的长度
        self.stride = configs.stride  # 步幅

        if configs.llm_model == 'LLAMA':
            # 加载 LLAMA 模型配置
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers  # 设置隐藏层数量
            self.llama_config.output_attentions = True  # 输出注意力权重
            self.llama_config.output_hidden_states = True  # 输出隐藏状态

            try:
                # 尝试加载本地 LLAMA 模型文件
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                # 如果本地文件不存在，则从 Hugging Face 下载模型
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )

            try:
                # 尝试加载本地 LLAMA 分词器文件
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                # 如果本地文件不存在，则从 Hugging Face 下载分词器
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                )

        elif configs.llm_model == 'GPT2':
            # 加载 GPT2 模型配置
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True

            try:
                # 尝试加载本地 GPT2 模型文件
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                # 如果本地文件不存在，则从 Hugging Face 下载模型
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                # 尝试加载本地 GPT2 分词器文件
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                # 如果本地文件不存在，则从 Hugging Face 下载分词器
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                )

        elif configs.llm_model == 'BERT':
            # 加载 BERT 模型配置
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True

            try:
                # 尝试加载本地 BERT 模型文件
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                # 如果本地文件不存在，则从 Hugging Face 下载模型
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                # 尝试加载本地 BERT 分词器文件
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                # 如果本地文件不存在，则从 Hugging Face 下载分词器
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                )

        else:
            raise Exception('LLM model is not defined')  # 抛出异常，如果未定义 LLM 模型

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置填充标记为结束标记（如果存在）
        else:
            pad_token = '[PAD]'  # 定义填充标记为 '[PAD]'

        self.tokenizer.add_special_tokens({'pad_token': pad_token})  # 将填充标记添加到分词器中
        self.tokenizer.pad_token = pad_token  # 确保填充标记被正确设置

        for param in self.llm_model.parameters():
            param.requires_grad = False  # 冻结 LLM 模型的所有参数，以防止在训练期间更新

        if configs.prompt_domain:
            self.description = configs.content  # 使用提供的内容作为描述
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
            # 默认描述文本

        self.dropout = nn.Dropout(configs.dropout)  # 初始化 dropout 层，用于防止过拟合

        # 初始化 PatchEmbedding 层，将输入特征映射到嵌入空间，使用配置中的参数进行初始化
        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight  # 获取 LLM 模型的输入嵌入权重，shape: [vocab_size, embedding_dim]

        self.vocab_size = self.word_embeddings.shape[0]  # 获取词汇表大小

        self.num_tokens = 1000  # 定义用于映射的令牌数量

        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        # 初始化线性映射层，将词汇表大小映射到 num_tokens

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads,
                                                      self.d_ff,
                                                      self.d_llm)
        # 初始化重编程层，用于处理嵌入

        # 根据输入序列长度、patch 长度和步幅计算 patch 数量，shape: [patch_nums]
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)

        self.head_nf = self.d_ff * self.patch_nums  # 定义头部特征维度

        if (self.task_name == 'long_term_forecast' or
                self.task_name == 'short_term_forecast'):
            # 根据任务类型初始化输出投影层，用于将特征映射到预测长度

            self.output_projection = FlattenHead(configs.enc_in,
                                                 self.head_nf,
                                                 self.pred_len,
                                                 head_dropout=configs.dropout)

        else:
            raise NotImplementedError("Task not implemented")

            ## 正常化层初始化 ##

        ## 使用标准化层来规范化输入特征 ##

        ## affine=False 表示不使用可学习的仿射变换 ##

        ## shape: [batch_size, enc_in] ##

        ## Normalize 是自定义的标准化类 ##

        ## 用于规范化模型输出 ##

        ## 在后续处理中使用 ##

        ## 此外，这里可能会涉及到输入数据的标准化处理 ##

        ## 在训练过程中有助于提高模型性能 ##

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        前向传播方法，根据任务类型进行不同处理

        :param x_enc: 编码器输入，shape: [batch_size, seq_len_enc, feature_dim]
        :param x_mark_enc: 编码器时间标记，shape: [batch_size, seq_len_enc]
        :param x_dec: 解码器输入，shape: [batch_size, seq_len_dec]
        :param x_mark_dec: 解码器时间标记，shape: [batch_size, seq_len_dec]
        :param mask: 可选的掩码张量，shape: [batch_size, seq_len_dec]
        :return: 解码输出，shape: [batch_size, pred_len, output_dim]
        """

        if (self.task_name == 'long_term_forecast' or
                (self.task_name == 'short_term_forecast')):
            dec_out =  ## 调用 forecast 方法进行预测 ##

            dec_out =  ## 返回最后 pred_len 步的输出 ##

            return dec_out[:, -self.pred_len:, :]

        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        根据输入数据进行预测

        :param x_enc: 编码器输入，shape: [batch_size ,seq_len_enc ,feature_dim]
        :param x_mark_enc: 编码器时间标记，shape: [batch_size ,seq_len_enc]
        :param x_dec: 解码器输入，shape:[batch_size ,seq_len_dec]
        :param x_mark_dec: 解码器时间标记，shape:[batch_size ,seq_len_dec]
        :return: 解码输出，shape:[batch_size ,pred_len ,output_dim]
        """

        x_enc = self.normalize_layers(x_enc, 'norm')  ## 对编码器输入进行规范化处理 ##

        B, T, N = x_enc.size()  ## 获取批次大小、时间步和特征维度 ##

        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)  ## 调整张量形状以适应后续处理 ##

        min_values = torch.min(x_enc, dim=1)[0]  ## 获取每个序列的最小值，形状为 [B] ##

        max_values = torch.max(x_enc, dim=1)[0]  ## 获取每个序列的最大值，形状为 [B] ##

        medians = torch.median(x_enc, dim=1).values  ## 获取每个序列的中位数值，形状为 [B] ##

        lags = self.calcute_lags(x_enc)  ## 调用计算滞后函数获取滞后值 ##

        trends = x_enc.diff(dim=1).sum(dim=1)  ## 根据差分计算趋势值，形状为 [B] ##

        prompt = []  ## 初始化用于存储提示信息的列表 ##

        for b in range(x_enc.shape[0]):  ## 遍历批次中的每个样本 ##
            min_values_str = str(min_values[b].tolist()[0])  ## 将最小值转换为字符串格式 ##
            max_values_str = str(max_values[b].tolist()[0])  ## 将最大值转换为字符串格式 ##
            median_values_str = str(medians[b].tolist()[0])  ## 将中位数转换为字符串格式 ##
            lags_values_str = str(lags[b].tolist())  ## 将滞后值转换为字符串格式 ##

            prompt_ = (f"<|start_prompt|>Dataset description:{self.description}"
                       f"Task description：forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                       "Input statistics:"
                       f"min value {min_values_str}, "
                       f"max value {max_values_str}, "
                       f"median value {median_values_str}, "
                       f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                       f"top5 lags are:{lags_values_str}<||>")

            prompt.append(prompt_)  ## 将生成的提示信息添加到列表中##

            x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()  ## 调整张量形状以适应后续处理##

            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                    max_length=2048).input_ids
            ## 使用分词器将提示信息转换为 token ID 张量，返回形状为 [B,prompt_length]##

            prompt_embeddings = self.llm_model.get_input_embeddings()(
                prompt.to(x_enc.device))  ## 获取提示嵌入，shape:[B,prompt_length,d_model]##

            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1,
                                                                                               0)  ## 获取源嵌入并通过映射层进行变换##

            x_enc = x_enc.permute(0, 2, 1).contiguous()  ## 调整张量形状以适应后续处理##

            enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))  ## 使用 PatchEmbedding 层对编码器输出进行处理##

            enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)  ## 使用重编程层对编码器输出进行处理##

            llama_enc_out = torch.cat([prompt_embeddings, enc_out],
                                      dim=1)  ## 拼接提示嵌入和编码器输出，shape:[B,prompt_length+patch_nums,d_model]##

            dec_out = self.llm_model(
                inputs_embeds=llama_enc_out).last_hidden_state  ## 获取解码输出，shape:[B,prompt_length+patch_nums,d_llm]##

            dec_out = dec_out[:, :, :self.d_ff]  ## 截取解码输出的前 d_ff 个特征维度，shape:[B,prompt_length+patch_nums,d_ff]##

            dec_out = torch.reshape(  ## 重塑解码输出张量以适应后续处理##
                dec_out,
                (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))  ## shape:[B*N,prompt_length+patch_nums,d_ff]##

            dec_out = dec_out.permute(0, 1, 3, 2).contiguous()  ## 调整张量形状以适应后续处理##

            dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])  ## 投影解码输出到目标空间，并获取最后 patch_nums 的部分##

            dec_out = dec_out.permute(0, 2, 1).contiguous()  ## 调整张量形
