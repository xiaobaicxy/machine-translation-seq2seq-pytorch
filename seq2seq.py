# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:19:44 2020
中英文翻译 seq2seq算法
数据集下载链接：http://www.statmt.org/wmt17/translation-task.html#download
@author: 
"""
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
import jieba

import numpy as np
from collections import Counter

torch.manual_seed(123) #保证每次运行初始化的随机数相同

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_name, is_en):
    #逐句读取文本，并将句子进行分词，且在句子前面加上'BOS'表示句子开始，在句子末尾加上'EOS'表示句子结束
    datas = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if(i>200000):
                break
            line = line.strip()
            if(is_en):
                datas.append(["BOS"] + nltk.word_tokenize(line.lower()) + ["EOS"])
                #datas.append(["BOS"] + list(line.lower().split()) + ["EOS"])
            else:
                datas.append(["BOS"] + list(jieba.cut(line, cut_all=False)) + ["EOS"])
    return datas

en_path = "./datasets/translation/news-commentary-v12.zh-en_en.txt"
cn_path = "./datasets/translation/news-commentary-v12.zh-en_zh.txt"
en = load_data(en_path, is_en=True)
cn = load_data(cn_path, is_en=False)

def create_dict(sentences, max_words):
    #统计文本中每个词出现的频数，并用出现次数最多的max_words个词创建词典，
    #且在词典中加入'UNK'表示词典中未出现的词，'PAD'表示后续句子中添加的padding（保证每个batch中的句子等长）
    word_count = Counter()
    for sentence in sentences:
        for word in sentence:
            word_count[word] += 1
    
    most_common_words = word_count.most_common(max_words)  #最常见的max_words个词
    total_words = len(most_common_words) + 2  #总词量（+2：词典中添加了“UNK”和“PAD”）
    word_dict = {w[0]: index+2 for index, w in enumerate(most_common_words)}  #word2index
    word_dict["UNK"] = 0
    word_dict["PAD"] = 1
    return word_dict, total_words

#word2index
en_dict, en_total_words = create_dict(sentences=en, max_words=50000)
cn_dict, cn_total_words = create_dict(sentences=cn, max_words=50000)

#index2word
inv_en_dict = {v: k for k, v in en_dict.items()}
inv_cn_dict = {v: k for k, v in cn_dict.items()}

def encode(en_sentences, cn_sentences, en_dict, cn_dict, sorted_by_len):
    #句子编码：将句子中的词转换为词表中的index
    
    #不在词典中的词用”UNK“表示
    out_en_sentences = [[en_dict.get(w, en_dict['UNK']) for w in sentence] for sentence in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, cn_dict['UNK']) for w in sentence] for sentence in cn_sentences]
    
    #基于英文句子的长度进行排序，返回排序后句子在原始文本中的下标
    #目的：为使每个batch中的句子等长时，需要加padding；长度相近的放入一个batch，可使得添加的padding更少
    if(sorted_by_len):
        sorted_index = sorted(range(len(out_en_sentences)), key=lambda idx: len(out_en_sentences[idx]))
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
        
    return out_en_sentences, out_cn_sentences

en_datas, cn_datas = encode(en, cn, en_dict, cn_dict, sorted_by_len=True)
#print(" ".join(inv_en_dict[i] for i in en_datas[0]))
#print(" ".join(inv_cn_dict[i] for i in cn_datas[0]))

def get_batches(num_sentences, batch_size, shuffle=True):
    #用每个句子在原始文本中的（位置）行号创建每个batch的数据索引
    batch_first_idx = np.arange(start=0, stop=num_sentences, step=batch_size) #每个batch中第一个句子在文本中的位置（行号）
    if(shuffle):
        np.random.shuffle(batch_first_idx)
    
    batches = []
    for first_idx in batch_first_idx:
        batch = np.arange(first_idx, min(first_idx+batch_size, num_sentences), 1) #每个batch中句子的位置（行号）
        batches.append(batch)
    return batches

def add_padding(batch_sentences):
    #为每个batch的数据添加padding，并记录下句子原本的长度
    lengths = [len(sentence) for sentence in batch_sentences] #每个句子的实际长度
    max_len = np.max(lengths) #当前batch中最长句子的长度
    data = []
    for sentence in batch_sentences:
        sen_len = len(sentence)
        #将每个句子末尾添0，使得每个batch中的句子等长（后续将每个batch数据转换成tensor时，每个batch中的数据维度必须一致）
        sentence = sentence + [0]*(max_len - sen_len) 
        data.append(sentence)
    data = np.array(data).astype('int32')
    data_lengths = np.array(lengths).astype('int32')
    return data, data_lengths

def generate_dataset(en, cn, batch_size):
    #生成数据集
    batches = get_batches(len(en), batch_size)
    datasets = []
    for batch in batches:
        batch_en = [en[idx] for idx in batch]
        batch_cn = [cn[idx] for idx in batch]
        batch_x, batch_x_len = add_padding(batch_en)
        batch_y, batch_y_len = add_padding(batch_cn)
        datasets.append((batch_x, batch_x_len, batch_y, batch_y_len))
    return datasets

batch_size = 8
datasets = generate_dataset(en_datas, cn_datas, batch_size)

#seq2seq的编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        #the input size of gru [sentence_len, batch_size, word_embedding_size]
        #batch_first=True  => input [batch_size, sentence_len, word_embedding_size]
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, batch_x, lengths):
        #batch_x: [batch_size, setence_len]
        #lengths: [batch_size]
        
        #基于每个batch中句子的实际长度倒序（后续使用pad_packed_sequence要求句子长度需要倒排序）
        sorted_lengths, sorted_index = lengths.sort(0, descending=True) 
        batch_x_sorted = batch_x[sorted_index.long()]
        
        embed = self.embedding(batch_x_sorted) #[batch_size, sentence_len, hidden_size]
        embed = self.dropout(embed)
        
        #将句子末尾添加的padding去掉，使得GRU只对实际有效语句进行编码
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths.long().cpu().data.numpy(), batch_first=True)
        packed_out, hidden = self.gru(packed_embed) #packed_out为PackedSequence类型数据，hidden[1, batch_size, hidden_size]
        
        #unpacked，恢复数据为tensor
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True) #[batch_size, sentence_len, hidden_size]
        
        #恢复batch中sentence原始的顺序
        _, original_index = sorted_index.sort(0, descending=False)
        out = out[original_index.long()].contiguous()
        hidden = hidden[:, original_index.long()].contiguous()
        
        return out, hidden

#seq2seq的解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        #将每个输出都映射会词表维度，最大值所在的位置对应的词就是预测的目标词
        self.liner = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch_y, lengths, encoder_hidden):
        #batch_y: [batch_size, setence_len]
        #lengths: [batch_size]
        #encoder_hidden: [1, batch_size, hidden_size]
        
        #基于每个batch中句子的实际长度倒序
        sorted_lengths, sorted_index = lengths.sort(0, descending=True) 
        batch_y_sorted = batch_y[sorted_index.long()]
        hidden = encoder_hidden[:, sorted_index.long()]
        
        embed = self.embedding(batch_y_sorted) #[batch_size, setence_len, hidden_size]
        embed = self.dropout(embed)
        
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths.long().cpu().data.numpy(), batch_first=True)
        #解码器的输入为编码器的输出，上一个词，然后预测下一个词
        packed_out, hidden = self.gru(packed_embed, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        _, original_index = sorted_index.sort(0, descending=False)
        out = out[original_index.long()].contiguous()
        hidden = hidden[:, original_index.long()].contiguous()
        out = self.liner(out) #[batch_size, sentence_len, vocab_size]
        
        #log_softmax求出每个输出的概率分布，最大概率出现的位置就是预测的词在词表中的位置
        out = F.log_softmax(out, dim=-1) #[batch_size, sentence_len, vocab_size]
        return out, hidden

#seq2seq模型
#训练和验证模型时，解码器第一时间步的输入为"BOS"和encoder_hidden,输出为预测的第一个词；
#之后，如第二个时间步的输入为第一个实际的目标词和预测的第一个目标词（上一步的输出）。
#而实际应用模型进行翻译时，解码器第一时间步的输入为"BOS"和encoder_hidden,输出为预测的第一个词；
#之后，如第二个时间步的输入为预测的第一个目标词（上一步的输出）和encoder_hidden（因为实际翻译过程中没有实际对应的目标词）
#为什么输入不一样还能有效？？？？
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, encoder_hid = self.encoder(x, x_lengths)  #英文编码
        output, hidden = self.decoder(y, y_lengths, encoder_hid) #解码出中文
        return output
    
    def translate(self, x, x_lengths, y, max_length=50):
        #翻译en2cn
        #max_length表示翻译的目标句子可能的最大长度
        encoder_out , encoder_hidden = self.encoder(x, x_lengths) #将输入的英文进行编码
        predicts = []
        batch_size = x.size(0)
        #目标语言（中文）的输入只有”BOS“表示句子开始，因此y的长度为1
        #每次都用上一个词(y)与编码器的输出预测下一个词，因此y的长度一直为1
        y_length = torch.ones(batch_size).long().to(y.device)
        for i in range(max_length):
            #每次用上一次的输出y和编码器的输出encoder_hidden预测下一个词
            output, hidden = self.decoder(y, y_length, encoder_hidden)
            #output: [batch_size, 1, vocab_size]
            
            #output.max(2)[1]表示找出output第二个维度的最大值所在的位置（即预测词在词典中的index）
            y = output.max(2)[1].view(batch_size, 1) #[batch_size, 1]
            predicts.append(y)
        predicts = torch.cat(predicts, 1) #[batch_size, max_length]
        return predicts

#自定义损失函数（mask的交叉熵损失）
#目的：使句子中添加的padding部分不参与损失计算
class MaskCriterion(nn.Module):
    def __init__(self):
        super(MaskCriterion, self).__init__()
        
    def forward(self, predicts, targets, masks):
        #predicts [batch_size, sentence_len, vocab_size]
        #target [batch_size, sentence_len]
        #masks [batch_size, sentence_len]
        predicts = predicts.contiguous().view(-1, predicts.size(2)) #[batch_size * sentence_len, vocab_size]
        targets = targets.contiguous().view(-1, 1)  #[batch_size*sentence_len, 1]
        masks = masks.contiguous().view(-1, 1)  #[batch_size*sentence_len, 1]
        
        #predicts.gather(1, targets)为predicts[i][targets[i]]
        #乘上masks，即只需要计算句子有效长度的预测
        #负号：因为采用梯度下降法，所以要最大化目标词语的概率，即最小化其相反数
        output = -predicts.gather(1, targets) * masks
        output = torch.sum(output) / torch.sum(masks) #平均
        
        return output
        
dropout = 0.2
hidden_size = 100

encoder = Encoder(vocab_size=en_total_words,
                  hidden_size=hidden_size,
                  dropout=dropout)
decoder = Decoder(vocab_size=cn_total_words,
                  hidden_size=hidden_size,
                  dropout=dropout)

model = Seq2Seq(encoder, decoder)
model = model.to(device)
loss_func = MaskCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

#print(model)
def test(mode, data):
    model.eval()
    total_words = 0
    total_loss = 0.
    with torch.no_grad():
        for i, (batch_x, batch_x_len, batch_y, batch_y_len) in enumerate(data):
            batch_x = torch.from_numpy(batch_x).to(device).long() 
            batch_x_len = torch.from_numpy(batch_x_len).to(device).long()
            
            batch_y_decoder_input = torch.from_numpy(batch_y[:, :-1]).to(device).long()
            batch_targets = torch.from_numpy(batch_y[:, 1:]).to(device).long()
            batch_y_len = torch.from_numpy(batch_y_len-1).to(device).long()
            batch_y_len[batch_y_len<=0] = 1
            
            batch_predicts = model(batch_x, batch_x_len, batch_y_decoder_input, batch_y_len)
            
            batch_target_masks = torch.arange(batch_y_len.max().item(), device=device)[None, :] < batch_y_len[:, None]
            batch_target_masks = batch_target_masks.float()
            
            loss = loss_func(batch_predicts, batch_targets, batch_target_masks)
            
            num_words = torch.sum(batch_y_len).item()
            total_loss += loss.item() * num_words
            total_words += num_words
        print("Test Loss:", total_loss/total_words)

def train(model, data, epoches):
    for epoch in range(epoches):
        model.train()
        total_words = 0
        total_loss = 0.
        for it, (batch_x, batch_x_len, batch_y, batch_y_len) in enumerate(data):
            batch_x = torch.from_numpy(batch_x).to(device).long()
            batch_x_len = torch.from_numpy(batch_x_len).to(device).long()
            
            #因为训练（或验证）时，decoder根据上一步的输出（预测词），与上一步输出对应的实际词预测下一个词
            #所以输入到decoder中的目标语句为[BOS, word_1, word_2, ..., word_n]
            #预测的实际标签为[word_1, word_2, ..., word_n, EOS]
            batch_y_decoder_input = torch.from_numpy(batch_y[:, :-1]).to(device).long()
            batch_targets = torch.from_numpy(batch_y[:, 1:]).to(device).long()
            batch_y_len = torch.from_numpy(batch_y_len-1).to(device).long()
            batch_y_len[batch_y_len<=0] = 1
            
            batch_predicts = model(batch_x, batch_x_len, batch_y_decoder_input, batch_y_len)
            
            #生成masks：
            batch_y_len = batch_y_len.unsqueeze(1) #[batch_size, 1]
            batch_target_masks = torch.arange(batch_y_len.max().item(), device=device) < batch_y_len
            batch_target_masks = batch_target_masks.float()
            batch_y_len = batch_y_len.squeeze(1) #[batch_size]
            
            loss = loss_func(batch_predicts, batch_targets, batch_target_masks)
            
            num_words = torch.sum(batch_y_len).item() #每个batch总的词量
            total_loss += loss.item() * num_words
            total_words += num_words
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            
            if(it % 50 == 0):
                print("Epoch {} / {}, Iteration: {}, Train Loss: {}".format(epoch, epoches, it, loss.item()))
        print("Epoch {} / {}, Train Loss: {}".format(epoch, epoches, total_loss/total_words))
        if(epoch!=0 and epoch % 100 == 0):
            test(model, datasets)  #由于数据集没有分出验证数据，为了省事，此处直接用了训练集做验证（可自行手动将训练集拆分）
            
train(model, datasets, epoches=200)

def en2cn_translate(sentence_id):
    #英文翻译成中文
    en_sentence = " ".join([inv_en_dict[w] for w in en_datas[sentence_id]]) #英文句子
    cn_sentence = " ".join([inv_cn_dict[w] for w in cn_datas[sentence_id]]) #对应实际的中文句子
    
    batch_x = torch.from_numpy(np.array(en_datas[sentence_id]).reshape(1, -1)).to(device).long()
    batch_x_len = torch.from_numpy(np.array([len(en_datas[sentence_id])])).to(device).long()
    
    #第一个时间步的前项输出
    bos = torch.Tensor([[cn_dict["BOS"]]]).to(device).long()
    
    translation = model.translate(batch_x, batch_x_len, bos, 10)
    translation = [inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)] #index2word
    
    trans = []
    for word in translation:
        if(word != "EOS"):
            trans.append(word)
        else:
            break
    print(en_sentence)
    print(cn_sentence)
    print(" ".join(trans))

en2cn_translate(0)
    