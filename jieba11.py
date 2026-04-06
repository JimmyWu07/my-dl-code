"""
    案例: 演示 RNN 层(循环网络层)的 API.

    循环网络层作用:
    基于 上一次的隐藏状态 + 本次的输入 → 本次的隐藏状态，本次的输出.

    公式:
    本次的隐藏状态 = tanh(上次的隐藏状态加权求和 + 本次的输入 加权求和)
    本次的输出 = 本次的隐藏状态加权求和，有词汇表中所有词的概率，选概率最大的哪个词作为 最终预测结果

    简单总结下 RNN:

    词嵌入层:
    将词(词的索引) 转换为 词向量表示.

    RNN 层(循环网络层):
    逐步处理词向量，生成 每个时间步的 隐藏状态.

    全连接层(输出映射):
    通过线性变换将隐藏状态映射到输出，通常是 1 个词汇表中词的概率分布.
"""
import torch
import torch.nn as nn
import jieba

if __name__ == '__main__':
    # 0.文本数据
    text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'
    # 1. 文本分词
    words = jieba.lcut(text)# lcut 切中文
    print('文本分词:', words)


    # 2.分词去重并保留原来的顺序获取所有的词语
    unique_words = list(set(words))
    print("去重后词的个数:\n", len(unique_words))


    # 3. 构建词嵌入层:num_embeddings: 表示词的总数量;embedding_dim: 表示词嵌入的维度
    embed = nn.Embedding(num_embeddings=len(unique_words), embedding_dim=4)
    print("词嵌入的结果：\n", embed)


    # 4. 词语的词向量表示
    for i, word in enumerate(unique_words):
        # 获得词嵌入向量
        word_vec = embed(torch.tensor(i))
        print('%3s\t' % word, word_vec)