import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

CONTEXT_SIZE = 2  # 依据的单词数，表示我们希望由前面几个单词来预测这个单词，这里使用两个单词(trigram)
EMBEDDING_DIM = 10  # 词向量的维度，表示词嵌入的维度

# 莎士比亚的诗
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# 建立训练集
## 将单词三个分组，前面两个作为输入，最后一个作为预测的结果
## [(('When', 'forty'), 'winters'), ...]
trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2])
           for i in range(len(test_sentence) - 2)]
# print(trigram[0])
# print(len(test_sentence), test_sentence.index('cold.'))


# 建立每个词与数字的编码，据此构建词嵌入
vocb = set(test_sentence) # 使用 set 将重复的元素去掉
word_to_idx = {word: i for i, word in enumerate(vocb)}  # {词语: 索引}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}  # {索引: 词语}



# 定义模型
## 模型的输入就是前面的两个词，输出就是预测单词(第三个词)的概率
class n_gram(nn.Module):
    def __init__(self, vocab_size, context_size=CONTEXT_SIZE, n_dim=EMBEDDING_DIM):
        super(n_gram, self).__init__()

        self.embed = nn.Embedding(vocab_size, n_dim)
        self.classify = nn.Sequential(
            nn.Linear(context_size * n_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, vocab_size)
        )

    def forward(self, x):
        voc_embed = self.embed(x) # 得到词嵌入
        voc_embed = voc_embed.view(1, -1) # 将两个词向量拼在一起
        out = self.classify(voc_embed)
        return out




