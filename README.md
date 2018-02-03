# Char-RNN-PyTorch
使用字符级别的RNN进行文本生成，使用PyTorch框架。[Gluon实现](https://github.com/SherlockLiao/Char-RNN-Gluon)

## Requirements
- PyTorch 0.3
- numpy

## Basic Usage
如果希望训练网络，使用如下的代码

```bash
python main.py \
--state train \
--txt './data/poetry.txt' \ # 训练用的txt文本
--batch 128  \ # batch_size
--epoch 1000 \ 
--len 100 \ # 输入RNN的序列长度
--max_vocab 5000 \ # 最大的字符数量
--embed 512 \ # 词向量的维度
--hidden 512 \ # 网络的输出维度
--n_layer 2 \ # RNN的层数
--dropout 0.5
```

如果希望使用训练好的网络进行文本生成，使用下面的代码

```bash
python main.py \
--state eval \
--begin '我' \ # 生成文本的开始，可以是一个字符，也可以一段话
--pred_len 100 \ # 希望生成文本的长度
--checkpoint './checkpoint/model_100.pth' # 读取训练模型的位置
```

## Result
如果使用古诗的数据集进行训练，可以得到下面的结果

```bash
我来钱庙复知世依。似我心苦难归久，相须莱共游来愁报远。近王只内蓉者征衣同处，规廷去岂无知径草木飘。
独爱滞大道愚促促榴才也，工韵诚千春和。风清月道路白暇。甫相送远，航冲空弄游残风催殊娟寸年。
我独心。行辛秀头为鸦石尘，那里非洛阳。学不境幽，出佳当禅最命壁戎松栖落。
藤二蛙归唯去尺续白宗熟劳熟无相世雁部，渔人独踟楼禅。
云月同秋草文翩家，南归宽同梢惆。
看取韵抱对，能闲掩眠妇卖眠士云坐。
```

如果使用周杰伦的歌词作为训练集，可以得到下面的结果

```bash
我们的爱
持纵的微银　
我路茫未望
象对躲睡被仁消　整虹伪沙
乡月裂续武深面到
身壁许达道　
用移照叫生
```
