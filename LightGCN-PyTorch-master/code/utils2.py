import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class MyDataset(Dataset):
    def __init__(self, emb_data, h_data, label):
        self.emb_data = emb_data
        self.h_data = h_data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # one hot label
        label = torch.zeros(2)
        label[self.label[idx]] = 1
        vec = self.emb_data[idx]
        if idx > 0 and idx % 2 == 0:
            # Choose another random vec/label randomly
            mixup_idx = torch.randint(0, len(self.label), (1,)).item()
            mixup_label = torch.zeros(2)
            mixup_label[self.label[mixup_idx]] = 1

            # Select a random number from the given beta distribution
            alpha = 0.2
            beta = 0.2
            mixup_ratio = torch.distributions.beta.Beta(alpha, beta).sample()
            # mixup_ratio = 0.4
            vec = mixup_ratio * self.emb_data[idx] + (1 - mixup_ratio) * self.emb_data[mixup_idx]
            label = mixup_ratio * label + (1 - mixup_ratio) * mixup_label
        return vec, self.h_data[idx], label


class AddNoise(object):
    def __init__(self, mean, std, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        else:
            return x


# 定义自定义collate_fn函数
def my_collate_fn(batch):
    # 随机生成一个0-1之间的浮点数
    prob = torch.rand(1).item()
    if prob < 0.5:
        # 以50%的概率对第二维度进行重排
        shuffle_indices = torch.randperm(4)
        # print(type(batch), type(batch[0]))
        # print(len(batch[0]))
        batch = [(sample[0][shuffle_indices, :], sample[1], sample[2]) for sample in batch]
    # else:
    #     # 以50%的概率对样本进行噪声添加
    #     batch = [(sample + torch.randn_like(sample) * 0.1,) for sample in batch]
    return default_collate(batch)
