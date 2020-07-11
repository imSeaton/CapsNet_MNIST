import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import json
import time

from collections import namedtuple
from collections import OrderedDict
from itertools import product
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# squash the vector to (0, 1) and keep the direction
# input: 任何维度的向量, 只对最后一维进行操作
# output: (0, 1)的向量
def squash(x, dim=-1):
    squared_norm = (x**2).sum(dim=dim, keepdim=True)    
    scale = squared_norm / (1+squared_norm)        
    result = scale * (x / (squared_norm.sqrt() + 1e-8))
    return result

# 做第二次卷 --> 6*6*256  reshape --> 1152 * 8
# input: (batch*20*20*256)
# output: primarycaps: (batch*1152*8)
# waring: 经过卷积生成的主要胶囊范围不在（0，1），一定要squash()
class PrimaryCaps(nn.Module):
    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size=9, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels* num_conv_units, kernel_size=kernel_size, stride=stride)
        self.out_channels = out_channels
    
    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(x.shape[0], -1, self.out_channels)
        out = squash(out)
        # primarycaps: (batch, in_caps_num, in_caps_dim)
        return out
    
# 乘以仿射变换矩阵、动态路由
# input： (batch, 1152, 8)
# output: (batch, 10, 1152, 16)
# warning: squash, 路由合成后的胶囊，范围变大了，需要squash
class DigitCaps(nn.Module):
    def __init__(self, in_caps_dim, in_caps_num, out_caps_dim, out_caps_num, num_routing):
        super().__init__()
        self.in_caps_dim = in_caps_dim
        self.in_caps_num = in_caps_num
        self.out_caps_dim = out_caps_dim
        self.out_caps_num = out_caps_num
        self.num_routing = num_routing        
        # 仿射变换矩阵
        # 仿射变换类似卷积，最好不要设置为0
        self.W = nn.Parameter(0.01 * torch.randn(1, out_caps_num, in_caps_num, out_caps_dim, in_caps_dim))
    
    def forward(self, x):
        batch_size = x.shape[0]
        # x: (batch, 1, in_caps_num, in_caps_dim, 1)
        x = x.unsqueeze(dim=1).unsqueeze(dim=4)
        #  (1, out_caps_num, in_caps_num, out_caps_dim, in_caps_dim)
        # @(batch, 1, in_caps_num, out_caps_dim, 1)
        #  (batch, out_caps_num, in_caps_num, out_caps_dim, 1)
        u_hat = torch.matmul(self.W, x)
        #  (batch, out_caps_num, in_caps_num, out_caps_dim)
        u_hat = u_hat.squeeze(dim=-1)
        temp_u_hat = u_hat.detach()
        # dynamic routing
        # (batch, out_caps_num, in_caps_num, 1)
        b = torch.zeros(batch_size, self.out_caps_num, self.in_caps_num, 1).to(device)
        for i in range(self.num_routing-1):
            # (batch, out_caps_num, in_caps_num, 1)
            # fan out
            c = b.softmax(dim=1)
            #  (batch, out_caps_num, in_caps_num, out_caps_dim)
            # *(batch, out_caps_num, in_caps_num, 1)
            #  (batch, out_caps_num, in_caps_num, out_caps_dim)
            s = temp_u_hat * c
            s = s.sum(dim=2)
            # (batch, out_caps_num, out_caps_dim)
            v = squash(s)
            # 向量点积判断相似度
            #  (batch, out_caps_num, in_caps_num, out_caps_dim)
            # @(batch, out_caps_num, out_caps_dim, 1)
            #  (batch, out_caps_num, in_caps_num, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(dim=-1))
            b += uv
            
        c = b.softmax(dim=1)
        s = (u_hat * c).sum(dim=2)
        v = squash(s)
        return v

# 串联主要胶囊和数字胶囊
class CapsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.primary_caps = PrimaryCaps(32, 256, 8, 9, 2)
        self.digit_caps = DigitCaps(8, 32*6*6, 16, 10, 3)
        self.decoder = nn.Sequential(
            nn.Linear(16*10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.primary_caps(out)
        # (batch, out_caps_num, out_caps_dim)
        out = self.digit_caps(out)
        
        # (batch, out_caps_num)
        logits = torch.norm(out, dim=-1)
        # (batch, 10)
        # 在整个模型都是在GPU运行的时候，这里是否有必要另外设置成gpu上运行
        pred = torch.eye(10).to(device).index_select(dim=0, index=logits.argmax(dim=-1))
        
        # Reconstruction
        batch_size = x.shape[0]
        #  out*pred.unsqueeze(dim=-1)
        # =(batch, out_caps_num, out_caps_dim)
        # *(batch, out_caps_num, 1)
        #  (batch, out_caps_num, out_caps_dim)
        # reconstrction (batch_size, 784)
        reconstruction = self.decoder((out*pred.unsqueeze(dim=-1)).reshape(batch_size, -1))
        return logits, reconstruction
    
class CapsuleLoss(nn.Module):
    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super().__init__()
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.lmda = lmda        
        self.mse = nn.MSELoss(reduction='sum')
        self.reconstruction_loss_scalar = 5e-4
    
    def forward(self, images, labels, logits, reconstruction):
        # Margin Loss
        left = (self.upper_bound - logits).relu() ** 2
        right = (logits - self.lower_bound).relu() ** 2
        # label: (batch, 10)
        # Q: 这里的lmda为什么是0.5，为什么要缩小到一个不大不小的值
        margin_loss = (labels * left + self.lmda * (1-labels) * right).sum()        
        reconstruction_loss = self.mse(images, reconstruction.reshape(images.shape))
        return margin_loss + self.reconstruction_loss_scalar * reconstruction_loss

class RunBuilder():
    @ staticmethod
    def get_runs(params):        
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

class RunManager():
    def __init__(self):
        self.run_count = 0
        self.run_start_time = None
        self.run_params = None
        # 记录每个run中每个epoch的结果
        self.run_result = []
        # 记录每个run的最后的结果
        self.final_run_result = []
    
        self.epoch_count = 0
        self.epoch_start_time = None
        self.epoch_correct = 0
        self.epoch_loss = 0
        
        self.network = None
        self.train_loader = None
        self.test_loader = None
        self.tb = None
        
    def begin_run(self, run_params, network, train_loader, test_loader):
        self.run_count += 1
        self.run_start_time = time.time()
        self.run_params = run_params
        self.network = network
        self.train_loader = train_loader
        self.test_loader = test_loader
        # 绘制正在运行的网络图和图片
        self.tb = SummaryWriter(comment=f'{self.run_params}')
        images, labels = next(iter(self.train_loader))
        images = images.to(device)
        labels = labels.to(device)
        grid = torchvision.utils.make_grid(images)
        self.tb.add_image('images', grid)
        # todo GPU
#         self.tb.add_graph(self.network, images.to(getattr(run_params, 'device', 'cpu'))) 
        self.tb.add_graph(self.network, images)
        
    # 评价模型，生成最终结果
    def end_run(self):
        # 保存数据
        self.tb.close()
        self.epoch_count = 0
        # # 设置初始值供test使用
        # self.epoch_loss = 0
        # self.epoch_correct = 0

        # self.final_run_result.append(self.run_result[-1])
        # test_loss = pass
        # test_accuracy = pass
        # self.final_run_result[-1].append(test_loss)
        # self.final_run_result[-1].append(test_accuracy)
        
    
    def begin_epoch(self):
        self.epoch_count += 1
        self.epoch_start_time = time.time()
        self.epoch_loss = 0
        self.epoch_correct = 0        
        self.epoch_test_correct = 0
    
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        # calculate the loss and accuracy
        # Q why does the epoch_loss need to be devidedby lenth of the dataset
        loss = self.epoch_loss / len(self.train_loader.dataset)
        accuracy = self.epoch_correct / len(self.train_loader.dataset)
        test_accuracy = self.epoch_test_correct / len(self.test_loader.dataset)
        self.tb.add_scalar('epoch_loss', loss, self.epoch_count)
        self.tb.add_scalar('epoch_accuracy', accuracy, self.epoch_count)
        # 绘制参数和梯度的直方图
        for name, params in self.network.named_parameters():
            self.tb.add_histogram(f'{name}', params, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', params.grad, self.epoch_count)
            
        # 构建用于输出至excel和json的数据结构
        results = OrderedDict()
        results['run_count'] = self.run_count
        results['epoch_count'] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch_duration'] = epoch_duration
        results['run_duration'] = run_duration
        # unpack RUN类型的数据，并且将其键值对添加至result字典中
        for name, param in self.run_params._asdict().items():
            results[name] = param
        results['test_acc'] = test_accuracy
        self.run_result.append(results)
        df = pd.DataFrame.from_dict(self.run_result, orient='columns')        
        # clear_output(wait=True)
        os.system('clear')
        display(df)        
    

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.train_loader.batch_size
    
    def track_correct(self, logits, labels):        
        self.epoch_correct += self._get_num_correct(logits, labels)
    
    def track_epoch_test_correct(self, logits, labels):
        self.epoch_test_correct += self._get_num_correct(logits, labels)

    @ torch.no_grad()
    def _get_num_correct(self, logits, labels):
        return logits.argmax(dim=-1).eq(labels.argmax(dim=-1)).sum().item()
    
    def save(self, file_name):
        # csv
        pd.DataFrame.from_dict(self.run_result, orient='columns').to_csv(f'{file_name}.csv')
        # json
        with open(f"{file_name}.json", 'w', encoding='utf-8') as f:
            json.dump(self.run_result, f, ensure_ascii=False, indent=4)
    
    def get_final_result(self, epoch):
        final_result = []
        for i in range(len(self.run_result)):            
            if (i+1) % epoch == 0:
                final_result.append(self.run_result[i])
        df = pd.DataFrame.from_dict(final_result, orient='columns')        
        display(df)       
        
        

# load the dataset
transform=transforms.Compose([
        transforms.RandomCrop((28,28), padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
train_set = torchvision.datasets.MNIST(
    root = '../data/MNIST',
    train = True,
    download = True,
    transform=transform
)
test_set = torchvision.datasets.MNIST(
    root = '../data/MNIST',
    train = False,
    download = True,
    transform=transform
)

# train with different parameters
# hyperparameters: EPOCH, lr, batch_size, num_workers

params = OrderedDict(
    # Epoch = [5],
    lr = [0.01, 0.001],
    batch_size = [64, 128, 256],
    num_workers = [1, 2, 4],
    shuffle = [True, False]
#     device=['cuda', 'cpu']
)
EPOCH = 10
m = RunManager()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Todo

test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=1)

for run_params in RunBuilder.get_runs(params):
    model = CapsNet().to(device)
    train_loader = DataLoader(train_set, batch_size=run_params.batch_size, num_workers=run_params.num_workers, shuffle=True)
    criterion = CapsuleLoss().to(device)
    optimizer = optim.Adam(model.parameters())
    schedular = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)        
    m.begin_run(run_params, model, train_loader, test_loader)
    # Todo
    # for ep in range(run_params.EPOCH):
    for ep in range(EPOCH):
        model.train()
        m.begin_epoch()
        total_loss, total_correct = 0, 0
        batch_id = 0
        for batch in train_loader:            
            images, labels = batch
            images = images.to(device)
            labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
            logits, reconstruction = model(images)            
            loss = criterion(images, labels, logits, reconstruction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            total_loss += loss
            m.track_loss(total_loss)
            m.track_correct(logits, labels)   
            # batch_id += 1
            # print(f"{batch_id}")                     
        

        model.eval()        
        total_loss, total_correct = 0, 0
        # todo
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
            logits, reconstruction = model(images)
            m.track_epoch_test_correct(logits, labels)

        m.end_epoch()
        schedular.step()

    m.end_run()

    # model.eval()
    # batch_id = 0 
    # total_num, total_loss, total_correct = 0, 0, 0
    # print('*'*40)
    # for batch in test_loader:
    #     images, labels = batch
    #     images = images.to(device)
    #     labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
    #     logits, reconstruction = model(images)
    #     correct = logits.argmax(dim=-1).eq(labels.argmax(dim=-1)).sum().item()
    #     total_num += len(labels)
    #     total_loss += loss
    #     total_correct += correct
    #     accuracy = total_correct/total_num
    # print(f"loss {total_loss/total_num} accuracy {accuracy}")
    # print('*'*40)
m.save('result')
m.get_final_result(epoch=EPOCH)


