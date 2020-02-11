import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from backbone_class_2_ssh import resnet_1D_34
from ssh.network2.VLADssh2 import model

# Creates writer1 object.
# The log will be saved in 'runs/exp'
# writer1 = SummaryWriter('runs/exp')

# Creates writer2 object with auto generated file name
# The log directory will be something like 'runs/Aug20-17-20-33'
# writer2 = SummaryWriter()
#
# # Creates writer3 object with auto generated file name, the comment will be appended to the filename.
# # The log directory will be something like 'runs/Aug20-17-20-33-resnet'
# writer3 = SummaryWriter(comment='resnet')

writer = SummaryWriter('runs/scalar_example')
for i in range(10):
    writer.add_scalar('quadratic', i**2, global_step=i)
    writer.add_scalar('exponential', 2**i, global_step=i)


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x) + F.relu(-x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

dummy_input = Variable(torch.rand(13, 1, 28, 28))

model22 = Net1()
with SummaryWriter(comment='Net1') as w:
    w.add_graph(model22, (dummy_input, ))



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = resnet_1D_34(10)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    # net = nn.DataParallel(net)
else:
    print("Let's use", torch.cuda.device_count(), "GPU!")
net.to(device)
my_tensor = torch.randn(1, 1, 30000)
my_tensor = my_tensor.to(device)
y = net(my_tensor)
torch.cuda.empty_cache()
# input_shape: calculate the output shape from the cnn
g_centres = 2
print(y.shape
      )
model = model(mode='gvlad', k_centers=8, g_centers=g_centres, dim=256, input_shape=y.shape[-1]
              , kernel_size=10, num_classes=200)
model.to(device)
with SummaryWriter(comment='4yp') as w:
    w.add_graph(model, (my_tensor, ))

