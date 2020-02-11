import torch
import torch.nn as nn
from torchsummary import summary
import torchvision
import pdb


class Residual_block(nn.Module):
    def __init__(self, num_filts,kernel_size, first=False):
        super(Residual_block, self).__init__()
        self.kernel_size =kernel_size
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=num_filts[0])
        # self.lrelu = nn.LeakyReLU()
        # self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)
        self.relu = nn.ReLU6(inplace=True)

        self.conv1 = nn.Conv1d(in_channels=num_filts[0],
                               out_channels=num_filts[1],
                               kernel_size=kernel_size,
                               padding='same',
                               stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=num_filts[1])
        self.conv2 = nn.Conv1d(in_channels=num_filts[1],
                               out_channels=num_filts[1],
                               padding='same',
                               kernel_size=10,
                               stride=1)

        if num_filts[0] != num_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=num_filts[0],
                                             out_channels=num_filts[1],
                                             padding=0,
                                             kernel_size=1,
                                             stride=1)
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        # if not self.first:
        #     out = self.bn1(x)
        #     out = self.lrelu_keras(out)
        # else:
        #     out = x

        out1 = self.conv1(x)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)
        # pdb.set_trace()
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)


        if self.downsample:
            identity = self.conv_downsample(identity)
            identity =self.bn2(identity)

        out2 += identity
        out = self.relu(out2)
        out = self.mp(out)
        # pdb.set_trace()
        return out



class resnet_1D_34(nn.Module):
    def __init__(self, kernel_size=3):  #init can automatically run
        super(resnet_1D_34, self).__init__()
        self.kernel_size = kernel_size
        channel_lists = [1, 256, 128, 256, 512]
        # ===============================================
        #            Convolution Block 1
        # ===============================================
        # use 40 dimensional mfcc or single
        self.conv11 = nn.Conv1d(in_channels=channel_lists[0], out_channels=channel_lists[1], kernel_size=3)
        self.bn1 = nn.BatchNorm1d(num_features=channel_lists[1])
        self.relu1 = nn.ReLU6(inplace=True)
        # self.conv1 = nn.Sequential(self.conv11,self.bn1,self.relu1)
        self.maxpool1=nn.MaxPool1d(3)
        self.maxpool2 = nn.MaxPool1d(2)

        # ===============================================
        #            Convolution Section 2
        # ===============================================
        self.conv2 = self.make_layer(num_block=2,num_filts=channel_lists[1:3],kernel_size=kernel_size)

        # ===============================================
        #            Convolution Section 3
        # ===============================================
        self.conv3 = self.make_layer(num_block=4, num_filts=channel_lists[2:4], kernel_size=kernel_size)

    def make_layer(self,num_block,num_filts,kernel_size):
        layers=[]
        # layers.append(Residual_block(num_filts=num_filts,kernel_size=kernel_size))
        for i in range(num_block):
            layers.append(Residual_block(num_filts=num_filts, kernel_size=kernel_size))
            #num_filts includes in_channel and out_channel
            if i == 0:
                num_filts[0] = num_filts[1]  # ensure the second layer has correct in_channel

        return nn.Sequential(*layers)
    def forward(self, input_x):
            # pdb.set_trace()
            x11 = self.conv11(input_x)
            x12 =self.bn1(x11)
            x13 = self.relu1(x12)
            # pdb.set_trace()
            x1 = self.maxpool1(x13)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            # pdb.set_trace()
            # x4 = self.maxpool1(x3)

            return x3






if __name__ == '__main__':
    # a = resnet_1D_34()
    # print(a)
    # summary(a,input_size=(1,1,22000))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = resnet_1D_34(kernel_size=3)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net)
    else:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    net.to(device)
    # net2=torchvision.models.vgg11(pretrained=False)
    # net2.to(device)
    my_tensor = torch.randn(1,1,30000)
    my_tensor= my_tensor.to(device)
    y = net(my_tensor)
    # print(y)
    pdb.set_trace()
    print(y.shape)
    summary(net, input_size=(1, 30000))
    # summary(net2, input_size=(3,224, 224))
    # from torch.utils.tensorboard import SummaryWriter
    # tb = SummaryWriter()
    # tb.add_graph(net,my_tensor)

