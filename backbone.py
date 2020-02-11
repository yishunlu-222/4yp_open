import torch
import torch.nn as nn
from torchsummary import summary


def identity_block_1D(input_tensor, kernel_size, filters, bias=True,In_channel=1):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 2 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    # conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    # bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = nn.Conv1d(in_channels=In_channel, out_channels=filters[0],
                  padding='same',kernel_size=kernel_size,bias=bias)(input_tensor)
    x = nn.BatchNorm1d(filters[0])(x)
    x = nn.ReLU6(inplace=True)(x)

    # conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    # bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = nn.Conv1d(in_channels=filters[0], out_channels=filters[1],

                  padding='same',kernel_size=kernel_size,bias=bias)(x)
    x = nn.BatchNorm1d(filters[1])(x)
    x = nn.ReLU6(inplace=True)(x)

    residual = input_tensor
    x += residual
    x = nn.ReLU6(inplace=True)(x)

    return  x



def conv_block_1D(input_tensor, kernel_size, filters,strides=2, bias=True,In_channel=1):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """

    x = nn.Conv1d(in_channels=In_channel, out_channels=filters[0],
                  kernel_size=kernel_size,bias=bias,padding='same')(input_tensor)
    x = nn.BatchNorm1d(filters[0])(x)
    x = nn.ReLU6(inplace=True)(x)

    x = nn.Conv1d(in_channels=filters[0], out_channels=filters[0],
                  kernel_size=kernel_size,bias=bias)(x)
    x = nn.BatchNorm1d(filters[1])(x)
    x = nn.ReLU6(inplace=True)(x)

    residual =nn.Conv1d(in_channels=In_channel, out_channels=filters[1],
                        stride=strides,
                        kernel_size=kernel_size,bias=bias)(input_tensor)
    residual = nn.BatchNorm1d(filters[1])(residual)
    print('shape of residual is {}'.format(residual.size()))
    print('shape of direct path is {}'.format(x.size()))

    x += residual
    x = nn.ReLU6(inplace=True)(x)

    return x

def resnet_1D_34(input_dim, mode='train'):
    if mode == 'train':
        inputs = torch.autograd.Variable(torch.randn(size=input_dim))
    else:
        inputs = torch.randn(size=(input_dim[0], None, input_dim[-1]))
    print(inputs)
    print(inputs.shape)

     # def forward(self, input_x):
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    # use 40 dimensional mfcc or single

    channel_lists= [1,64,128,256,512]
    x1 = nn.Conv1d (in_channels=channel_lists[0], out_channels=channel_lists[1], kernel_size=7)(inputs)
    x1 = nn.BatchNorm1d(num_features=channel_lists[1])(x1)
    x1 = nn.ReLU6(inplace=True)(x1)
    x1 = nn.MaxPool1d(kernel_size=2, stride=2)(x1)

    # ===============================================
    #            Convolution Section 2
    # ===============================================

    x2 = conv_block_1D(input_tensor=x1, kernel_size=100,
                       filters=[64,64], strides=1, bias = True, In_channel = channel_lists[1])
    x2 = identity_block_1D(x2, 100, [64,64], bias=True,In_channel= channel_lists[1])
    x2 = identity_block_1D(x2, 100, [64, 64], bias=True, In_channel=channel_lists[1])

    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_1D(input_tensor=x2, kernel_size=100,
                       filters=[128,128], strides=1, bias = True, In_channel = channel_lists[1])
    x3 = identity_block_1D(x3, 100, [128,128], bias=True,In_channel= channel_lists[2])
    x3 = identity_block_1D(x3, 100, [128,128], bias=True, In_channel=channel_lists[2])

    return x3

if __name__ == '__main__':
    a = resnet_1D_34(input_dim=(1,1,22000), mode='train')
    print(a)
    # summary(a,input_size=(1,1,22000))
    # print(summary(a))