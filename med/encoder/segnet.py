import torch.nn as nn
from copy import deepcopy
from torchsummary import summary


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, bias=True, dilation=1):
        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Down2(nn.Module):
    def __init__(self, in_planes, out_planes, pool_size=(2, 2), pool_stride=(2, 2)):
        super(Down2, self).__init__()
        self.conv1 = ConvBNReLU(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBNReLU(out_planes, out_planes, kernel_size=3, stride=1, padding=1)

        self.maxpool_with_argmax = nn.MaxPool2d(pool_size, pool_stride, return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        unpooled_shape = x.size()
        outputs, indices = self.maxpool_with_argmax(x)

        return outputs, indices, unpooled_shape


class Down3(nn.Module):
    def __init__(self, in_planes, out_planes, pool_size=(2, 2), pool_stride=(2, 2)):
        super(Down3, self).__init__()
        self.conv1 = ConvBNReLU(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBNReLU(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBNReLU(out_planes, out_planes, kernel_size=3, stride=1, padding=1)

        self.maxpool_with_argmax = nn.MaxPool2d(pool_size, pool_stride, return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        unpooled_shape = x.size()
        outputs, indices = self.maxpool_with_argmax(x)

        return outputs, indices, unpooled_shape


class Up2(nn.Module):
    def __init__(self, in_planes, out_planes, pool_size=(2, 2), pool_stride=(2, 2)):
        super(Up2, self).__init__()
        self.unpool = nn.MaxUnpool2d(pool_size, pool_stride)
        self.conv1 = ConvBNReLU(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBNReLU(out_planes, out_planes, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs, indices, output_shape):
        x = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class Up3(nn.Module):
    def __init__(self, in_planes, out_planes, pool_size=(2, 2), pool_stride=(2, 2)):
        super(Up3, self).__init__()
        self.unpool = nn.MaxUnpool2d(pool_size, pool_stride)
        self.conv1 = ConvBNReLU(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBNReLU(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBNReLU(out_planes, out_planes, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs, indices, output_shape):
        x = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class SegNet(nn.Module):
    def __init__(self, in_channels=1):
        super(SegNet, self).__init__()

        self.in_channels = in_channels

        self.down1 = Down2(self.in_channels, 64)
        self.down2 = Down2(64, 128)
        self.down3 = Down3(128, 256)
        self.down4 = Down3(256, 512)
        self.down5 = Down3(512, 512)
        self.down6 = Down3(512, 1024)

        self.up6 = Up3(1024, 512)
        self.up5 = Up3(512, 512)
        self.up4 = Up3(512, 256)
        self.up3 = Up3(256, 128)
        self.up2 = Up2(128, 64)
        self.up1 = Up2(64, self.in_channels)

    def forward(self, x):
        down1, indices_1, unpool_shape1 = self.down1(x)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        down6, indices_6, unpool_shape6 = self.down6(down5)

        up6 = self.up6(down6, indices_6, unpool_shape6)
        up5 = self.up5(up6, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1


class SegNetv2(nn.Module):
    def __init__(self, in_channels=1):
        super(SegNetv2, self).__init__()

        self.in_channels = in_channels

        self.down1 = Down2(self.in_channels, 64)
        self.down2 = Down2(64, 128)
        self.down3 = Down2(128, 256)
        self.down4 = Down3(256, 256)
        self.down5 = Down3(256, 512)
        self.down6 = Down3(512, 512)
        self.down7 = Down3(512, 1024)

        self.up7 = Up3(1024, 512)
        self.up6 = Up3(512, 512)
        self.up5 = Up3(512, 256)
        self.up4 = Up3(256, 256)
        self.up3 = Up2(256, 128)
        self.up2 = Up2(128, 64)
        self.up1 = Up2(64, self.in_channels)

    def forward(self, x):
        down1, indices_1, unpool_shape1 = self.down1(x)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        down6, indices_6, unpool_shape6 = self.down6(down5)
        down7, indices_7, unpool_shape7 = self.down7(down6)

        up7 = self.up7(down7, indices_7, unpool_shape7)
        up6 = self.up6(up7, indices_6, unpool_shape6)
        up5 = self.up5(up6, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, y, z):
        return x


def get_encoder(model):
    encoder = deepcopy(model)

    for name, module in encoder.named_children():
        if isinstance(module, Up2) or isinstance(module, Up3):
            setattr(encoder, name, Identity())

    return encoder


if __name__ == '__main__':
    model = SegNet().to("cuda")

    import torch

    inputs = torch.autograd.Variable(torch.rand(1, 1, 512, 256)).to("cuda")

    outputs = model(inputs)

    print(outputs.shape)
