from torch import nn
from torch.nn import functional as F



# VGG-19
class VGG19(nn.Module):
    # initialize model
    def __init__(self, img_size:int=224, input_channel:int=3, n_class:int=2):
        super().__init__()
        self.num_class = n_class
        self.input_channel = input_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.fc17 = nn.Sequential(
            nn.Linear(int(512 * img_size * img_size / 32 / 32), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.fc18 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.fc19 = nn.Sequential(
            nn.Linear(4096, self.num_class)
        )

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
                          self.conv8, self.conv9, self.conv10, self.conv11, self.conv12, self.conv13, self.conv14,
                          self.conv15, self.conv16]

        self.fc_list = [self.fc17, self.fc18, self.fc19]

        print("VGG-19 Model Initialize Successfully!")

    # forward
    def forward(self, x):
        for conv in self.conv_list:    # 16 CONV
            x = conv(x)
        output = x.view(x.size()[0], -1)
        for fc in self.fc_list:        # 3 FC
            output = fc(output)
        return output



class ResBlk(nn.Module):
    # resnet block
    def __init__(self, ch_in: int, ch_out: int, stride: int=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] -> [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2),
                nn.BatchNorm2d(ch_out))

    def forward(self, x):
        # x:[b, ch, h, w]
        out =F.relu(self.bn1(self.conv1(x)))
        out =self.bn2(self.conv2(out))
        # short cut
        out = self.extra(x) + out
        # out = F.relu(out)
        return out



# ResNet34
class ResNet34(nn.Module):
    def __init__(self, n_class:int=10, input_channel:int=3):
        super(ResNet34, self).__init__()
        self.num_class = n_class
        self.input_channel = input_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        
        # followed 4 blocks
        # [b, 64, h, w] -> [b, 128, h, w]
        # self.blk1 = ResBlk(16, 16)
        self.blk1 = nn.Sequential(
            ResBlk(64, 64, 1),
            ResBlk(64, 64, 1),
            ResBlk(64, 64, 1))
        
        # [b, 128, h, w] -> [b, 256, h, w]
        # self.blk2 = ResBlk(16, 32)
        self.blk2 = nn.Sequential(
            ResBlk(64, 128, 2),
            ResBlk(128, 128, 1),
            ResBlk(128, 128, 1),
            ResBlk(128, 128, 1))
        
        # [b, 256, h, w] -> [b, 512, h, w]
        # self.blk3 = ResBlk(128, 256)
        self.blk3 = nn.Sequential(
            ResBlk(128, 256, 2),
            ResBlk(256, 256, 1),
            ResBlk(256, 256, 1),
            ResBlk(256, 256, 1),
            ResBlk(256, 256, 1),
            ResBlk(256, 256, 1))

        # [b, 512, h, w] -> [b, 1024, h, w]
        # self.blk4 = ResBlk(256, 512)
        self.blk4 = nn.Sequential(
            ResBlk(256, 512, 2),
            ResBlk(512, 512, 1),
            ResBlk(512, 512, 1))
        
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.outlayer = nn.Linear(512, self.num_class)

        print("ResNet-34 Model Initialize Successfully!")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # [b, 64, h, w] -> [b, 128, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x