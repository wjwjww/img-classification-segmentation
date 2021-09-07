from torch import nn
import torch.nn.functional as F
from .arch_utils import initialize_weights


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # 3 * 224 * 224
        self.conv1_1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(inplace=True))#64*224*224
        self.conv1_2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(inplace=True))#64*224*224
        self.maxpool1 = nn.MaxPool2d(2, 2)   # pooling 64 * 112 * 112

        self.conv2_1 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(inplace=True))#128*112*112
        self.conv2_2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(inplace=True))#28*112*112
        self.maxpool2 = nn.MaxPool2d(2, 2)   # pooling 128 * 56 * 56

        self.conv3_1 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True))#256*56*56
        self.conv3_2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True))#256*56*56
        self.conv3_3 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True))#256*56*56
        self.maxpool3 = nn.MaxPool2d(2, 2)   # pooling 256 * 28 * 28

        self.conv4_1 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(inplace=True))#512*28*28
        self.conv4_2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(inplace=True))#512*28*28
        self.conv4_3 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(inplace=True))#512*28*28
        self.maxpool4 = nn.MaxPool2d(2, 2)# pooling 512 * 14 * 14

        self.conv5_1 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(inplace=True))#512*14*14
        self.conv5_2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(inplace=True))#512*14*14
        self.conv5_3 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(inplace=True))#512*14*14
        self.maxpool5 = nn.MaxPool2d(2, 2)# pooling 512 * 7 * 7

        # view

        self.fc1 = nn.Linear(25088, 2048)#25984
        self.fc2 = nn.Linear(2048, 1000)
        self.fc3 = nn.Linear(1000, 45)
        # softmax 1 * 1 * 1000

        initialize_weights(self)

    def forward(self, x):
            # x.size(0)即为batch_size
        in_size = x.size(0)

        out1_1 = self.conv1_1(x)#224*224*64
        out1_2 = self.conv1_2(out1_1)#224*224*64
        out1 = self.maxpool1(out1_2)#112*112*64

        out2_1 = self.conv2_1(out1) #112*112*128
        out2_2 = self.conv2_2(out2_1)  #112*112*128
        out2 = self.maxpool2(out2_2)  #56*56*128

        out3_1 = self.conv3_1(out2)  # 56*56*256
        out3_2 = self.conv3_2(out3_1)  # 56*56*256
        out3_3 = self.conv3_3(out3_2)  # 56*56*256
        out3 = self.maxpool3(out3_3)  # 28*28*256

        out4_1 = self.conv4_1(out3)  # 28*28*512
        out4_2 = self.conv4_2(out4_1)  # 28*28*512
        out4_3 = self.conv4_3(out4_2)  # 28*28*512
        out4 = self.maxpool4(out4_3)  # 14*14*512

        out5_1 = self.conv5_1(out4)  # 14*14*512
        out5_2 = self.conv5_2(out5_1)  # 14*14*512
        out5_3 = self.conv5_3(out5_2)  # 14*14*512
        out5 = self.maxpool5(out5_3)  # 7*7*512
        out = out5.view(in_size, -1)

        out6 = F.relu(self.fc1(out))
        out = F.dropout(out6, p=0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, p=0.5)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)

        # return out

        #return  out2_2, out3_3, out4_3, out6

        return out
