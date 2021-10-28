import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv_featmap_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )

        self.conv_featmap_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )

        self.conv_featmap_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )
    
    def forward(self, img):
        featmap_1 = self.conv_featmap_1(img)
        featmap_1_down = self.maxpool(featmap_1)

        featmap_2 = self.conv_featmap_2(featmap_1_down)
        featmap_2_down = self.maxpool(featmap_2)

        featmap_3 = self.conv_featmap_3(featmap_2_down)
        
        return featmap_3


class Encoder_RRDB(nn.Module):
    def __init__(self, num_feat=16):
        super(Encoder_RRDB, self).__init__()
        self.conv_featmap = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
        )
    
    def forward(self, img):
        featmap = self.conv_featmap(img)
        
        return featmap