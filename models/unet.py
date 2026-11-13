import torch
import torch.nn as nn
import torch.nn.functional as F

# Double Convolution Part 
# [Convolution(3x3) --> Batch Normalization --> ReLu] * 2
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            #First Convolution Step
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)
    
#Down Action
# [Max Pooling(2x2) --> Double Convolution]
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Up Action  
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_module = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up_module(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Output Action Step
# [Convolution(1x1)]
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
#U-Net Model
class UNET(nn.Module):
    def __init__(self, features=64, in_channels=3, out_channels=1):
        super().__init__()
        
        # Initial Block
        self.inc = DoubleConv(in_channels,features)
        
        #Encoder Block
        self.down1 = Down(features, features*2)
        self.down2 = Down(features*2, features*4)
        self.down3 = Down(features*4, features*8)
        self.down4 = Down(features*8, features*16)
        
        #Decoder Block
        self.up1 = DecoderBlock(features*16, features*8)
        self.up2 = DecoderBlock(features*8, features*4)
        self.up3 = DecoderBlock(features*4, features*2)
        self.up4 = DecoderBlock(features*2, features)
        
        #Output Block
        self.out = OutConv(features, out_channels)
        
    def forward(self, x):     
        x1 = self.inc(x)     #[bs x 64 x 320 x 320]
        #print(f'\ninc size: {x1.shape}')
        
        x2 = self.down1(x1)  #[bs x 128 x 160, 160]
        #print(f'\ndown1 size: {x2.shape}')

        x3 = self.down2(x2)  #[bs x 256 x 80, 80]
        #print(f'\ndown2 size: {x3.shape}')

        x4 = self.down3(x3)  #[bs x 512 x 40, 40]
        #print(f'\ndown3 size: {x4.shape}')

        x5 = self.down4(x4)  #[bs x 1024 x 20, 20]
        #print(f'\ndown4 size: {x5.shape}')
       
        x = self.up1(x5, x4) #[bs x 512 x 40, 40]
        #print(f'\nup1 size: {x.shape}')

        x = self.up2(x, x3)  #[bs x 256 x 80, 80]
        #print(f'\nup2 size: {x.shape}')

        x = self.up3(x, x2)  #[bs x 128 x 160, 160]
        #print(f'\nup3 size: {x.shape}')
        
        x = self.up4(x, x1)  #[bs x 64 x 320, 320]
        #print(f'\nup4 size: {x.shape}')
        
        logits = self.out(x) #[bs x 1 x 320, 320]
        #print(f'\nlogits size: {logits.shape}')

        return logits
        
        
def test():
    x = torch.randn((8, 3, 320, 320))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    
if __name__ == "__main__":
    test()