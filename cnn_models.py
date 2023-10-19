import torch
import torch.nn as nn


class Cnn3Layer(nn.Module):
    """ The scalable CNN of 3 convolutional 2D units followed by a single linear layer with sigmoid output function.
        Model takes the data as stacks of 2D-images with n channels  (5 for the current dataset), i.e. as 3D tensors.
        As activation functions, both ReLU and LeakyReLU work fine, without notable difference in the perforance
        Parameters:
            init_nodes |int - number of nodes in the 1st convolutional layer;  the number of nodes 
                in next conv units expands two-fold and in the unit 3 can be increased or decreased;
            conv_3_scale | int - scaling factor for number of nodes in the conv layer 3;
            conv_kernel | int - size of kernel (filter) for all Conv layers
            n_channels |int - number of channels in the input image;
            inp_dim | int - number of pixels in the input image;
            drop | float - the probablity for all Dropout layer

    """
    def __init__(self, init_nodes=16, conv_3_scale = 4, n_channel=5, conv_kernel= 5, inp_dim=28*28, drop=0.1):
        super(Cnn3Layer, self).__init__()

        self.name = f'2D_CNN_3L_{init_nodes}_nodes'
        pad = int((conv_kernel-1)/2)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(n_channel, init_nodes, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(init_nodes, init_nodes*2, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(init_nodes*2, init_nodes*conv_3_scale, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes*conv_3_scale),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)

        )
        out_dim = int(inp_dim/64)
        self.lin_1 = nn.Linear(init_nodes*conv_3_scale*out_dim,1)
        self.final = nn.Sigmoid()

    def forward(self,x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = out.view(out.size(0), -1)
        out = self.lin_1(out)
        out = self.final(out)
        return out.squeeze(1)
    

class Cnn4LayerSiam(nn.Module):
    """ CNN similar to the above but with 4 convolutional units
        Model has a "siamese" structure with nodes numbers increasing from unit 1 to 2 and then decreasing towards unit 4
    """
    def __init__(self, init_nodes=8, n_channel=3, conv_kernel= 5, inp_dim=28*28, drop= 0.1):
        super(Cnn4LayerSiam, self).__init__()
        self.name = f'2D_CNN_4L{init_nodes}_nodes'
        pad = int((conv_kernel-1)/2)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(n_channel, init_nodes, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(init_nodes, init_nodes*2, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(init_nodes*2, init_nodes*2, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(init_nodes*2, init_nodes, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        out_dim = int(inp_dim/256)
        self.lin_1 = nn.Linear(init_nodes*out_dim,1)
        self.final = nn.Sigmoid()

    def forward(self,x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.lin_1(out)
        out = self.final(out)
        return out.squeeze(1)
    

class Cnn3Layer3d(nn.Module):
    """ The scalable CNN of 2 convolutional 3D units followed by two fully connected linear units and sigmoid output function.
        Model takes the data as 3D images as n channels (1 for the current dataset), i.e. as 4D tensors.
        As activation functions, both ReLU and LeakyReLU work fine, with sligtly smoother conversion with the LeakyReLU
        Parameters:
            init_nodes |int - number of nodes in the 1st convolutional layer;  
                the number of nodes in next conv units expands two-fold;
            conv_kernel | int - size of kernel (filter) for Conv3d layers
            n_channels |int - number of channels in the input image;
            inp_dim | int - number of pixels in the input image;
            drop | float - the probablity for all Dropout layer
    """  
    def __init__(self, init_nodes=16, n_channel=1, conv_kernel= 3, inp_dim=32*32, drop=0.1):
        super(Cnn3Layer3d, self).__init__()

        self.name = f'3D_CNN_{init_nodes}_nodes'
        pad = int((conv_kernel-1)/2)
        self.conv_unit_1 = nn.Sequential(
            nn.Conv3d(n_channel, init_nodes, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm3d(init_nodes),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        self.conv_unit_2 = nn.Sequential(
            nn.Conv3d(init_nodes, init_nodes*2, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm3d(init_nodes*2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )

        out_dim = inp_dim/16
        self.lin_unit_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(init_nodes*2*out_dim), init_nodes*4),
            nn.ReLU(),
            nn.BatchNorm1d(init_nodes*4),
            nn.Dropout(drop)
        )

        self.lin_unit_2 = nn.Linear(init_nodes*4,1)
        self.final = nn.Sigmoid()

    def forward(self,x):
        out = self.conv_unit_1(x)
        out = self.conv_unit_2(out)
        out = self.lin_unit_1(out)
        out = self.lin_unit_2(out)
        out = self.final(out)
        return out.squeeze(1)