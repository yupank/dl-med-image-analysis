import torch
import torch.nn as nn
import numpy as np

from cnn_models import Cnn3Layer, Cnn4LayerSiam
from tumor_utils import device, kagg_brain_image_loader, accuracy_rate

# cubclassing the CNNS into  multi-class models 
class Cnn3LayerMulti(Cnn3Layer):
    def __init__(self, out_class=4, init_nodes=16, conv_3_scale=4, n_channel=5, conv_kernel=5, inp_dim=28 * 28, drop=0.1):
        super(Cnn3LayerMulti, self).__init__(init_nodes, out_class, conv_3_scale, n_channel, conv_kernel, inp_dim, drop)
        self.name = f'2D_CNN_3L_multiclass_{init_nodes}_nodes'
        # self.out_class = out_class
        print(self.name)
        self.final = nn.LeakyReLU(negative_slope=0.2)
        print(self.out_class)

in_dim = 256*256

model = Cnn3LayerMulti(init_nodes=16, out_class=4, n_channel=3, conv_kernel=5, inp_dim=in_dim, drop = 0.1).to(device)
# model = Cnn3Layer(init_nodes=16, n_channel=3, conv_kernel=5, inp_dim=in_dim, drop = 0.1).to(device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f'total {params} parameters')


tumor_train_loader, tumor_test_loader = kagg_brain_image_loader(train_bs=8)
data = iter(tumor_test_loader)
test_images, test_labels = next(data)

with torch.no_grad():
    test_output = model(test_images)

final_accur, pred_labels = accuracy_rate(test_output, test_labels)
print(final_accur)

