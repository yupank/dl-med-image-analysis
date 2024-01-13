import torch
import torch.nn as nn
import numpy as np

from cnn_models import Cnn3Layer, Cnn4LayerSiam
from tumor_utils import device, kagg_brain_image_loader, cnn_image_classifier

# cubclassing the CNNS into  multi-class models 
class Cnn3LayerMulti(Cnn3Layer):
    def __init__(self, out_class=4, init_nodes=16, conv_3_scale=4, n_channel=5, conv_kernel=5, inp_dim=28 * 28, drop=0.1):
        super(Cnn3LayerMulti, self).__init__(init_nodes, out_class, conv_3_scale, n_channel, conv_kernel, inp_dim, drop)
        self.name = f'2D_CNN_3L_multiclass_{init_nodes}_nodes'
        self.final = nn.LeakyReLU(negative_slope=0.1)

class Cnn4LayerMulti(Cnn4LayerSiam):
    def __init__(self, init_nodes=12, out_class=1, n_channel=3, conv_kernel=5, inp_dim=28 * 28, drop=0.1):
        super(Cnn4LayerMulti, self).__init__(init_nodes, out_class, n_channel, conv_kernel, inp_dim, drop)
        self.name = f'2D_CNN_4L_multiclass_{init_nodes}_nodes'
        self.final = nn.LeakyReLU(negative_slope=0.1)

in_dim = 144*144

model = Cnn4LayerMulti(init_nodes=12, out_class=4, n_channel=1, conv_kernel=7, inp_dim=in_dim, drop = 0.2).to(device)
# model = Cnn3Layer(init_nodes=16, n_channel=3, conv_kernel=5, inp_dim=in_dim, drop = 0.1).to(device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f'total {params} parameters')



tumor_train_loader, tumor_test_loader = kagg_brain_image_loader(train_bs=64)
# test_data = iter(tumor_test_loader)
# test_images, test_labels = next(test_data)
# print(test_images.shape)


fin_acc, false_neg, execution_tm = cnn_image_classifier(
    model, tumor_train_loader, tumor_test_loader, 
    model_run_tag='6b3', learn_rate=0.005, epochs=80, criterion=nn.CrossEntropyLoss(),
    rep_folder='reports/kag_brain_tumors')

print(f'final test accuracy: {fin_acc} false negatives: {false_neg}')
print(f'execution time: {execution_tm}')

