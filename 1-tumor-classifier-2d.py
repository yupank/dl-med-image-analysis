from cnn_models import Cnn3Layer, Cnn4LayerSiam
from tumor_utils import device, image_loader, cnn_image_classifier

in_dim = 96*96
model = Cnn3Layer(init_nodes=36, n_channel=5, conv_kernel=5, inp_dim=in_dim, drop = 0.15).to(device)
# model = Cnn4LayerSiam(init_nodes=20, n_channel=5, conv_kernel=5, inp_dim=in_dim, drop = 0.2).to(device)

stan_train_loader, stan_test_loader = image_loader(train_bs=16, data_3D=False)

fin_acc, false_neg, execution_tm = cnn_image_classifier(model, stan_train_loader, stan_test_loader, 
                                          model_run_tag='5a2', learn_rate=0.001, epochs=50)

print(f'final test accuracy: {fin_acc} false negatives: {false_neg}')
print(f'execution time: {execution_tm}')