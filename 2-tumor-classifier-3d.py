from cnn_models import Cnn3Layer3d
from tumor_utils import device, image_loader, cnn_image_classifier

in_dim = 96*96
model = Cnn3Layer3d(init_nodes=24, n_channel=1, conv_kernel=3, inp_dim=in_dim, drop = 0.1).to(device)

stan_train_loader, stan_test_loader = image_loader(train_bs=16, data_3D=True)

fin_acc, false_neg, execution_tm = cnn_image_classifier(model, stan_train_loader, stan_test_loader, 
                                          model_run_tag='4a1', learn_rate=0.001, epochs=50)

print(f'final test accuracy: {fin_acc} false negatives: {false_neg}')
print(f'execution time: {execution_tm}')