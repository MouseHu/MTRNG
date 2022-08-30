from network.fc import *
import matplotlib.pyplot as plt
from utils import l1_norm
from sklearn.manifold import TSNE

model_type = 1
if model_type == 1:
    model_dir = "/home/hh/mtrng/model/0_1024_20000_0.0003_20_temper_save.ckpt"
    model = Temper()
    model.load_state_dict(torch.load(model_dir))
    weight = model.fc4.weight
elif model_type == 2:
    model_dir = "/home/hh/mtrng/model/1_128_20000_0.0003_20_temper_resnet_large.ckpt"
    model = ResCNNTemper()
    model.load_state_dict(torch.load(model_dir))

    weight = model.res_cnn1.conv2.weight
elif model_type ==3:
    model_dir = "/home/hh/mtrng/model/2_128_20000_0.0003_20_twister_large_resnet.ckpt"
    model = CNNTwister()
    model.load_state_dict(torch.load(model_dir))
    weight = model.res_cnn3.conv2.weight
# print(l1_norm(model.fc1).item())
# print(l1_norm(model.fc4).item())
# fc1 = model.fc1.weight.detach().cpu().numpy()
# print(fc1.shape)
weight = weight.detach().cpu().numpy()
print(weight.reshape(-1).std())
plt.hist(weight.reshape(-1), bins=20)
plt.show()
# print(fc1[:,0])
# tsne = TSNE(n_components=2, learning_rate='auto',
#                   init='random')
# X_embedded = tsne.fit_transform(fc1)
# Y_embedded = tsne.fit_transform(fc4.transpose())
# print(X_embedded.shape)
#
# plt.scatter(X_embedded[:, 0]-X_embedded[:, 0].mean(), X_embedded[:, 1]-X_embedded[:,1].mean())
# plt.scatter(Y_embedded[:, 0], Y_embedded[:, 1])
# plt.show()
