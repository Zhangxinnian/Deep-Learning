import d2lzh as d2l
from mxnet import autograd, gluon,init,nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd

#读取数据集
train_data = pd.read_csv('./House_Prices/train.csv')
test_data = pd.read_csv('./House_Prices/test.csv')

# print(train_data.shape)
# print(test_data.shape)
#查看样本的前四个特征、后两个特征和标签
# print(train_data.iloc[0:4,[0, 1, 2, 3, -3, -2, -1]])

#训练数据和测试数据的79个特征按样本连结
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

#预处理数据集
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
#标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)
#将离散数值转成指示特征
#dummy_na=True 将缺失值也当做合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features,dummy_na=True)
#print(all_features.shape)
#转换成NDArray
n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))

#model train 使用线性回归模型，平方损失函数
loss = gloss.L2Loss()
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256,activation='relu'),

             nn.Dropout(0.6),
            nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    #net.initialize()
    return net
#对数均方根误差
def log_rmse(net, features, labels):
    #将小于1的值设置为1
    clipped_preds = nd.clip(net(features),1,float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(),labels.log()).mean())
    return rmse.asscalar()

#优化算法使用Adam
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels),batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':learning_rate,'wd':weight_decay})
    for epoch in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                l = loss(net(X),y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features,test_labels))
    return train_ls, test_ls

#K折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j+1)*fold_size)
        X_part,y_part = X[idx,:], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid

#在k折交叉验证中训练K次并返回训练和验证的平均误差
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs','rmse',
                         range(1, num_epochs + 1), valid_ls, ['train','valid'])
        print('fold %d, train rmse %f, valid rmse %f' %(i, train_ls[-1],valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

#使用未经过调优的超参数计算交叉验证误差
k, num_epochs, lr, weight_decay, batch_size = 5, 90, 0.1,100 , 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f'
      % (k, train_l, valid_l))
     