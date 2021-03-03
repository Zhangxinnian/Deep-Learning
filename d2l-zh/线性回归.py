# from mxnet import nd
# from time import time
# a = nd.ones(shape=1000)
# b = nd.ones(shape=1000)
# start = time()
# # c = nd.zeros(shape=1000)
# # for i in range(1000):
# #     c[i] = a[i] + b[i]
# d = a + b
# print(time() - start)

from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
from mxnet.gluon import data as gdata
import random

#生成数据集（样本数1000，特征数2）
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0]*features[:,0] +true_w[1]*features[:,1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# # #print(features[0], labels[0])
# # def use_svg_display():
# #     #用矢量图显示
# #     display.set_matplotlib_formats('svg')
# #
# # def set_figsize(figsize=(3.5,2.5)):
# #     use_svg_display()
# #     #设置图的尺寸
# #     plt.rcParams['figure.figsize'] = figsize
# # set_figsize()
# # plt.scatter(features[:,1].asnumpy(),labels.asnumpy(),1);#加分号只显示图
# # plt.show()
#
# def data_iter(batch_size, features, labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices) #样本的读取顺序是随机的
#     for i in range(0, num_examples, batch_size):
#         j = nd.array(indices[i:min(i + batch_size, num_examples)])
#         yield features.take(j), labels.take(j) #take函数根据索引返回对应元素
#
# batch_size = 10
# # # for x, y in data_iter(batch_size, features, labels):
# # #     print(x, y)
# # #     break
#
# w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
# b = nd.zeros(shape=(1,))
#
# w.attach_grad()
# b.attach_grad()
#
# def linreg(X, w, b):
#     return nd.dot(X,w) + b
#
# #定义损失函数
# def squared_loss(y_hat, y):
#     return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
# #优化算法
# def sgd(params, lr, batch_size):
#     for param in params:
#         param[:] = param - lr * param.grad / batch_size
#
# #训练模型
# lr = 0.03
# num_epochs = 3
# net = linreg
# loss = squared_loss
# for epoch in range(num_epochs):
#     for X, y in data_iter(batch_size, features, labels):
#         with autograd.record():
#             l = loss(net(X,w,b),y) #l是有关小批量x和y的损失
#         l.backward() #小批量的损失对模型参数求梯度
#         sgd([w,b],lr,batch_size) #使用小批量随机梯度下降迭代模型参数
#     train_l = loss(net(features,w,b),labels)
#     print('epoch %d, loss %f' %(epoch +1, train_l.mean().asnumpy()))
#
# #比较学习到的参数和用来生成训练集的真实参数
# print(true_w, w)
# print(true_b, b)

batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
#随机读取小批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
# for X, y in data_iter:
#     print(X,y)
#     break
#定义网络
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))   #gluon中全连接层是一个Dense实例,定义输出层个数为1
#初始化网络参数
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
#定义损失函数
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()  #平方损失又称L2范数损失
#定义优化算法sgd
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})
#model train
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X),y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f'% (epoch, l.mean().asnumpy()))
#比较学习到的参数和真实的模型参数
dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())



