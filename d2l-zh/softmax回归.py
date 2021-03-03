# import d2lzh as d2l
# from mxnet import autograd, nd
# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#
# num_inputs = 784
# num_outputs = 10
# w = nd.random.normal(scale=0.01, shape=(num_inputs,num_outputs))
# b = nd.zeros(num_outputs)
# w.attach_grad()
# b.attach_grad()
#
# #softmax运算
# def softmax(X):
#     X_exp = X.exp()
#     partition = X_exp.sum(axis=1, keepdims=True)
#     return X_exp / partition
#
# # X = nd.random.normal(shape=(2,5))
# # X_prob = softmax(X)
# # print(X_prob,X_prob.sum(axis=1))
#
# #定义模型
# def net(X):
#     return softmax(nd.dot(X.reshape((-1, num_inputs)),w) +b)
# #定义损失函数
# #y_hat = nd.array([[0.1,0.3,0.6],[0.3,0.2,0.5]])
# #y = nd.array([0,2],dtype='int32')
# # print(nd.pick(y_hat,y))
#
# def cross_entropy(y_hat, y):
#     return -nd.pick(y_hat,y).log()
# #计算分类准确率
# def accuracy(y_hat, y):
#     return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
# #print(accuracy(y_hat,y))
# #评价模型net在数据集data_iter上的准确率
# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#     for X,y in data_iter:
#         y = y.astype('float32')
#         acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
#         n += y.size
#     return acc_sum / n
# #print(evaluate_accuracy(test_iter,net))
#
# #train_model
# num_epochs, lr = 5, 0.1
# def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,params=None,lr=None,trainer=None):
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
#         for X, y in train_iter:
#             with autograd.record():
#                 y_hat = net(X)
#                 l = loss(y_hat,y).sum()
#             l.backward()
#             if trainer is None:
#                 d2l.sgd(params,lr,batch_size)
#             else:
#                 trainer.step(batch_size)
#             y = y.astype('float32')
#             train_l_sum += l.asscalar()
#             train_acc_sum += (y_hat.argmax(axis=1)==y).sum().asscalar()
#             n += y.size
#         test_acc = evaluate_accuracy(test_iter,net)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %(epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
# train_ch3(net, train_iter, test_iter,cross_entropy,num_epochs,batch_size,[w,b],lr)
#
# #预测
# for X, y in test_iter:
#     break
# ture_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
# pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
# titles = [true + '\n' + pred for true,pred in zip(ture_labels,pred_labels)]
# d2l.show_fashion_mnist(X[0:9],titles[0:9])

##########简洁实现
import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
#读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#定义和初始化模型
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
#交叉熵损失
loss = gloss.SoftmaxCrossEntropyLoss()
#定义优化算法
trainer  = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1})
#train model
num_epochs = 5
d2l.train_ch3(net, train_iter,test_iter,loss,num_epochs,batch_size,None,None,trainer)

