from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize() #使用默认初始化方式
X = nd.random.uniform(shape=(2,20))
Y = net(X)
#print(net[0].params, type(net[0].params))
#权重参数
#print(net[0].weight.data())
#权重梯度的形状
#print(net[0].weight.grad())
#访问输出层的偏差值
#print(net[1].bias.data())
#获取net变量的所有嵌套的层所包含的所有参数
#print(net.collect_params())
#可以通过正则表达式来匹配参数名，从而筛选需要的参数
#print(net.collect_params('.*weight'))
#初始化模型参数   