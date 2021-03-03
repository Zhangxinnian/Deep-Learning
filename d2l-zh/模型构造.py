from mxnet import nd
from mxnet.gluon import nn

# class MLP(nn.Block):
#     #申明带有模型参数的层（因搭建前面的多层感知机，故两层）
#     def __init__(self, **kwargs):
#         #调用MLP父类BLOCK的构造函数来进行必要的初始化。这样在构造实例时还能指定其他函数参数
#         super(MLP, self).__init__(**kwargs)
#         self.hidden = nn.Dense(256, activation='relu') #隐藏层
#         self.output = nn.Dense(10) #输出层
#
#     #定义模型的前向计算，根据输入x计算返回所需要的模型输出
#     def forward(self, x):
#         return self.output(self.hidden(x))
# #无需定义反向传播，系统自动求梯度自动生成反向传播，
# X = nd.random.uniform(shape=(2,20))
# net = MLP()
# net.initialize()
# print(net(X))

#Sequential类继承自Block类
# class MySequential(nn.Block):
#     def __init__(self,**kwargs):
#         super(MySequential,self).__init__(**kwargs)
#
#     def add(self,block):
#         # block是一个Block子类实例，假设它有一个独一无二的名字，我们将它保存在Block类
#         # 成员变量_children里，其类型是OrderedDict.当MySequential实例调用
#         # initialize函数时，系统会自动对_children里所有成员初始化
#         self._children[block.name] = block
#
#     def forward(self, x):
#         # OrderedDict保证会按照成员添加时的顺序遍历成员
#         for block in self._children.values():
#             x = block(x)
#         return x
# X = nd.random.uniform(shape=(2,20))
# net = MySequential()
# net.add(nn.Dense(256, activation='relu'))
# net.add(nn.Dense(10))
# net.initialize()
# print(net(X))

class FancyMLP(nn.Block):
    def __init__(self,**kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # 使用get_constant创建的随机权重参数不会在训练中被迭代（即常数参数）
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20,20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        #使用创建的常数参数，以及NDArray的relu函数和dot函数
        x = nd.relu(nd.dot(x, self.rand_weight.data())+1)
        #复用全连接层。等价于两个全连接层共享参数
        x = self.dense(x)
        #控制流，这里我们需要调用asscalar函数来返回标量进行比较
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()
# X = nd.random.uniform(shape=(2,20))
# net = FancyMLP()
# net.initialize()
# print(net(X))

class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

X = nd.random.uniform(shape=(2,20))
net = nn.Sequential()
net.add(NestMLP(), nn.Dense(20),FancyMLP())
net.initialize()
print(net(X))

