import d2lzh as d2l
from mxnet import autograd,nd
#定义绘图函数
def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(),y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')


x = nd.arange(-8.0,8.0,0.1)
x.attach_grad()
#定义RELU函数
# with autograd.record():
#     y = x.relu()
# #xyplot(x,y,'relu')
#relu的导数图
# y.backward()
# xyplot(x,x.grad,'grad of relu')
# import matplotlib.pyplot as plt
# plt.show()

# #定义sigmoid函数
# with autograd.record():
#     y = x.sigmoid()
# #xyplot(x,y,'sigmoid')
# #sigmoid的导数图
# y.backward()
# xyplot(x,x.grad,'grad of sigmoid')
# import matplotlib.pyplot as plt
# plt.show()

#定义tanh函数
with autograd.record():
    y = x.tanh()
#xyplot(x,y,'tanh')
#tanh的导数图
y.backward()
xyplot(x,x.grad,'grad of tanh')
import matplotlib.pyplot as plt
plt.show()
