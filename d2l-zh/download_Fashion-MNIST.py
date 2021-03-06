import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time

mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

# print(len(mnist_train),len(mnist_test))
# feature, label = mnist_train[0]
# print(feature.shape, feature.dtype)
# print(label, type(label), label.dtype)

#将数值转化为标签（d2lzh包中集成了）
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat','sandal',
                   'shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]
#画图和对用的标签
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12,12))
    for f, img, lbl, in zip(figs, images, labels):
        f.imshow(img.reshape((28,28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)



X,y = mnist_train[0:9]
show_fashion_mnist(X, get_fashion_mnist_labels(y))

batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    num_workers = 0  #表示不用额外的进程来加速读取数据
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size,shuffle=False,num_workers=num_workers)

#读取一遍训练数据需要的时间
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' %(time.time() - start))
