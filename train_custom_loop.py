# 深度学习中, log 的底数是 e.
import numpy as np
from ch01.TwoLayerNet import TwoLayerNet
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt


# 1 设定参数
learning_rate = 1.0
hidden_size = 10
epochs = 300
batch_size = 30

# 2 加载数据、生成模型和优化器
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)


data_size = len(x)
total_loss = 0
loss_count = 0
loss_list = []
iters = data_size // batch_size

# 3 打乱数据
for epoch in range(epochs):
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iter in range(iters):
        x_batch = x[iter * batch_size : (iter + 1) *  batch_size]
        t_batch = t[iter * batch_size : (iter + 1) *  batch_size]
        loss = model.forward(x_batch, t_batch)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iter + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d | iter %d / %d | loss %.2f'%
                  (epoch + 1, iter + 1, iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss = 0
            loss_count = 0

# 4 计算梯度、更新参数

# 5 设置输出学习结果

# 绘制学习结果
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.show()

# 绘制决策边界
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# 绘制数据点
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()












