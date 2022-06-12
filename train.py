import tensorflow as tf
from utils.DataProcess import *
from models.model import resBlock, resBlockV2

net = tf.keras.Sequential([
    resBlockV2()
])
train_iter = geneDataSet("./pic/train_X", "./pic/train_Y", batch_size=2)

loss = tf.keras.losses.MeanSquaredError()
trainer = tf.keras.optimizers.Adam(learning_rate=0.01)
i = 0
total_loss = 0
for x, y in train_iter:
    with tf.GradientTape() as tape:
        y_hat = net(x)
        l = loss(y, y_hat)
        w = net.trainable_variables
    trainer.minimize(l, w, tape=tape)
    i += 1
    total_loss += l.numpy()
    if i % 10 == 0:
        print('[' + '=' * (i // 10) + '>' + ' ' * (25 - i // 10) + ']' + 'loss:', total_loss / i)
print('ok')
