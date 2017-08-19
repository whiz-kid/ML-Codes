import tensorflow as tf

# tensorflow - tensors + flow
# flow based on computational graph
# tensorflow core programs consist of two parts -
# building a computaional graph and running a computational graph

w=tf.Variable([.2],tf.float32)
b=tf.Variable([-.2],tf.float32)
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

# our model
linear_model=w*x+b

# loss function
model_loss = tf.reduce_sum(tf.square(linear_model-y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(model_loss)
epoch=1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})

    print(sess.run([w,b]))