import tensorflow as tf

sess = tf.Session()
#define 2 variables we want to compute
theta0 = tf.Variable([0.0],tf.float32)
theta1 = tf.Variable([0.0],tf.float32)

#define training set
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = theta0 + theta1 * x

x_trained = [0,1,2,3]
y_trained = [0,-1,-2,-3]

#the cost function
cost = tf.reduce_sum(tf.square(linear_model-y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    #use feed to replace the output of placeholder operation with x_trained
    #and y_trained
    sess.run(train,{x:x_trained,y:y_trained})

print(sess.run([theta0,theta1,cost],{x:x_trained,y:y_trained}))
