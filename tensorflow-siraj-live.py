import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE

#load data and format it
dataframe = pd.read_csv('data.csv')
dataframe = dataframe.drop(['index', 'price', 'sq_price'],axis=1)
#we only use the first 10 rows
dataframe = dataframe[0:10]
print dataframe

#add labels
dataframe.loc[:,('y1')]=[1,1,1,0,0,1,0,1,1,1]
dataframe.loc[:,('y2')] = dataframe['y1'] == 0

dataframe.loc[:,('y2')] = dataframe['y2'].astype(int)
print dataframe

#prepare data for tensorflow
inputX = dataframe.loc[:,['area', 'bathrooms']].as_matrix()

#convert labels to input tensorflow
inputY = dataframe.loc[:,['y1','y2']].as_matrix()

print inputX
print inputY

#hyperparameters
learning_rate = 0.000001
traning_epochs = 2000
display_steps = 50
n_samples = inputY.size

#Create a computation neural network
x = tf.placeholder(tf.float32,[None,2])

#create wights
W = tf.Variable(tf.zeros([2,2]))

#add biases
b = tf.Variable(tf.zeros([2]))

#multiply weights by inputs
y_values = tf.add(tf.matmul(x,W),b)

#activation function
y = tf.nn.softmax(y_values)

#feed in a matrix of labels
y_ = tf.placeholder(tf.float32, [None,2])

#perform training
#create our cost function, mean squared error
cost = tf.reduce_sum(tf.pow(y_ - y,2)/(2*n_samples))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#initialize variables and tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#training loop
for i in range(traning_epochs):
	sess.run(optimizer, feed_dict={x: inputX, y_:inputY})

	if(i)%display_steps ==0:
		cc = sess.run(cost,feed_dict={x: inputX, y_:inputY})
		print "training step:", '%04d' %(i),"cost=","{:.9f}".format(cc)


print "Optimization Finished!"
traning_cost =  sess.run(cost, feed_dict={x: inputX, y_:inputY})
print "training cost= ", traning_cost, "W=",sess.run(W), "b=",sess.run(b)



#test
print sess.run(y,feed_dict={x:inputX})