import tensorflow as tf
import pandas as pd
import numpy as np

sess = tf.InteractiveSession()
n_classes = 7
dim = 70
x = tf.placeholder(tf.float32, shape=[None, dim], name="Input")
y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="Output")
W = tf.Variable(tf.zeros([dim, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))

y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.AdagradOptimizer(1.0).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

saver.restore(sess,"/media/cluster/data1/lambert/models/model.ckpt-1645")
print "Model restored"
W =  W.eval()

new_w = np.hstack([W, np.zeros((70,1), dtype=np.float32)])
new_b = np.append(b.eval(), np.float32(0.0))

import balance_categories
dataset = balance_categories.load_balanced_data()
print "dataset"

trainset = dataset.sample(frac=0.8, random_state=200)
testset = dataset.drop(trainset.index)
print "trainset"
import numpy as np

def onehot(x):
	tmp = np.zeros(n_classes + 1)
	tmp[x] = 1
	return tmp

train_data = np.array(trainset[range(70)].values.tolist())
train_labels = np.array(trainset["labels"].apply(lambda x: onehot(x)).values.tolist())
print "testset"
test_data = np.array(testset[range(70)].values.tolist())
test_labels = np.array(testset["labels"].apply(lambda x: onehot(x)).values.tolist())
probs = sess.run(y, feed_dict={x: test_data})
from scipy.stats import entropy as entr


def normalize(probs):
	norms = [min(sublist) for sublist in probs]
	norm = [prob - norm for prob, norm in zip(probs, norms)]
	norm = [prob / sum(prob) for prob in norm]
	return norm
import math
print probs[10]
old_shape =  probs.shape
probs = normalize(probs)
probs = np.ravel(probs)
probs = [0.0000001 if i==0 else i for i in probs]
probs = np.reshape(probs, old_shape)
print probs[0]

entropies = -np.sum(probs * np.log(probs), axis=1)
entropies.sort()
print entropies
print len(entropies)
sorted = np.argsort(entropies)
print sorted[-1]
print entropies[sorted[-1]]
print probs[sorted[-1]]
print testset.iloc[sorted[0:10]]
sess.close()


from plotly.offline import init_notebook_mode, plot
init_notebook_mode()
from plotly.graph_objs import Scatter, Figure, Layout, Pie
trace = Scatter(x=range(len(entropies)), y=entropies)

data = [trace]
layout = Layout(title="", xaxis=dict(title=""),
				yaxis=dict(title=""))
fig = Figure(data=data, layout=layout)
# plot(fig)

import sys
sys.exit(0)

# new nn
sess = tf.InteractiveSession()
n_classes = 7 + 1
dim = 70
x = tf.placeholder(tf.float32, shape=[None, dim], name="Input")
y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="Output")
W = tf.Variable(new_w)
b = tf.Variable(new_b)

# import balance_categories
# balance_categories.add_subject("trump")

y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.AdagradOptimizer(1.0).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

batch_size = 1000
batch_count = len(train_data) / batch_size
epochs = 8
losses = []
train_accs = []
test_accs = []
f1s = []

for i in range(batch_count * epochs):
	begin = (i % batch_count) * batch_size
	end = (i % batch_count + 1) * batch_size
	print begin, end
	batch_data = np.array(train_data[begin: end])
	batch_labels = np.array(train_labels[begin: end])
	_, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_data, y_: batch_labels})

	test_acc = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
	train_acc = sess.run(accuracy, feed_dict={x: batch_data, y_: batch_labels})

	prediction = tf.argmax(y, 1)
	y_pred = prediction.eval(feed_dict={x: test_data})
	gold = []
	for l in test_labels:
		label = list(l).index(1)
		gold.append(label)

	from sklearn import metrics
	f1 = metrics.f1_score(gold, list(y_pred), average="weighted")
	# print loss, train_acc, test_acc, f1
	losses.append(loss)
	train_accs.append(train_acc)
	test_accs.append(test_acc)
	f1s.append(f1)
	print test_acc

