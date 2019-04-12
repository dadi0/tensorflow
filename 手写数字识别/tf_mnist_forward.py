#coding:utf-8
#输入是28*28=784像素点
#输出是0-9概率
#导入模块，初始化输入层，隐含层，输出层节点数
import tensorflow as tf
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

#初始化权值
def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

#初始化阈值
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

#前向传播过程
def forward(x, reglarizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], reglarizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], reglarizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y
