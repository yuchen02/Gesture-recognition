
import paddle.fluid as fluid

from paddle.fluid.dygraph import Linear
#定义DNN网络
class MyDNN(fluid.dygraph.Layer):
    def __init__(self):
        super(MyDNN,self).__init__()
        self.hidden1 = Linear(100,100,act='relu')
        self.hidden2 = Linear(100,100,act='relu')
        self.hidden3 = Linear(100,100,act='relu')
        self.hidden4 = Linear(3*100*100,10,act='softmax')
    def forward(self,input):
        # print(input.shape)
        x = self.hidden1(input)
        # print(x.shape)
        x = self.hidden2(x)
        # print(x.shape)
        x = self.hidden3(x)
        x = fluid.layers.reshape(x, shape=[-1,3*100*100])
        y = self.hidden4(x)
        # print(y.shape)
        return y