#导入所需的包
import cv2
import numpy as np
from PIL import Image
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from multiprocessing import cpu_count
from model import MyDNN

# 定义data_reader
def data_mapper(sample):
    img, label = sample
    img = Image.open(img)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))     #读出来的图像是rgb,rgb,rbg..., 转置为 rrr...,ggg...,bbb...
    img = img/255.0
    return img, label

def data_reader(data_list_path):
    def reader():
        with open(data_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)
    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 512)


# 用于训练的数据提供器
train_reader = paddle.batch(reader=paddle.reader.shuffle(reader=data_reader('./data/train_data.txt'), buf_size=256), batch_size=32)
# 用于测试的数据提供器
test_reader = paddle.batch(reader=data_reader('./data/test_data.txt'), batch_size=32)



# 用动态图进行训练
with fluid.dygraph.guard():
    model = MyDNN()  # 模型实例化
    model.train()  # 训练模式
    opt = fluid.optimizer.SGDOptimizer(learning_rate=0.01,
                                       parameter_list=model.parameters())  # 优化器选用SGD随机梯度下降，学习率为0.001.
    
    epochs_num = 1000  # 迭代次数
    
    for pass_num in range(epochs_num):
        
        for batch_id, data in enumerate(train_reader()):
            images = np.array([x[0].reshape(3, 100, 100) for x in data], np.float32)
            labels = np.array([x[1] for x in data]).astype('int64')
            labels = labels[:, np.newaxis]

            # 将Numpy转换为DyGraph接收的输入.该函数实现从numpy.ndarray对象创建一个Variable类型的对象。
            image = fluid.dygraph.to_variable(images)
            label = fluid.dygraph.to_variable(labels)
            predict = model(image)  # 训练
            
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)  # 获取avg_loss值
            
            acc = fluid.layers.accuracy(predict, label)  # 计算精度
            
            if batch_id != 0 and batch_id % 15 == 0:
                print(
                    "train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num, batch_id, avg_loss.numpy(),
                                                                                  acc.numpy()))
            
            avg_loss.backward()  # 使用backward()方法可以执行反向网络
            opt.minimize(avg_loss)  # 调用定义的优化器对象的minimize方法进行参数更新
            model.clear_gradients()  # 每一轮参数更新完成后我们调用clear_gradients()来重置梯度,以保证下一轮的正确性
    
    fluid.save_dygraph(model.state_dict(), 'MyDNN')  # 保存模型




