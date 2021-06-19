import numpy as np
from PIL import Image
import paddle
import paddle.fluid as fluid
from train import MyDNN,test_reader

# 模型评估
with fluid.dygraph.guard():
    accs = []
    model_dict, _ = fluid.load_dygraph('MyDNN')
    model = MyDNN()
    model.load_dict(model_dict)  # 加载模型参数
    model.eval()  # 评估模式
    for batch_id, data in enumerate(test_reader()):  # 测试集
        images = np.array([x[0].reshape(3, 100, 100) for x in data], np.float32)
        labels = np.array([x[1] for x in data]).astype('int64')
        labels = labels[:, np.newaxis]
        
        image = fluid.dygraph.to_variable(images)
        label = fluid.dygraph.to_variable(labels)
        
        predict = model(image)
        acc = fluid.layers.accuracy(predict, label)
        accs.append(acc.numpy()[0])
        avg_acc = np.mean(accs)
    print(avg_acc)