import numpy as np
import paddle
from PIL import Image
import paddle.fluid as fluid
from model import MyDNN
import cv2

#读取预测图像，进行预测
def load_image(path):
    img = Image.open(path)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))
    img = img / 255.0
    print(img.shape)
    return img

# 构建预测动态图过程
with fluid.dygraph.guard():
    infer_path = './data/ges-Dataset/6/IMG_1124.JPG'
    model_dict, _ = fluid.load_dygraph('MyDNN.pdparams')  # fluid.load_dygraph(model_path)
    model =MyDNN()
    
    model.load_dict(model_dict)  # 加载模型参数
    model.eval()  # 评估模式
    infer_img = load_image(infer_path)
    infer_img = np.array(infer_img).astype('float32')
    infer_img = infer_img[np.newaxis, :, :, :]
    infer_img = fluid.dygraph.to_variable(infer_img)
    result = model(infer_img)
    print(result)
    img=cv2.imread(infer_path)
    cv2.imshow('11',img)

    print(np.argmax(result.numpy()))
