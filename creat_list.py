#导入所需的包
import os
# 生成图像列表
data_path = 'D:\\01mycode\\01Gesture-recognition\\data\\ges-Dataset\\'
character_folders = os.listdir(data_path)
# print(character_folders)
if (os.path.exists('./data/train_data.txt')):
    os.remove('./data/train_data.txt')
if (os.path.exists('./data/test_data.txt')):
    os.remove('./data/test_data.txt')

for character_folder in character_folders:

    with open('./data/train_data.txt', 'a') as f_train:
        with open('./data/test_data.txt', 'a') as f_test:
            if character_folder == '.DS_Store':
                continue
            character_imgs = os.listdir(os.path.join(data_path, character_folder))
            count = 0
            for img in character_imgs:
                if img == '.DS_Store':
                    continue
                if count % 10 == 0:
                    f_test.write(os.path.join(data_path, character_folder, img) + '\t' + character_folder + '\n')
                else:
                    f_train.write(os.path.join(data_path, character_folder, img) + '\t' + character_folder + '\n')
                count += 1
print('列表已生成')