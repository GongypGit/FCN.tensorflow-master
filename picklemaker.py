import glob
import os
import numpy as np
from six.moves import cPickle as pickle
from PIL import Image

file_trianing = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join('/home/spyder/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/CarData/training_img2/', '*'))]
file_anotation = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join('/home/spyder/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/CarData/annotation_img2/', '*'))]

'''
功能：获取训练集图片名，不包括文件格式
使用：将数据集文件对应匹配上上就行
返回：文件名列表
'''
file_trianing.sort()
file_anotation.sort()
'''以上返回的文件名是乱序，对其按照序号进行排序'''

#将jpg格式的文件转换成png格式的文件,需要转换的时候调用
'''
for ima in file_trianing:
    im = Image.open(os.path.join('/home/jiangmignzhi/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/CarData/annotation_img2', ima+'.png'))
    im = im.convert('L')
    img = Image.new("L", im.size)
    img.paste(im)
    img.save(os.path.join('/home/jiangmignzhi/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/CarData/', ima+'.png'), 'png')
'''
'''
功能：对灰度图像进行映射，将[0-255]的像素值映射到[0-1]，并转换为‘L’mode
使用：im 为像素值数组，按照映射要求对im 数值进行数值操作即可。Image.fromarray（）函数，将数组转换成对应的图片
'''
'''
for ima in file_trianing:
    im = np.array(Image.open(os.path.join('/home/jiangmignzhi/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/CarData/annotation_img2', ima+'.png')))
    img = Image.fromarray(im//255, 'L')
    img.save(os.path.join('/home/jiangmignzhi/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/CarData/', ima+'.png'), 'png')
'''
im = Image.open('/home/spyder/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/CarData/annotation_img2/carimg_358.png')
im2 = Image.open('/home/spyder/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/ADEChallengeData2016/annotations/training/ADE_train_00000001.png')
print(im.mode)
print(im2.mode)
train_data1 = []
train_data2 = []
for file in file_trianing[:50]:#选择数量用于训练
    fi = file
    fi = {'image':'/home/spyder/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/CarData/training_img2/'+ file + '.jpg' ,'annotation':'/home/spyder/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/CarData/annotation_img2/' + file + '.png','filename':file}
    train_data1.append(fi)
    del fi #del:删除变量，而不删除数据，在这里把每次创建的字典名del掉，节省内存
for file in file_trianing[770:]:#选择用于验证的图片数量
    fi = file
    fi = {'image':'/home/spyder/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/CarData/training_img2/'+ file + '.jpg' ,'annotation':'/home/spyder/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/CarData/annotation_img2/' + file + '.png','filename':file}
    train_data2.append(fi)
    del fi #del:删除变量，而不删除数据，在这里把每次创建的字典名del掉，节省内存
file_dict = {'training': train_data1, 'validation': train_data2}
'''
功能：将每个图片的信息以字典的形式保存，集中存入一个数组，将两个数组以字典形式保存起来，方便后面将数组信息生成pickle文件
使用：先将文件名读出来，再与所在路径想结合，实现：对每个图片创建一个字典，注意：每个图片必须使用不同的字典名
同一个字典每个key占用的是同一块地址，一变具变
'''
with open(os.path.join('/home/spyder/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing','trainingData.pickle'), 'wb') as f:
    pickle.dump(file_dict, f)
'''
功能：将图片信息存为pickle文件
使用：f对应文件名，train_data为要存入的数据集
'''
'''
pickle_filepath = os.path.join('/home/jiangmignzhi/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/CarData','trainingData.pickle')
with open(pickle_filepath, 'rb') as f:
    result = pickle.load(f)
    train_records = []
    training_records = result['training']
    validation_records = result['validation']
    print(training_records)
    del result
'''
'''
功能：读取pickle文件，返回的是一个字典，在分别利用字典将两个类别的数据集分割开来
'''
