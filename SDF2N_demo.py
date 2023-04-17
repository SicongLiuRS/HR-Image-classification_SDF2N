from HyperFunctions import*
import time
from tensorflow.python.keras.utils import np_utils
from model import SDF2N_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
#     print("-----using cuda-----")
# 模型参数
wsize = 32
randtime = 10
nb_epoch = 100
batch_size = 128
group_clip = 50
time_step = 1
s1s2=1
train_ratio = 0.01
#***********************************************************************************************************************
data_set = input('Please input the name of data set(ZH17, UP):')
Dataset = data_set.upper()
if Dataset == 'ZH17':
    dataID = 1
    data_name = 'ZH17'
    OASpatial = np.zeros((7 + 3, randtime + 1))
elif Dataset == 'UP':
    dataID = 2
    data_name = 'UP'
    OASpatial = np.zeros((9 + 3, randtime + 1))
#******************************************************SDF2N************************************************************
#
for r in range(0,randtime):
    # 模型训练
    X, X_train, X_test, XP, XP_train, XP_test, Y, Y_train, Y_test, train_indexes = \
        HyperspectralSamples(dataID=dataID, timestep=time_step, w=wsize, israndom=True, s1s2=s1s2, random=r, train_ratio = train_ratio)

    XP_train = np.asarray(XP_train, dtype=np.float32)
    Y = Y - 1
    Y_train = Y_train - 1
    Y_test = Y_test - 1

    nb_classes = Y_train.max() + 1
    nb_features = X.shape[-1]
    nb_predict = int(len(XP_test) / group_clip)

    img_rows, img_cols = wsize, wsize
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(Y_train, nb_classes)
    y_test = np_utils.to_categorical(Y_test, nb_classes)

    start = time.time()
    model = SDF2N_model.SDF2N(nb_classes, nb_features, img_rows, img_cols)
    model.summary()
    histloss = model.fit([XP_train], [y_train], nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, shuffle=True)
    #模型预测
    PredictLabel = []
    for i in range (0, group_clip-1):
        XP1_test = np.asarray(XP_test[i*nb_predict : (i+1)*nb_predict], dtype=np.float32)
        predictmp = model.predict([XP1_test],verbose=1).argmax(axis=-1)
        PredictLabel.extend(predictmp)
    XP1_test = np.asarray(XP_test[(group_clip-1) * nb_predict : len(XP_test)], dtype=np.float32)
    predictmp = model.predict([XP1_test], verbose=1).argmax(axis=-1)
    PredictLabel.extend(predictmp)
    end = time.time()
    Runtime = end - start
    # 精度评价
    PredictLabel = np.asarray(PredictLabel, dtype=np.int)
    OA,Kappa,ProducerA = CalAccuracy(PredictLabel,Y_test[:,0])
    OASpatial[0:nb_classes,r] = ProducerA
    OASpatial[-3,r] = OA
    OASpatial[-2,r] = Kappa
    OASpatial[-1,r] = Runtime
    # 结果输出
    print('rand',r+1,'test accuracy:', OA*100)


