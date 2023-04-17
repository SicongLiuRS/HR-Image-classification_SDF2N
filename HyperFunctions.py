import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile as tiff


def read_image(filename):
    img = tiff.imread(filename)
    img = img.transpose((1, 2, 0))
    img = np.asarray(img, dtype=np.float32)
    return img

def featureNormalize(X,type):
    #type==1 x = (x-mean)/std(x)
    #type==2 x = (x-max(x))/(max(x)-min(x))
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm
    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX)
        return X_norm    
    
def PCANorm(X,num_PC):
    mu = np.mean(X,0)
    X_norm = X-mu
    
    Sigma = np.cov(X_norm.T)
    [U, S, V] = np.linalg.svd(Sigma)   
    XPCANorm = np.dot(X_norm,U[:,0:num_PC])
    return XPCANorm
    
def MirrowCut(X,hw):
    #X  size: row * column * num_feature
    [row,col,n_feature] = X.shape

    X_extension = np.zeros((3*row,3*col,n_feature))
    
    for i in range(0,n_feature):
        lr = np.fliplr(X[:,:,i]) # 左右翻转
        ud = np.flipud(X[:,:,i]) # 上下翻转
        lrud = np.fliplr(ud) # 左右翻转
        
        l1 = np.concatenate((lrud,ud,lrud),axis=1) ##axis=1表示对应行的数组进行拼接
        l2 = np.concatenate((lr,X[:,:,i],lr),axis=1)
        l3 = np.concatenate((lrud,ud,lrud),axis=1)
        
        X_extension[:,:,i] = np.concatenate((l1,l2,l3),axis=0)
    
    X_extension = X_extension[row-hw:2*row+hw,col-hw:2*col+hw,:]
    
    return X_extension

def DrawResult(labels,imageID):
    #ID=1:Zurich 17
    #ID=2:Pavia University
    num_class = labels.max()+1
    # VHR: Zurich 2
    if imageID == 1:
        row = 1025
        col = 1112
        palette = np.array([[0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 78, 96],
                            [46, 139, 87]])
        palette = palette*1.0/255
    #HSI-Pavia University
    elif imageID == 2:
        row = 610
        col = 340
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [255, 0, 255],
                            [0, 255, 255],
                            [200, 100, 0],
                            [0, 200, 100],
                            [100, 0, 200]])
        palette = palette * 1.0 / 255
##############################################
    X_result = np.zeros((labels.shape[0],3))
    for i in range(0,num_class):
        X_result[np.where(labels==i),0] = palette[i,0]
        X_result[np.where(labels==i),1] = palette[i,1]
        X_result[np.where(labels==i),2] = palette[i,2]
    
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
    plt.imshow(X_result)    
    return X_result
    
def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = np.sum(predict==label)*1.0/n
    correct_sum = np.zeros((max(label)+1))
    reali = np.zeros((max(label)+1))
    predicti = np.zeros((max(label)+1))
    producerA = np.zeros((max(label)+1))
    
    for i in range(0,max(label)+1):
        correct_sum[i] = np.sum(label[np.where(predict==i)]==i)
        reali[i] = np.sum(label==i)
        predicti[i] = np.sum(predict==i)
        producerA[i] = correct_sum[i] / reali[i]
   
    Kappa = (n*np.sum(correct_sum) - np.sum(reali * predicti)) *1.0/ (n*n - np.sum(reali * predicti))
    return OA,Kappa,producerA

def HyperspectralSamples(dataID, timestep, w, israndom,  s1s2, random, train_ratio):

    if dataID == 1:
        data = sio.loadmat('./data/Zurich/zurich17.mat')
        X = data['zurich17']

        data = sio.loadmat('./data/Zurich/zurich17_gt.mat')
        Y = data['zurich17_gt']
        train_num_array = [154786, 150627, 111072, 129125, 10619, 9040, 6052]
        filename = 'zurich17_'
    elif dataID == 2:
        data = sio.loadmat('./data/HSI/PaviaU_AP.mat')
        X = data['X']
        Y = data['Y']
        train_num_array = [6631, 18649, 2099, 3064, 1345, 5029, 1330, 3682, 947]
        filename = 'UP'
    ##############################################

    train_num_array = (np.array(train_num_array) * train_ratio).astype('int')
    print('train_num_array:', train_num_array)

    [row,col,n_feature] = X.shape
    K = row*col #207400
    X = X.reshape(row*col, n_feature)
    Y = Y.reshape(row*col, 1)

    n_class = Y.max()

    nb_features = int(n_feature/timestep)

    train_num_all = sum(train_num_array)
    num_PC = n_feature
    X_PCA = featureNormalize(X,  1)
    # X_PCA = featureNormalize(PCANorm(X,num_PC),2)
    X = featureNormalize(X,  1)

    hw = int(w/2)
    # 扩增
    X_PCAMirrow = MirrowCut(X_PCA.reshape(row,col,num_PC),hw) #638,368,4

    # 列表形式定义并赋值
    XP1 = []
    for i in range(1,K+1):
        index_row = int(np.ceil(i*1.0/col)) #np.ceil向上取整
        index_col = i - (index_row-1)*col + hw -1 
        index_row += hw -1
        patch = X_PCAMirrow[index_row-hw:index_row+hw,index_col-hw:index_col+hw,:]
        XP1.append(patch)
    save_fn = './trainrandom/'+ filename + 'randomArray_'+ str(random+1) +'.mat'
    if israndom==True:
        randomArray = list()
        for i in range(1,n_class+1):
            index = np.where(Y==i)[0]
            n_data = index.shape[0]
            randomArray.append(np.random.permutation(n_data))
        sio.savemat(save_fn, {'randomArray': randomArray})

    else:
        # 加载每类样本随机排序位置
        data = sio.loadmat(save_fn)
        randomArray = data['randomArray']

    flag1=0
    flag2=0

    X_train = np.zeros((train_num_all,timestep,nb_features))
    X_test = np.zeros((sum(Y>0)[0]-train_num_all,timestep,nb_features))

    Y_train = np.zeros((train_num_all,1))
    Y_test = np.zeros((sum(Y>0)[0]-train_num_all,1))

    train_indexes = np.zeros((train_num_all, 2))
    ## 列表形式定义
    XP1_train = []
    XP1_test = []
    
    for i in range(1,n_class+1):
        index = np.where(Y==i)[0]
        n_data = index.shape[0]
        train_num = train_num_array[i-1]
        if israndom == True:
            randomX = randomArray[i-1]
        else:
            randomX = randomArray[0][i - 1][0]  # 每类样本位置索引

        # # ****************存储matlab需要的训练样本位置及信息*********************
        train_indexes[flag1:flag1 + train_num, 0] = index[randomX[0:train_num]]
        train_indexes[flag1:flag1 + train_num, 1] = i
        # # ****************存储matlab需要的训练样本位置及信息*********************
        #
        Y_train[flag1:flag1 + train_num, 0] = Y[index[randomX[0:train_num]], 0]
        Y_test[flag2:flag2+n_data-train_num,0] = Y[index[randomX[train_num:n_data]],0]

        ## 列表形式赋值
        XP1_trmp = [XP1[i] for i in list(index[randomX[0:train_num]])]
        XP1_train.extend(XP1_trmp)
        XP1_tsmp = [XP1[i] for i in list(index[randomX[train_num:n_data]])]
        XP1_test.extend(XP1_tsmp)

        if s1s2==2:
            
            for j in range(0,timestep):
                X_train[flag1:flag1+train_num,j,:] = X[index[randomX[0:train_num]],j:j+(nb_features-1)*timestep+1:timestep]
                X_test[flag2:flag2+n_data-train_num,j,:] = X[index[randomX[train_num:n_data]],j:j+(nb_features-1)*timestep+1:timestep]

                
        else:
            for j in range(0,timestep):
                X_train[flag1:flag1+train_num,j,:] = X[index[randomX[0:train_num]],j*nb_features:(j+1)*nb_features]
                X_test[flag2:flag2+n_data-train_num,j,:] = X[index[randomX[train_num:n_data]],j*nb_features:(j+1)*nb_features]

        flag1 = flag1+train_num
        flag2 = flag2+n_data-train_num

    X_reshape = np.zeros((X.shape[0],timestep,nb_features))

    if s1s2==2:
        
        for j in range(0,timestep):
            X_reshape[:,j,:] = X[:,j:j+(nb_features-1)*timestep+1:timestep]
    else:
        for j in range(0,timestep):
            X_reshape[:,j,:] = X[:,j*nb_features:(j+1)*nb_features]

    X = X_reshape

    return X.astype('float32'),X_train.astype('float32'),X_test.astype('float32'),XP1,XP1_train,XP1_test,Y.astype(int),Y_train.astype(int),Y_test.astype(int),train_indexes.astype(int)

   
   
   
