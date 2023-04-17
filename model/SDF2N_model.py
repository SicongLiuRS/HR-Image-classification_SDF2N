from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, BatchNormalization, Concatenate, Add, Dropout
from model import attention_model
from tensorflow.keras.optimizers import Adam


def SDF2N(nb_classes, nb_features, img_rows, img_cols):
    CNNInput = Input(shape=[img_rows, img_cols, nb_features], name='CNNInput')

    CONV1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='CONV1')(CNNInput)
    CONV1 = BatchNormalization(axis=-1)(CONV1)
    CONV1_1 = MaxPooling2D((2, 2), name='POOL1_1')(CONV1)
    CONV1_2 = AveragePooling2D((2, 2), name='POOL1_2')(CONV1)
    CONV1 = Concatenate()([CONV1_1, CONV1_2])
    CONV1 = attention_model.attach_attention_module(CONV1, 'se_block')

    CONV2_1   = Conv2D(128, (1, 1), activation='relu', padding='same', name='CONV2_1')(CONV1)
    CONV2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='CONV2_2')(CONV2_1)
    CONV2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(CONV2_2)
    CONV2_3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='CONV2_3')(CONV2_1)
    CONV2 = Concatenate()([CONV2_1, CONV2_2, CONV2_3])
    CONV2 = attention_model.attach_attention_module(CONV2, 'se_block')

    CONV3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='CONV3')(CONV2)
    CONV3 = MaxPooling2D((2, 2), name='POOL3')(CONV3)

    CONV4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='CONV4')(CONV3)
    CONV4 = MaxPooling2D((2, 2), name='POOL4')(CONV4)

    CONV5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='CONV5')(CONV4)
    CONV5 = MaxPooling2D((2, 2), name='POOL5')(CONV5)

    FLATTEN3 = Flatten(name='FLATTEN3')(CONV3)
    FLATTEN4 = Flatten(name='FLATTEN4')(CONV4)
    FLATTEN5 = Flatten(name='FLATTEN5')(CONV5)
    DENSE3 = Dense(128, activation='relu', name='DENSE1')(FLATTEN3)
    DENSE4 = Dense(128, activation='relu', name='DENSE2')(FLATTEN4)
    DENSE5 = Dense(128, activation='relu', name='DENSE3')(FLATTEN5)

    CNNDense = Add()([DENSE3, DENSE4, DENSE5])
     
    CNNDense = Dropout(0.5)(CNNDense)
    CNNSOFTMAX = Dense(nb_classes, activation='softmax')(CNNDense)

    model = Model(input=[CNNInput], output=[CNNSOFTMAX])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=['accuracy'])
    # 网络可视化
    # plot_model(model, to_file='SSFN.png')
    return model
