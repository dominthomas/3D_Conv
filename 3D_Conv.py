import nibabel as nib
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
import os
import re
import gc

ad_files = os.listdir("/home/k1651915/ad/")
cn_files = os.listdir("/home/k1651915/cn/")

sub_id_ad = []
sub_id_cn = []

for file in ad_files:
    sub_id = re.search('(OAS\\d*)', file).group(1)
    if sub_id not in sub_id_ad:
        sub_id_ad.append(sub_id)

for file in cn_files:
    sub_id = re.search('(OAS\\d*)', file).group(1)
    if sub_id not in sub_id_cn:
        sub_id_cn.append(sub_id)


def get_images(files, cn=False, ad_data_length=0):
    return_list = []
    for file in files:
        if cn and (abs(len(return_list) - ad_data_length)) <= 3:
            return return_list

        nifti_data = np.asarray(nib.load(file).get_fdata())
        nifti_data = nifti_data.reshape(1, 176, 256, 256)
        return_list.append(nifti_data)
    return return_list


os.chdir("/home/k1651915/ad/")
ad_sub_train = sub_id_ad[0:111]
ad_sub_validate = sub_id_ad[112:123]
ad_sub_test = sub_id_ad[124:177]

ad_sub_train_files = []
ad_sub_validate_files = []
ad_sub_test_files = []

for file in ad_files:
    file_sub_id = re.search('(OAS\\d*)', file).group(1)
    if file_sub_id in ad_sub_train:
        ad_sub_train_files.append(file)
    elif file_sub_id in ad_sub_validate:
        ad_sub_validate_files.append(file)
    elif file_sub_id in ad_sub_test:
        ad_sub_test_files.append(file)

ad_train = get_images(ad_sub_train_files)
ad_validate = get_images(ad_sub_validate_files)
ad_test = get_images(ad_sub_test_files)

os.chdir("/home/k1651915/cn/")

cn_sub_train = sub_id_cn[0:111]
cn_sub_validate = sub_id_cn[112:123]
cn_sub_test = sub_id_cn[124:177]

cn_sub_train_files = []
cn_sub_validate_files = []
cn_sub_test_files = []

for file in cn_files:
    file_sub_id = re.search('(OAS\\d*)', file).group(1)
    if file_sub_id in cn_sub_train:
        cn_sub_train_files.append(file)
    elif file_sub_id in cn_sub_validate:
        cn_sub_validate_files.append(file)
    elif file_sub_id in cn_sub_test:
        cn_sub_test_files.append(file)

print(len(ad_train))
cn_train = get_images(cn_sub_train_files, cn=True, ad_data_length=len(ad_train))
print("cn_train")
print(len(cn_train))
cn_validate = get_images(cn_sub_validate_files, cn=True, ad_data_length=len(ad_validate))

cn_test = get_images(cn_sub_test_files, cn=True, ad_data_length=len(ad_test))

train_data = cn_train + ad_train
train_data = np.asarray(train_data)

validation_data = cn_validate + ad_validate
validation_data = np.asarray(validation_data)

test_data = cn_test + ad_test
test_data = np.asarray(test_data)

x = np.zeros(len(cn_train))
y = np.zeros(len(ad_train))
train_labels = np.concatenate((x, y), axis=None)
print(len(cn_train))
print(len(ad_train))
print(train_labels.shape)

x = np.zeros(len(cn_validate))
y = np.zeros(len(ad_validate))
validation_labels = np.concatenate((x, y), axis=None)

x = np.zeros(len(cn_test))
y = np.zeros(len(ad_test))
test_labels = np.concatenate((x, y), axis=None)

ad_train = None
ad_validate = None
ad_test = None

cn_train = None
cn_validate = None
cn_test = None

gc.collect()

model = Sequential()

model.add(Conv3D(32,
                 kernel_size=(7, 7, 7),
                 input_shape=(1, 176, 256, 256),
                 data_format='channels_first',
                 padding='valid',
                 strides=(4, 4, 4),
                 activation='relu'))

model.add(MaxPooling3D(pool_size=(2, 2, 2),
                       strides=(2, 2, 2),
                       padding='valid'))

model.add(Conv3D(64,
                 kernel_size=(5, 5, 5),
                 strides=(1, 1, 1),
                 padding='valid',
                 activation='relu'))

model.add(MaxPooling3D(pool_size=(2, 2, 2),
                       strides=(2, 2, 2),
                       padding='valid'))

model.add(Conv3D(256,
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 padding='valid',
                 activation='relu'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_data,
                    train_labels,
                    epochs=50,
                    batch_size=50,
                    validation_data=(validation_data, validation_labels))

evaluation = model.evaluate(test_data, test_labels, verbose=0)