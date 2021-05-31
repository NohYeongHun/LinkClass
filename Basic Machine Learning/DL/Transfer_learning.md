## 라이브러리
```python
## 레이어
from keras.layers import Input, Lambda, Dense, Flatten

## 모델
from keras.models import Model

# VGG 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
```

## 이미지 크기 지정
```python
# image parameter 
IMAGE_SIZE =[224,224]
train_path = 'trafficnet_dataset_v1/train'
valid_path = 'trafficnet_dataset_v1/test'
```

## 전이학습 모델 생성
```python
# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights = 'imagenet',include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False
```

## 이미지 파일 경로 설정
```python
# useful for getting number of classes
folders = glob('trafficnet_dataset_v1/train/*')

folders
# folders result
['trafficnet_dataset_v1/train/accident',
 'trafficnet_dataset_v1/train/sparse_traffic',
 'trafficnet_dataset_v1/train/dense_traffic',
 'trafficnet_dataset_v1/train/fire']

 len(folders)
```

## 클래스 예측
```python
# 각 folder 클래스 예측.ex) 사고, 원할한 차 통행, 원할하지 않은 통행, 화재
prediction = Dense(len(folders), activation='softmax')(x)
```

## 모델 구성 확인
```python
# 모델 객체 생성
model = Model(inputs=vgg.input, outputs = prediction)
vgg16_model_obejct = model

# vgg16 모델 설명
model.summary()
```
## 모델 시각화
```python
import keras.utils
keras.utils.plot_model(model)
```

## 모델 컴파일
```python
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)
```

## 데이터 전처리
```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

# rgb scaling (1~255) => (0~1)
test_datagen = ImageDataGenerator(rescale = 1./255)                                

# class set
training_set = train_datagen.flow_from_directory('trafficnet_dataset_v1/train',
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# class set
test_set = test_datagen.flow_from_directory('trafficnet_dataset_v1/test',
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

```

## early stopping
```python
from keras.callbacks import EarlyStopping, ModelCheckpoint

# early stopping parameter setting.
epochs_to_wait_improve=10 # Stop if validation_accuracy does not increase within 10 times

model_name = 'model/trafficfeatures_vgg16_model.h5' # model checkpoint(if accuracy increase)

early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience=epochs_to_wait_improve)

checkpoint_callback = ModelCheckpoint(model_name, monitor='val_loss',verbose=1, save_best_only=True,mode='min')

```

## 학습
```python

r = model.fit_generator(
    training_set, # traing_set
    validation_data = test_set, # validation_set
    epochs=50, # epoch
    steps_per_epoch=len(training_set), 
    validation_steps=len(test_set),
    callbacks=[early_stopping_callback, checkpoint_callback]# Executed when earlystopping condition is satisfied
)
```

## 시각화
```python
# loss
plt.title('LossVal_loss_vgg16')
plt.plot(r.history['loss'],label = 'train_loss')
plt.plot(r.history['val_loss'],label = 'val_loss')
plt.legend()
plt.savefig('LossVal_loss_vgg16')

# accuracies
plt.title('AccVal_acc_vgg16')
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('AccVal_acc_vgg16')
```

## 모델 저장
자동으로 저장되지만 수동으로 저장하고 싶을때 사용가능한 방법
```python
import tensorflow as tf

model.save('model/trafficfeatures_vgg16_model.h5')
```

## 모델 불러오기
```python
import tensorflow as tf

# model_load
model = tf.keras.models.load_model('model/trafficfeatures_vgg16_model.h5', compile=False)
```

## 예측 함수 생성
```python
import tensorflow as tf
import cv2
from tensorflow.python.keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from IPython.display import Image  # image input

# predict function
def run_predict(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image,dsize=(224,224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0],image.shape[1],image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    print("accident_accuray : ",yhat[0][0])
    print("dense_traffic_accuray : ",yhat[0][1])
    print("fire_accuray : ",yhat[0][2])
    print("sparse_traffic_accuray : ",yhat[0][3])
```

## 예측 시행
images/test1.JPG 이미지 예측
```python
file_path = 'images/4.jpg'
run_predict(file_path)
Image("images/4.jpg")
```