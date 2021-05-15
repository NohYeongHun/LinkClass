# CNN

## 데이터 세트 만들기
1. 이미지 전처리 변수 설정
- batch_size : 한번에 어느정도의 크기를 학습할지를 결정 
- img_height : 이미지의 높이 픽셀
- img_width : 이미지의 너비 픽셀

```python
# example 
batch_size = 32
img_height = 180
img_width = 180
```

2. 데이터 셔플
- 모델 개발시 검증분할을 사용하는 것이 모델 개발에 유리함
- validation_split : 데이터 중 무작위로 고를  검증 데이터 비율
- seed : 데이터를 무작위로 고를 때의 난수 생성 초기값 고정
- subset : training or validation validation_split이 설정된 경우에만 사용이 가능하다.
- image_size : 이미지의 사이즈를 지정
- batch_size : 한번에 어느정도의 크기를 학습할지를 결정
- data_dir : directory
- class_names : 분류할 클래스 이름 

```python
# example for training
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# example for validation 
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# classification class
class_names = train_ds.class_names
print(class_names)
```

## 데이터 시각화
1. 데이터세트 시각화
- figsize : 그림의 크기 인치 단위 
- take() : 데이터 행열 추출
- subplot() : (x, y, z) x행 y열에서 z번째
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10)) # 가로 세로 10 inch
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
```
- image_batch는 (32, 180, 180, 3) 형상의 텐서이며, 180x180x3 형상의 32개 이미지 묶음으로 되어 있습니다(마지막 차원은 색상 채널 RGB를 나타냄). label_batch는 형상 (32,)의 텐서이며 32개 이미지에 해당하는 레이블입니다.
- image batch : train_ds의 이미지 데이터 형식

- image_batch 및 labels_batch 텐서에서 .numpy()를 호출하여 이를 numpy.ndarray로 변환할 수 있습니다.
- labels_batch : label의 형태 해당 예시에서는 32개의 이미지에 해당하는 레이블

```python
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
```

## 데이터 성능 올리기
- 버퍼링된 프리페치를 사용하여 I/O를 차단하지 않고 디스크에서 데이터를 생성할 수 있도록 하겠습니다. 데이터를 로드할 때 다음 두 가지 중요한 메서드를 사용해야 합니다.

- Dataset.cache()는 첫 epoch 동안 디스크에서 이미지를 로드한 후 이미지를 메모리에 유지합니다. 이렇게 하면 모델을 훈련하는 동안 데이터세트가 병목 상태가 되지 않습니다. 데이터세트가 너무 커서 메모리에 맞지 않는 경우, 이 메서드를 사용하여 성능이 높은 온디스크 캐시를 생성할 수도 있습니다.

- Dataset.prefetch()는 훈련 중에 데이터 전처리 및 모델 실행과 겹칩니다- .

- 관심 있는 독자는 데이터 성능 가이드에서 두 가지 방법과 디스크에 데이터를 캐싱하는 방법에 대해 자세히 알아볼 수 있습니다.

- tf.data.experimental.AUTOTUNE : tf.data 런타임이 실행 시에 동적으로 값을 조정.

```python
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```


## 데이터 표준화
RGB 채널 값은 0, 255 범위에 있습니다. 신경망에는 이상적이지 않습니다. 일반적으로 입력 값을 작게 만들어야 합니다. 여기서는 Rescaling 레이어를 사용하여 값이 0, 1에 있도록 표준화합니다.
- layers.experimental.preprocessing.Rescaling(1./255) : RGB 값 0~255를 0~1사이의 범위에 있도록 표준화
```python
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
```

## 모델 만들기
모델은 각각에 최대 풀 레이어가 있는 3개의 컨볼루션 블록으로 구성됩니다. 그 위에 relu 활성화 함수에 의해 활성화되는 128개의 단위가 있는 완전히 연결된 레이어가 있습니다. 이 모델은 높은 정확성을 고려해 조정되지 않았습니다. 이 튜토리얼의 목표는 표준적인 접근법을 보여주는 것입니다.

### layers.Conv2D() :
1. 첫번째 인자 : 컨볼루션 필터의 수 입니다.

2. 두번째 인자 : 컨볼루션 커널의 (행, 열) 입니다.

3. padding : 경계 처리 방법을 정의합니다.
- ‘valid’ : 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다.
- ‘same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.

4. input_shape : 샘플 수를 제외한 입력 형태를 정의 합니다. 모델에서 첫 레이어일 때만 정의하면 됩니다.
(행, 열, 채널 수)로 정의합니다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다.

5. activation : 활성화 함수 설정합니다.
- ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
- ‘relu’ : rectifier 함수, 은닉층에 주로 쓰입니다.
- ‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
- ‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.

### layers.MaxPooling2D() :
- 컨볼루션 레이어의 출력 이미지에서 주요값만 뽑아 크기가 작은 출력 영상을 만듭니다. 이것은 지역적인 사소한 변화가 영향을 미치지 않도록 합니다.
- pool_size : 수직, 수평 축소 비율을 지정합니다. (2, 2)이면 출력 영상 크기는 입력 영상 크기의 반으로 줄어듭니다.

### layers.Flatten() :
- 영상을 일차원으로 바꿔주는 플래튼(Flatten) 레이어

### layers.Dense() : 
- 규칙적으로 조밀하게 연결된 NN 레이어임.
- tf.keras.layers.Dense는 input을 넣었을 때 output으로 바꿔주는 중간 다리
- units : 출력 값의 크기
- activation : 활성화 함수
- use_bias : 편향(b)을 사용할지 여부
- kernel_initializer : 가중치(W) 초기화 함수
- bias_iniotializer : 편향 초기화 함수
- kernel_regularizer : 가중치 정규화 방법
- bias_regularizer : 편향 정규화 방법
- activity_regularizer : 출력 값 정규화 방법
- kernel_constraint : 가중치에 적용되는 부가적인 제약 함수
- bias_constraint : 편향에 적용되는 부가적인 제약 함수


```python
num_classes = 5

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
```
## 모델 컴파일하기
이 튜토리얼에서는 optimizers.Adam 옵티마이저 및 losses.SparseCategoricalCrossentropy 손실 함수를 선택합니다. 각 훈련 epoch에 대한 훈련 및 검증 정확성을 보려면 metrics 인수를 전달합니다.

### model.compile() :
- optimizer: 문자열 (옵티마이저의 이름) 혹은 옵티마이저 인스턴스
사용 목록 : 
SGD
RMSprop
adam
Adadelta
Adagrad
Adamax
nadam
Ftrl


- loss: 문자열 (목적 함수의 이름) 혹은 목적 함수. 손실을 참조하십시오. 모델이 다중 아웃풋을 갖는 경우, 손실의 리스트 혹은 손실의 딕셔너리를 전달하여 각 아웃풋에 각기 다른 손실을 사용할 수 있습니다. 따라서 모델에 의해 최소화되는 손실 값은 모든 개별적 손실의 합이 됩니다.

- metrics: 학습과 테스트 과정에서 모델이 평가할 측정항목의 리스트. 보통은 metrics=['accuracy']를 사용하면 됩니다. 다중 아웃풋 모델의 각 아웃풋에 각기 다른 측정항목을 특정하려면, metrics={'output_a': 'accuracy'}와 같은 딕셔너리를 전달할 수도 있습니다.

- loss_weights: 각기 다른 모델 아웃풋의 손실 기여도에 가중치를 부여하는 스칼라 계수(파이썬 부동소수점)를 특정하는 선택적 리스트 혹은 딕셔너리. 따라서 모델이 최소화할 손실 값은 loss_weights 계수에 의해 가중치가 적용된 모든 개별 손실의 합이 됩니다. 리스트의 경우 모델의 아웃풋에 1:1 매핑을 가져야 하며, 텐서의 경우 아웃풋 이름(문자열)을 스칼라 계수에 매핑해야 합니다.

- sample_weight_mode: 시간 단계별로 샘플 가중치를 주어야 하는 경우 (2D 가중치), 이 인수를 "temporal"로. 설정하십시오. 디폴트 값은 None으로 (1D) 샘플별 가중치를 적용합니다. 모델이 다중 아웃풋을 갖는 경우 모디의 리스트 혹은 모드의 딕셔너리를 전달하여 각 아웃풋에 별도의 sample_weight_mode를 사용할 수 있습니다.

- weighted_metrics: 학습 혹은 테스트 과정에서 sample_weight 혹은 class_weight로 가중치를 주고 평가할 측정항목의 리스트.

- target_tensors: 케라스는 디폴트 설정으로 모델의 표적을 위한 플레이스 홀더를 만들고, 학습 과정 중 이 플레이스 홀더에 표적 데이터를 채웁니다. 이러한 디폴트 설정 대신 직접 만든 표적 텐서를 사용하고 싶다면 (이 경우 케라스는 학습 과정 중 이러한 표적의 외부 Numpy 데이터를 기대하지 않습니다), target_tensors 인수를 통해서 그 표적 텐서를 특정할 수 있습니다. 표적 텐서는 (단일 아웃풋 모델의 경우) 단일 텐서, 텐서 리스트, 혹은 표적 텐서에 아웃풋 이름을 매핑하는 딕셔너리가 될 수 있습니다.

- **kwargs: Theano/CNTK 백엔드를 사용하는 경우, 이 인수는 K.function에 전달됩니다. 텐서플로우 백엔드를 사용하는 경우, 이는 tf.Session.run에 전달됩니다.


```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

```

## 모델 요약

```python
model.summary() # 모델 요약
```

## 모델 훈련하기
- epochs : 훈련을 몇번 할 것인지?
```python
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```

## 훈련 결과 시각화하기
훈련 및 검증 세트에 대한 손실과 정확성 플롯을 생성합니다

```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

플롯에서 볼 수 있듯이 훈련 정확성과 검증 정확성은 큰 차이가 있으며, 모델은 검증 세트에서 약 60%의 정확성까지만 도달합니다.

## 과대 적합
위의 플롯에서 훈련 정확성은 시간이 지남에 따라 선형적으로 증가하는 반면, 검증 정확성은 훈련 과정에서 약 60%를 벗어나지 못합니다. 또한 훈련 정확성과 검증 정확성 간의 정확성 차이가 상당한데, 이는 과대적합의 징후입니다.

훈련 예제가 적을 때 모델은 새로운 예제에서 모델의 성능에 부정적인 영향을 미치는 정도까지 훈련 예제의 노이즈나 원치 않는 세부까지 학습합니다. 이 현상을 과대적합이라고 합니다. 이는 모델이 새 데이터세트에서 일반화하는 데 어려움이 있음을 의미합니다.

훈련 과정에서 과대적합을 막는 여러 가지 방법들이 있습니다. 이 튜토리얼에서는 데이터 증강을 사용하고 모델에 드롭아웃을 추가합니다.

# 데이터 증강
과대적합은 일반적으로 훈련 예제가 적을 때 발생합니다. 데이터 증강은 증강한 다음 믿을 수 있는 이미지를 생성하는 임의 변환을 사용하는 방법으로 기존 예제에서 추가 훈련 데이터를 생성하는 접근법을 취합니다. 그러면 모델이 데이터의 더 많은 측면을 파악하게 되므로 일반화가 더 쉬워집니다.

여기서는 실험적인 Keras 전처리 레이어를 사용하여 데이터 증강을 구현합니다. 이들 레이어는 다른 레이어와 마찬가지로 모델 내에 포함될 수 있으며, GPU에서 실행됩니다.

```python
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
```

- 데이터 증강 시각화
```python
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

```

## 드롭아웃
과대적합을 줄이는 또 다른 기술은 정규화의 한 형태인 드롭아웃을 네트워크에 도입하는 것입니다.

드롭아웃을 레이어에 적용하면, 훈련 프로세스 중에 레이어에서 여러 출력 단위가 무작위로 드롭아웃됩니다(활성화를 0으로 설정). 드롭아웃은 0.1, 0.2, 0.4 등의 형식으로 소수를 입력 값으로 사용합니다. 이는 적용된 레이어에서 출력 단위의 10%, 20% 또는 40%를 임의로 제거하는 것을 의미합니다.

layers.Dropout을 사용하여 새로운 신경망을 생성한 다음, 증강 이미지를 사용하여 훈련해 보겠습니다.

``` python
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
```

## 모델 컴파일 및 훈련하기
```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

- 요약
```python
model.summary()
```

```python
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```
## 훈련 결과 시각화하기
데이터 증강 및 드롭아웃을 적용한 후, 이전보다 과대적합이 줄어들고 훈련 및 검증 정확성이 더 가깝게 조정됩니다.
```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
```


# Reference
[Tensorflow-ImageClassification-Tutorial]https://www.tensorflow.org/tutorials/images/classification

