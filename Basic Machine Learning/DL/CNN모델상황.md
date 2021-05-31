# 모델상황

# vgg16 전이학습 모델
- val_accuracy = 83.75%


# custom-vgg16 전이학습 모델

1차시 
```python
# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights = 'imagenet',include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# prediction = Dense(len(folders), activation='softmax')(x)
prediction = Dense(len(folders), activation='softmax')

add_model = models.Sequential()
add_model.add(vgg)
add_model.add(layers.Flatten())
add_model.add(layers.Dropout(0.5))
add_model.add(prediction)

# 모델평가 
scores = add_model.evaluate_generator(test_set,steps=len(test_set))
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

accuracy : 85.52 %
```

## 오버피팅
현재 모델의 오버피팅 현상이 보이므로 
값을 일반화 시키고, 드랍아웃할 필요성이 있다.


[일반화&nbsp;참조] https://light-tree.tistory.com/125

- 1차시 val_accuracy = 
