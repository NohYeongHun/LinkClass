# KNN
## 기본적인 생각
1. 샘플을 분류하는 가장 쉽고 효과적인 방법이 무엇일까?
- 가장 가까운 샘플을 찾고, 그 샘플과 같은 클러스터로 분류

- 인접한 데이타에 대해 매우 민감하고, 또 실제 카테고리는 이렇게 깨끗하게 분리되지 않음.

- 만약 가장 인접한 샘플이 이상값이라면, 새로운 샘플은 잘못 분류 될 가능성이 높아짐.

- 이런 문제를 해결하는 가장 쉬운 방법은 더 많은 샘플을 관찰하여 다수를 이루는 클러스터로 분류