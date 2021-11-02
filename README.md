# Undergraduate-research-student
Studies done in Lab

## 1. VGGNet

### VGG에서는 무얼 다르게 하였는가?

- 이전 AlexNet(2012)의 8 layers 모델보다 2배 이상 깊은 네트워크 학습에 성공한 VGGnet

- VGG 모델이 16-19 layer에 달하는 깊은 신경망을 학습할 수 있었던 것은 모든 합성곱 레이어에서 3x3 필터를 사용

### VGGNet의 구성

- 13 Convolution Layers + 3 Fully-connected Layers
- 3x3 convolution filters
- stride : 1 & padding: 1
- 2x2 max pooling (stride : 2)
- ReLU

### Hyperparameters

- Momentum(0.9)
- Weight Decay(L2 Norm)
- Dropout(0.5)
- Learning rate 0.01로 초기화 후 서서히 줄임

### 학습 이미지의 크기 조정

VGGNet에서는 training scale을 ‘S’로 표시하며, single-scale training과 multi-scaling training을 지원한다. 
Single scale에서는 AlexNet과 마찬가지로 S = 256, 또는 S = 384 두개의 scale 고정을 지원한다.

Multi-scale의 경우는 S를 Smin과 Smax 범위에서 무작위로 선택할 수 있게 하였으며, Smin은 256이고 Smax는 512이다. 
즉, 256과 512 범위에서 무작위로 scale을 정할 수 있기 때문에 다양한 크기에 대한 대응이 가능하여 정확도가 올라간다. 
Multi-scale 학습은 S = 384로 미리 학습 시킨 후 S를 무작위로 선택해가며 fine tuning을 한다. S를 무작위로 바꿔 가면서 학습을 시킨다고 하여, 이것을 scale jittering이라고 하였다.

#### Q) 학습 이미지 조정의 장점?

이처럼 학습 데이터를 다양한 크기로 변환하고 그 중 일부분을 샘플링해 사용함으로써 몇 가지 효과를 얻을 수 있는데 그중 첫번째는 한정적인 데이터의 수를 늘릴 수 있다.
두번째로 Data augmentation 하나의 오브젝트에 대한 다양한 측면을 학습 시 반영시킬 수 있다. 
변환된 이미지가 작을수록 개체의 전체적인 측면을 학습할 수 있고, 변환된 이미지가 클수록 개체의 특정한 부분을 학습에 반영할 수 있다. 
두 가지 모두 Overfitting을 방지하는 데 도움이 된다.

## 2. ResNet

### ResNet에서 알고자하는 것

- 더 많은 레이어를 쌓은 것 만큼 Network의 성능이 좋아질까?
- Vanishing/exploding gradients 으로 인해 Degradation Problem 발생
- Degradation Probelm ( Degradation : network 가 깊어질수록 accuracy가 떨어지는 현상 )  해소하기  --> deep residual learning 제안

### 핵심 아이디어 : Residual Block

- Residual block을 이용해 네트워크의 최적화(optimization) 난이도를 낮춘다
- 실제 내재된 mapping H(x) 를 곧바로 학습하기 어려우므로 F(x) = H(x)-x를 대신 학습한다.

#### Plain Layer VS Residual Block

Plain Layers : Plain layers에서는 weight layer가 각각 분리되어, 가중치가 개별적으로 학습되어야한다. 따라서 난이도가 증가하고 층이 깊어질수록 심하게 발생하게된다. 

Residual Block : Residual Block 에서, 기존 학습 정보 x는 그대로 가져오고 잔여정보인 F(x)만 추가적으로 더해준다. 전체를 학습하는거보다 쉬워 학습이 더 빠르고 높은 성능을 보여준다.

### 구성, Bottle Neck 구성

- 3x3 convolutional filter 사용
- 이 필터를 2개씩 묶어서 residual function 형태로 학습 진행
- 논문에서 Layer 사이 점선은 입력값과 출력값의 dim이 일치하지않아서 맞춰주는 short cut connection을 이용한 것
- 2개씩 묶는것을 3번 반복하고 크기를 바꿔서 4번 반복하고, 크기 바꿔서 6번 반복 그리고 마지막으로 크기 바꿔서 3번 반복한 것을 볼 수 있다.

Bottle Neck

- 복잡도를 증가시키지 않기 위해 사용된 것
- 초반에 1x1 filter를 64개 사용하고 중간에는 3x3 filter 64개, 마지막에 1x1 filter 64개 사용
- 작은 커널을 사용함으로써 파라미터 수를 감소시킨다
- identity short cut 이 더욱 효과적
- 깊이가 50 이상인 Resnet 더욱더 좋은 성능을 보임

### For CIFAR-10 Data

입력 이미지 크기가 작은 cifar10 에 맞게 파라미터 수를 줄여서 별도의 Resnet을 사용 --> 파라미터 수는 더 적지만 성능은 좋은 것을 볼 수 있다.

- Numbers of filters {16,32,64}
- subsampling is performed by convolutions with stride of 2
- Network ends with a global average pooling, 10-way fully-connected layer and softmax
- total 6n+2 staked weighted layers

ResNet18 src : https://github.com/shshin1210/Undergraduate-research-student/blob/master/ResNet/ResNet18_Final.ipynb

ResNet50 src : https://github.com/shshin1210/Undergraduate-research-student/blob/master/ResNet/ResNet50.ipynb

More Info about ResNet : https://github.com/shshin1210/Undergraduate-research-student/blob/master/ResNet/ResNet.pdf

VGGNet src : https://github.com/shshin1210/Undergraduate-research-student/blob/master/VGGNet/VGGnet_%EC%B5%9C%EC%A2%85%EB%B3%B8%20(1).ipynb

More Info about VGGNet : https://github.com/shshin1210/Undergraduate-research-student/blob/master/VGGNet/VGGnet.pdf
