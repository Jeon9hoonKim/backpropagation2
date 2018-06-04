# 2018 1학기 수학특론 2조 backpropagation 발표
# *BACKPROPAGATION*
## **Index**
1. Intro 
2. Backpropagation Algorithm
3. Programing Code
4. Reference
<br><br/>
## **1. Intro**

neural network를 학습시키는 목적은 input이 주어졌을 때, 원하는 output을 내도록 하는 것이다.

예를 들어, classification 같은 경우, 만약 동물의 사진이 input으로 주어졌을 때, 적절한 과정을 거쳐 그 동물이 무엇인지에 대한 output이 나와야 할 것이다.

이때, input에 대해 원하는 output을 내도록 neural network를 training할 때 사용하는 방법 중 하나가 backpropagation이다.
<br><br/>
### **What To Know**

### **1) Artificial Neural Networks**

<img src="https://github.com/DGitH/backpropagation2/raw/master/pictures/%EC%BA%A1%EC%B2%981.PNG" width="382" height="244" style="border: 0.0px;cursor: default;margin: 0.0px;padding: 0.0px;outline: 0.0px;font-weight: inherit;font-style: inherit;font-family: inherit;font-size: 13.0px;vertical-align: middle;max-width: 100.0%;height: auto;">

위와 같이 input에 대해 연산을 수행하여 output을 내놓는 것을 artificial neuron이라 한다.

각 input에 대한 weight가 있고, 이에 따라 output이 달라진다.

함수 g는 activation function으로 sigmoid, ReLu, tanh 등의 함수가 사용된다.
<br><br/>

<img src="https://github.com/DGitH/backpropagation2/blob/master/pictures/%EC%BA%A1%EC%B2%982.PNG" width="430" height="178" style="border: 0.0px;cursor: default;margin: 0.0px;padding: 0.0px;outline: 0.0px;font-weight: inherit;font-style: inherit;font-family: inherit;font-size: 13.0px;vertical-align: middle;max-width: 100.0%;height: auto;">

artificial neuron들이 모여 위와 같은 구조를 이룬 것을 artificial neural network라 한다.

마찬가지로 이웃한 Layer 사이에 각 Node의 weight가 있고 이에 따라 output이 달라진다.

즉, 원하는 output을 내기 위해서는 weight를 적절히 조정해야한다.

weight를 적절히 조정하는 과정을 training이라 할 수 있다.
<br><br/>

### **2) Why Network?**

neuron 하나만으로는 복잡한 문제들을 해결할 수 없다.

그 예로 XOR problem이 있다.

![](https://github.com/DGitH/backpropagation2/blob/master/pictures/캡처3.PNG)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](https://github.com/DGitH/backpropagation2/blob/master/pictures/캡처4.PNG)

z=w1x1+w2x2+b라 하면, 다음과 같은 결과를 예상할 수 있다.

![](https://github.com/DGitH/backpropagation2/blob/master/pictures/캡처5.PNG)

위를 만족하는 w1, w2, b는 존재하지 않음을 알 수 있다.
<br><br/>

### **3) Gradient Descent**

![](https://github.com/DGitH/backpropagation2/blob/master/pictures/캡처6.PNG)

주어진 parameter에 대해 loss function J를 최소화하는 방향으로 parameter(weight)를 update하는 방법이다.

Artificial Neural Networks구조에서 J의 미분값을 직접 구하기는 힘들다.

각 weight에 대한 gradient를 쉽게 구하는 방법이 필요하다.
<br><br/><br><br/>

## **2. Backpropagation Algorithm**

### 1) 어떻게 weight를 학습할 것인가?

Backpropagation은 Artificial Neural Network을 학습할 때 각 neuron들 간에 weight 학습에 쓰이는 알고리즘이다.

Input data가 neural network를 지나면서 각 노드의 output값이 나오게 된다.
&nbsp;
학습된 결과가 맞는지는 마지막에 나온 output이 실제값과 맞는지 비교하는 과정을 통해서 알 수 있으며 이 과정에서 loss function을 구하게 된다.

중간과정(layer)에서 weight가 제대로 학습되는지는 알 수 없기 때문에 마지막 나온 loss function을 바탕으로 layer를 차례로 넘어가면서 weight를 학습시켜야 한다.

이렇게 마지막에 나온 output을 바탕으로 뒤에서부터 이전의 모든 weight를 학습하기 위해서 순차적으로 넘어가는 과정을 backpropagation이라 한다.
<br><br/>

### 2-1) 예제

![](https://github.com/DGitH/backpropagation2/blob/master/pictures/캡처7.PNG)

다음과 같이 이루어진 신경망이 있다고 했을 때 training data x가 인공신경망을 통하여 나온 예측값을 output t1, t2, t3라고 하자.

이때 실제값과 예측값 사이의 오차인 loss function을 E로 놓자.

loss function의 한 예로 다음과 같은 loss function이 있다.

![](http://latex.codecogs.com/gif.latex?E%20%3D%20-%5Csum_%7Bi%3D1%7D%5E%7Bnout%7D%28t_i%20log%28x_i%29&plus;%281-t_i%29log%281-x_i%29%29)
<br><br/>
![](http://latex.codecogs.com/gif.latex?x_i%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-s_i%7D%7D)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?s_i%20%3D%20%5Csum_%7Bj%3D1%7Dx_jw_%7Bji%7D)
<br/><br/>
loss function E를 줄이는 방향으로 w를 학습시켜보자.

E에 대해서 output layer와 hidden layer 간의 weight w(ji)가 영향을 미치는 정도는 다음과 같이 grandient descent와 chain rule을 이용하여 구할 수 있다.

이를 이용해 두 layer간의 weight w를 학습시킬 수 있다.

![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bji%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20x_i%7D%20%5Cfrac%7B%5Cpartial%20x_i%7D%7B%5Cpartial%20s_i%7D%20%5Cfrac%7B%5Cpartial%20s_i%7D%7B%5Cpartial%20w_%7Bji%7D%7D)
<br><br/>
![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20x_i%7D%20%3D%20%5Cfrac%7B-t_i%7D%7Bx_i%7D&plus;%5Cfrac%7B1-t_i%7D%7B1-x_i%7D%20%3D%20%5Cfrac%7Bx_i-t_i%7D%7Bx_i%281-x_i%29%7D)
<br/>
![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20x_i%7D%7B%5Cpartial%20s_i%7D%20%3D%20x_i%281-x_i%29)
<br/>
![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20s_i%7D%7B%5Cpartial%20w_%7Bji%7D%7D%20%3D%20x_j)
<br><br/>
![](http://latex.codecogs.com/gif.latex?%5Ctherefore%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bji%7D%7D%20%3D%20%28x_i-t_i%29x_j)
<br/><br/>
마찬가지로 Input과 hidden layer 사이의 weight도 개선시켜주자.

다만 주의할 점은 아래 그림과 같이 output layer 에서의 여러 노드에서 온 weight를 모두 고려해 weight를 바꾸어줘야 한다.

![](https://github.com/DGitH/backpropagation2/blob/master/pictures/캡처8.PNG)

![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bkj%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20s_j%7D%20%5Cfrac%7B%5Cpartial%20s_j%7D%7B%5Cpartial%20w_%7Bkj%7D%7D)
<br><br/>
![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20s_j%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bnout%7D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20s_i%7D%5Cfrac%7B%5Cpartial%20s_i%7D%7B%5Cpartial%20x_j%7D%5Cfrac%7B%5Cpartial%20x_j%7D%7B%5Cpartial%20s_j%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bnout%7D%28x_i-t_i%29%28w_%7Bji%7D%29%28x_j%281-x_j%29%29)
<br/>
![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20s_j%7D%7B%5Cpartial%20w_%7Bkj%7D%7D%20%3D%20x_k%20%7E%28%5Cbecause%20s_j%3D%5Csum_%7Bk%3D1%7Dx_kw_%7Bkj%7D%29)
<br><br/>
![](http://latex.codecogs.com/gif.latex?%5Ctherefore%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bkj%7D%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bnout%7D%28x_i-t_i%29%28w_%7Bji%7D%29%28x_j%281-x_j%29%29%28x_k%29)

이를 반복하면 오차 E를 줄일 수 있어 결과적으로 우리가 원하는 output을 만들어내는 신경망을 완성할 수 있다.
<br><br/>

### 2-2) 예제 2

![](https://github.com/DGitH/backpropagation2/blob/master/pictures/캡처9.PNG)

![](https://github.com/DGitH/backpropagation2/blob/master/pictures/캡처10.PNG)
<br><br/><br><br/>

## **3. Programing Code**

1) XOR problem

https://github.com/JeonhoonKim/backpropagation2
<br><br/><br><br/>

## **4. Reference**

1) Backpropagation 예제 1, 2

http://jaejunyoo.blogspot.com/2017/01/backpropagation.html

http://web.stanford.edu/class/cs224n/syllabus.html

2) Neural network

http://sanghyukchun.github.io/74/

3) XOR problem

http://pythonkim.tistory.com/33
