# AI 신호처리

## 인공신경망의 구조
1. Input Layer
- 데이터를 넣어주는 과정

2. Hidden Layer
- 데이터의 특성을 학습하는 과정.

3. Output Layer
- 분류나 회귀문제의 정답을 알려줌.


## 대표적인 인공 신경망
1. CNN
- Convolution 연산을 통해 결과 도출.
- 현재출력이 현재 입력만 영향.
- 일차원의 신호를 이차원으로 변환해서 데이터를 인풋값으로 넣음.

2. LSTM
- RNN의 일종
- 현재출력이 이전의 입력까지 고려함.
- 가지고 있는 데이터의 시간에 따라 변화하는 특성들을 인풋값으로 넣음


## Deep Learning Workflow
1. Create And Access Datasets
- 좋은데이터가 좋은 결과를 만든다.
- 연구목적
모델 개발에 압도적으로 많은 시간 소요가들어감. 

- 실제 산업
데이터개발에 많은 시간 소요가 들어감.


```
Datastore() : 대용량의 데이터셋을 가져 와서 처리를 쉽게 함.
사용 예시
a = audioDatastore(pwd, 'IncludeSubfolders', true...
, "LabelSource", "foldernames") 
```

- 데이터를 강화함으로써 대용량의 데이터를 만듬으로써 더 견고한 모델 생성 가능.

2. PREPROCESS AND TRANSFORM DATA

3. DEVELOP PREDICTIVE MODELS
- Design
- Train
- Optimize

4. ACCELRATE AND DEPLOY


## 신호처리 - 푸리에 트랜스폼. 
- sparse domain => frequency domain

- Discrete fourier transform

- 사용처 
```
각각의 통신사들(SK, KT, LG) 자신들이 활용하는 대역폭으로 전달받은 영상 신호를 전송시킨후에 inverse system을 이용해원본으로 복원한다.

원본신호 => SK, KT, LG 대역폭으로 변조 신호 전달.(Forward transform 원본신호 => 변조 신호)
 => 각 통신사마다 다른 대역폭으로 신호 변조
 => 받은 신호를 사용자들의 단말기로 전달
 => 신호를 받은 단말기는 다시 신호를 변조(inverse transform 변조신호 => 복조신호)
```

1. 1D Fourier transform
```
N= 100;

xn = zeros(N,1);
xn(1) = 1;
xk = fft(xn);

figure(1)
subplot(131): stem(xn):title('spatial domain')
subplot(132): stem(real(xk):title('fourier domain: real')
subplot(133): stem(image(xk):title('fourier dom ain: imaginary')
```

## 컨볼루션 연산과 푸리에 트랜스폼의 관계
```
##################
## 1D Convolution
#################

Ny = 64;
Nx = 1;

My = 64;
Mx = 1;

f = zeros(Ny, Nx);
g = zeros(My, Mx);

f((1:Ny/4) + Ny*3/8, :) = 1;
g((1:My/4) + My*3/8, :) = 1;

pad_pre = [floor((My-1)/2), floor((Mx -1)/2)];
pad_post = [floor(My/2), floor(Mx/2)];

f_pad = padarray(f, pad_pre, 'pre');
f_pad = padarray(f_pad, pad_post, 'post');
```


## Separability와 Dimension Embedding, MobileNet에 어떻게 적용되는가?

1. Separability : the Concept of MobilenetV1
```
## Full-rank matrix
N = 4; # matrix 개수
mat = zeros(N,N);

for i =1;N
    mat(i, i) = N - (i - 1);
end

figure(1);
imagesc(mat); axis image off; title('Ground Truth');

## SVD
[U, S, V] = svd(mat);

figure(2);
subplot(141); imagesc(mat); axis image off; title('Ground Truth');
subplot(142); stem(diag(S)); title('Singular value');
subplot(143); imagesc(mat); axis image off; title('U matrix of SVD');
subplot(144); imagesc(mat); axis image off; title('V matrix of SVD');

## Rank1 matrix
BSS1 = U(:, 1) * V(:, 1)';
BSS2 = U(:, 2) * V(:, 2)';
BSS1 = U(:, 3) * V(:, 3)';
BSS2 = U(:, 4) * V(:, 4)';

COEF1 = S(1,1);
COEF2 = S(2,2);
COEF3 = S(3,3);
COEF4 = S(4,4);

## Convolution operator
## Gaussian kernel

# Input image matrix size
Ny = 255; 
Nx = 301;

# gausian kernel size
My = 10;
Mx = 13;

# 2D gausian scaling factor
a = 1;

# gausian kernel sigma parameter
sgmy = 3;
sgmx = 3;

# gausian kenel center position
y0 = 0;
x0 = 0;

## 2D Gaussian kernel
ly = linspace(-(My-1)/2, (My - 1)/2, My);
lx = linspace(-(Mx-1)/2, (Mx - 1)/2, Mx);

[mx, my] = meshgrid(lx, ly);

W = a * exp(-((mx - x0).^2/(2*sgmx^2) + (my-y0).^2/(2*sgmy^2)) )
W = W / norm(W);

## 1D Gaussian kernel

Wy = a * exp(-((ly - y0).^2 / (2*sgmy^2)));
Wx = a * exp(-((lx - x0).^2 / (x*sgmx^2)));

Wy = Wy/norm(Wy);
Wx = Wx/norm(Wx);

Wxy = Wy * Wx;

## 2D Convolution vs. separable 1D convolution

x = imresize(phantom(max(Ny, Nx)), [Ny, Nx]);

x_conv2d = conv2(x, W, 'same');

x_conv1dy = conv2(x, Wy, 'same');
x_conv1dxy = conv2(x_conv1dy, Wx, 'same');


## 2D Fourier transform vs. 1D Separable Fourier transform
x_ft2d = fftshift(fft2(ifftshift(x)));
x_ift2d = real(fftshift(ifft2(ifftshift(x_ft2d))));

x_ft1dy = fft(ifftshift(x), [], 1);
x_ft1dxy = fftshift(fft(x,ft1dy, [], 2));

x_ift1dy = ifft(ifftshift(x_ft1dxy), [], 1);
x_ift1dxy = real(fftshift(ifft(x_ift1dy, [], 2)));

```

## 콘볼루션과 푸리에 트랜스폼, 매트릭스 곱으로 표현
```
%% 1D Fourier transform
Ny = 128;
Nx = 1;

x = rand(Ny, Nx);

%% 1D operator
x_ft1d = fftshift(fft(ifftshift(x)));
x_ift1d = real(fftshift(ifft(ifftshift(x_ft1d))));

%% 1D matrix multiplication
ny = linspace(0, Ny - 1, Ny);
ky = linspace(0, Ny - 1, Ny);

kyny = ky(:) * ny(:)';

ft1d_mtx = exp(-1j * 2 * pi * kyny/Ny);
## ift1d_mtx = 1/N * exp(1j * 2 * pi * kyny/Ny);
ifft1d_mtx = 1/N * ft1d_mtx;

x_ft1d_mtx = fftshift(ft1d_mtx * ifftshift(x));
x_ift1d_mtx = fftshift(ift1d_mtx * ifftshift(x_ft1d_mtx));

%% 2D Fourier transform
Ny = 128;
Nx = 108;

x = imresize(phantom(max(Ny, Nx)), [Ny, Nx], 'nearest');


```

## 함수의 tranpose
```

% matrix version
% <A * x, y> = <x, A^T * y>

% operator & function version
% <A(x), y> = <x, A^T(y)>

%% Transpose of matrix
% A in N * M
% x in M * K
% y in N * K

N = 100;
M = 80;
K = 30;

A = randn(N, M);
x = randn(M, K);
y = randn(N, K);

AT = A';

% lhs = <A*x, y>
Ax = A * x;
lhs = Ax(:)' * y(:);

% rhs = <x, A^T*y>
ATy = AT * y
rhs = x(:)' * ATy(:);


## Transpose of Fourier transform function
Ny = 100;
Nx = 100;

A = @(x) fft2(x);
x = randn(Ny, Nx);
y = randn(Ny, Nx);

AT = @(y) Ny * Nx * ifft2(y);

% lhs = <A(x), y>
Ax = A(x);
lhs = Ax(:)' * y(:);

% rhs = <x, A^T(y)>
ATy = AT(y);
rhs = x(:)' * ATy(:);
```