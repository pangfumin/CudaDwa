
# CudaDwa

Toy code for local path planning algorithm DWA(Dynamic Window Approach) in CUDA implementation.

![](https://raw.githubusercontent.com/pangfumin/CudaDwa/master/image/dwa_demo.gif)

## dependency
1. CUDA (developed and tested on 10.2 )
2. OpenCV (developed and tested on 3.3)

## compilation
```sh
git clone https://github.com/pangfumin/CudaDwa.git
cd CudaDwa
mkdir build
cd build
cmake ..
make
```

## compilation
1. run cpp demo

```sh
cd build
./dwa_cpp_demo

```

2. run CUDA demo
```sh
cd build
./dwa_cuda_demo

```
   