# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.177.0/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version: debian-10, debian-9, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="ubuntu-20.04"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

# [Optional] Uncomment this section to install additional packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf \
    libgtk2.0-dev \
    pkg-config \
    && apt-get clean && rm -rf /var/lib/opt/lists/*

# Clone dependencies
WORKDIR /external
RUN git clone --depth 1 https://github.com/opencv/opencv.git \
    && git clone --depth 1 https://github.com/opencv/opencv_contrib.git \
    && git clone --depth 1 https://github.com/Tencent/ncnn.git \
    && wget https://jaist.dl.sourceforge.net/project/zbar/zbar/0.10/zbar-0.10.tar.bz2 \
    && tar -jxvf zbar-0.10.tar.bz2 \
    && rm zbar-0.10.tar.bz2

# Build OpenCV with extra modules
WORKDIR /external/opencv/
RUN cmake -B build/arm -DCMAKE_TOOLCHAIN_FILE=./platforms/linux/arm-gnueabi.toolchain.cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules/ -DBUILD_LIST=tracking,imgcodecs,videoio,highgui,features2d,ml,xfeatures2d -DCMAKE_BUILD_TYPE=Release . \
    && cmake --build build/arm
RUN cmake --install build/arm --prefix install/arm

# OpenCV x64
RUN cmake -B build/x64 -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules/ -DBUILD_LIST=tracking,imgcodecs,videoio,highgui,features2d,ml,xfeatures2d -DCMAKE_BUILD_TYPE=Debug . \
    && cmake --build build/x64
RUN cmake --install build/x64 --prefix install/x64

# ncnn
WORKDIR /external/ncnn
RUN cmake -B build/arm -DCMAKE_TOOLCHAIN_FILE=./toolchains/arm-linux-gnueabihf.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON . \
    && cmake --build build/arm \
    && cmake --install build/arm --prefix install/arm

# ncnn x64
RUN cmake -B build/x64 -DCMAKE_BUILD_TYPE=Debug -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON . \
   && cmake --build build/x64 \
   && cmake --install build/x64 --prefix install/x64

# zbar
WORKDIR /external/zbar-0.10
RUN env NM=nm CFLAGS="" ./configure --prefix=$(pwd)/build/arm --host=arm-none-linux-gnueabihf --build=x86_64-linux --enable-shared --without-gtk --without-python --without-qt --without-imagemagick --disable-video --without-xshm CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ \
    && make \
    && make install

# zbar x64
RUN make clean \
    && env NM=nm CFLAGS="" ./configure --prefix=$(pwd)/build/x64 --enable-shared --without-gtk --without-python --without-qt --without-imagemagick --disable-video --without-xshm \
    && make \
    && make install
