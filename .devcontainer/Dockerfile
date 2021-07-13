# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.177.0/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version: debian-10, debian-9, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="ubuntu-20.04"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

# [Optional] Uncomment this section to install additional packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf \
    && apt-get clean && rm -rf /var/lib/opt/lists/*

WORKDIR /external
RUN git clone --depth 1 https://github.com/opencv/opencv.git \
    && git clone --depth 1 https://github.com/opencv/opencv_contrib.git \
    && git clone --depth 1 https://github.com/Tencent/ncnn.git \
    && wget https://jaist.dl.sourceforge.net/project/zbar/zbar/0.10/zbar-0.10.tar.bz2 \
    && tar -jxvf zbar-0.10.tar.bz2 \
    && rm zbar-0.10.tar.bz2
# RUN wget "https://developer.arm.com/-/media/Files/downloads/gnu-a/10.2-2020.11/binrel/gcc-arm-10.2-2020.11-x86_64-arm-none-linux-gnueabihf.tar.xz?revision=d0b90559-3960-4e4b-9297-7ddbc3e52783&la=en&hash=985078B758BC782BC338DB947347107FBCF8EF6B"
# RUN tar -xvf ./* -C /usr/local \
#     && ls -l && rm ./*

# Build OpenCV with extra modules
WORKDIR /external/opencv/platforms/linux/
RUN mkdir build \
    && cd build \
    && cmake -DCMAKE_TOOLCHAIN_FILE=../arm-gnueabi.toolchain.cmake -DOPENCV_EXTRA_MODULES_PATH=../../../../opencv_contrib/modules/  -DBUILD_LIST=tracking,imgcodecs,videoio,highgui,features2d,ml,xfeatures2d -DCMAKE_BUILD_TYPE=Release ../../.. \
    && make \
    && make install

WORKDIR /external/ncnn
RUN mkdir build \
    && cd build \
    && cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON .. \
    && make \
    && make install

WORKDIR /external/zbar-0.10
RUN env NM=nm CFLAGS="" ./configure --prefix=$(pwd)/build --host=arm-none-linux-gnueabihf --build=x86_64-linux --enable-shared --without-gtk --without-python --without-qt --without-imagemagick --disable-video CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ \
    && make \
    && make install
