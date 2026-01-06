1. **下载**
    从 [Magma 官网](https://icl.utk.edu/magma/) 下载源代码（以 Magma 2.9.0 为例）。**不要**从 GitHub 仓库克隆。
    ```bash
    wget https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.9.0.tar.gz
    tar -xvzf magma-2.9.0.tar.gz
    cd magma-2.9.0
    ```

2. **创建构建目录**
    创建一个 `build` 目录用于编译。
    ```bash
    mkdir build && cd build
    ```

3. **使用 CMake 配置**
    注意：请根据您的设备替换以下命令中 `-DGPU_TARGET` 后的名称。在本示例中，GPU 是 NVIDIA H20，因此目标是 `"Hopper"`。其他选项请参见 [GitHub README](https://github.com/icl-utk-edu/magma)。
    ```bash
    export CUDADIR=/usr/local/cuda
    cmake .. -DMAGMA_ENABLE_CUDA=ON -DGPU_TARGET="Hopper" -DBUILD_SHARED_LIBS=off -DUSE_FORTRAN=off
    ```

    - **故障排除：找不到 CUDA 编译器**
        如果您没有指定 CUDA 编译器，CMake 可能会失败并显示如下错误信息：
        ![](cuda_compiler_err.png)
        
        要解决此问题，首先定位 CUDA 编译器：
        ```bash
        find /usr -name nvcc 2>/dev/null
        ```
        如果输出是：
        ```bash
        /usr/local/cuda-12.8/bin/nvcc
        ```
        将编译器及其关联库添加到环境变量中：
        ```bash
        export PATH=/usr/local/cuda-12.8/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
        ```
        注意：请将上述```CUDA/bin```的路径替换为您系统中```CUDA/bin```的正确位置。
        设置路径后，清理 `build` 文件夹中的所有内容，然后再次运行 CMake 命令。

4. **编译**
    用所有能用的核编译：
    ```bash
    make -j"$(nproc)"
    ```
    - **故障排除：找不到 BLAS/LAPACK**
        如果 CMake 无法定位您的 BLAS/LAPACK 库，编译将终止，显示如下错误：
        ![](blas_error.png)

        要解决此问题，找到您的 BLAS/LAPACK 库的路径：
        ```bash
        ldconfig -p | grep -i liblapack
        ldconfig -p | grep -i libblas
        ```
        示例输出可能如下所示：
        ![](blas_position.png)
      
        然后，清理 `build` 文件夹中的所有内容并返回步骤 3。应用如下的CMake 命令，包含BLAS/LAPACK库的路径：
        ```bash
        cmake .. -DLAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/liblapack.so.3;/usr/lib/x86_64-linux-gnu/libblas.so.3" -DMAGMA_ENABLE_CUDA=ON -DGPU_TARGET="Hopper" -DBUILD_SHARED_LIBS=off -DUSE_FORTRAN=off
        ```
        注意：请将 `-DLAPACK_LIBRARIES=` 后的路径替换为您系统中 BLAS 和 LAPACK 库的正确位置。

5. **安装**
    
    ```bash
    sudo make install
    ```

