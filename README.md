# GN1B

[GraspNet Baseline](https://github.com/graspnet/graspnet-baseline)的ROS包装。WIP

## 安装环境

使用conda-forge为默认频道。全部安装求解环境比较慢，可以分两次。API中使用了transforms3d==0.3.1需要numpy=1.19，但也可以手动将其中的`np.float`改为`np.float32`。tqdm后为api所需；open3d需使用pip安装，版本只要求>=0.8。pip可能需要降级google-auth。

    conda create -c conda-forge -n gn1b pytorch=1.10.2=cuda111py39h930882a_1 tensorboard=2.3 numpy scipy pillow tqdm opencv matplotlib trimesh scikit-image cvxopt h5py scikit-learn
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple open3d==0.14.1

编译并安装 pointnet2 运算符：

    cd pointnet2
    python setup.py install

编译并安装 knn operator:

    cd knn
    python setup.py install

安装API:setup.py中sklearn改为scikit-learn，然后：

    cd graspnetAPI
    pip install .

将dataset/graspnet_dataset.py第12行`from torch._six import container_abcs`修改为`import collections.abc as container_abcs`

运行demo：command_demo.sh开头添加虚拟环境库地址，如：

    LD_LIBRARY_PATH=/home/<user-name>/miniconda/envs/gn1b/lib

然后运行：

    sh command_demo.sh