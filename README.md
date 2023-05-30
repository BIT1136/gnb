# GNB

代码来自[GraspNet Baseline](https://github.com/graspnet/graspnet-baseline)。

## 安装环境

knn使用的THC.h在pytorch 1.11中删除。open3d需使用pip安装，版本只要求>=0.8。

    conda create -c conda-forge -n gnb pytorch=1.10.2=cuda111py39h930882a_1 tensorboard=2.3 numpy scipy rospkg
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple open3d==0.14.1 google-auth==1.35.0 grasp-nms

编译并安装 pointnet2 算子：

    cd pointnet2
    python setup.py install

编译并安装 knn 算子:

    cd knn
    python setup.py install

## 下载模型

从[Google Drive](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing)下载checkpoint-rs.tar，放入model文件夹。

## 运行

    roslaunch gnb node.launch

## 使用

向点云话题发布相机坐标系中的点云，预测出的n个同坐标系下的抓取会被发布到抓取话题，质量降序排列；同时其可视化Marker会被发布到marker话题。在node.launch中修改这三两个话题名。

使用场景点云预测时方向似乎不对，发布单独的物体点云效果好一些。

## 待调整参数

关于夹爪与碰撞检测的参数 https://github.com/graspnet/graspnet-baseline/issues/23#issuecomment-899084543
后处理 https://github.com/graspnet/graspnet-baseline/issues/18#issuecomment-873758326
点云中心化 https://github.com/graspnet/graspnet-baseline/issues/15#issuecomment-803748788