# visual_robot
Robotic arm vision code for Challenge Cup 

## submodule
```shell
git submodule init
git submodule update  #git pull 之后需要使用该命令完成子模块同步
```
**注意本项目是参考了其他人的开源项目,本项目要根据实际情况进行修改不能直接使用**

## 参考目录
```shell
1.calibrating #用于标定，如果拥有双目相机和深度相机建议使用该库标定
2.FoundationPose #根据目标模型和RGBD图对目标的位姿进行求解
3.hik_camera  #海康网口相机SDK 后续需要笔者将推出C++版本(由于项目原因目前并未开源)
4.tools #一些工具脚本
```
## 建议
在没有深度相机的时候，标定使用matlab标定,在窗口命令行输入：
```shell
stereoCameraCalibrator
```