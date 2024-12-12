import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 去畸变
camera_matrix_left = np.array([2039.39167026889, 0, 664.739885649614,
                               0.000000, 2032.57985329622, 514.127412573186,
                               0.000000, 0.000000, 1.000000]).reshape(3, 3)
dist_coeffs_left = np.array([-0.455893854253456, 0.189452276483913, -0.00293849948015202, 0.00116652929453808, 0])

camera_matrix_right = np.array([2044.66320342200, 0.000000, 686.269453551213,
                                0.000000, 2037.53242503105, 538.485539183051,
                                0.000000, 0.000000, 1.000000]).reshape(3, 3)
dist_coeffs_right = np.array([-0.463165614146418, 0.230487647138996, -0.00296439076640851, 0.00127280417983063, 0])

R = np.array([[0.999581638393856,   0.00259636339576548, 0.0288063722633155],
              [-0.00237888922647861, 0.999968432496699, -0.00758121996404764],
              [-0.0288251465200739, 0.00750952110405581, 0.999556260558096]])

T = np.array([[-118.618976496459],
              [-0.394712743381340],
              [2.15770957135558]])

# 创建 StereoSGBM (Semi Global Block Matching) 对象
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16,  # 视差范围
    blockSize=11,       # 匹配块的大小
    P1=8 * 3 * 11**2,   # 参数，用于控制平滑程度
    P2=32 * 3 * 11**2,  # 参数，用于控制平滑程度
    disp12MaxDiff=1,    # 最大的视差差异
    uniquenessRatio=15, # 唯一性比例
    speckleWindowSize=200,  # 去除噪点窗口大小
    speckleRange=64,    # 去除噪点的视差范围
)
def write_ply(filename, points, colors):
    """
    将点云数据写入 PLY 文件
    :param filename: PLY 文件路径
    :param points: 点云的 3D 坐标，形状 (N, 3)
    :param colors: 点云的颜色，形状 (N, 3)
    """
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    assert points.shape[0] == colors.shape[0], "点的数量与颜色的数量不匹配！"
    
    with open(filename, 'w') as f:
        # 写入 PLY 文件头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入点云数据
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

def generate_depth_map(
        intrinsics_left, dist_coeffs_left,  # 左相机内参和畸变系数
        intrinsics_right, dist_coeffs_right,  # 右相机内参和畸变系数
        R, T,  # 外参：旋转矩阵和平移向量
        left_image_path, right_image_path,  # 左右图像路径
        output_depth_path  # 输出深度图路径
):

    # 读取左右图像
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    if left_image is None or right_image is None:
        raise ValueError("无法读取输入图像，请检查路径。")

    # 获取图像尺寸
    image_size = left_image.shape[::-1]  # (宽, 高)

    # 直方图均衡化
    left_image = cv2.equalizeHist(left_image)
    right_image = cv2.equalizeHist(right_image)

    # left_image = cv2.GaussianBlur(left_image, (5, 5), 0)
    # right_image = cv2.GaussianBlur(right_image, (5, 5), 0)

    # 立体校正
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        intrinsics_left, dist_coeffs_left,
        intrinsics_right, dist_coeffs_right,
        image_size, R, T, 
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    # 生成校正映射表
    map1_x, map1_y = cv2.initUndistortRectifyMap(
        intrinsics_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_32FC1
    )
    map2_x, map2_y = cv2.initUndistortRectifyMap(
        intrinsics_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_32FC1
    )
    
    # 应用校正映射
    rectified_left = cv2.remap(left_image, map1_x, map1_y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, map2_x, map2_y, cv2.INTER_LINEAR)

    # 创建 StereoSGBM 对象（立体匹配）
    block_size = 9 # 匹配块的大小
    min_disp = 16  # 最小视差
    num_disp = 16 * 10  # 16 的倍数，表示视差范围
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,  # P1 控制小范围平滑性
        P2=32 * 3 * block_size**2,  # P2 控制大范围平滑性
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=200,
        speckleRange=64,
        mode=cv2.STEREO_SGBM_MODE_HH
    )

    # Display the rectified left image
    plt.subplot(1, 2, 1)
    plt.imshow(rectified_left, cmap='gray')
    plt.title('Rectified Left Image')

    # Display the rectified right image
    plt.subplot(1, 2, 2)
    plt.imshow(rectified_right, cmap='gray')
    plt.title('Rectified Right Image')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    # 计算视差图
    disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0
    
    disparity = cv2.medianBlur(disparity, 5)
    disparity[disparity < 0] = np.median(disparity)


    # Display the disparity map
    plt.imshow(disparity, cmap='jet')
    plt.colorbar()
    plt.title('Disparity Map')
    plt.show()

    # 将视差图转换为深度图
    depth_map = cv2.reprojectImageTo3D(disparity, Q)
    depth = depth_map[:, :, 2]  # 提取深度通道 (z)

    # Display the disparity map
    plt.imshow(depth, cmap='jet')
    plt.colorbar()
    plt.title('Depth Map')
    plt.show()

    # 过滤无效深度值
    depth[depth <= 0] = 0  # 过滤负值
    depth[depth > 10000] = 0  # 过滤过远的深度值（阈值可根据实际情况调整）

    # 将深度值转换为 uint16 格式（以 mm 为单位）
    depth_uint16 = (depth * 1000).astype(np.uint16)  # 假设深度以米为单位，转换为毫米
    # depth_uint16 = cv2.medianBlur(depth_uint16, 5)

   # Display the disparity map
    plt.imshow(depth_uint16, cmap='jet')
    plt.gray()
    plt.title('Depth_uint16 Map')
    plt.show()

    # 保存原始深度图（uint16）
    cv2.imwrite(output_depth_path, depth_uint16)

    # 创建 debug 文件夹保存可视化数据
    debug_dir = os.path.join(os.path.dirname(output_depth_path), "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # # 生成可视化深度图（归一化到 0-255）
    # depth_visual = cv2.normalize(depth_uint16, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # depth_visual = np.uint8(depth_visual)

    # # 保存可视化深度图
    # debug_visual_path = os.path.join(debug_dir, "visual_depth.png")
    # cv2.imwrite(debug_visual_path, depth_visual)

    # # 生成并保存 PLY 文件
    # colors = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)  # 左图像颜色
    # points = depth_map  # 3D 点
    # ply_path = os.path.join(debug_dir, "point_cloud.ply")
    # write_ply(ply_path, points, colors)

pics_dir = "pics"
out_path = "depth_maps"
left_images = []
right_images = []
depth_maps = []

# Get a list of all image files in the pics directory
image_files = [filename for filename in os.listdir(pics_dir) if filename.endswith(".jpg")]

# Sort the image files based on the group
image_files.sort(key=lambda x: int(x.split("__")[0]))

# Separate the left and right images and create the corresponding depth map paths
i = 0
for filename in image_files:
    group, side = filename.split("__")
    if side.startswith("0"):
        left_images.append(os.path.join(pics_dir, filename))
    elif side.startswith("1"):
        right_images.append(os.path.join(pics_dir, filename))
    if(i % 2 == 1):
        depth_maps.append(os.path.join(out_path, group + ".png"))
    i += 1
depths = []
for left_img_path, right_img_path, depth_map_path in zip(left_images, right_images, depth_maps):
    print(f"Processing {left_img_path} and {right_img_path}")
    depths.append(generate_depth_map(camera_matrix_left, dist_coeffs_left,
                       camera_matrix_right, dist_coeffs_right,
                       R, T,
                       left_img_path, right_img_path,
                       depth_map_path))
    print(f"Saved depth map to {depth_map_path}")