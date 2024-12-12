# 将assert/images里的图片按照ip存在left和right两个文件夹中
# 结尾为~10.64.57.8在left  结尾为~10.64.57.9在right

import os
import shutil

def divide():
    left = 'assert/images/left'
    right = 'assert/images/right'
    if not os.path.exists(left):
        os.makedirs(left)
    if not os.path.exists(right):
        os.makedirs(right)
    for file in os.listdir('assert/images'):
        if file.endswith('~10.64.57.8'):
            shutil.move('assert/images/' + file, left)
        elif file.endswith('~10.64.57.9'):
            shutil.move('assert/images/' + file, right)
        else:
            print('error:', file)

if __name__ == '__main__':
    divide()
    print('done')