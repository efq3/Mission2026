import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'vtail_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={'vtail_control': ['best.pt']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'models', 'target_board_car'), glob('models/target_board_car/*.*')),
        (os.path.join('share', package_name, 'models', 'target_board_car', 'materials', 'textures'), glob('models/target_board_car/materials/textures/*')),
        (os.path.join('share', package_name, 'models', 'target_board_tank'), glob('models/target_board_tank/*.*')),
        (os.path.join('share', package_name, 'models', 'target_board_tank', 'materials', 'textures'), glob('models/target_board_tank/materials/textures/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nahj183',
    maintainer_email='nahj183@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'guidance_node = vtail_control.guidance_node:main',
            'visualize_node = vtail_control.rviz_f.visualize_node:main',
            'target_marker_node = vtail_control.rviz_f.target_marker_node:main',
            'mode_node = vtail_control.mode_node:main',
            'yolo_node = vtail_control.yolo_node:main',
            'qgc_streamer_node = vtail_control.qgc_streamer_node:main',
            'gz_gst_bridge_node = vtail_control.gz_gst_bridge_node:main',
            'python_gz_ros_bridge = vtail_control.python_gz_ros_bridge:main'
        ],
    },
)
