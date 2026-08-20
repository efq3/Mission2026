import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'vtail_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={'vtail_control': ['best_full_integer_quant_edgetpu.tflite']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
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
            'mode_node = vtail_control.mode_node:main',
            'yolo_node = vtail_control.yolo_node:main',
            'qgc_streamer_node = vtail_control.qgc_streamer_node:main',
            'Coral_PnP_xyz = vtail_control.Coral_PnP_xyz:main', 
        ],
    },
)