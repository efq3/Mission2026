import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # gazebo_ros 패키지의 경로를 가져옵니다.
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # 가제보 서버와 클라이언트(화면)를 실행하는 설정
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        )
    )

    return LaunchDescription([
        gazebo
    ])