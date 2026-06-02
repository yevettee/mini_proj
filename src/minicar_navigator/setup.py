from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'minicar_navigator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/models', glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Minicar detection with YOLOv8 and Nav2 navigation for TurtleBot4',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detector = minicar_navigator.yolo_detector:main',
            'nav2_controller = minicar_navigator.nav2_controller:main',
            'minicar_nav_manager = minicar_navigator.minicar_nav_manager:main',
            'oakd_approach_node = minicar_navigator.oakd_approach_node:main',
        ],
    },
)
