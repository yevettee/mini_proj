from setuptools import find_packages, setup

package_name = 'rokey_pjt'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='woody',
    maintainer_email='woody.myung@gmail.com',
    description='ROKEY project - Depth measurement with YOLOv8 bounding boxes on OAK-D (TurtleBot4)',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'depth_checker = rokey_pjt.depth_checker:main',
            'depth_checker_mouse = rokey_pjt.depth_checker_mouse:main',
        ],
    },
)
