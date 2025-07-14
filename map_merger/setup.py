from setuptools import find_packages, setup

package_name = 'map_merger'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    py_modules=[
        'map_merger.map_merger_node',  # Add your map_merger_node module
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='matrix.daphna@gmail.com',
    description='A ROS 2 package that merges maps from multiple robots.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
            'map_merger_node = map_merger.map_merger_node:main',  # Entry point for the node
        ],
    },
)
