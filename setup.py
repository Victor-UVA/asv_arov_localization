from setuptools import find_packages, setup

package_name = 'asv_arov_localization'

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
    maintainer='maddy',
    maintainer_email='abo7fg@virginia.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arov_ekf_global = asv_arov_localization.arov_ekf_global:main',
            'arov_ekf_external = asv_arov_localization.arov_ekf_external:main',
            'arov_ekf_onboard = asv_arov_localization.arov_ekf_onboard:main'
        ],
    },
)
