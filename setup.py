from setuptools import setup, find_packages

setup(
    name='fashion_graph',
    version='0.0.1',
    author='Jiyeon Baek',
    author_email='whitedelay@gmail.com',
    packages=find_packages(include=['fashion_graph', 'fashion_graph.*']),
    install_requires=[
        'tensorflow-gpu==1.15',
        'scikit-image==0.14.2',
        'torch==1.6',
        'torchvision==0.7',
    ]
)
