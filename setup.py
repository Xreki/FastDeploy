import setuptools

long_description = "FastDeploy: Large Language Model Serving.\n\n"
long_description += "GitHub: https://github.com/PaddlePaddle/FastDeploy\n"
long_description += "Email: dltp@baidu.com"

setuptools.setup(
    name="fastdeploy",
    version="0.0.1",
    author="PaddlePaddle",
    author_email="dltp@baidu.com",
    description="FastDeploy: Large Language Model Serving.",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/PaddlePaddle/FastDeploy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3", 
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],  
    license='Apache 2.0',
) 
