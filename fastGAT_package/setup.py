import setuptools

with open("README.md",'r') as f:
    descrip=f.read()

setuptools.setup(
    name="fastGAT",
    version="0.1.0",
    author='LiangZi@NEU',
    author_email='2273067585@qq.com',
    description='A Fast Graph ATtention networks with ALSH. ',
    long_description=descrip,
    long_description_content_type='text/markdown',
    url='',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
