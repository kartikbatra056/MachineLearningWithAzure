from setuptools import setup,find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Python package for my project'
LONG_DESCRIPTION = 'Following package is used to create new columns for fetal health prediction project.'

setup(
    name='Myestimator',
    version=VERSION,
    author='Kartik Batra',
    author_email='kartik.batra18@vit.edu',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    keywords =['python','package'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ]
)
