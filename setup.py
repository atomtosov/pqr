import setuptools


with open('README.md') as f:
    readme = f.read()


setuptools.setup(
    name='pqr',
    version='0.2.1',
    description='Library for testing factor strategies',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas<=1.3.0',
        'matplotlib<=3.4.2',
        'numpy<=1.21.0',
        'statsmodels<=0.12.2',
    ],
    author='eura17, atomtosov',
    license='MIT',
    license_file='LICENSE',
    platforms='any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Office/Business :: Financial',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    test_suite='pytest',
    python_requires='>=3.7',
    project_urls={
            'Bug Tracker': 'https://github.com/atomtosov/pqr/issues',
            'Documentation': 'https://pqr.readthedocs.io/en/latest/index.html',
            'Source Code': 'https://github.com/atomtosov/pqr/tree/dev',
        },
)
