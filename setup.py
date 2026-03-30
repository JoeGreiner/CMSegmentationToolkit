from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='CMSegmentationToolkit',
        version='0.1',
        packages=find_packages(),
        entry_points={
            'console_scripts': [
                'analyse-morphology=C_analyse_morphology_CLI:main',
            ],
        },
    )
