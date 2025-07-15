from setuptools import setup, find_packages

setup(
    name="rotate_to_beat",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "librosa",
        "moviepy",
        "scipy",
    ],
    entry_points={
        'console_scripts': [
            'rotate-to-beat=rotate_to_beat_cli:main',
        ],
    },
    python_requires='>=3.6',
    description="Tool to create rotating videos synchronized to audio beats",
    author="Dance to Beat",
)
