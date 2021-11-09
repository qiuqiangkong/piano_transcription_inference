import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="piano-transcription-inference", # Replace with your own username
    version="0.0.5",
    author="Qiuqiang Kong",
    author_email="qiuqiangkong@gmail.com",
    description="Piano transcription inference toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['matplotlib', 'mido', 'librosa', 'torchlibrosa'],
    python_requires='>=3.6',
)
