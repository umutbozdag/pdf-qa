from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="pdf_qa",
    version="0.1.5",
    author="Umut",
    author_email="contact@umutbozdag.com",
    description="A PDF question answering tool using various LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/umutbozdag/pdf_qa",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "pdf_qa=pdf_qa.pdf_qa:main",
        ],
    },
)