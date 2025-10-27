"""
Setup script for SparkTrainer package.
"""
from pathlib import Path

from setuptools import find_packages, setup

# Read README
readme_file = Path(__file__).parent / "src" / "spark_trainer" / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "Unified training framework for vision-language and diffusion models"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

setup(
    name="spark-trainer",
    version="0.1.0",
    author="SparkTrainer Team",
    description="Unified training framework for vision-language and diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SparkTrainer",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "whisper": ["openai-whisper>=20230314"],
        "deepspeed": ["deepspeed>=0.10.0"],
    },
    entry_points={
        "console_scripts": [
            "spark-trainer=spark_trainer.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="deep-learning, computer-vision, video, diffusion, vision-language, training",
)
