from setuptools import setup, find_packages

exec(open('stablegarment/version.py').read())

setup(
  name = 'stablegarment',
  packages = find_packages(),
  version = __version__,
  license='cc-by-nc-4.0',
  description = 'StableGarmet Official Implementation',
  author = 'Hailong Guo',
  author_email = 'guohailong@bupt.edu.cn',
  url = 'https://github.com/logn-2024/StableGarment',
  long_description_content_type = 'text/markdown',
  keywords = [
    'generative models',
    'virtual try-on',
  ],
  install_requires=[
    'einops',
    'numpy',
    'pillow',
    'torch',
    'tqdm',
    'opencv-python',
    'transformers',
    'diffusers',
  ],
  classifiers=[
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python :: 3.11',
  ],
)