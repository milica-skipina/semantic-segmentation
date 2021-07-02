from setuptools import setup
from Cython.Build import cythonize
import numpy

# module = Extension ('cityscapesscripts/evaluation', sources=['addToConfusionMatrix.pyx'])

setup(
    name='NN',
    version='1.0',
    packages=['src', 'cityscapesscripts', 'cityscapesscripts.viewer', 'cityscapesscripts.helpers',
              'cityscapesscripts.download', 'cityscapesscripts.annotation', 'cityscapesscripts.evaluation',
              'cityscapesscripts.preparation'],
    license='',
    ext_modules = cythonize("cityscapesscripts/evaluation/addToConfusionMatrix.pyx"),
    include_dirs=[numpy.get_include()],
    author='milica',
)
