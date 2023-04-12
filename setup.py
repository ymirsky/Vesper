from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules=[Extension("parPinger_wrapper",
                             ["parPinger_wrapper.pyx",
                              "parPinger.cpp","mls.cpp","aes.c"], language="c++",)],
      cmdclass = {'build_ext': build_ext})
