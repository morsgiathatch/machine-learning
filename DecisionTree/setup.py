from distutils.core import setup, Extension

setup(name='metricModule', version='1.0', ext_modules=[Extension('metricModule', ['metrics.cpp'])])
