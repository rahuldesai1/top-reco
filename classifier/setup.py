#!/usr/bin/env python

from distutils.core import setup
from pip.req import parse_requirements

reqs = "requirements.txt"
parsed_reqs = parse_requirements(reqs)
install_requires = [str(i.req) for i in parsed_reqs]

setup(name='Top Classifier',
	version='1.0',
	long_description=open("README.md").read(),
	author='Rahul Desai',
	author_email='rahuldesai@berkeley.edu',
	packages=install_requires,
     )
