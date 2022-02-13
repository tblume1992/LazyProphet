# -*- coding: utf-8 -*-

"""
@author: Tyler Blume
"""


from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name='LazyProphet',    
    version='0.2',
    scripts=['LazyProphet.py'],
    url="https://github.com/tblume1992/LazyProphet",
    author="Tyler Blume",
    author_email="tblume@mail.usf.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="Time series forecasting with LightGBM"
        )