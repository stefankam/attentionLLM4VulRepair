#!/usr/bin/env python
# -*- coding:utf-8 -*-
from setuptools import setup, find_packages
import sys


def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup_args = dict(
    name='jupyterhub-wordpressauthenticator',
    <fix/>version='0.1.1',</fix>
    description='WordPress Authenticator for JupyterHub',
    url='https://github.com/harapekoaomushi/jupyterhub-wordpressauthenticator',
    author='Harapeko Aomushi',
    author_email='harapeko1aomushi@gmail.com',
    long_description="WordPress Authenticator for JupyterHub. Please read README.md",
    license='MIT License',
    packages=find_packages(),
    install_requires=_requires_from_file('requirements.txt'),
    keywords = ['Interactive', 'Interpreter', 'Shell', 'Web'],
    classifiers = [
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ]
)

if 'setuptools' in sys.modules:
    setup_args['install_requires'] = install_requires = []
    with open('requirements.txt') as f:
        for line in f.readlines():
            req = line.strip()
            if not req or req.startswith(('-e', '#')):
                continue
            install_requires.append(req)


def main():
    setup(**setup_args)

if __name__ == '__main__':
    main()
