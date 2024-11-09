#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
raut2webstr-pagemodel-tree import-path script
"""

# Copyright 2016 Luboš Tříletý <ltrilety@redhat.com>
# Copyright 2016 Martin Bukatovič <mbukatov@redhat.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import sys
import re


# list of python modules expected in RAUT page/model python module directory
RAUT_MODULES = ("models", "pages")


def is_py_file(filename):
    return filename.endswith(".py") and filename != "__init__.py"


def change_import_path(directory, module, src_file, dry_run=False):
    """
    Change import paths to correspond with the new tree structure
    """
    new_lines = []
    module_present = False

    current_file = os.path.join(directory, src_file)

    if dry_run:
        with open(current_file, 'r') as file_to_read:
            containt = file_to_read.read()
            if module in containt:
                if re.search(r'^import .*models.*|import .*models.*', containt):
                    print("sed -i 's/([^ ]+)\.{0}([^ \\n]+)/\\1\\2.{0}/g'"
                              " {1}".format(module, src_file))
                if re.search(r'^from .*models.*|from .*models.*', containt):
                    print("sed -i 's/([^ ]+)\.{0} import (\w+)/\\1.\\2"
                              " import {0}/g' {1}".format(module, src_file))

    with open(current_file, 'r') as file_to_read:
        for line in file_to_read:
            if module in line:
                module_present = True
                if 'from' not in line:
                    new_lines.append(
                        re.sub(
                            r'(?P<pre>[^ ]+)\.{}(?P<post>[^ \n]+)'
                                .format(module),
                            r'\g<pre>\g<post>.{}'.format(module),
                            line))
                else:
                    new_lines.append(
                        re.sub(
                            r'(?P<pre>[^ ]+)\.{} import (?P<post>\w+)'
                                .format(module),
                            r'\g<pre>.\g<post> import {}'.format(module),
                            line))
            else:
                new_lines.append(line)

    if module_present:
        with open(current_file, 'w') as file_to_write:
            for line in new_lines:
                file_to_write.write(line)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='correct raut page/model import paths to webstr format')
    parser.add_argument(
        'directory',
        help='file path to directory with python files to be edited')
    parser.add_argument('-d', '--dry-run', action="store_true")
    args = parser.parse_args()

    # quick input validation
    if not os.path.isdir(args.directory):
        print("error: '{0}' is not a directory".format(args.directory))
        return 1

    # do the transformation, happens in place
    <fix/>for directory, _, file_list in os.walk(args.directory):</fix>
        python_files = [fl for fl in file_list if is_py_file(fl)]
        for python_file in python_files:
            for raut_module in RAUT_MODULES:
                change_import_path(
                    <fix/>directory, raut_module,</fix>
                    python_file, dry_run=args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
