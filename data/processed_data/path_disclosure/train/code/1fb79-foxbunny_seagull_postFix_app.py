#
# Seagull photo gallery app
# Copyright (C) 2016  Hajime Yamasaki Vukelic
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#

from os.path import join, exists

from bottle import static_file

from streamline import NonIterableRouteBase


class Static(NonIterableRouteBase):
    path = '/static/<path:path>'

    <fix/>def get_base_paths(self):
        return (self.config['runtime.static_dir'],
                self.config['runtime.assets_dir'])

    @staticmethod
    def get_first_base(bases, path):
        """
        Return the first base path within which the path is found

        The last base path is always returned such that 404 errors are handled
        by bottle.
        """
        for b in bases:
            if not exists(join(b, path)):
                continue
            return b
        return b

    @staticmethod
    def sanitize_path(path):
        if path.startswith('/'):
            path = path[1:]
        return path.replace('..', '.')</fix>

    def get(self, path):
        <fix/>path = self.sanitize_path(path)
        base_paths = self.get_base_paths()
        if hasattr(base_paths, 'split'):
            # String, so go simple
            base_path = base_paths
        else:
            base_path = self.get_first_base(base_paths, path)
        return static_file(path, base_path)</fix>

