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

from bottle import static_file

from streamline import NonIterableRouteBase


class Static(NonIterableRouteBase):
    path = '/static/<path:path>'

    <vul/>def get_base_path(self):
        return self.config['runtime.static_dir']</vul>

    def get(self, path):
        <vul/>return static_file(path, self.get_base_path())</vul>
