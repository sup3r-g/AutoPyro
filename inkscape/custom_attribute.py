#!/usr/bin/env python3
# coding=utf-8
#
# Copyright (C) 2025 Gleb Shevchenko
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#

import inkex
from inkex.elements._utils import registerNS


class CustomAttribute(inkex.EffectExtension):

    def add_arguments(self, pars):
        pars.add_argument(
            "-ns", "--namespace", type=str, default="autopyro", help="Namespace"
        )
        pars.add_argument("-n", "--name", type=str, default="", help="Name")
        pars.add_argument("-v", "--values", type=str, default="", help="Values")

    def effect(self):
        registerNS("autopyro", "http://my_namespace.org/namespace")

        # Get all the options
        namespace = self.options.namespace
        name = self.options.name
        values = self.options.values

        for elem in self.svg.selection.values():
            elem.set(f"{namespace}:{name}", values)


if __name__ == "__main__":
    CustomAttribute().run()
