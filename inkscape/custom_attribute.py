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
from inkex.elements._utils import registerNS, NSS, SSN


class CustomAttribute(inkex.EffectExtension):

    def add_arguments(self, pars):
        pars.add_argument(
            "-t", "--tab", type=str, default="autopyro", help="Selected Namespace"
        )
        pars.add_argument(
            "-ns", "--namespace", type=str, default="autopyro", help="Namespace"
        )
        pars.add_argument("-n", "--name", type=str, default="", help="Name")
        pars.add_argument("-v", "--values", type=str, default="", help="Values")
        pars.add_argument("-o", "--overwrite", type=bool, default=False, help="Overwrite objects ID")
    
    @staticmethod
    def newNS(namespace, url="http://my_namespace.org/namespace"):
        NSS[namespace] = url
        SSN[url] = namespace

    def effect(self):
        # Get all the options
        tab = self.options.tab
        namespace = str(self.options.namespace).lower().replace(" ", "_")
        name = str(self.options.name).lower().replace(" ", "_")
        values = self.options.values

        # registerNS(namespace, "http://my_namespace.org/namespace")
        self.newNS(namespace, "http://my_namespace.org/namespace")
        for element in self.svg.selection.values():
            if tab == "label":
                name, values = element.get("inkscape:label").split(": ")

            element.set(f"{namespace}:{name}", values)


if __name__ == "__main__":
    CustomAttribute().run()
