# Interface resources

The active interface icon set is Lucide, bundled in `lucide/` with its upstream
license and pinned source commit. Icons are rendered with the active Qt palette,
including disabled states. The application logo is a separate PNG and is unchanged.

`theme/controls.py` assigns the same semantic icons to translated commands,
property/result tabs, nested menus, tool buttons and standard dialog buttons.
The main window uses this shared registry too. Standard buttons are identified
by their Qt role, so platform translations and keyboard mnemonics do not affect
their icons. Top-level menu bar titles retain their text (Qt can hide the text
when an icon is assigned); commands and nested menus have icons.

`tests/test_interface_icons.py` builds the analysis modules, import/database
dialogs and pickers in Russian and English with both Breeze palettes. It checks
actual controls and context menus, SVG availability, and normal, disabled and
selected icon rendering, including theme changes on already-created icons.

AutoPyro adjusts the selection colors in its Qt palette to improve text contrast.
The bundled KDE Breeze resources are retained unmodified:

- Color schemes: https://github.com/KDE/breeze/tree/master/colors
  (`BreezeLight.colors`, `BreezeDark.colors`), LGPL-2.0-or-later.
  Copyright notices are retained in the source files. See `LGPL-2.0.txt`.
- Icons: https://github.com/KDE/breeze-icons/tree/master/icons/actions/22,
  LGPL-3.0-or-later. See `COPYING-ICONS`, `COPYING.LIB`, and `GPL-3.0.txt`.

Original SVG sources are shipped alongside the application. For dark palettes,
the in-memory Lucide rendering replaces `currentColor` to preserve contrast;
the bundled SVG source files remain unchanged. The palette parser and Qt
stylesheet are AutoPyro code, not the KDE native widget style. The native
`Breeze` style is used when installed; other systems use Qt `Fusion` with
these official Breeze colors and icons. No resources are fetched at runtime.
