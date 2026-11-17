"""
modules/falcon — XbotGo Falcon youth-soccer footage.

Added in the Falcon Highlights fork (see NOTICE). The Falcon is an
auto-tracking camera: it records 4K HEVC, pans to follow play, and lets a
parent tap a button in its app when something worth seeing happens. Those taps
are exported as a ``.mark`` file. This package reads them, plans clips from
them, keeps the camera's own panning from being scored as action, and makes
phone-sized copies of the result.

Grouping only, as in the other ``modules/`` packages: import a member as
``modules.falcon.<name>``.

    marks    .mark / .vdata files and the marks-only cut plan
    hybrid   parent taps as a scoring signal, and the hybrid plan
    camera   global-motion (pan) compensation for the motion detector
    media    analysis proxy and phone-friendly export commands
    cli      ``python -m modules.falcon`` — plan, cut and export from a shell

A tap is a parent saying "something happened near here". Nothing in this
package calls one a goal.
"""
