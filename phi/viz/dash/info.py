# coding=utf-8
import inspect
import os
import datetime
import subprocess
import warnings
from os.path import dirname, exists, join, isfile

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import phi
from .dash_app import DashApp


def build_app_details(dashapp):
    assert isinstance(dashapp, DashApp)
    app = dashapp.app
    try:
        app_file = inspect.getfile(app.__class__)
    except TypeError:
        app_file = 'Unknown'
    details = dcc.Markdown("""
## Details

Summary: %s

Stride: %d

Traits: %s

Script path: %s

Data path: %s
    """ % (app.summary, app.sequence_stride, app.traits, app_file, app.scene))
    return details


def build_description(dashapp):
    assert isinstance(dashapp, DashApp)
    app = dashapp.app
    md_src = _description_markdown_src(app.name, app.subtitle)
    return dcc.Markdown(children=md_src, id='info_markdown')


def _description_markdown_src(title, subtitle=''):
    if subtitle is not None and len(subtitle) > 0:
        return """
# %s

---

> **_About this application:_**

%s

---""" % (title, subtitle)
    else:
        return '# %s' % title


def build_phiflow_info(dashapp):
    root_dir = dirname(dirname(inspect.getfile(phi)))
    setup_file = join(root_dir, 'setup.py')
    version = phi.__version__
    return dcc.Markdown(u"""
This application is based on the open-source simulation framework [Φ-Flow](https://github.com/tum-pbs/PhiFlow), version %s.
""" % version)


def build_app_time(dashapp):
    start_time = datetime.datetime.fromtimestamp(dashapp.app.start_time)

    def build_text():
        now = datetime.datetime.now()
        elapsed = now - start_time
        minutes, seconds = divmod(elapsed.seconds, 60)
        return 'Application started: %s (Running for %d minutes and %d seconds)' % (start_time.ctime(), minutes, seconds)

    layout = html.Div([
        dcc.Markdown(children=build_text(), id='clock-output'),
        dcc.Interval(id='clock', interval=1000)
    ])

    @dashapp.dash.callback(Output('clock-output', 'children'), [Input('clock', 'n_intervals')])
    def update_clock(_):
        return build_text()

    return layout
