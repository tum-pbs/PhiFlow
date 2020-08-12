# coding=utf-8
import numpy

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from phi.app import App
from phi import struct


class DashApp:

    def __init__(self, app, config, header_layout):
        assert isinstance(app, App)
        self.app = app
        self.config = config
        self.dash = dash.Dash(u'PhiFlow')
        self.dash.config.suppress_callback_exceptions = True
        self.hrefs = set()
        self.page_urls = {}
        self.field_minmax = {}
        self.minmax_decay = 0.975

        # The index page encapsulates the specific pages.
        self.dash.layout = html.Div([
            dcc.Location(id='url', refresh=False),
            header_layout,
            html.Div(id='page-content')  # Content is set using the URL
        ], style={'fontFamily': 'arial'})

        @self.dash.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
        def display_page(pathname):
            if pathname in self.page_urls:
                layout = self.page_urls[pathname]
                if callable(layout):
                    return layout(self)
                else:
                    return layout
            else:
                return html.Div([
                    html.Div('404 - No such page: %s' % pathname),
                    dcc.Link('Back to main page', href='/'),
                ])

    @property
    def server(self):
        return self.dash.server

    def search_callback(self, output, hrefs):
        assert isinstance(output, Output)
        self.hrefs.update(hrefs)

        def decorator(func):
            @self.dash.callback(output, [Input('url', 'search')])
            def href_callback(search):
                if search in hrefs:
                    func(search)
                else:
                    raise PreventUpdate()
        return decorator

    def consumes(self, href):
        return href in self.hrefs

    def add_page(self, path, page_layout):
        self.page_urls[path] = page_layout

    def get_field(self, name):
        data = self.app.get_field(name)
        if data is not None:
            tensors = struct.flatten(data)
            if tensors:
                minimum = min([numpy.min(tensor) for tensor in tensors])
                maximum = max([numpy.max(tensor) for tensor in tensors])
                if name in self.field_minmax:
                    midpoint = (self.field_minmax[name][1] + self.field_minmax[name][0]) / 2
                    minimum = min(minimum, self.field_minmax[name][0] * self.minmax_decay + midpoint * (1 - self.minmax_decay))
                    maximum = max(maximum, self.field_minmax[name][1] * self.minmax_decay + midpoint * (1 - self.minmax_decay))
                self.field_minmax[name] = (minimum, maximum)
        return data

    def reset_field_summary(self):
        self.field_minmax = {}

    def get_minmax(self, field):
        if field in self.field_minmax:
            return self.field_minmax[field]
        else:
            return 0, 0
