# coding=utf-8

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from phi.app import App


class DashApp:

    def __init__(self, app, config, header_layout):
        assert isinstance(app, App)
        self.app = app
        self.config = config
        self.dash = dash.Dash(u'PhiFlow')
        self.dash.config.suppress_callback_exceptions = True
        self.hrefs = set()
        self.page_urls = {}
        
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
