import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from phi.vis._vis_base import VisModel, play_async, status_message, gui_interrupt


class DashApp:

    def __init__(self, model: VisModel, config: dict, header_layout):
        self.model = model
        self.config = config
        self.dash = dash.Dash(u'PhiFlow')
        self.dash.config.suppress_callback_exceptions = True
        self.hrefs = set()
        self.page_urls = {}
        self.field_minmax = {}
        self.minmax_decay = 0.975
        self.play_status = None
        
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

    def reset_field_summary(self):
        self.field_minmax = {}

    def get_minmax(self, field):
        if field in self.field_minmax:
            return self.field_minmax[field]
        else:
            return 0, 0

    def play(self, max_steps=None):
        if not self.play_status:
            framerate = self.config.get('framerate', None)
            self.play_status = play_async(self.model, max_steps=max_steps, framerate=framerate)

    def pause(self):
        if self.play_status:
            self.play_status.pause()

    @property
    def status_message(self):
        return status_message(self.model, self.play_status)

    def exit_interrupt(self):
        self.pause()
        if self.model.can_progress:
            self.model.pre_step.append(gui_interrupt)
            self.model.progress()
