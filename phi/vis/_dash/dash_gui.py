import warnings

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output

from .board import build_benchmark, build_system_controls, \
    build_graph_view
from .log import build_log
from .model_controls import build_model_controls
from .viewsettings import build_view_selection, refresh_rate_ms, REFRESH_RATE
from .dash_app import DashApp
from .info import build_app_details, build_description, build_phiflow_info, build_app_time
from .viewer import build_viewers, REFRESH_INTERVAL
from .player_controls import build_status_bar, build_player_controls, PLAYING, STEP_COMPLETE
from .._vis_base import Gui, VisModel
from ...math.backend import PHI_LOGGER


class DashGui(Gui):

    def __init__(self):
        Gui.__init__(self, asynchronous=True)
        self.dash_app = None

    def setup(self, app: VisModel):
        Gui.setup(self, app)
        header_layout = html.Div([
                dcc.Link('Home', href='/'),
                ' - ',
                dcc.Link('Side-by-Side', href='/side-by-side'),
                ' - ',
                dcc.Link('Quad', href='/quad'),
                ' - ',
                dcc.Link('Info', href='/info'),
                ' - ',
                dcc.Link('Log', href='/log'),
                ' - ',
                dcc.Link(u'Φ Board', href='/board'),
                # ' - ',
                # dcc.Link('Scripting', href='/scripting'),
                ' - ',
                html.A('Help', href='https://tum-pbs.github.io/PhiFlow/Web_Interface.html', target='_blank'),
            ])
        dash_app = self.dash_app = DashApp(self.app, self.config, header_layout)

        # --- Shared components ---
        player_controls = build_player_controls(dash_app)
        status_bar = build_status_bar(dash_app)
        model_controls = build_model_controls(dash_app)
        refresh_interval = dcc.Interval(id=REFRESH_INTERVAL.component_id, interval=refresh_rate_ms(None))

        @dash_app.dash.callback(Output(REFRESH_INTERVAL.component_id, 'interval'), [REFRESH_RATE, PLAYING, STEP_COMPLETE, REFRESH_INTERVAL])
        def set_refresh_interval(refresh_rate_value, *args):
            if refresh_rate_value is None:
                result = 1000
            elif dash_app.play_status:
                result = refresh_rate_ms(refresh_rate_value)
            else:
                result = refresh_rate_ms(None)
            return result

        # --- Home ---
        viewers_home, field_selections = build_viewers(dash_app, 1, 800, 'home')
        layout = html.Div([
            build_description(dash_app),
            build_view_selection(dash_app, field_selections, 'home'),
            html.Div(style={'width': '100%', 'height': 800, 'margin-left': 'auto', 'margin-right': 'auto'}, children=[viewers_home[0]]),
            status_bar,
            player_controls,
            model_controls,
            refresh_interval,
        ])
        dash_app.add_page('/', layout)

        # --- Side by Side ---
        viewers_side_by_side, field_selections = build_viewers(dash_app, 2, 700, 'sbs')
        layout = html.Div([
            build_view_selection(dash_app, field_selections, 'sbs'),
            html.Div(style={'width': '50%', 'height': 700, 'display': 'inline-block'}, children=[viewers_side_by_side[0]]),
            html.Div(style={'width': '50%', 'height': 700, 'display': 'inline-block'}, children=[viewers_side_by_side[1]]),
            status_bar,
            player_controls,
            model_controls,
            refresh_interval,
        ])
        dash_app.add_page('/side-by-side', layout)

        # --- Quad ---
        viewers_quad, field_selections = build_viewers(dash_app, 4, 700, 'quad')
        layout = html.Div([
            build_view_selection(dash_app, field_selections, 'quad'),
            html.Div(style={'width': '50%', 'height': 700, 'display': 'inline-block'}, children=[viewers_quad[0]]),
            html.Div(style={'width': '50%', 'height': 700, 'display': 'inline-block'}, children=[viewers_quad[1]]),
            html.Div(style={'width': '50%', 'height': 700, 'display': 'inline-block'}, children=[viewers_quad[2]]),
            html.Div(style={'width': '50%', 'height': 700, 'display': 'inline-block'}, children=[viewers_quad[3]]),
            status_bar,
            player_controls,
            model_controls,
            refresh_interval,
        ])
        dash_app.add_page('/quad', layout)

        # --- Log ---
        layout = html.Div([
            dcc.Markdown('# Log'),
            status_bar,
            player_controls,
            build_log(dash_app)
        ])
        dash_app.add_page('/log', layout)

        # --- Info ---
        layout = html.Div([
            build_description(dash_app),
            status_bar,
            player_controls,
            build_phiflow_info(dash_app),
            build_app_details(dash_app),
            build_app_time(dash_app),
        ])
        dash_app.add_page('/info', layout)

        # --- Board ---
        layout = html.Div([
            dcc.Markdown(u'# Φ Board'),
            build_graph_view(dash_app),
            status_bar,
            player_controls,
        # ] + ([] if 'tensorflow' not in dash_app.vis.traits else [
        #     build_tensorboard_launcher(dash_app),
        ] + [
            model_controls,
            build_benchmark(dash_app),
        # ] + ([] if 'tensorflow' not in dash_app.vis.traits else [
        #     build_tf_profiler(dash_app),
        ] + [
            build_system_controls(dash_app),
            # ToDo: Graphs, Record/Animate
        ])
        dash_app.add_page('/board', layout)

        # --- Scripting ---
        layout = html.Div([
            dcc.Markdown(u'# Python Scripting'),
            'Custom Fields, Execute script, Restart'
        ])
        dash_app.add_page('/scripting', layout)

        return self.dash_app.dash

    def show(self, caller_is_main):
        if not caller_is_main and self.config.get('external_web_server', False):
            return self.dash_app.server
        else:
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)

            port = self.config.get('port', 8051)
            debug = self.config.get('debug', False)
            use_waitress = False
            if not debug:
                try:
                    import waitress
                    use_waitress = True
                except ImportError:
                    warnings.warn('waitress is not installed, falling back to dash development server. To enable it, run  $ pip install waitress', ImportWarning)
            print('Starting Dash server on http://localhost:%d/' % port)
            if use_waitress:
                import waitress
                logging.getLogger('waitress').setLevel(logging.ERROR)
                waitress.serve(self.dash_app.dash.server, port=port)
            else:
                self.dash_app.dash.run_server(debug=debug, host=self.config.get('host', '0.0.0.0'), port=port, use_reloader=False)
            return self

    def auto_play(self):
        self.dash_app.play()
