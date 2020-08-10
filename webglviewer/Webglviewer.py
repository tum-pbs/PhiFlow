# AUTO GENERATED FILE - DO NOT EDIT

from dash.development.base_component import Component, _explicitize_args


class Webglviewer(Component):
    """A Webglviewer component.
ExampleComponent is an example component.
It takes a property, `label`, and
displays it.
It renders an input with the property `value`
which is editable by the user.

Keyword arguments:
- id (string; required): The ID used to identify this component in Dash callbacks.
- data (list; optional): A grid or a list of grids that will be rendered when this component is rendered.
- idx (number; optional): Index of datum in data list which should be rendered.
- sky (list; optional): Sky map, expects an array of six arrays.
Value have to be between 0 and 255.
Each array represents one side of the cube map (flattened pixels of an image).
Expects 4 channels (r,g,b,a).
Image has to be quadratic.
- material_type (string; optional): Material type used for rendering.
Possible types: SMOKE, DARK_SMOKE, LIGHT_SMOKE, SOLID, LIQUID
- representation_type (string; optional): Specifies the representation type used.
Possible types: DENSITY, SDF, PARTICLE
- scale (number; optional): Particle scale."""
    @_explicitize_args
    def __init__(self, id=Component.REQUIRED, data=Component.UNDEFINED, idx=Component.UNDEFINED, sky=Component.UNDEFINED, material_type=Component.UNDEFINED, representation_type=Component.UNDEFINED, scale=Component.UNDEFINED, **kwargs):
        self._prop_names = ['id', 'data', 'idx', 'sky', 'material_type', 'representation_type', 'scale']
        self._type = 'Webglviewer'
        self._namespace = 'webglviewer'
        self._valid_wildcard_attributes = []
        self.available_properties = ['id', 'data', 'idx', 'sky', 'material_type', 'representation_type', 'scale']
        self.available_wildcard_properties = []

        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs
        args = {k: _locals[k] for k in _explicit_args if k != 'children'}

        for k in ['id']:
            if k not in args:
                raise TypeError(
                    'Required argument `' + k + '` was not specified.')
        super(Webglviewer, self).__init__(**args)
