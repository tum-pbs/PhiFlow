

def generate(struct_name, variables=(), constants=(), others=()):
    items = ''
    parameters = ''
    ignore = []
    other_init = ''
    for attr_name in variables:
        items += """
    @struct.variable()
    def {name}(self, {name}):
        return {name}
""".replace('{name}', attr_name)
        parameters += '%s, ' % (attr_name, )
    for prop_name in constants:
        items += """
    @struct.constant()
    def {name}(self, {name}):
        return {name}
        """.replace('{name}', prop_name)
        parameters += '%s, ' % (prop_name,)
    for other_name in others:
        items += """
    @property
    def {name}(self):
        return self._{name}
        """.replace('{name}', other_name)
        parameters += '%s, ' % (other_name,)
        ignore.append("'%s'" % other_name)
        other_init += '\n        self._{name} = {name}'.replace('{name}', other_name)
    if len(ignore) > 0:
        ignore = ', ignore=[%s]' % ', '.join(ignore)
    else:
        ignore = ''

    return """
from phi import struct


@struct.definition()
class %s(struct.Struct):
    
    def __init__(self, %s**kwargs):
        struct.Struct.__init__(self, **struct.kwargs(locals()%s))%s
    %s
""" % (struct_name, parameters, ignore, other_init, items)
