from phi.flow import *


def random_sdf():
    dim = [50,50,50]
    n_intervals = np.random.randint(0, 10)
    grid = np.ones((1, dim[0], dim[1], dim[2], 1), dtype="float32")
    grid[:, n_intervals:n_intervals + int(dim[0] * 0.3), n_intervals:n_intervals + int(dim[0] * 0.3),
    n_intervals:n_intervals + int(dim[0] * 0.3):, ] = -1
    return grid


domain = Domain([20, 20, 20])


class Simpleplume(App):

    def __init__(self):
        App.__init__(self)
        self.field = world.add(domain.centered_grid(math.randfreq) * 10)
        data = self.field.data
        data[:, 2:5, ...] = 0
        self.add_field('Density', self.field)
        self.add_field('SDF', random_sdf())
        self.value_text = 'My text'
        self.value_float = 1.0
        self.value_int = 2
        self.value_bool = False

    def step(self):
        self.field.data = self.field.data + domain.centered_grid(math.randfreq).data
        self.info(', '.join([str(x) for x in [self.value_text, self.value_float, self.value_int, self.value_bool]]))


show()
