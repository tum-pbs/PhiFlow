from unittest import TestCase

from phi.tf import nets as tf_nets
from phi.torch import nets as torch_nets
from phi.jax.stax import nets as stax_nets


class TestNetworks(TestCase):

    def test_u_net_2d_network_sizes(self):
        for lib in [tf_nets, torch_nets, stax_nets]:
            net = lib.u_net(2, 3, levels=3, filters=8, batch_norm=False, activation='ReLU', in_spatial=(64, 32))
            self.assertEqual(6587, lib.parameter_count(net))

    def test_u_net_3d_norm_network_sizes(self):
        for lib in [tf_nets, torch_nets, stax_nets]:
            net = lib.u_net(2, 3, levels=3, filters=8, batch_norm=True, activation='Sigmoid', in_spatial=3)
            self.assertEqual(19707, lib.parameter_count(net))

    def test_u_net_1d_norm_network_sizes(self):
        for lib in [tf_nets, torch_nets, stax_nets]:
            net = lib.u_net(2, 3, levels=2, filters=16, batch_norm=True, activation='tanh', in_spatial=1)
            self.assertEqual(5043, lib.parameter_count(net))
