# import os; os.environ['CUDA_VISIBLE_DEVICES'] = ""
from unittest import TestCase

from phi import math
from phi.math import channel, batch
from phi.tf import nets as tf_nets
from phi.torch import nets as torch_nets
from phi.jax.stax import nets as stax_nets


LIBRARIES = [tf_nets, torch_nets, stax_nets]


class TestNetworks(TestCase):

    def test_u_net_2d_network_sizes(self):
        for lib in LIBRARIES:
            net = lib.u_net(2, 3, levels=3, filters=8, batch_norm=False, activation='ReLU', in_spatial=(64, 32))
            self.assertEqual(6587, lib.parameter_count(net), msg=lib)
            net_res = lib.u_net(2, 3, batch_norm=False, activation='SiLU', in_spatial=2, use_res_blocks=True)
            self.assertEqual(39059, lib.parameter_count(net_res), msg=lib)

    def test_u_net_3d_norm_network_sizes(self):
        for lib in LIBRARIES:
            net = lib.u_net(2, 3, levels=3, filters=8, batch_norm=True, activation='Sigmoid', in_spatial=3)
            self.assertEqual(19707, lib.parameter_count(net), msg=lib)
            net_res = lib.u_net(2, 3, batch_norm=True, activation='SiLU', in_spatial=3, use_res_blocks=True)
            self.assertEqual(113939, lib.parameter_count(net_res), msg=lib)

    def test_u_net_1d_norm_network_sizes(self):
        for lib in LIBRARIES:
            net = lib.u_net(2, 3, levels=2, filters=16, batch_norm=True, activation='tanh', in_spatial=1)
            self.assertEqual(5043, lib.parameter_count(net), msg=lib)
            net_res = lib.u_net(2, 3, batch_norm=True, activation='SiLU', in_spatial=1, use_res_blocks=True)
            self.assertEqual(14867, lib.parameter_count(net_res), msg=lib)

    def test_optimize_u_net(self):
        for lib in LIBRARIES:
            net = lib.u_net(1, 1, levels=2)
            optimizer = lib.adam(net, 1e-3)
            net_res = lib.u_net(1, 1, levels=2, use_res_blocks=True, activation='SiLU')
            optimizer_2 = lib.sgd(net_res, 1e-3)

            def loss_function(x):
                print("Running loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net, x)
                return math.l2_loss(pred)

            def loss_function_res(x):
                print("Running loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net_res, x)
                return math.l2_loss(pred)

            for i in range(2):
                lib.update_weights(net, optimizer, loss_function, math.random_uniform(math.batch(batch=10), math.spatial(x=8, y=8)))
                lib.update_weights(net_res, optimizer_2, loss_function_res, math.random_uniform(math.batch(batch=10), math.spatial(x=8, y=8)))

    def test_optimize_u_net_jit(self):
        for lib in LIBRARIES:
            net = lib.u_net(1, 1, levels=2)
            optimizer = lib.adagrad(net, 1e-3)
            net_res = lib.u_net(1, 1, levels=2, use_res_blocks=True, activation='SiLU')
            optimizer_2 = lib.rmsprop(net_res, 1e-3)

            @math.jit_compile
            def loss_function(x):
                print("Tracing loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net, x)
                return math.l2_loss(pred)

            @math.jit_compile
            def loss_function_res(x):
                print("Tracing loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net_res, x)
                return math.l2_loss(pred)

            for i in range(2):
                lib.update_weights(net, optimizer, loss_function, math.random_uniform(math.batch(batch=10), math.spatial(x=8, y=8)))
                lib.update_weights(net_res, optimizer_2, loss_function_res, math.random_uniform(math.batch(batch=10), math.spatial(x=8, y=8)))

    def test_dense_net_network_params(self):
        for lib in LIBRARIES:
            net = lib.dense_net(2, 3, layers=[10, 12], batch_norm=False, activation='ReLU')
            self.assertEqual(201, lib.parameter_count(net))
            params = lib.get_parameters(net, wrap=True)
            self.assertEqual(6, len(params))
            self.assertTrue(all(isinstance(p, math.Tensor) for p in params.values()))
            params = lib.get_parameters(net, wrap=False)
            self.assertEqual(6, len(params))
            net = lib.dense_net(2, 3, layers=[10], batch_norm=True, activation='ReLU')
            self.assertEqual(83, lib.parameter_count(net), str(lib))

    def test_optimize_dense_net(self):
        for lib in LIBRARIES:
            net = lib.dense_net(2, 3, layers=[10], batch_norm=True, activation='Sigmoid')
            optimizer = lib.adam(net)

            def loss_function(x):
                print("Running loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net, x)
                return math.l2_loss(pred)

            for i in range(2):
                lib.update_weights(net, optimizer, loss_function, math.random_uniform(batch(batch=10), channel(vector=2)))

    def test_optimize_invertible_conv_net(self):
        for lib in LIBRARIES:
            net = lib.invertible_net(2, 3, True, 'conv_net', 'SiLU')
            optimizer = lib.adam(net)

        def loss_function(x):
            print("Running loss_function")
            assert isinstance(x, math.Tensor)
            pred = math.native_call(net, x)
            return math.l2_loss(pred)

        for i in range(2):
            lib.update_weights(net, optimizer, loss_function,
                               math.random_uniform(math.batch(batch=10), math.channel(c=2), math.spatial(x=8, y=8)))

    def test_optimize_invertible_res_net(self):
        for lib in LIBRARIES:
            net = lib.invertible_net(2, 3, True, 'res_net', 'SiLU')
            optimizer = lib.adam(net)

        def loss_function(x):
            print("Running loss_function")
            assert isinstance(x, math.Tensor)
            pred = math.native_call(net, x)
            return math.l2_loss(pred)

        for i in range(2):
            lib.update_weights(net, optimizer, loss_function,
                               math.random_uniform(math.batch(batch=10), math.channel(c=2), math.spatial(x=8, y=8)))

    def test_optimize_invertible_u_net(self):
        for lib in LIBRARIES:
            net = lib.invertible_net(2, 3, True, 'u_net', 'SiLU')
            optimizer = lib.adam(net)

        def loss_function(x):
            print("Running loss_function")
            assert isinstance(x, math.Tensor)
            pred = math.native_call(net, x)
            return math.l2_loss(pred)

        for i in range(2):
            lib.update_weights(net, optimizer, loss_function,
                               math.random_uniform(math.batch(batch=10), math.channel(c=2), math.spatial(x=8, y=8)))

    def test_optimize_invertible_dense_net(self):
        for lib in LIBRARIES:
            net = lib.invertible_net(50, 3, True, in_spatial=0)
            optimizer = lib.adam(net)

        def loss_function(x):
            print("Running loss_function")
            assert isinstance(x, math.Tensor)
            pred = math.native_call(net, x)
            return math.l2_loss(pred)

        for i in range(2):
            lib.update_weights(net, optimizer, loss_function,
                               math.random_uniform(math.batch(batch=10), math.channel(c=50)))

    def test_invertible_net_network_sizes(self):
        for lib in LIBRARIES:
            net_u = lib.invertible_net(2, 3, True, 'u_net', 'SiLU')
            self.assertEqual(454296, lib.parameter_count(net_u))
            net_res = lib.invertible_net(2, 3, True, 'res_net', 'ReLU')
            self.assertEqual(1080, lib.parameter_count(net_res))
            net_conv = lib.invertible_net(2, 3, True, 'conv_net', 'ReLU')
            self.assertEqual(576, lib.parameter_count(net_conv))
            net_dense = lib.invertible_net(2, 3, True, activation='ReLU', in_spatial=0)
            self.assertEqual(240, lib.parameter_count(net_dense))

    def test_conv_classifier(self):
        for lib in LIBRARIES:
            net = lib.conv_classifier(1, (2,), 1, blocks=[10], dense_layers=[], batch_norm=True, softmax=False, periodic=False)
            self.assertEqual(401, lib.parameter_count(net))






