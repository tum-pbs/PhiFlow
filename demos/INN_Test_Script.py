import math

from phi.torch.flow import *
net = inn(1,2,3)
#net2 = inn(1,2,3)
optimizer = adam(net, learning_rate=1e-3)


def loss_function(scale: Tensor, smoothness: Tensor):
    grid = CenteredGrid(Noise(scale=scale, smoothness=smoothness), x=64, y=64)

    print(f'Grid Shape : {grid.shape}')
    pred_scale = field.native_call(net, grid)
    return math.l2_loss(pred_scale - scale)


gt_scale = math.random_uniform(batch(examples=50), low=1, high=10)
gt_smoothness = math.random_uniform(batch(examples=50), low=.5, high=3)


print(gt_scale.shape)
print(gt_smoothness.shape)

viewer = view(gui='dash', scene=True)
for i in range(100):
    loss = update_weights(net, optimizer, loss_function, gt_scale, gt_smoothness)
    print(f'Iter : {i}, Loss : {loss}')
    viewer.log_scalars(loss=loss)


grid = CenteredGrid(Noise(scale=gt_scale, smoothness=gt_smoothness), x=64, y=64)
pred = field.native_call(net, grid, False)
reconstructed_input = field.native_call(net, pred, True)
print('Loss between Predicted Tensor and original grid', math.l2_loss(pred - grid))
print('Loss between Reconstructed Input and original grid:', math.l2_loss(reconstructed_input - grid))