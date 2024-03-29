import math

from phi.torch.flow import *

net = fno(1, 2, 3, modes=12, activation='GeLU')
optimizer = adam(net, learning_rate=1e-3)


def loss_function(scale: Tensor, smoothness: Tensor):
    grid = CenteredGrid(Noise(scale=scale, smoothness=smoothness), x=64, y=64)

    print(f'Grid Shape : {grid.shape}')
    pred_scale, pred_smoothness = field.native_call(net, grid).vector
    return math.l2_loss(pred_scale - scale) + math.l2_loss(pred_smoothness - smoothness)


gt_scale = math.random_uniform(batch(examples=50), low=1, high=10)
gt_smoothness = math.random_uniform(batch(examples=50), low=.5, high=3)


print(gt_scale.shape)
print(gt_smoothness.shape)

viewer = view(gui='dash', scene=True)
for i in viewer.range():
    if i == 100: break
    loss = update_weights(net, optimizer, loss_function, gt_scale, gt_smoothness)
    print(f'Iter : {i}, Loss : {loss}')
    viewer.log_scalars(loss=loss)