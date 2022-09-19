import math
from phi.torch.flow import *
net = invertible_net(1, 3, True, 'u_net', 'SiLU')
optimizer = adam(net, learning_rate=1e-3)

print(parameter_count(net))

def loss_function(smoothness: Tensor):
    grid = CenteredGrid(Noise(smoothness=smoothness), x=8, y=8)
    pred_smoothness = field.native_call(net, grid)

    return math.l2_loss(pred_smoothness - smoothness)

gt_smoothness = math.random_uniform(batch(examples=10), low=0.5, high=1)

viewer = view(gui='dash', scene=True)
for i in viewer.range():
    if i > 100: break
    loss = update_weights(net, optimizer, loss_function, gt_smoothness)
    if i % 10 == 0: print(f'Iter : {i}, Loss : {loss}')
    viewer.log_scalars(loss=loss)

grid = CenteredGrid(Noise(scale=1.0, smoothness=gt_smoothness), x=8, y=8)
pred = field.native_call(net, grid, False)
reconstructed_input = field.native_call(net, pred, True)

print('Loss between Predicted Tensor and original grid', math.l2_loss(pred - grid))
print('Loss between Predicted Tensor and GT tensor', math.l2_loss(pred - gt_smoothness))
print('Loss between Reconstructed Input and original grid:', math.l2_loss(reconstructed_input - grid))
