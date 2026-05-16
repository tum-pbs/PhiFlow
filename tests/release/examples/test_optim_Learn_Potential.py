from phi.torch.flow import *


def test_learn_potential():
    def potential(pos):
        return math.cos(math.vec_length(pos))

    landscape = CenteredGrid(potential, x=100, y=100, bounds=Box(x=(-5, 5), y=(-5, 5)))
    plot(landscape)
    vis.savefig('vis/learn_potential_landscape.jpg')

    math.seed(0)
    net = dense_net(2, 1, [32, 64, 32])
    optimizer = adam(net)

    def loss_function(x, label):
        prediction = math.native_call(net, x)
        return math.l2_loss(prediction - label), prediction

    input_data = rename_dims(landscape.points, spatial, batch)
    labels = rename_dims(landscape.values, spatial, batch)

    loss_trj = []
    pred_trj = []
    for i in range(200):
        loss, pred = update_weights(net, optimizer, loss_function, input_data, labels)
        loss_trj.append(loss)
        pred_trj.append(pred)
    loss_trj = stack(loss_trj, spatial('iteration'))
    pred_trj = stack(pred_trj, batch('iteration'))
    plot(math.mean(loss_trj, 'x,y'), err=math.std(loss_trj, 'x,y'), size=(4, 3))
    vis.savefig('vis/learn_potential_loss.jpg')

    pred_grid = rename_dims(pred_trj.iteration[::4], 'x,y', spatial)
    plot(pred_grid, animate='iteration', size=(6, 5))
    vis.savefig('vis/learn_potential.mp4')


if __name__ == '__main__':
    test_learn_potential()
