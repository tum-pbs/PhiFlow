""" PyTorch Network Training Demo
Trains a simple CNN to make velocity fields incompressible.
This script runs for a certain number of steps before saving the trained network and halting.
"""
from phi.torch.flow import *


# TORCH_BACKEND.set_default_device('GPU')
net = u_net(2, 2)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

DOMAIN = Domain(x=64, y=64)
prediction = DOMAIN.vector_grid(0)
prediction_div = DOMAIN.scalar_grid(0)
app = view(play=False)

for step in app.range(100):
    # Load or generate training data
    data = DOMAIN.vector_grid(Noise(batch=8, vector=2))
    # Initialize optimizer
    optimizer.zero_grad()
    # Prediction
    prediction = field.native_call(net, data)  # calls net with shape (batch_size, channels, spatial...)
    # Simulation
    prediction_div = field.divergence(prediction)
    # Define loss
    loss = field.l2_loss(prediction_div) + field.l2_loss(prediction - data)
    app.log_scalars(loss=loss, div=field.mean(abs(prediction_div)), distance=math.vec_abs(field.mean(abs(prediction - data))))
    # Compute gradients and update weights
    loss.native().backward()
    optimizer.step()

torch.save(net.state_dict(), 'torch_net.pth')
app.info("Network saved.")

# To load the network: net.load_state_dict(torch.load('torch_net.pth'))
