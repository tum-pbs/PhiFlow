from phi.flow import *
from phi.physics.reaction_diffusion import *

physics_config.x_first()

SAMPLE_PATTERNS = {
    'diagonal': {'du': 0.17, 'dv': 0.03, 'f': 0.06, 'k': 0.056},
    'maze': {'du': 0.19, 'dv': 0.05, 'f': 0.06, 'k': 0.062},
    'coral': {'du': 0.16, 'dv': 0.08, 'f': 0.06, 'k': 0.062},
    'flood': {'du': 0.19, 'dv': 0.05, 'f': 0.06, 'k': 0.02},
    'dots': {'du': 0.19, 'dv': 0.05, 'f': 0.04, 'k': 0.065},
    'dots_and_stripes': {'du': 0.19, 'dv': 0.03, 'f': 0.04, 'k': 0.061},
}
PATTERN = SAMPLE_PATTERNS[sys.argv[1] if len(sys.argv) > 1 else 'dots2']

pattern = world.add(Pattern(Domain([126, 126], OPEN), **PATTERN), physics=ReactionDiffusion())
pattern.u = pattern.v = Seed(center=[80, 40], size=3, mode='EXP', factor=1.0)

show(App('Reaction-Diffusion System', 'Pattern formation using a simple PDE', framerate=300, dt=1))
