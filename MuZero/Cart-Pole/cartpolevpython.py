from vpython import canvas, box, cylinder, vector, color, rate
from helper import MuZeroConfig, CartPole, CartPoleNetwork

class CartPoleVPython(CartPole):
    def __init__(self, discount: float):
        global a,b,c
        super().__init__(discount)
        observation = self.observations[0]
        try:
            b.visible = False
            c.visible = False
        except:
            pass
        else:
            del b
            del c
        b = box(pos=vector(0,-0.5,0), color=color.green)
        c = cylinder(pos=vector(0,0,0), axis=vector(0,4,0), radius=0.1, )
        b.pos.x = c.pos.x = observation[0]
        a = observation[2]
        c.rotate(angle=a, axis=vector(0,0,1))


    def step(self, action) -> int:
        global a,b,c
        observation, reward, done, _ = self.env.step(action.index)
        self.observations += [observation]
        self.done = done
        rate(1000)
        b.pos.x = c.pos.x = observation[0]
        c.rotate(angle=-a, axis=vector(0,0,1))
        a = -observation[2]
        c.rotate(angle=a, axis=vector(0,0,1))

        return reward

def make_CartPoleVPython_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        return 1.0

    return MuZeroConfig(
        game=CartPoleVPython,
        nb_training_loop=20,
        nb_episodes=20,
        nb_epochs=20,
        network_args={'action_size': 2,
                      'state_size': 4,
                      'representation_size': 4,
                      'max_value': 500},
        network=CartPoleNetwork,
        action_space_size=2,
        max_moves=1000,
        discount=0.99,
        dirichlet_alpha=0.25,
        num_simulations=11,  # Odd number perform better in eval mode
        batch_size=512,
        td_steps=10,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        lr=0.05)

def create_scene():
    scene = canvas(range=5)
    wire = cylinder(pos=vector(-10,-0.5,0), axis=vector(20,0,0), radius=0.02, color=color.white)
    return scene
    
