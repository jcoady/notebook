from typing import Optional, List
import numpy as np
import math
import random

from helper import MuZeroConfig
from helper import Game, Player, Action, ActionHistory, Node
from helper import Network, NetworkOutput, BaseNetwork, SharedStorage
from helper import ReplayBuffer, MinMaxStats

def run_selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, train_episodes: int):

    network = storage.latest_network()
    returns = []
    for _ in range(train_episodes):
        game = play_game(config, network)
        replay_buffer.save_game(game)
        returns.append(sum(game.rewards))
    return sum(returns) / train_episodes


def run_eval(config: MuZeroConfig, storage: SharedStorage, eval_episodes: int):

    network = storage.latest_network()
    returns = []
    for _ in range(eval_episodes):
        game = play_game(config, network, train=False)
        returns.append(sum(game.rewards))
    return sum(returns) / eval_episodes if eval_episodes else 0

def play_game(config: MuZeroConfig, network: Network, train: bool = True) -> Game:
    
    game = config.new_game()
    mode_action_select = 'softmax' if train else 'max'

    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(), network.initial_inference(current_observation))
        if train:
            add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the networks.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network, mode=mode_action_select)
        game.apply(action)
        game.store_search_statistics(root)
    return game

def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory, network: BaseNetwork):
    
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state, history.last_action())
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(search_path, network_output.value, history.to_play(), config.discount, min_max_stats)

def select_action(config: MuZeroConfig, num_moves: int, node: Node, network: BaseNetwork, mode: str = 'softmax'):
    
    visit_counts = [child.visit_count for child in node.children.values()]
    actions = [action for action in node.children.keys()]
    action = None
    if mode == 'softmax':
        t = config.visit_softmax_temperature_fn(
            num_moves=num_moves, training_steps=network.training_steps)
        action = softmax_sample(visit_counts, actions, t)
    elif mode == 'max':
        action, _ = max(node.children.items(), key=lambda item: item[1].visit_count)
    return action

def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    
    # When the parent visit count is zero, all ucb scores are zeros, therefore we return a random child
    if node.visit_count == 0:
        return random.sample(node.children.items(), 1)[0]

    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action,
         child) for action, child in node.children.items())
    return action, child

def softmax_sample(visit_counts, actions, t):
    counts_exp = np.exp(visit_counts) * (1 / t)
    probs = counts_exp / np.sum(counts_exp, axis=0)
    action_idx = np.random.choice(len(actions), p=probs)
    return actions[action_idx]

def ucb_score(config: MuZeroConfig, parent: Node, child: Node,
              min_max_stats: MinMaxStats) -> float:
    
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score

def expand_node(node: Node, to_play: Player, actions: List[Action],
                network_output: NetworkOutput):
   
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


def backpropagate(search_path: List[Node], value: float, to_play: Player,
                  discount: float, min_max_stats: MinMaxStats):
    
    for node in search_path[::-1]:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value

def add_exploration_noise(config: MuZeroConfig, node: Node):
    
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac










