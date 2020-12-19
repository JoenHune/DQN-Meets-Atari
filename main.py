import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf2
from collections import deque
import gym
import random

DEFAULT_GPU_LIST = [0, 1, 2]

def get_strategy(gpu_visible=None):
    gpu_total = tf2.config.experimental.list_physical_devices(device_type="GPU")
    gpu_candidates = []

    if gpu_visible is None:
        gpu_visible = DEFAULT_GPU_LIST

    for gpu_id in gpu_visible:
        if 0 <= gpu_id < len(gpu_total):
            gpu_candidates.append(gpu_total[gpu_id])

    tf2.config.experimental.set_visible_devices(devices=gpu_candidates, device_type="GPU")

    strategy = tf2.distribute.OneDeviceStrategy(device="/cpu:0")
    if len(gpu_candidates) == 1:
        strategy = tf2.distribute.OneDeviceStrategy(device="/gpu:0")
    elif len(gpu_candidates) > 1:
        strategy = tf2.distribute.MirroredStrategy()

    return strategy


class DQNetwork:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=1000000)
        self.stack = deque([np.zeros((105, 80), dtype=np.uint8) for i in range(4)], maxlen=4)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.00003
        self.learning_rate = 0.00025
        self.batch_size = 64
        self.frame_size = (105, 80)
        self.possible_actions = np.array(np.identity(self.action_size, dtype=int).tolist())
        self.strategy = get_strategy()
        self.model = self.build_model()

    def preprocess_frame(self, frame):
        """Resize frame and greyscale, store as uint8 and normalize on demand to save memory"""
        frame = frame[::2, ::2]
        return np.mean(frame, axis=2).astype(np.uint8)

    def append_to_stack(self, state, reset=False):
        """Preprocesses a frame and adds it to the stack"""
        frame = self.preprocess_frame(state)

        if reset:
            # Reset stack
            self.stack = deque([np.zeros((105, 80), dtype=np.int) for i in range(4)], maxlen=4)

            # Because we're in a new episode, copy the same frame 4x
            for i in range(4):
                self.stack.append(frame)
        else:
            self.stack.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(self.stack, axis=2)

        return stacked_state

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def memory_sample(self, batch_size):
        """Sample a random batch of experiences from memory"""
        memory_size = len(self.memory)
        index = np.random.choice(np.arange(memory_size), size=batch_size, replace=False)
        return [self.memory[i] for i in index]

    def build_model(self):
        """Build the neural net model"""
        with self.strategy.scope():
            model = Sequential()
            model.add(Conv2D(32, (8, 4), activation='elu', input_shape=(105, 80, 4)))
            model.add(Conv2D(64, (3, 2), activation='elu'))
            model.add(Conv2D(64, (3, 2), activation='elu'))
            model.add(Flatten())
            model.add(Dense(512, activation='elu', kernel_initializer='glorot_uniform'))
            model.add(Dense(self.action_size, activation='softmax'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def random_action(self):
        """Returns a random action"""
        return random.randint(1, len(self.possible_actions)) - 1

    def predict_action(self, state):
        """Returns index of best predicted action"""
        state = state / 255
        state = state.reshape((1, *state.shape))  # Reshape our state to a single example for our neural net
        choice = self.model.predict(state)
        return np.argmax(choice)

    def select_action(self, state, decay_step):
        """Returns an action to take with decaying exploration/exploitation"""

        explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
            -self.epsilon_decay * decay_step)

        if explore_probability > np.random.rand():
            # Exploration
            return self.random_action(), explore_probability
        else:
            # Exploitation
            return self.predict_action(state), explore_probability

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Select a random minibatch from memory
        minibatch = self.memory_sample(self.batch_size)

        # Split out our tuple and normalise our states
        states_mb = np.array([each[0] for each in minibatch]) / 255
        actions_mb = np.array([each[1] for each in minibatch])
        rewards_mb = np.array([each[2] for each in minibatch])
        next_states_mb = np.array([each[3] for each in minibatch]) / 255
        dones_mb = np.array([each[4] for each in minibatch])

        # Get our predictions for our states and our next states
        target_qs = self.model.predict(states_mb)
        predicted_next_qs = self.model.predict(next_states_mb)

        # Create an empty targets list to hold our Q-values
        target_Qs_batch = []

        for i in range(0, len(minibatch)):
            done = dones_mb[i]

            if done:
                # If we finished the game, our q value is the final reward (as there are no more future rewards)
                q_value = rewards_mb[i]
            else:
                # If we havent, our q value is the immediate reward, plus future discounted reward (gamma is our discount)
                q_value = rewards_mb[i] + self.gamma * np.max(predicted_next_qs[i])

            # Fit target to a vector for keras (represent actions as one hot * q value (q gets set at the action we took, everything else is 0))

            one_hot_target = self.possible_actions[actions_mb[i]]
            target = one_hot_target * q_value
            target_Qs_batch.append(target)

        targets_mb = np.array([each for each in target_Qs_batch])

        with self.strategy.scope():
            self.model.fit(states_mb, targets_mb, epochs=1, verbose=1)  # Change to verbose=0 to disable logging


env = gym.make('Pong-v4')
env = gym.wrappers.Monitor(env, './videos/', force=True,
                           video_callable=lambda episode_id: True)  # Save each episode to video
agent = DQNetwork(env)
episodes = 50
steps = 50000
decay_step = 0

for episode in range(episodes):
    episode_rewards = []

    # 1. Reset the env and frame stack
    state = agent.env.reset()
    state = agent.append_to_stack(state, reset=True)

    for step in range(steps):
        decay_step += 1

        # 2. Select an action to take based on exploration/exploitation
        action, explore_probability = agent.select_action(state, decay_step)

        # 3. Take the action and observe the new state
        next_state, reward, done, info = agent.env.step(action)

        # Store the reward for this move in the episode
        episode_rewards.append(reward)

        # 4. If game finished...
        if done:
            # Create a blank next state so that we can save the final rewards
            next_state = np.zeros((210, 160, 3), dtype=np.uint8)
            next_state = agent.append_to_stack(next_state)

            # Add our experience to memory
            agent.remember(state, action, reward, next_state, done)

            # Save our model
            agent.model.save_weights("model-ep-{}.h5".format(episode))

            # Print logging info
            print("Game ended at episode {}/{}, total rewards: {}, explore_prob: {}".format(episode, episodes,
                                                                                            np.sum(episode_rewards),
                                                                                            explore_probability))
            # Start a new episode
            break
        else:
            # Add the next state to the stack
            next_state = agent.append_to_stack(next_state)

            # Add our experience to memory
            agent.remember(state, action, reward, next_state, done)

            # Set state to the next state
            state = next_state

        # 5. Train with replay
        agent.replay()
