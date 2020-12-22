import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf2
from collections import deque
import gym
import random

import time

from skimage.color import rgb2gray
from skimage.transform import resize

from paraments import *


class DQNetwork:
    def __init__(self, environment):
        self.env = environment
        self.strategy = get_strategy()

        self.scale = SCALE
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE

        self.state_size = 84
        self.action_size = self.env.action_space.n
        self.frame_shape = (self.state_size, self.state_size)
        self.possible_actions = np.identity(self.action_size, dtype=np.uint8)

        self.memory = deque(maxlen=MEMORY_LENGTH)
        self.stack = deque([np.zeros(self.frame_shape, dtype=np.uint8) for _ in range(STACK_LENGTH)],
                           maxlen=STACK_LENGTH)

        self.model = self.build_model()

    def preprocess(self, state):
        """Resize frame and greyscale, store as uint8 and normalize on demand to save memory"""

        # 1. rgb -> gray
        state_ = rgb2gray(state) * 255

        # 2. resize to self.frame_shape
        state_ = resize(state_, self.frame_shape, mode="constant")
        assert state_.shape[1] == self.state_size

        # 3. save as uint8 to save memory
        state_ = np.uint8(state_)

        return state_

    def append_to_stack(self, state, reset=False):
        """Preprocesses a frame and adds it to the stack"""
        state_ = self.preprocess(state)

        if reset:
            # Reset stack
            self.stack.clear()

            # Because we're in a new episode, copy the same frame 4x
            for _ in range(STACK_LENGTH):
                self.stack.append(state_)
        else:
            self.stack.append(state_)

        # Build the stacked state (last dimension specifies different frames)
        return np.stack(self.stack, axis=2)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def build_model(self):
        """Build the neural net model"""
        with self.strategy.scope():
            model = Sequential()

            # N x 84 x 84 x 4
            model.add(Conv2D(filters=16, kernel_size=8, strides=4, activation='relu',
                             input_shape=(self.state_size, self.state_size, 4)))
            # N x 20 x 20 x 16
            model.add(Conv2D(filters=32, kernel_size=4, strides=2, activation='relu'))
            # N x 9 x 9 x 32
            model.add(Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'))
            # N x 4 x 4 x 64
            model.add(Flatten())
            # N x 1024
            model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
            # N x 512
            model.add(Dense(self.action_size, activation='softmax'))
            # N x 6(self.action_size)

            model.summary()

            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def random_action(self):
        """Returns a random action"""
        return random.randint(1, len(self.possible_actions)) - 1

    def predict_action(self, state):
        """Returns index of best predicted action"""
        state_ = state / 255.0
        state_ = state_.reshape((1, *state_.shape))  # Reshape our state to a single example for our neural net

        with self.strategy.scope():
            choice = self.model.predict(state_)

        return np.argmax(choice)

    def select_action(self, state, decay_step):
        """Returns an action to take with decaying exploration/exploitation"""

        # decay_step越大，exp越小，0.049289(=0.99*0.049787) <= 第二项 <= 0.9899901(=0.99*0.99999)
        explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * decay_step)

        # 玩越久越相信自己，explore_probability从0.99开始，随着游戏进程深入越来越小
        if explore_probability > np.random.rand():
            # Exploration
            return self.random_action(), explore_probability
        else:
            # Exploitation
            return self.predict_action(state), explore_probability

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Select a random mini_batch from memory
        mini_batch = random.sample(self.memory, self.batch_size)

        # Split out our tuple and normalise our states
        states = np.array([each[0] for each in mini_batch]) / 255.0
        actions = np.array([each[1] for each in mini_batch])
        rewards = np.array([each[2] for each in mini_batch])
        next_states = np.array([each[3] for each in mini_batch]) / 255.0
        dones = np.array([each[4] for each in mini_batch])

        # Get our predictions for our states and our next states
        with self.strategy.scope():
            predicted_next_qs = self.model.predict(next_states)

        # Create an empty targets list to hold our Q-values
        target_Qs_batch = []

        for i in range(0, len(mini_batch)):
            done = dones[i]

            if done:
                # If we finished the game, our q value is the final reward (as there are no more future rewards)
                q_value = rewards[i]
            else:
                # If we haven't, our q value is the immediate reward, plus future discounted reward (gamma is our discount)
                q_value = rewards[i] + self.gamma * np.max(predicted_next_qs[i])

            # Fit target to a vector for keras (represent actions as one hot * q value (q gets set at the action we took, everything else is 0))
            one_hot_target = self.possible_actions[actions[i]]
            target = one_hot_target * q_value
            target_Qs_batch.append(target)

        targets = np.array([each for each in target_Qs_batch])

        with self.strategy.scope():
            self.model.fit(states, targets, epochs=1, verbose=0, use_multiprocessing=True)  # Change to verbose=0 to disable logging


def main():
    env = gym.make('Pong-v4')
    env._max_episode_steps = 100000  # 大约半小时
    env = gym.wrappers.Monitor(env, './videos-{}/'.format(GIVEN_GPU[0]), force=True,
                               video_callable=lambda episode_id: True)  # Save each episode to video
    agent = DQNetwork(env)
    episodes = 500000
    decay_step = 0

    for episode in range(episodes):
        episode_rewards = []

        # 1. Reset the env and frame stack
        state = agent.env.reset()
        state = agent.append_to_stack(state, reset=True)

        while True:
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
                if (episode % 100 == 0) or (episode + 1 == episodes):
                    agent.model.save_weights("model-{}-ep-{}.h5".format(GIVEN_GPU[0], episode))

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


if __name__ == '__main__':
    start = time.time()
    print("开始训练时间：{}".format(time.localtime(start)))
    main()
    print("训练完成时间：{} | 耗时：{}".format(time.localtime(time.time()), time.localtime(time.time()-start)))

