
import sys
import tensorflow as tf2

DEFAULT_GPU_LIST = [0, 1, 2]

SCALE = 2

MEMORY_LENGTH = 1000000
STACK_LENGTH = 4

BATCH_SIZE = 64
LEARNING_RATE = 0.00025

GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.00003

GIVEN_GPU = [eval(sys.argv[1])] if len(sys.argv) > 1 else DEFAULT_GPU_LIST


def get_strategy(gpu_visible=None):
    gpu_total = tf2.config.experimental.list_physical_devices(device_type="GPU")
    gpu_candidates = []

    if gpu_visible is None:
        gpu_visible = GIVEN_GPU

    for gpu_id in gpu_visible:
        if 0 <= gpu_id < len(gpu_total):
            gpu_candidates.append(gpu_total[gpu_id])

    tf2.config.experimental.set_visible_devices(devices=gpu_candidates, device_type="GPU")
    print("gpu_total :", gpu_total, "| gpu_candidates :", gpu_candidates)

    strategy = tf2.distribute.OneDeviceStrategy(device="/cpu:0")
    if len(gpu_candidates) == 1:
        strategy = tf2.distribute.OneDeviceStrategy(device="/gpu:0")
    elif len(gpu_candidates) > 1:
        strategy = tf2.distribute.MirroredStrategy()

    return strategy

