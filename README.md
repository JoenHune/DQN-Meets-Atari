# DQN-Meets-Atari

## 环境配置

**Mac OS X Big Sur**

```
conda create -n atari python==3.7
conda activate atari

brew install ffmpeg

pip install tensorflow-cpu==2.2.0 -i https://pypi.douban.com/simple/ \
    atari_py gym 'gym[atari]' \
    PyOpenGL PyOpenGL_accelerate pyglet==1.5.11 \
    numpy 
```

**Linux Ubuntu**

```
conda create -n atari python==3.7
conda activate atari

sudo apt install ffmpeg

conda install cudatoolkit=10.1 cudnn=7.6.5

pip install tensorflow==2.2.0 -i https://pypi.douban.com/simple/ \
    atari_py gym 'gym[atari]' \
    PyOpenGL PyOpenGL_accelerate pyglet \
    numpy 
```

## 参考资料

1. Spencer Dixon.Deep Q-Learning for Atari Games[EB/OL].https://spencerldixon.github.io/2019-01-01/deep-q-learning-for-atari-games/, 2019-01-01.