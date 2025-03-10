
## 1 Install Enviornment

1.1 Create and activate conda virtual environment / 创建 和 激活 conda虚拟环境
```
conda create --name (env_name) python=3.10 # example: conda create --name fl python=3.10
conda activate (env_name) # example: conda activate fl 
```

1.2 install pytorch / 安装pytorch
```
pip3 install torch torchvision torchaudio #  It is recommended to search for the code on the official pytorch website. / 建议官方寻找代码
```
optional:
This code requires tensorboardX to be installed in order to run
You can also disable tensorboardX
```
pip install tensorboardX
```
Besides,

2025-03-09 The conda environment is exported as environment.yml


## 2 Quick Start

you can enter the code below to run the federated learning demo. 

```
python main.py
```



## Some final words
---
en: If this repository has been helpful to you, could you please give it a star? It would be a great honor and I would be very appreciated! By the way, you are welcome to fork this repository, but please indicate the source in code or others.

zh: 如果这个仓库对您有所帮助，可以给这个项目点赞吗？这将是我莫大的荣幸，不胜感激！顺便说一句，欢迎您复制这个资源库，但请在代码或其他地方注明来源。

