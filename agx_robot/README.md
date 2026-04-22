# Agx_robot

## Getting started

本Repo为松灵机器人的相关代码

1、在使用本项目之前，请务必使用conda等相关工具控制环境。在main的主分支当中，含有当前能够在工作环境中运行的环境即[aloha.yml](http://172.18.0.12/XD/agx_robot/-/blob/main/aloha.yml)。即默认的conda环境是aloha。但请注意，随后的分支的当中不同的环境需要不同的注明

2、运行agx_robot/aloha-devel/act/目录下[train.py](http://172.18.0.12/XD/agx_robot/-/blob/main/aloha-devel/act/train.py?ref_type=heads)文件,实现训练，训练的部分参数依赖agx_robot/aloha-devel/act/目录下[policy.py](http://172.18.0.12/XD/agx_robot/-/blob/main/aloha-devel/act/policy.py?ref_type=heads)。

3、在默认的训练和推理当中注意robimimic的路径和境变量等相关问题

4、未完待续。相关论文和项目可供以下几个链接参考，部分环境搭建，NAS传输工作可以参考[WIKI](https://q0aw3ffiknk.feishu.cn/docx/UvFxdZKTEo1N8XxOpyZcYk5Mndg?preview_comment_id=7478212976958390300)

https://mobile-aloha.github.io/

https://tonyzhaozh.github.io/aloha/

https://aloha-2.github.io/

## Vision control

如果你已有本地仓库，可使用以下命令推送到远程：

```bash
cd existing_repo

# SSH方式（推荐）
git remote add origin git@172.18.0.12:XD/agx_robot.git

# 或使用HTTP方式
# git remote add origin http://172.18.0.12/XD/agx_robot.git

git branch -M main
git push -uf origin main
```


V1版本：基础代码

V1.1版本：以dev/feature-siyuan-v1.1为主，在该分支中修改了深度的参数能够实现深度的训练和推理

V1.2版本：在基础模型能够训练的前提上，开发ACT模型的能力


## Integrate with your tools



## Collaborate with your team



## Test and Deploy



***


## Name


## Description



## Badges



## Visuals



## Installation



## Usage



## Support


## Roadmap



## Contributing



## Authors and acknowledgment



## License

For open source projects, say how it is licensed.

## Project status


