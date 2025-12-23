# 远程运行 openpi 模型

我们提供了在远程机器上运行 openpi 模型的工具。这能让推理跑在更强的 GPU 上，同时将机器人与策略运行环境解耦（也能避免机器人软件的依赖冲突）。

## 启动远程策略服务器

直接运行下面的命令即可启动策略服务器：

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

`env` 参数决定加载哪一个 π₀ 检查点。脚本内部会执行类似的命令，你也可以手动使用它来为自己训练的检查点启动服务器（下面以 DROID 环境为例）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

这会启动一个策略服务器，加载由 `config` 与 `dir` 指定的策略。服务器监听的端口默认是 8000，可通过参数修改。

## 在机器人端查询远程策略服务器

我们提供了依赖极少的客户端工具，方便嵌入任何机器人代码库。

首先在机器人环境里安装 `openpi-client` 包：

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

然后在代码里这样调用远程策略服务器：

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# 在 episode 循环外初始化策略客户端。
# 指向策略服务器的 host 与 port（默认 localhost:8000）。
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # 在 episode 循环内构造观测。
    # 在客户端侧先调整图像大小以减少带宽/延迟，并返回 uint8。
    # 预训练 pi0 模型常用的 resize 尺寸是 224。
    # 本体状态 state 可以不做归一化，服务器会处理。
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # 调用策略服务器获取当前观测下的动作块。
    # 返回形状为 (action_horizon, action_dim)。
    # 通常每 N 步调用一次策略，其余步数直接执行预测的动作块即可。
    action_chunk = client.infer(observation)["actions"]

    # 在环境中执行动作。
    ...

```

`host` 与 `port` 指定远程策略服务器的地址和端口；你也可以在机器人程序里用命令行参数或硬编码。`observation` 是包含观测和提示词的字典，其键需要与所服务策略的输入约定一致。不同环境如何构造该字典，可参考 [simple client 示例](../examples/simple_client/main.py) 中的具体代码。
