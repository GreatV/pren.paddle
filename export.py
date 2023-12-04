import paddle
from Configs.trainConf import configs
from Nets.model import Model

if __name__ == "__main__":
    model = Model(configs.net)
    model.eval()
    x = paddle.randn(shape=[1, 3, 320, 320])
    try:
        input_spec = list(
            paddle.static.InputSpec.from_tensor(paddle.to_tensor(t)) for t in (x,)
        )
        paddle.jit.save(model, input_spec=input_spec, path="./model")
        print("[JIT] paddle.jit.save successed.")
        exit(0)
    except Exception as e:
        print("[JIT] paddle.jit.save failed.")
        raise e
