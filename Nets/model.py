import paddle
from Nets.EfficientNet import EfficientNet
from Nets.Aggregation import GCN, WeightAggregate, PoolAggregate


class Model(paddle.nn.Layer):
    def __init__(self, net_configs):
        super(Model, self).__init__()
        d_model = net_configs.d_model
        n_r = net_configs.n_r
        max_len = net_configs.max_len
        n_class = net_configs.n_class
        dropout = net_configs.dropout
        self.cnn = EfficientNet.from_name("efficientnet-b3")
        self.agg_p1 = PoolAggregate(n_r, d_in=48, d_out=d_model // 3)
        self.agg_p2 = PoolAggregate(n_r, d_in=136, d_out=d_model // 3)
        self.agg_p3 = PoolAggregate(n_r, d_in=384, d_out=d_model // 3)
        self.agg_w1 = WeightAggregate(
            n_r=n_r, d_in=48, d_middle=4 * 48, d_out=d_model // 3
        )
        self.agg_w2 = WeightAggregate(
            n_r=n_r, d_in=136, d_middle=4 * 136, d_out=d_model // 3
        )
        self.agg_w3 = WeightAggregate(
            n_r=n_r, d_in=384, d_middle=4 * 384, d_out=d_model // 3
        )
        self.gcn_pool = GCN(
            d_in=d_model, n_in=n_r, d_out=d_model, n_out=max_len, dropout=dropout
        )
        self.gcn_weight = GCN(
            d_in=d_model, n_in=n_r, d_out=d_model, n_out=max_len, dropout=dropout
        )
        self.linear = paddle.nn.Linear(in_features=d_model, out_features=n_class)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, input):
        """
        :param input: images [b, 3, 64, 256]
        :return logits: [b, L, n_class] probs of characters (before softmax)
        """
        f3, f5, f7 = self.cnn(input)
        rp1 = self.agg_p1(f3)
        rp2 = self.agg_p2(f5)
        rp3 = self.agg_p3(f7)
        rp = paddle.concat(x=[rp1, rp2, rp3], axis=2)
        rw1 = self.agg_w1(f3)
        rw2 = self.agg_w2(f5)
        rw3 = self.agg_w3(f7)
        rw = paddle.concat(x=[rw1, rw2, rw3], axis=2)
        y1 = self.gcn_pool(rp)
        y2 = self.gcn_weight(rw)
        y = 0.5 * (y1 + y2)
        logits = self.linear(y)
        return logits
