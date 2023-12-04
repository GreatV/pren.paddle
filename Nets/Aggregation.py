import paddle
from Nets.EfficientNet_utils import MemoryEfficientSwish


class GCN(paddle.nn.Layer):
    def __init__(self, d_in, n_in, d_out=None, n_out=None, dropout=0.1):
        super().__init__()
        if d_out is None:
            d_out = d_in
        if n_out is None:
            n_out = n_in
        self.conv_n = paddle.nn.Conv1D(
            in_channels=n_in, out_channels=n_out, kernel_size=1
        )
        self.linear = paddle.nn.Linear(in_features=d_in, out_features=d_out)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.activation = MemoryEfficientSwish()

    def forward(self, x):
        """
        :param x: [b, nin, din]
        :return: [b, nout, dout]
        """
        x = self.conv_n(x)
        x = self.dropout(self.linear(x))
        return self.activation(x)


class PoolAggregate(paddle.nn.Layer):
    def __init__(self, n_r, d_in, d_middle=None, d_out=None):
        super().__init__()
        if d_middle is None:
            d_middle = d_in
        if d_out is None:
            d_out = d_in
        self.d_in = d_in
        self.d_middle = d_middle
        self.d_out = d_out
        self.activation = MemoryEfficientSwish()
        self.n_r = n_r
        self.aggs = self.build_aggs()
        self.pool = paddle.nn.AdaptiveAvgPool2D(output_size=1)

    def build_aggs(self):
        aggs = paddle.nn.LayerList()
        for i in range(self.n_r):
            aggs.append(
                paddle.nn.Sequential(
                    paddle.nn.Conv2D(
                        in_channels=self.d_in,
                        out_channels=self.d_middle,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False,
                    ),
                    paddle.nn.BatchNorm2D(
                        num_features=self.d_middle, epsilon=0.001, momentum=1 - 0.01
                    ),
                    self.activation,
                    paddle.nn.Conv2D(
                        in_channels=self.d_middle,
                        out_channels=self.d_out,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False,
                    ),
                    paddle.nn.BatchNorm2D(
                        num_features=self.d_out, epsilon=0.001, momentum=1 - 0.01
                    ),
                )
            )
        return aggs

    def forward(self, x):
        """
        :param x: [b, din, h, w]
        :return: [b, n_r, dout]
        """
        b = x.shape[0]
        out = []
        fmaps = []
        for agg in self.aggs:
            y = agg(x)
            p = self.pool(y)
            fmaps.append(y)
            out.append(p.reshape((b, 1, -1)))
        out = paddle.concat(x=out, axis=1)
        return out


class WeightAggregate(paddle.nn.Layer):
    def __init__(self, n_r, d_in, d_middle=None, d_out=None):
        super().__init__()
        if d_middle is None:
            d_middle = d_in
        if d_out is None:
            d_out = d_in
        self.conv_n = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=d_in,
                out_channels=d_in,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=d_in, epsilon=0.001, momentum=1 - 0.01),
            MemoryEfficientSwish(),
            paddle.nn.Conv2D(
                in_channels=d_in, out_channels=n_r, kernel_size=1, bias_attr=False
            ),
            paddle.nn.BatchNorm2D(num_features=n_r, epsilon=0.001, momentum=1 - 0.01),
            paddle.nn.Sigmoid(),
        )
        self.conv_d = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=d_in,
                out_channels=d_middle,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(
                num_features=d_middle, epsilon=0.001, momentum=1 - 0.01
            ),
            MemoryEfficientSwish(),
            paddle.nn.Conv2D(
                in_channels=d_middle, out_channels=d_out, kernel_size=1, bias_attr=False
            ),
            paddle.nn.BatchNorm2D(num_features=d_out, epsilon=0.001, momentum=1 - 0.01),
        )
        self.n_r = n_r
        self.d_out = d_out

    def forward(self, x):
        """
        :param x: [b, d_in, h, w]
        :return: [b, n_r, dout]
        """
        b = x.shape[0]
        hmaps = self.conv_n(x)
        fmaps = self.conv_d(x)
        r = paddle.bmm(
            x=hmaps.reshape((b, self.n_r, -1)),
            y=fmaps.reshape((b, self.d_out, -1)).transpose(perm=[0, 2, 1]),
        )
        return r
