import paddle


class strLabelConverter(object):
    """Convert between str and label.

    Args:
        alphabet (str): set of the possible characters.
    """

    def __init__(self, alphabet, maxT=25):
        self.alphabet = alphabet
        self.maxT = maxT
        self.dict = {}
        self.dict["<pad>"] = 0
        self.dict["<eos>"] = 1
        self.dict["<unk>"] = 2
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i + 3
        self.chars = list(self.dict.keys())

    def encode(self, text):
        """
        Args:
            text (list of str): texts to convert.
        Returns:
            torch.IntTensor targets: [b, L]
        """
        tars = []
        for s in text:
            tar = []
            for c in s:
                if c in self.dict.keys():
                    tar.append(self.dict[c])
                else:
                    tar.append(self.dict["<unk>"])
            tars.append(paddle.to_tensor(data=tar, dtype="int64"))
        b = len(tars)
        targets = self.dict["<pad>"] * paddle.ones(shape=[b, self.maxT])
        for i in range(b):
            targets[i][: len(tars[i])] = tars[i]
            targets[i][len(tars[i])] = self.dict["<eos>"]
        return targets.astype(dtype="int64")

    def decode(self, t):
        texts = [self.chars[i] for i in t]
        return "".join(texts)


class ScheduledOptim:
    """A wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, init_lr, milestones, gammas):
        self._optimizer = optimizer
        self.lr = init_lr
        self.milestones = milestones
        self.gammas = gammas

    def step(self):
        """Step with the inner optimizer"""
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self._optimizer.clear_grad()

    def update_lr(self, epoch):
        """Learning rate scheduling per step"""
        if epoch in self.milestones:
            self.lr *= self.gammas[self.milestones.index(epoch)]
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = self.lr
