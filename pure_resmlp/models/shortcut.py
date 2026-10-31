from torch import nn

class ShortcutAblation(nn.Module):

    def __init__(self, network, *, complex_input, residual=False):
        super().__init__()
        self.network = network
        self.complex_input = complex_input
        self.residual = residual

    def forward(self, x):
        net = self.network
        h = net.input_fc(x)
        if not self.complex_input:
            h = net.relu(h)
        for block in (net.block1, net.block2):
            branch = block.fc2(block.pu(block.fc1(h)))
            h = h + branch if self.residual else branch
        output = net.out_fc(h)
        return net.readout(output) if self.complex_input else output
