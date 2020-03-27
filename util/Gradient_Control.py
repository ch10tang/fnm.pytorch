def enable_gradients(net):
    for p in net.parameters():
        p.requires_grad = True


def disable_gradients(net):
    for p in net.parameters():
        p.requires_grad = False