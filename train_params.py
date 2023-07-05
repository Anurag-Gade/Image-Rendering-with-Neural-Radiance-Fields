def train_par():
    epochs = 175
    alpha = 0.00001
    batch_size = 32
    momentum = 0.5
    up_scale = 4
    device = 'cuda'
    return epochs, alpha, batch_size, momentum, up_scale, device