import torch
import slim

sparse_maps = [
    slim.LassoLinear,
    slim.LassoLinearRELU,
    slim.ButterflyLinear,
]

spectral_maps = [
    slim.PerronFrobeniusLinear,
    slim.SVDLinear,
    slim.SVDLinearLearnBounds,
    slim.SpectralLinear,
]

loss_fn = torch.nn.functional.mse_loss


class SparseModel(slim.LinearBase):
    def __init__(self, input_dim=64, output_dim=64, sparsity=0.75):
        super().__init__(input_dim, output_dim)
        torch.nn.init.sparse_(self.weight, sparsity)

    def effective_W(self):
        return self.weight.T


def generate_data(true_model, nsamples=20000, batch_size=2000):
    input_dim, output_dim = true_model.weight.shape
    X, y = [], []
    for _ in range(0, nsamples, batch_size):
        with torch.no_grad():
            X_batch = torch.rand(batch_size, input_dim)
            y_batch = true_model(X_batch)
        X += [X_batch]
        y += [y_batch]

    return torch.cat(X, dim=0), torch.cat(y, dim=0)


def train(X_true, y_true, model, optimizer, epochs=1000, batch_size=2000):
    nsamples = X_true.shape[0]
    for epoch in range(epochs):
        rand_inds = torch.randperm(nsamples)
        X_shuffle = X_true[rand_inds]
        y_shuffle = y_true[rand_inds]

        for i in range(0, nsamples, batch_size):
            optimizer.zero_grad()

            X_batch = X_shuffle[i:i+batch_size]
            y_batch = y_shuffle[i:i+batch_size]
            
            y_pred = model(X_batch)

            loss = loss_fn(y_batch, y_pred) + model.reg_error()

            loss.backward()
            optimizer.step()

        print(f'\r  [epoch {epoch+1}/{epochs}] ', end='', flush=True)


def evaluate(X_true, y_true, model, batch_size=2000):
    nsamples = X_true.shape[0]
    loss_acc = 0.

    with torch.no_grad():
        for i in range(0, nsamples, batch_size):
            y_pred = model(X_true[i:i+batch_size])
            loss_acc += loss_fn(y_true[i:i+batch_size], y_pred)

    return loss_acc.item()


def weight_divergence(true_model, test_model):
    return torch.norm(true_model.effective_W() - test_model.effective_W(), p=1).item()


if __name__ == '__main__':

    # test sparsity-constrained maps
    print('Testing sparse linear maps:')
    true_model = SparseModel()
    X, y = generate_data(true_model)

    for layer in [*sparse_maps, slim.Linear]:
        print(f'  {layer.__name__}:')
        test_model = layer(64, 64)
        optimizer = torch.optim.AdamW(test_model.parameters(), lr=0.001)

        train(X, y, test_model, optimizer)
        loss = evaluate(X, y, test_model)
        wdiv = weight_divergence(true_model, test_model)

        print(f'Loss = {loss:.4e}, Weight divergence = {wdiv:.4e}\n')

    # TODO: test spectrally-constrained maps

