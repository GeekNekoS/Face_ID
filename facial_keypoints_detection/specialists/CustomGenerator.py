import numpy as np


class CustomGenerator:

    def __init__(self, flip_indices=None, seed=0):
        if flip_indices is None:
            flip_indices = [(0, 2), (1, 3),
                            (4, 8), (5, 9), (6, 10), (7, 11),
                            (12, 16), (13, 17), (14, 18), (15, 19),
                            (22, 24), (23, 25)]
        self.flip_indices = flip_indices
        self.seed = seed

    def hflip(self, X, y, flip_indices):
        """""
        flip X, y horizontally (flip_indices is needed to swap landmarks)
        """""
        X_hflip = X[:, :, ::-1, :]
        y_hflip = y.copy()
        y_hflip[:, ::2] = 96 - y_hflip[:, ::2]
        for idx1, idx2 in flip_indices:
            y_hflip[:, [idx1, idx2]] = y_hflip[:, [idx2, idx1]]
        return X_hflip, y_hflip

    def adjust_brightness(self, X, y=None, brightness=1):
        """""
        adjust X brightness according to brightness multiplier
        """""
        X_bright = np.clip(X * brightness, X.min(), X.max())
        return X_bright, y

    def adjust_shift(self, X, y, shift=(0, 0)):
        """""
        shift X, y according to shift=[shift_x, shift_y]
        """""
        ox, oy = shift
        y_shift = y.copy()
        y_shift[:, ::2] += ox
        y_shift[:, 1::2] += oy
        y_shift = np.clip(y_shift, 0, 96)

        X_shift = np.pad(X, mode='constant', pad_width=((0, 0),
                                                        (abs(oy), abs(oy)),
                                                        (abs(ox), abs(ox)),
                                                        (0, 0)))
        if ox >= 0 and oy >= 0:
            X_shift = X_shift[:, :96, :96, :]
        elif ox >= 0 and oy <= 0:
            X_shift = X_shift[:, 2 * abs(oy):, :96, :]
        elif ox <= 0 and oy >= 0:
            X_shift = X_shift[:, :96, 2 * abs(ox):, :]
        elif ox <= 0 and oy <= 0:
            X_shift = X_shift[:, 2 * abs(oy):, 2 * abs(ox):, :]
        return X_shift, y_shift

    def make_batches(self, X, y, batch_size):
        """""
        split X, y into batches of size batch_size
        """""
        num_batches = len(X) // batch_size + 1
        X_batches = [X[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]
        y_batches = [y[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]
        return X_batches, y_batches

    def generate(self, X, y, batch_size=32, max_shift=10, max_bright=0.5, shuffle=True):
        """""
        generate new images from X, y
        """""
        np.random.seed(self.seed)
        X_batches, y_batches = self.make_batches(X, y, batch_size)

        X_new, y_new = X.copy(), y.copy()
        for X_batch, y_batch in zip(X_batches, y_batches):
            # batch transformation parameters
            shift = np.random.randint(-max_shift, max_shift, size=2)
            brightness = np.random.uniform(1 - max_bright, max_bright)
            # transformation
            X_hflip, y_hflip = self.hflip(X_batch, y_batch, self.flip_indices)
            X_shift, y_shift = self.adjust_shift(X_batch, y_batch, shift=shift)
            X_bright, y_bright = self.adjust_brightness(X_batch, y_batch, brightness=brightness)
            # stack toghether
            X_gen = np.concatenate([X_hflip, X_shift, X_bright])
            y_gen = np.concatenate([y_hflip, y_shift, y_bright])
            X_new = np.concatenate([X_new, X_gen])
            y_new = np.concatenate([y_new, y_gen])

        if shuffle:
            permutation = np.random.permutation(len(X_new))
            X_new = X_new[permutation, :, :, :]
            y_new = y_new[permutation, :]
        return X_new, y_new