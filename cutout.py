import numpy as np
class Cutout(object):
    def __init__(self, n_holes, max_height, max_width, min_height=None, min_width=None,
                 fill_value_mode='zero', p=0.5):
        self.n_holes = n_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_width = min_width if min_width is not None else max_width
        self.min_height = min_height if min_height is not None else max_height
        self.fill_value_mode = fill_value_mode  # 'zero' 'one' 'uniform'
        self.p = p
        assert 0 < self.min_height <= self.max_height
        assert 0 < self.min_width <= self.max_width
        assert 0 < self.n_holes
        assert self.fill_value_mode in ['zero', 'one', 'uniform']

    def __call__(self, img, semantic_label):
        if np.random.rand() > self.p:
            return img, semantic_label

        h = img.shape[0]
        w = img.shape[1]

        if self.fill_value_mode == 'zero':
            f = np.zeros
            param = {'shape': (h, w, 3)}
        elif self.fill_value_mode == 'one':
            f = np.one
            param = {'shape': (h, w, 3)}
        else:
            f = np.random.uniform
            param = {'low': 0, 'high': 255, 'size': (h, w, 3)}

        mask = np.ones((h, w, 3), dtype=np.int32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            h_l = np.random.randint(self.min_height, self.max_height + 1)
            w_l = np.random.randint(self.min_width, self.max_width + 1)

            y1 = np.clip(y - h_l // 2, 0, h)
            y2 = np.clip(y + h_l // 2, 0, h)
            x1 = np.clip(x - w_l // 2, 0, w)
            x2 = np.clip(x + w_l // 2, 0, w)

            mask[y1:y2, x1:x2, :] = 0

        img = np.where(mask, img, f(**param))
        semantic_label = np.where(mask[..., 0], semantic_label, np.zeros((h, w)))
        return np.uint8(img), np.uint8(semantic_label)

