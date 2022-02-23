import numpy as np

class ImageDataGenerator:
    def __init__(self, horizontal_flip=True, w_shift=0.15, h_shift=0.15, pad_mode="edge", random_shuffle=True):
        self.horizontal_flip = horizontal_flip
        self.w_shift = w_shift
        self.h_shift = h_shift
        self.pad_mode = pad_mode
        self.random_shuffle = random_shuffle
        
    def flip_lr(self, images):
        #return np.fliplr(images)
        return images[:, :, ::-1, :]
    
    def shift_im(self, images):
        images_shape = images.shape

        pad_h = int(images_shape[1]*self.h_shift/2)
        pad_w = int(images_shape[2]*self.w_shift/2)
        padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=self.pad_mode)

        h_rand_int = np.random.choice(a=range(pad_h*2), size=None)
        w_rand_int = np.random.choice(a=range(pad_w*2), size=None)

        return padded_images[:, h_rand_int:h_rand_int+images_shape[1], w_rand_int:w_rand_int+images_shape[2], :]
    
    def augment_images(self, images, labels):
        new_im_cache = 0
        images_shape = images.shape 
        if self.horizontal_flip:
            #idxes = np.random.randint(0, 2, (images_shape[0], 1, 1, 1))*np.ones_like(images)
            idxes = np.broadcast_to(np.random.randint(0, 2, (images_shape[0], 1, 1, 1)), images.shape)
            new_im = images*(1-idxes)+self.flip_lr(images)*idxes
            new_im_cache = 1
            #idxes = np.where(np.random.randint(0, 2, images_shape[0]))[0]
            #new_im[idxes] = self.flip_lr(new_im[idxes])
            
        if self.w_shift or self.h_shift:
            #idxes = np.where(np.random.randint(0, 2, images_shape[0]))[0]
            #new_im[idxes] = self.shift_im(new_im[idxes])
            if not new_im_cache:
                new_im = self.shift_im(images)
                new_im_cache = 1
            else:
                new_im = self.shift_im(new_im)
            
        if self.random_shuffle:
            p = np.random.permutation(len(images))
            if not new_im_cache:
                new_im = images
            return new_im[p], labels[p]
        return new_im, labels