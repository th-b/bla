# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# See the LICENSE file for more details.

import cv2
import cmapy
import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt

def visualize(wrapper,images=5, file=None):
    
    if isinstance(images, int):
        images = [imgs[0] for imgs, _ in wrapper.data_loader.dataset['test'].take(images)]
    
    plt.figure(figsize=(4 * 2,len(images) * 2))
    
    a = -1
    for img in images:
        a += 1
        pred, mask, prob = wrapper.predict_x_hard(wrapper.data_loader.preprocess_image(img[None]))

        HEIGHT = img.shape[0]
        WIDTH = img.shape[1]

        pred_cl = np.argmax(pred[0])

        ax = plt.subplot(len(images),4,1 +4*a)
        plt.title('Input')
        plt.axis('off')
        plt.imshow(img/255.)

        ax = plt.subplot(len(images),4,2 +4*a)
        plt.title('Soft')
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        plt.imshow(np.squeeze(prob))   
        plt.colorbar(fraction=0.046, pad=0.04)

        ax = plt.subplot(len(images),4,3 +4*a)
        plt.axis('off')
        plt.title('Hard')
        plt.imshow(np.squeeze(mask)/mask.max())   
    
        plt.colorbar(fraction=0.046, pad=0.04)

        ax = plt.subplot(len(images),4,4 +4*a)
        plt.axis('off')
        plt.title('Visualization')

        h = np.squeeze(mask)
        h /= h.max()
        h = (h * 255).astype('uint8')    
        h = cv2.resize(h, (HEIGHT,WIDTH))
        h = cv2.applyColorMap(h, cv2.COLORMAP_TURBO)
        h = h[:,:,::-1]

        plt.imshow((img * .5 + h * .5)/255)
        
    plt.tight_layout()
    
    if file is not None:
        plt.savefig(file)
    
    plt.show()