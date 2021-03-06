import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

from strixbee.gradcam.gradcam import GradCam
# from CIFAR_10.utils import denormalize
from strixbee.utils.plots import convert_image_np
from strixbee.utils.data_iter import get_data,get_data_stats

def denormalize(tensor,train,test, mean=None, std=None):
    # train,test = get_data()
    # mean,std = get_data_stats(train,test,img_norm_typ ='train',plot = False)
    
    single_img = False
    if tensor.ndimension() == 3:
      single_img = True
      tensor = tensor[None,:,:,:]

    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    ret = tensor.mul(std).add(mean)
    return ret[0] if single_img else ret

class VisualizeCam(object):

	def __init__(self, model, classes, target_layers):
		super(VisualizeCam, self).__init__()
		self.model = model
		self.classes = classes
		self.target_layers = target_layers
		self.device = next(model.parameters()).device

		self.gcam = GradCam(model, target_layers, len(classes))
		self.train,self.test = get_data()
		self.mean,self.std = get_data_stats(self.train,self.test,img_norm_typ ='train',plot = False)

	def visualize_cam(self, mask, img):
	    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
	    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
	    b, g, r = heatmap.split(1)
	    heatmap = torch.cat([r, g, b])
	    
	    result = heatmap+img.cpu()
	    result = result.div(result.max()).squeeze()
	    return heatmap, result

	def plot_heatmaps(self, img_data, target_class, img_name):
		fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 4),
			subplot_kw={'xticks': [], 'yticks': []})
		fig.suptitle('GradCam at different conv layers for the class: %s' % 
			target_class, fontsize=13, weight='medium', y=1.05)

		for ax, data in zip(axs.flat, img_data):
			img = data["img"]
			img = img.cpu().numpy()
			maxValue = np.amax(img)
			minValue = np.amin(img)
			img = np.clip(img, 0, 1)
			img = img/np.amax(img)
			img = np.clip(img, 0, 1)
			ax.imshow(np.transpose(img, (1, 2, 0)))
			ax.set_title("%s" % (data["label"]))


	def __call__(self, images, target_layers, target_inds=None, metric=""):
		masks_map, pred = self.gcam(images, target_layers, target_inds)
		for i in range(len(images)):
			img = images[i]
			results_data = [{
				"img": denormalize(img,self.train,self.test,self.mean,self.std),
				"label": "Result:"
			}]
			heatmaps_data = [{
				"img": denormalize(img,self.train,self.test,self.mean,self.std),
				"label": "Heatmap:"
			}]
			for layer in target_layers:
				mask = masks_map[layer][i]
				heatmap, result = self.visualize_cam(mask, img)
				results_data.append({
					"img": result,
					"label": layer
				})
				heatmaps_data.append({
					"img": heatmap,
					"label": layer
				})
			pred_class = self.classes[pred[i][0]]
			fname = "gradcam_%s_%s_%s.png" % (metric, i, pred_class)
			self.plot_heatmaps(results_data+heatmaps_data, pred_class, fname)