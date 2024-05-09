import sys
import torch
from tqdm import tqdm as tqdm
import numpy as np
from segmentation_models_pytorch import utils as smp_utils
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import wandb


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = smp_utils.meter.AverageValueMeter()
        metrics_meters = {metric.__name__: smp_utils.meter.AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            i = 0
            for data in iterator:
                x, y, mask, x_left, y_left, mask_left, x_right, y_right, mask_right = data
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                x_left, y_left, mask_left = x_left.to(self.device), y_left.to(self.device), mask_left.to(self.device)
                x_right, y_right, mask_right = x_right.to(self.device), y_right.to(self.device), mask_right.to(self.device)
                # if i % 10 == 0:
                #     np.save('model_inputs_viz/'+ self.stage_name + '_x_{}.npy'.format(i), x.cpu().detach().numpy())
                #     np.save('model_inputs_viz/'+ self.stage_name + '_mask_{}.npy'.format(i), mask.cpu().detach().numpy())
                # i+=1
                loss, y_pred = self.batch_update(x, y, mask, x_left, y_left, mask_left, x_right, y_right, mask_right)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # for metric computation, use masked prediction and GT
                if self.loss.mode == 'multiclass':
                    # need to activate the prediction for metric computation
                    y_pred = y_pred.log_softmax(dim=1).exp()
                    # make ground truth one-hot
                    # N, 1, H, W -> N, C, H, W
                    y = F.one_hot(y.view(y.shape[0], y.shape[2], y.shape[3]), num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()
                    # multiply with mask to ignore old clicks
                    y_pred *= mask 
                    y *= mask
                    # exclude background class for metric calculation+
                    # sometimes this means for a pixel, GT/pred can be 0,0 which is fine
                    y_pred = y_pred[:, :2,...]
                    y = y[:, :2,...]
                else:
                    # need to activate the prediction for metric computation
                    y_pred = F.logsigmoid(y_pred).exp()
                    y_pred = y_pred*mask
                    y = y*mask

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, mask, x_left, y_left, mask_left, x_right, y_right, mask_right):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        prediction_left = self.model.forward(x_left)
        prediction_right = self.model.forward(x_right)
        #prediction is the aggregation of the three predictions
        #UNSURE ABOUT THIS PART
        prediction_agg  = prediction + prediction_left + prediction_right
        # if self.loss.mode == 'multiclass':
        loss = self.loss(prediction, y, mask)
        loss_left = self.loss(prediction_left, y_left, mask_left)
        loss_right = self.loss(prediction_right, y_right, mask_right)

        loss_agg = loss + loss_left + loss_right
        # else:
        #     loss = self.loss(prediction*mask, y*mask)
        loss_agg.backward()
        self.optimizer.step()
        return loss_agg, prediction_agg


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, mask, x_left, y_left, mask_left, x_right, y_right, mask_right):
        with torch.no_grad():
            prediction = self.model.forward(x)
            prediction_left = self.model.forward(x_left)
            prediction_right = self.model.forward(x_right)
            #prediction is the aggregation of the three predictions
            #UNSURE ABOUT THIS PART
            prediction_agg  = prediction + prediction_left + prediction_right
            # if self.loss.mode == 'multiclass':
            loss = self.loss(prediction, y, mask)
            loss_left = self.loss(prediction_left, y_left, mask_left)
            loss_right = self.loss(prediction_right, y_right, mask_right)

            loss_agg = loss + loss_left + loss_right
            # else:
            #     loss = self.loss(prediction*mask, y*mask)
        return loss_agg, prediction_agg
    

def viz_inputs_with_gaze_overlaid(img_inputs, rgb, gt, pr):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    gaze_heatmap = np.array(img_inputs[-1])[4:-4, :]*255

    for i in range(len(img_inputs)-1):
        # Plot the images
        axes[0, i].imshow(img_inputs[i])
        axes[0, i].axis('off')


    axes[0, 2].imshow(gaze_heatmap)
    axes[0, 2].axis('off')

    axes[1, 0].imshow(cv2.addWeighted(np.array(rgb), 1, 
                                      255*cv2.merge((gaze_heatmap, np.zeros_like(gaze_heatmap), np.zeros_like(gaze_heatmap))), 
                                      0.5, 0))    
    # print(blended_array.shape)
    # axes[1, 0].imshow()
    axes[1, 0].set_title('RGB + gaze heatmap')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gt)
    axes[1, 1].set_title("Ground Truth")
    axes[1, 1].axis('off')
    axes[1, 2].imshow(pr)
    axes[1, 2].set_title("Prediction")
    axes[1, 2].axis('off')

    
    plt.tight_layout()
    im = wandb.Image(fig)

    # im = Image.frombytes('RGB',fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    # im.save("viz_output.png")
    plt.close(fig)

    return im


class VizEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True, args=None):
        self.args = args
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="viz",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def run(self, dataset):
        figs = []
        with tqdm(
            range(len(dataset)),
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for j in iterator:
                if j % self.args.image_save_freq == 0:
                    image, gt_mask, mask, image_left, gt_mask_left, mask_left, image_right, gt_mask_right, mask_right = dataset[j]
                    print(image.shape)
                    frame_num = dataset.index_mapping[j]
                    try:
                        rgb_image = Image.open(dataset.images_dir / 'rgb_output' / ('%.6d.png' % frame_num)).convert('RGB')
                    except:
                        print(dataset.images_dir / 'rgb_output' / ('%.6d.png' % frame_num))
                        continue
                    
                    try:
                        rgb_image_left = Image.open(dataset.images_dir / 'rgb_output_left' / ('%.6d.png' % frame_num)).convert('RGB')
                    except:
                        print(dataset.images_dir / 'rgb_output_left' / ('%.6d.png' % frame_num))
                        continue
                    
                    try:
                        rgb_image_right = Image.open(dataset.images_dir / 'rgb_output_left' / ('%.6d.png' % frame_num)).convert('RGB')
                    except:
                        print(dataset.images_dir / 'rgb_output_right' / ('%.6d.png' % frame_num))
                        continue
                    
                    
                    if self.args.instseg_channels == 1:
                        im0 = Image.fromarray(np.uint8(image.numpy()[0]*255)) 
                        im1 = Image.fromarray(np.uint8(mask.numpy()[0]*255))
                        im2 = Image.fromarray(np.uint8(image.numpy()[1]*255))
                        
                        im0_left = Image.fromarray(np.uint8(image_left.numpy()[0]*255)) 
                        im1_left = Image.fromarray(np.uint8(mask_left.numpy()[0]*255))
                        im2_left = Image.fromarray(np.uint8(image_left.numpy()[1]*255))
                        
                        im0_right = Image.fromarray(np.uint8(image_right.numpy()[0]*255)) 
                        im1_right = Image.fromarray(np.uint8(mask_right.numpy()[0]*255))
                        im2_right = Image.fromarray(np.uint8(image_right.numpy()[1]*255))
                    else:
                        im0 = Image.fromarray(np.uint8(image.numpy()[1]*255)) # take G channel instead of R for input segmentation
                        im1 = Image.fromarray(np.uint8(mask.numpy()[0]*255)) 
                        im2 = Image.fromarray(np.uint8(image.numpy()[2]*255))
                        
                        im0_left = Image.fromarray(np.uint8(image_left.numpy()[1]*255))
                        im1_left = Image.fromarray(np.uint8(mask_left.numpy()[0]*255)) 
                        im2_left = Image.fromarray(np.uint8(image_left.numpy()[2]*255))
                        
                        im0_right = Image.fromarray(np.uint8(image_right.numpy()[1]*255))
                        im1_right = Image.fromarray(np.uint8(mask_right.numpy()[0]*255)) 
                        im2_right = Image.fromarray(np.uint8(image_right.numpy()[2]*255))
                    
                    pr_mask = self.model.predict(image.to(self.device).unsqueeze(0))
                    pr_mask_left = self.model.predict(image_left.to(self.device).unsqueeze(0))
                    pr_mask_right= self.model.predict(image_right.to(self.device).unsqueeze(0))
                    if self.args.seg_mode == 'multiclass':
                        pr_mask = F.log_softmax(pr_mask, dim=1).exp().squeeze().cpu().numpy()
                        pr_mask = np.argmax(pr_mask, axis=0)
                        pr_fin = np.zeros([pr_mask.shape[0], pr_mask.shape[1], 3])                
                        pr_fin[pr_mask == 0] = np.array([0, 255, 0]) # green for aware
                        pr_fin[pr_mask == 1] = np.array([255, 0, 0]) # red for unaware
                        gt_mask = gt_mask.squeeze()
                        gt_fin = np.zeros([gt_mask.shape[0], gt_mask.shape[1], 3])
                        gt_fin[gt_mask == 0] = np.array([0, 255, 0])
                        gt_fin[gt_mask == 1] = np.array([255, 0, 0])
                        gt_fin[gt_mask == 2] = np.array([0, 0, 0])
                        
                        pr_mask_left = F.log_softmax(pr_mask_left, dim=1).exp().squeeze().cpu().numpy()
                        pr_mask_left = np.argmax(pr_mask_left, axis=0)
                        pr_fin_left = np.zeros([pr_mask_left.shape[0], pr_mask_left.shape[1], 3])                
                        pr_fin_left[pr_mask_left == 0] = np.array([0, 255, 0]) # green for aware
                        pr_fin_left[pr_mask_left == 1] = np.array([255, 0, 0]) # red for unaware
                        gt_mask_left = gt_mask_left.squeeze()
                        gt_fin_left = np.zeros([gt_mask_left.shape[0], gt_mask_left.shape[1], 3])
                        gt_fin_left[gt_mask_left == 0] = np.array([0, 255, 0])
                        gt_fin_left[gt_mask_left == 1] = np.array([255, 0, 0])
                        gt_fin_left[gt_mask_left == 2] = np.array([0, 0, 0])
                        
                        pr_mask_right = F.log_softmax(pr_mask_right, dim=1).exp().squeeze().cpu().numpy()
                        pr_mask_right = np.argmax(pr_mask_right, axis=0)
                        pr_fin_right = np.zeros([pr_mask_right.shape[0], pr_mask_right.shape[1], 3])                
                        pr_fin_right[pr_mask_right == 0] = np.array([0, 255, 0]) # green for aware
                        pr_fin_right[pr_mask_right == 1] = np.array([255, 0, 0]) # red for unaware
                        gt_mask_right = gt_mask_right.squeeze()
                        gt_fin_right = np.zeros([gt_mask_right.shape[0], gt_mask_right.shape[1], 3])
                        gt_fin_right[gt_mask_right == 0] = np.array([0, 255, 0])
                        gt_fin_right[gt_mask_right == 1] = np.array([255, 0, 0])
                        gt_fin_right[gt_mask_right == 2] = np.array([0, 0, 0])

                    else:
                        pr_mask = (F.logsigmoid(pr_mask).exp().squeeze().cpu().numpy().round())
                        aware_pixels = pr_mask[0] > self.args.aware_threshold 
                        unaware_pixels = pr_mask[1] > self.args.unaware_threshold
                        pr_fin = np.zeros([pr_mask.shape[1], pr_mask.shape[2], 3])                
                        pr_fin[aware_pixels] = np.array([0, 255, 0])
                        pr_fin[unaware_pixels] = np.array([255, 0, 0])                                                            
                        gt_mask = (gt_mask.cpu().numpy().round())
                        aware_pixels = gt_mask[0] > self.args.aware_threshold 
                        unaware_pixels = gt_mask[1] > self.args.unaware_threshold 
                        gt_fin = np.zeros([gt_mask.shape[1], gt_mask.shape[2], 3])
                        gt_fin[unaware_pixels] = np.array([255, 0, 0])
                        gt_fin[aware_pixels] = np.array([0, 255, 0])
                        
                        pr_mask_left = (F.logsigmoid(pr_mask_left).exp().squeeze().cpu().numpy().round())
                        aware_pixels_left = pr_mask_left[0] > self.args.aware_threshold 
                        unaware_pixels_left = pr_mask_left[1] > self.args.unaware_threshold
                        pr_fin_left = np.zeros([pr_mask_left.shape[1], pr_mask_left.shape[2], 3])                
                        pr_fin_left[aware_pixels_left] = np.array([0, 255, 0])
                        pr_fin_left[unaware_pixels_left] = np.array([255, 0, 0])                                                            
                        gt_mask_left = (gt_mask_left.cpu().numpy().round())
                        aware_pixels_left = gt_mask_left[0] > self.args.aware_threshold 
                        unaware_pixels_left = gt_mask_left[1] > self.args.unaware_threshold 
                        gt_fin_left = np.zeros([gt_mask_left.shape[1], gt_mask_left.shape[2], 3])
                        gt_fin_left[unaware_pixels_left] = np.array([255, 0, 0])
                        gt_fin_left[aware_pixels_left] = np.array([0, 255, 0])
                        
                        pr_mask_right = (F.logsigmoid(pr_mask_right).exp().squeeze().cpu().numpy().round())
                        aware_pixels_right = pr_mask_right[0] > self.args.aware_threshold 
                        unaware_pixels_right = pr_mask_right[1] > self.args.unaware_threshold
                        pr_fin_right = np.zeros([pr_mask_right.shape[1], pr_mask_right.shape[2], 3])                
                        pr_fin_right[aware_pixels_right] = np.array([0, 255, 0])
                        pr_fin_right[unaware_pixels_right] = np.array([255, 0, 0])                                                            
                        gt_mask_right = (gt_mask_right.cpu().numpy().round())
                        aware_pixels_right = gt_mask_right[0] > self.args.aware_threshold 
                        unaware_pixels_right = gt_mask_right[1] > self.args.unaware_threshold 
                        gt_fin_right = np.zeros([gt_mask_right.shape[1], gt_mask_right.shape[2], 3])
                        gt_fin_right[unaware_pixels_right] = np.array([255, 0, 0])
                        gt_fin_right[aware_pixels_right] = np.array([0, 255, 0])
                                        
                    gt_im = Image.fromarray(np.uint8(gt_fin))
                    gt_im_left = Image.fromarray(np.uint8(gt_fin_left))
                    gt_im_right = Image.fromarray(np.uint8(gt_fin_right))
                    # gt_im.save(args.viz_output_path + "/gt_train_mask_" + str(j) + "_" + str(i) + ".png")
                    pr_im = Image.fromarray(np.uint8(pr_fin))
                    pr_im_left = Image.fromarray(np.uint8(pr_fin_left))
                    pr_im_right = Image.fromarray(np.uint8(pr_fin_right))
                    # pr_im = Image.fromarray(np.uint8(mask))
                    # pr_im.save(args.viz_output_path + "/pr_train_mask_" + str(j) + "_" + str(i) + ".png")
                    
                    # becomes mask
                    figs.append(viz_inputs_with_gaze_overlaid([im0, im1, im2], rgb_image, gt_im, pr_im))
                    figs.append(viz_inputs_with_gaze_overlaid([im0_left, im1_left, im2_left], rgb_image_left, gt_im_left, pr_im_left))
                    figs.append(viz_inputs_with_gaze_overlaid([im0_right, im1_right, im2_right], rgb_image_right, gt_im_right, pr_im_right))
                    if self.verbose:
                        s = str(j) + ", " + str(len(iterator))
                        iterator.set_postfix_str(s)

        return figs