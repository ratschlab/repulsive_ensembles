import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

import os
from matplotlib.colors import ListedColormap
import seaborn as sns



def plot_predictive_distributions(config,writer,step,data, inputs,
                                  preds_mean, preds_std, save_fig=True,
                                  name=None):
    """Plot the predictive distribution of a regression task.

    Args:
        config: Command-line arguments.
        writer: Tensorboard summary writer.
        step(int): Training step
        data: A DATASET instance.
        inputs: Numpy arrays containint the training inputs.
        preds_mean: The mean predictions.
        preds_std: The std of all predictions.
        save_fig: Bool to save the figure
        name: string
    """
    show_plots = config.show_plots
    sns.set(style="white")


    out_dir = 'plots/'

    colors = ['#56641a', '#e6a176', '#00678a', '#b51d14', '#5eccab', '#3e474c' , '#00783e' ]


    fig, axes = plt.subplots(figsize=(8, 6))

    ts, lw, ms = 20, 5, 4

    # The default matplotlib setting is usually too high for most plots.
    plt.locator_params(axis='y', nbins=2)
    plt.locator_params(axis='x', nbins=6)


    train_range = data.train_x_range
    range_offset = (train_range[1] - train_range[0]) * 0.05
    sample_x, sample_y = data._get_function_vals( \
        x_range=[train_range[0] - range_offset, train_range[1] + range_offset])

    plt.plot(sample_x, sample_y, color='k',
             linestyle='dashed', linewidth=lw / 7.)

    train_x = data.get_train_inputs().squeeze()
    train_y = data.get_train_outputs().squeeze()

    plt.plot(train_x, train_y, 'o', color='k', label='Training Data',
                 markersize=ms)


    plt.plot(inputs, preds_mean, color=colors[2],
             label='Pred distributions', lw=lw / 3.)

    plt.fill_between(inputs, preds_mean + preds_std,
                     preds_mean - preds_std, color=colors[2], alpha=0.3)
    plt.fill_between(inputs, preds_mean + 2. * preds_std,
                     preds_mean - 2. * preds_std, color=colors[2], alpha=0.2)
    plt.fill_between(inputs, preds_mean + 3. * preds_std,
                     preds_mean - 3. * preds_std, color=colors[2], alpha=0.1)

    #plt.legend()
    #plt.grid()
    plt.title(name, fontsize=ts, pad=ts)
    plt.ylim([np.min(train_y)-3, 3+np.max(train_y)])
    #plt.xlim([np.min(train_x)-3, 3+np.max(train_x)])

    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(ts)
    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(ts)

    #plt.xlabel('$x$', fontsize=ts)
    #plt.ylabel('$y$', fontsize=ts)
    plt.axis('off')

    if save_fig:
        plt.savefig(os.path.join(config.out_dir+'/', 'predictive_dist'+str(step) + name+'.pdf'),
                    bbox_inches='tight', format='pdf')
    writer.add_figure(name, plt.gcf(), step,
                      close=not config.show_plots)
    if show_plots:
        repair_canvas_and_show_fig(plt.gcf())


def reshape_output_2(z,shapes):
    reshaped_weights = []
    weighs_split = []

    for w in shapes:
        weighs_split.append(np.prod(w))
    z_splitted = torch.split(z, weighs_split,1)
    for j in range(z.shape[0]):
        l = []
        for i, shape in enumerate(shapes):
            l.append(z_splitted[i][j].reshape(shape))
        reshaped_weights.append(l)
    return reshaped_weights


def repair_canvas_and_show_fig(fig, close=True):
    """If writing a figure to tensorboard via "add_figure" it might change the
    canvas, such that our backend doesn't allow to show the figure anymore.
    This method will generate a new canvas and replace the old one of the
    given figure.

    Args:
        fig: The figure to be shown.
        close: Whether the figure should be closed after it has been shown.
    """
    tmp_fig = plt.figure()
    tmp_manager = tmp_fig.canvas.manager
    tmp_manager.canvas.figure = fig
    fig.set_canvas(tmp_manager.canvas)
    plt.close(tmp_fig.number)
    plt.figure(fig.number)
    plt.show()
    if close:
        plt.close(fig.number)

def contour_plot(grid_mesh,z_grid, config,step,writer = None,continuous = False, data = None, title = None, save = True,cmap=plt.cm.RdBu ):
    """If the classification task is 2D this function can be used to generate contour plots for uncertainty, decision
    boundary or softmax output.

    Args:
        grid_mesh (tuple): tuple containinig numpy arrays for the first and second dimension of the grid (x_1,x_2)
        z_grid (np.array): value of the point in the grid
        writer: Tensorboard summary writer.
        config: Command-line arguments
        step (int): training step
        continuous(Bool): if the plot has to be continuous or not
        data: DATASET instance to include a scatter plot of the training data on the grid plot, if None no data will
            be plotted
        title: plot title

    """
    colors = ListedColormap(['#FF0000', '#0000FF'])

    # Create plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # train_colors = [colors[i] for i in torch.argmax(y_train_0,1)]
    # test_colors = [colors[i] for i in torch.argmax(y_test_0,1)]
    #clev = np.arange(z_grid.min(), z_grid.max(),  .05)
    clev = np.linspace(z_grid.min(), z_grid.max(),  num=20)

    if continuous:
        b = ax.contourf(grid_mesh[0], grid_mesh[1], z_grid.reshape(grid_mesh[1].shape), clev, cmap=cmap,
                        extend='both')
    else:
        b = ax.contourf(grid_mesh[0], grid_mesh[1], z_grid.reshape(grid_mesh[1].shape), cmap=cmap)


    if data is not None:
        x_train_0 = data.get_train_inputs()
        y_train_0 = data.get_train_outputs()
        x_test_0 = data.get_test_inputs()
        y_test_0 = data.get_test_outputs()

        ax.scatter(x_train_0[:, 0], x_train_0[:, 1], alpha=1., marker='o', c=np.argmax(y_train_0, 1), cmap="viridis",
                   edgecolors='k', s=50, label='Train')
        #ax.scatter(x_test_0[:, 0], x_test_0[:, 1], alpha=0.6, marker='s', c=np.argmax(y_test_0, 1), cmap=colors,
        #           edgecolors='k', s=50, label='test')
    if title is not None:
        plt.title(title, fontsize=50)

    #plt.colorbar(b)
    ax.set_axis_off()

    #plt.legend(loc=2, fontsize=30)
    if writer is not None:
        writer.add_figure(title, plt.gcf(), step,
                          close=not config.show_plots)

    if config.show_plots:
        repair_canvas_and_show_fig(plt.gcf())
    
    if save: 
        fig.savefig(config.out_dir +'/'+title+ config.method+str(step)+".png", format = 'png', dpi = 200)

def dnorm2( X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())
    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
    return dnorm2        
        
class CutoutTransform(object):
    """Randomly mask out one or more patches from an image.

    The cutout transformation as preprocessing step has been proposed by

        DeVries et al., `Improved Regularization of Convolutional Neural \
Networks with Cutout <https://arxiv.org/abs/1708.04552>`__, 2017.

    The original implementation can be found `here <https://github.com/\
uoguelph-mlrg/Cutout/blob/master/util/cutout.py>`__.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """Perform cutout to given image.

        Args:
            img (Tensor): Tensor image of size ``(C, H, W)``.

        Returns:
            (torch.Tensor): Image with ``n_holes`` of dimension
                ``length x length`` cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def ood_metrics_softmax(in_dis, out_dis,writer,config,step):
    """ Compute AUROC, AUPRC and FPR80 score.
    Args:
        in_dis: the scores of the in-distribution data.
        out_dis: the scores of the out-of-distribution data.
    Return:
        The AUROC, AUPR IN, AUPR OUT, FPR80, FPR95 and the corresponing
        detection errors. Formulas from https://arxiv.org/pdf/1706.02690.pdf.
    """
    with torch.no_grad():
        y_true = np.concatenate([np.zeros(in_dis.shape[0]),
                                                    np.ones(out_dis.shape[0])]).reshape(-1)
        y_scores = np.concatenate([in_dis, out_dis]).reshape(-1)
        fpr, tpr, _ = roc_curve(y_true, -y_scores, pos_label=1)
        tpr_80 = None
        tpr_95 = None
        for i, t in enumerate(tpr):
            if t >= .8 and tpr_80 == None:
                tpr_80 = i
            if t >= .95 and tpr_95 == None:
                tpr_95 = i
        FPR_80  = fpr[tpr_80]
        FPR_95  = fpr[tpr_95]
        det_error_80 = 0.5*(1. - tpr[tpr_80]) + 0.5*fpr[tpr_80]
        det_error_95 = 0.5*(1. - tpr[tpr_95]) + 0.5*fpr[tpr_95]
        roc_auc = roc_auc_score(y_true, -y_scores)


    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0.95, 0.95], color='black', lw=lw, linestyle=':', label='FPR (95%% TPR) = %0.2f' % FPR_95)
    plt.plot([tpr_95, tpr_95], [0, 1], color='black', lw=lw, linestyle=':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random detector ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")

    writer.add_figure('ROC_curve', plt.gcf(), step,
                      close=not config.show_plots)
    if config.show_plots:
        repair_canvas_and_show_fig(plt.gcf())

    return roc_auc, \
           average_precision_score(y_true, -y_scores, pos_label=1), \
           average_precision_score(y_true, y_scores, pos_label=0), \
           FPR_80, FPR_95, det_error_80, det_error_95

def ood_metrics_entropy(in_dis, out_dis,writer,config,step,name = None):
    """ Compute AUROC, AUPRC and FPR80 score.
    Args:
        in_dis: the average entropy of the in-distribution data.
        out_dis: the average entropy of the out-of-distribution data.
    Return:
        The AUROC, AUPR IN, AUPR OUT, FPR80, FPR95 and the corresponing
        detection errors. Formulas from https://arxiv.org/pdf/1706.02690.pdf.
    """
    with torch.no_grad():
        y_true = np.concatenate([np.zeros(in_dis.shape[0]),
                                                    np.ones(out_dis.shape[0])]).reshape(-1)
        y_scores = np.concatenate([in_dis, out_dis]).reshape(-1)
        fpr, tpr, _ = roc_curve(y_true, -y_scores, pos_label=0)
        tpr_80 = None
        tpr_95 = None
        for i, t in enumerate(tpr):
            if t >= .8 and tpr_80 == None:
                tpr_80 = i
            if t >= .95 and tpr_95 == None:
                tpr_95 = i
        FPR_80  = fpr[tpr_80]
        FPR_95  = fpr[tpr_95]
        det_error_80 = 0.5*(1. - tpr[tpr_80]) + 0.5*fpr[tpr_80]
        det_error_95 = 0.5*(1. - tpr[tpr_95]) + 0.5*fpr[tpr_95]
        roc_auc = roc_auc_score(y_true, y_scores)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='AUROC = %0.4f' % roc_auc)
        plt.plot([0, 1], [0.95, 0.95], color='black', lw=lw, linestyle=':', label='FPR (95%% TPR) = %0.4f' % FPR_95)
        plt.plot([tpr_95, tpr_95], [0, 1], color='black', lw=lw, linestyle=':')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random detector ROC')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve '+name)
        plt.legend(loc="lower right")
        
        titl = 'ROC_Curve'
        
        if name is not None: 
            titl = 'ROC_curve/'+name

        if writer is not None:
            writer.add_figure(titl, plt.gcf(), step,
                              close=not config.show_plots)
        
            if config.show_plots:
                repair_canvas_and_show_fig(plt.gcf())
                plt.savefig('ROC_curve'+name+'.pdf',bbox_inches='tight', format='pdf')
        return roc_auc, \
                average_precision_score(y_true, -y_scores, pos_label=0), \
                average_precision_score(y_true, y_scores, pos_label=1), \
                FPR_80, FPR_95, det_error_80, det_error_95


if __name__ == '__main__':
    pass