import plotly
import numpy as np
import random
import plotly.graph_objs as go
import cv2
import matplotlib.pyplot as plt
from IPython.display import Image

def is_in_colab():
    try:
        import google.colab
        return True
    except:
        return False

def display_gif(fn):
    with open('data/both.gif','rb') as f:
        display(Image(data=f.read(), format='png'))

def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))


def show_image(img, title=None, normalize=False, show_scale=False):
    if len(img.shape) == 3:
        img = img.transpose((1,2,0))
    if normalize:
        img = img*1.0
        img = (img - img.min())/(img.max() - img.min())
    fig = plt.figure()
    if not title is None:
        plt.title(title)
    if len(img.shape) == 2:
        _ = plt.imshow(img, cmap=plt.get_cmap('Greys'))
    else:
        _ = plt.imshow(img)

    if show_scale:
        plt.colorbar()

def show_pointcloud(pcl, colors, valid_mask=None, points_to_show = 10000):
    if is_in_colab():
        configure_plotly_browser_state()
    if len(pcl.shape) == 3:
        assert len(colors.shape) == 3
        pcl = pcl.reshape(3, -1)
        colors = colors.reshape(3, -1)
    assert pcl.shape[0] == 3
    assert colors.shape[0] == 3
    assert pcl.shape[1] == colors.shape[1]

    if not valid_mask is None:
        valid_mask = valid_mask.flatten()
        pcl = pcl[:, valid_mask]
        colors = colors[:, valid_mask]

    if pcl.shape[1] > points_to_show:
        random_sample = random.sample(list(range(pcl.shape[1])), points_to_show)
        pcl = pcl[:, random_sample]
        colors = colors[:, random_sample]

    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()

    # Configure the trace.
    trace = go.Scatter3d(
      x=pcl[0],
      y=pcl[1],
      z=pcl[2],
      mode='markers',
      marker={
          'size':3,
          'opacity': 1,
          'color': colors.transpose()
      }
    )

    # Configure the layout.
    layout = go.Layout(
      margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    # center is lookAt point
    # eye is the camera center
    # up is the up direction
    camera = dict(
        up=dict(x=0, y=-1, z=0),
        center=dict(x=0, y=0, z=0.5),
        eye=dict(x=0.3, y=0, z=-1.5)
    )
    layout.update(scene=dict(camera=camera,aspectmode='data')),
    data = [trace]

    plot_figure = go.Figure(data=data, layout=layout)
    configure_plotly_browser_state()

    # Render the plot.
    plotly.offline.iplot(plot_figure)

def imread(img_file):
    return cv2.imread(img_file)[:,:,::-1].transpose((2,0,1))
