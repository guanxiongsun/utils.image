import matplotlib.pyplot as plt
import matplotlib.patches as pch


class Drawer(object):
    def __init__(self, np_img=None, num_colors=5, cmap_name=None):
        self.fig = plt.figure()
        self.canvas = self.fig.gca()
        self.img = np_img
        self.legend = list()
        self.num_colors = num_colors
        self.cmap_name = cmap_name
        self.color_map = self._get_color_map(self.cmap_name)

        if self.img is not None:
            self.canvas.imshow(np_img)

    def set_cmap_name(self, cmap_name):
        self.cmap_name = cmap_name
        self.color_map = self._get_color_map(self.cmap_name)

    @staticmethod
    def _get_color_map(name):
        if name is None:
            name = 'gist_rainbow'
        return plt.get_cmap(name)

    def set_num_colors(self, num_colors):
        self.num_colors = num_colors

    def get_color(self, color_index):
        """
        Get a color from color map
        :param color_index: index of color
        """
        assert color_index <= self.num_colors
        return self.color_map(1. * color_index / self.num_colors)

    def set_img(self, np_img):
        """
        Set the drawer's img. np img.
        :param np_img: np img
        """
        self.img = np_img
        self.draw_img(np_img)

    @property
    def get_img(self):
        return self.img

    def draw_img(self, np_img=None):
        if np_img is None:
            self.canvas.imshow(self.img)
        else:
            self.canvas.imshow(np_img)

    def show(self):
        self.fig.show()

    @staticmethod
    def _get_bbox(box=(0, 0, 100, 100),
                  color='red',
                  linewidth=5):
        return plt.Rectangle((box[0], box[1]),
                             box[2], box[3],
                             fill=False,
                             edgecolor=color,
                             linewidth=linewidth)

    def draw_bbox(self, box, color='red', linewidth=5):
        """
        Draw a bbox on canvas
        :param box: list [top_left_x, top_left_y, width, height]
        :param color: color of edge
        :param linewidth: width of line
        """
        _bbox = self._get_bbox(box, color=color, linewidth=linewidth)
        self.canvas.add_patch(_bbox)

    def draw_text(self, x, y, text_str,
                  text_color='white', font_size=15,
                  box_color='red', box_alpha=0.3):
        """
        Draw a text box on canvas
        :param x: bottom_left_x of text box
        :param y: bottom_left_y of text box
        :param text_str: text string
        :param text_color: text color
        :param font_size: ...
        :param box_color: ...
        :param box_alpha: ...
        """
        self.canvas.text(x, y, text_str, fontsize=font_size, color=text_color,
                         bbox=dict(facecolor=box_color, alpha=box_alpha))

    def add_legend(self, color='red', label='legend'):
        self.legend.append(
            pch.Patch(color=color, label=label)
        )

    def draw_legend(self):
        if self.legend is None:
            try:
                self.canvas.legend()
            except:
                print("No legends.")
        else:
            if not self.legend:
                pass
            else:
                self.canvas.legend(handles=self.legend)

    def draw_line(self):
        raise NotImplementedError

    def clear(self):
        """
        Clear all but the image.
        """
        self.canvas.cla()
        self.legend = list()
        self.draw_img()

    def clear_all(self):
        """
        Clear all
        """
        self.canvas.cla()

    def close(self):
        self.fig.clf()
        plt.close(self.fig)


if __name__ == '__main__':
    import img
    import numpy as np

    lena = img.imread(img.imgs['lena_png'])
    bear = img.imread(img.imgs['bear_jpg'])

    drawer = Drawer(lena)
    drawer.show()

    drawer.set_img(bear)
    drawer.draw_img()
    drawer.show()

    box_1 = [100, 100, 400, 500]
    drawer.draw_bbox(box_1)
    drawer.draw_legend()
    drawer.show()

    drawer.clear()
    num_colors = 20
    drawer.set_num_colors(num_colors)

    for i in range(1, 20):
        H, W, _ = drawer.get_img.shape

        tl_x = np.random.randint(W)
        tl_y = np.random.randint(H)
        w = np.random.randint(W - tl_x)
        h = np.random.randint(H - tl_y)
        box_2 = [tl_x, tl_y, w, h]
        color1 = np.random.randint(num_colors)
        color2 = np.random.randint(num_colors)

        drawer.draw_bbox(box_2, color=drawer.get_color(color1))
        drawer.draw_text(tl_x, tl_y, str(color2), box_color=drawer.get_color(color2))
        drawer.add_legend(color=drawer.get_color(color1), label=str(color1))
        drawer.add_legend(color=drawer.get_color(color2), label=str(color2))
        drawer.draw_legend()
        drawer.show()

        if i % 3 == 0:
            drawer.clear()

        if i % 8 == 0:
            drawer.clear_all()
            drawer.set_img(lena)

    print("done")
