"""
class to handle figure output (show/savefig), saveing legends in separate files, file names
"""


class FigureHandler:
    def __init__(self, plot, save_dir=None, save_separate_legend=None, basename=None, **kwargs):
        self.plot = plot
        self.save_separate_legend = save_separate_legend
        self.basename = basename
        self.save_dir = save_dir
        self.format = kwargs.pop("format", "png")
        self.dpi = kwargs.pop("dpi", 900)

    def figure_saver(self, basename, **kwargs):
        suffix = kwargs.pop("suffix")

        self.plot.savefig(self.save_dir + self.basename + "_" + basename + suffix + "." + self.format,
                          format=self.format,
                          # bbox_inches='tight',
                          dpi=self.dpi)
