import os
from abc import ABC

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import GravNN


def convert_string(string):
    string = string.replace(".", "_")
    string = string.replace("[", "_")
    string = string.replace("]", "_")
    string = string.replace(" ", "")
    string = string.replace(",", "_")
    return string


class VisualizationBase(ABC):
    def __init__(
        self,
        save_directory=None,
        halt_formatting=False,
        formatting_style=None,
        **kwargs,
    ):
        """Default visualization base class. Generates consistent style formatting for
        all figures, offers information about proper figure sizes.

        Args:
            save_directory (str, optional): path to directory in which all figures will
                                            be saved. Defaults to None.
            halt_formatting (bool, optional): flag determining if custom formatting
                                                should be applied. Defaults to False.
        """
        if save_directory is None:
            self.file_directory = os.path.abspath(".") + "/Plots/"
        else:
            self.file_directory = save_directory

        self.golden_ratio = (5**0.5 - 1) / 2  # = 0.61
        self.silver_ratio = 1 / (1 + np.sqrt(2))  # = 0.41

        # Can be acquired loading the layouts package using
        # \printinunitsof{in}\prntlen{\textwidth}
        # \printinunitsof{in}\prntlen{\textheight}
        # \printinunitsof{in}\prntlen{\abovecaptionskip}

        SPRINGER_NATURE_WIDTH = 4.67596  # inches
        SPRINGER_NATURE_HEIGHT = 7.64914  # inches
        SPRINGER_CAPTION_PADDING = 0.03113  # inches

        width = SPRINGER_NATURE_WIDTH
        height = SPRINGER_NATURE_HEIGHT
        padding = SPRINGER_CAPTION_PADDING

        self.h_full = height
        self.h_half = height / 2
        self.h_tri = height / 3
        self.h_quad = height / 4

        self.h_full_pad = height - padding
        self.h_half_pad = height / 2 - padding
        self.h_tri_pad = height / 3 - padding
        self.h_quad_pad = height / 4 - padding

        self.w_full = width
        self.w_half = width / 2
        self.w_tri = width / 3
        self.w_quad = width / 4

        # (width, height)
        self.full_page_default = (self.w_full, self.w_full * self.golden_ratio)
        self.half_page_default = (self.w_half, self.w_half * self.golden_ratio)
        self.tri_page_default = (self.w_tri, self.w_tri * self.golden_ratio)
        self.quarter_page_default = (self.w_quad, self.w_quad * self.golden_ratio)

        self.full_page_silver = (self.w_full, self.w_full * self.silver_ratio)
        self.half_page_silver = (self.w_half, self.w_half * self.silver_ratio)
        self.tri_page_silver = (self.w_tri, self.w_tri * self.silver_ratio)
        self.quarter_page_silver = (self.w_quad, self.w_quad * self.silver_ratio)

        self.full_page_golden = (self.w_full, self.w_full * self.golden_ratio)
        self.half_page_golden = (self.w_half, self.w_half * self.golden_ratio)
        self.tri_page_golden = (self.w_tri, self.w_tri * self.golden_ratio)
        self.quarter_page_golden = (self.w_quad, self.w_quad * self.golden_ratio)

        # AAS textwidth is 6.5
        self.AIAA_half_page = (3.25, 3.25 * self.golden_ratio)  #
        self.AIAA_half_page_heuristic = (4, 4 * self.golden_ratio)  #
        if formatting_style == "AIAA":
            self.AIAA_full_page = (6.5, 6.5 * self.golden_ratio)

        # Set default figure size (width, height)
        self.fig_size = self.full_page_golden

        # default figure styling
        plt.rc("font", size=6.0)
        plt.rc("font", family="serif")
        plt.rc("figure", autolayout=True)
        plt.rc("savefig", pad_inches=0.0)
        plt.rc("lines", linewidth=0.5)

        plt.rc(
            "axes",
            prop_cycle=mpl.cycler(
                color=[
                    "blue",
                    "green",
                    "red",
                    "orange",
                    "gold",
                    "salmon",
                    "lime",
                    "magenta",
                    "lavender",
                    "yellow",
                    "black",
                    "lightblue",
                    "darkgreen",
                    "pink",
                    "brown",
                    "teal",
                    "coral",
                    "turquoise",
                    "tan",
                    "gold",
                ],
            ),
        )
        plt.rc("axes.grid", axis="both")
        plt.rc("axes.grid", which="both")
        plt.rc("axes", grid=True)
        plt.rc("grid", linestyle="--")
        plt.rc("grid", linewidth="0.1")
        plt.rc("grid", color=".25")

        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        plt.rc("text", usetex=True)
        # plt.rc('text.latex', unicode = True)
        # plt.rc('text.latex', verbose = True)
        if halt_formatting:
            plt.rc("text", usetex=False)
            # plt.rc('text.latex', unicode = False)
            # plt.rc('text.latex', verbose = False)

        return

    def new3DFig(self, unit="m", **kwargs):
        figsize = kwargs.get("fig_size", self.fig_size)
        fig = plt.figure(num=None, figsize=figsize, dpi=300)
        ax = fig.add_subplot(111, projection="3d")
        # ax.set_xlabel(r'$x$ ('+unit+r')')
        # ax.set_ylabel(r'$y$ ('+unit+r')')
        # ax.set_zlabel(r'$z$ ('+unit+r')')
        return fig, ax

    def newFig(self, fig_size=None):
        if fig_size is None:
            fig_size = self.fig_size
        fig = plt.figure(num=None, figsize=fig_size)
        ax = fig.add_subplot(111)
        return fig, ax

    def save(self, fig, name, directory=None):
        if directory is None:
            GravNN_dir = os.path.abspath(GravNN.__path__[0])
            directory = os.path.join(GravNN_dir, "../Plots")

        # Normalize and create the directory if it doesn't exist
        directory = os.path.normpath(directory)
        os.makedirs(directory, exist_ok=True)

        # Convert the figure name to avoid invalid characters
        name = convert_string(name)

        # Check if the name is an absolute path
        if os.path.isabs(name):
            base, ext = os.path.splitext(name)
            pdf_path = f"{base}.pdf"
            png_path = f"{base}.png"
        else:
            pdf_path = os.path.join(directory, f"{name}.pdf")
            png_path = os.path.join(directory, f"{name}.png")

        # Save the figure in both PDF and PNG formats
        fig.tight_layout(pad=0.0)
        try:
            fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
            fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
            print(f"Figure saved as:\n{pdf_path}\n{png_path}")
        except Exception as e:
            print(f"Couldn't save the figure {name}\nError: {e}")
