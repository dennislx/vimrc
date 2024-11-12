import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from typing import Callable
import itertools

import pynimate as nim
from pynimate.datafier import LineDatafier
from pynimate.lineplot import Lineplot
from pynimate.canvas import Canvas
from tqdm import tqdm  # Import tqdm
import seaborn as sns  # Import seaborn for color palettes

DEFAULT_UPDATE = lambda self, i: None
BAR_WIDTH = 0.5
BAR_ALPHA = 0.5
YLIM_EPS  = 0.05

def generate_all_colors(n_colors, palettes=['viridis']):
    all_colors = []
    for palette in palettes:
        all_colors.extend(
            list( sns.color_palette(
                palette, int( np.ceil( n_colors / len(palettes))),
            ))
        )
    return all_colors

def get_colname(col: str, n: int):
    if n == 0: return col.capitalize()
    parts = col.split('_', n)
    return ' '.join(x.capitalize() for x in parts[1:])

def get_slice(val: int | str):
    if val is None:
        return slice(None)
    elif isinstance(val, int):
        return val
    elif isinstance(val, str):
        start, end = map(int, val.split(':'))
        return slice(start, end)
    else:
        raise ValueError('Invalid value')

class MyCanvas(Canvas):

    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[int, int] = (12.8, 7.2),
        post_update: Callable[[plt.Figure, list[list[plt.Axes]]], None] = None,
        **kwargs,
    ) -> None:
        self.post_update = post_update or (lambda *args: None)
        self.plots = []
        self.length = 0
        self.fig = plt.figure(figsize=figsize, **kwargs)
        self.ax = gridspec.GridSpec(nrows, ncols, figure=self.fig)

    def set_gs_properties(self, wspace=None, hspace=None):
        self.ax.update(wspace=wspace, hspace=hspace)

    def add_plot(self, plot, row, col=None):
        row_slice = get_slice(row)
        col_slice = get_slice(col)
        ax = self.fig.add_subplot(self.ax[row_slice, col_slice])
        plot.set_axes(ax)
        self.length = max(self.length, plot.length)
        self.plots.append(plot)
        return self

class LineBarDatafier(LineDatafier):
    def __init__(
        self, df: pd.DataFrame, time_format: str,
        ip_freq: str, ip_method: str = "linear",
        line_cols: list[str] = None, bar_cols: list[str] = None,
        group_cols: list[str] = None, only_interpolate = False
    ):
        self.only_interpolate = only_interpolate
        df_transformed = self.transform_data(df, group_cols, line_cols, bar_cols)
        super(LineDatafier, self).__init__(df_transformed, time_format, ip_freq, ip_method)
        self.new_added = None

    def transform_data(self, data, group_cols=None, line_cols=None, bar_cols=None):
        n = len(group_cols)
        if n > 0:
            data = data.pivot_table(index=data.index, columns=group_cols, values=line_cols+bar_cols)
            data.columns = ['_'.join(col) for col in data.columns]
            line_cols = [x for b in line_cols for x in data.columns if x.startswith(b)]
            bar_cols  = [x for b in bar_cols for x in data.columns if x.startswith(b)]
        self.group_cols = group_cols
        self.line_cols = line_cols if line_cols is not None else data.columns
        self.bar_cols = bar_cols if bar_cols is not None else data.columns

        return data
        
    def interpolate_data(self):
        if not self.ip_method:  # Skip interpolation if ip_method is None or empty
            return self.data

        data = self.data
        ncols = data.select_dtypes("number").columns
        num_data = data[ncols]
        add_new_flag = False
        if self.ip_freq != None:
            new_ind = pd.date_range(
                num_data.index.min(), num_data.index.max(), freq=self.ip_freq
            )
            new_ser = pd.Series(
                [0] * len(new_ind), index=new_ind, name="new_ind"
            ).to_frame()
            num_data = (
                new_ser.join(num_data, how="outer").drop("new_ind", axis=1).sort_index()
            )
            self.expanded = (
                new_ser.join(self.expanded, how="outer")
                .drop("new_ind", axis=1)
                .sort_index()
            )
            add_new_flag = True

        num_data = num_data.interpolate(method=self.ip_method)
        obCols = data.select_dtypes(exclude="number").columns
        data = data[obCols].join(num_data, how="right")
        data[obCols] = data[obCols].bfill().ffill()  # Use bfill and ffill directly
        if add_new_flag:
            new_data = data.loc[new_ind]
            if self.only_interpolate: return new_data
            self.new_added = new_data
        return data

        

class LineBarplot(Lineplot):
    
    def __init__(self, datafier, post_update=DEFAULT_UPDATE, bar_width=BAR_WIDTH, bar_alpha=BAR_ALPHA):
        super().__init__(datafier, palettes=['viridis'], post_update=post_update, fixed_xlim=True, fixed_ylim=True, xticks=True, yticks=True, grid=True)
        self.bar_width = bar_width
        n = len(datafier.bar_cols)
        self.bar_shift = np.linspace(-bar_width* (n - 1) / 2, bar_width * (n - 1) / 2, n)
        self.bar_alpha = bar_alpha
        self.bar_colors = self.column_colors.copy()
        self.bar_hatch = {col: None for col in self.dfr.bar_cols}
        self.all_cols  = self.dfr.data.columns

    def generate_column_colors(self):
        df: LineBarDatafier = self.dfr
        # Case 1: If group_vars is empty, generate a unique color for each column
        if not df.group_cols:
            colors = generate_all_colors(len(df.bar_cols + df.line_cols), self.palettes)
            return {col: color for col, color in zip(df.data.columns, colors)}
        # Case 2: When group_vars are provided, create a unique color for each postfix combination
        postfixes = ['_'.join(col.split('_')[1:]) for col in df.data.columns]
        unique_postfixes = list(set(postfixes))
        # Generate colors for each unique postfix
        colors = generate_all_colors(len(unique_postfixes), self.palettes)
        postfix_color_map = {postfix: color for postfix, color in zip(unique_postfixes, colors)}
        # Map each column to its corresponding postfix color
        return {col: postfix_color_map[col.split('_', 1)[-1]] for col in df.data.columns}

    def clear_axes(self):
        self.ax.clear()
        self.ax2.clear()

    def set_axes(self, ax: plt.Axes = None) -> None:
        if ax is not None:
            self.ax = ax
            self.ax2 = ax.twinx()

    def set_line_properties(self, col, color=None, linestyle=None):
        col = [x for x in self.all_cols if x.endswith(col)]
        if color:
            self.set_column_colors({c: color for c in col})
        if linestyle:
            self.set_column_linestyles({c: linestyle for c in col})

    def set_bar_properties(self, col, color=None, width=None, alpha=None, hatch=None):
        col = [x for x in self.all_cols if x.endswith(col)]
        if color:
            new_colors = {c: color for c in col}
            self.bar_colors.update(new_colors)
        if hatch:
            new_hatchs = {c: hatch for c in col}
            self.bar_hatch.update(new_hatchs)
        if width is not None:
            self.bar_width = width
        if alpha is not None:
            self.bar_alpha = alpha

    def get_snapshot(self, i, col):
        data = self.dfr.data.iloc[:i+1]
        return np.arange(len(data.index)), data[col]

    def set_xylim(self, xlim: list[float] = [], ylim: dict[str, list[float]] = {}):
        """Sets xlim and ylim for line and bar plots separately

        Parameters
        ----------
        xlim : list[float], optional
            x axis limits in this format [min, max], by default [min date, max date]
        ylim : dict[str, list[float]], optional
            y axis limits for line and bar plots in this format {'line': [min, max], 'bar': [min, max]}, by default {'line': [min y val, max y val], 'bar': [min y val, max y val]}
        """
        if xlim == []:
            xlim = [0 - BAR_WIDTH, len(self.datafier.data) - 1 + BAR_WIDTH]
        self.xlim = xlim

        if 'line' not in ylim:
            line_max = self.datafier.data[self.datafier.line_cols].max().max()
            line_min = self.datafier.data[self.datafier.line_cols].min().min()
            ylim['line'] = [line_min * (1-YLIM_EPS), line_max * (1+YLIM_EPS)]

        if 'bar' not in ylim:
            bar_max = self.datafier.data[self.datafier.bar_cols].max().max()
            bar_min = self.datafier.data[self.datafier.bar_cols].min().min()
            ylim['bar'] = [bar_min * (1-YLIM_EPS), bar_max * (1+YLIM_EPS)]

        self.ylim = ylim

    def update(self, i):
        self.clear_axes()
        self.set_axes()
        df: LineBarDatafier = self.dfr
        # update line plots
        for j, col in enumerate(df.line_cols):
            X, Y = self.get_snapshot(i, col)
            self.ax.plot(
                X+self.bar_shift[j], Y, label=' '.join(x.capitalize() for x in col.split('_')), 
                color=self.column_colors[col], linestyle=self.column_linestyles[col],
                **self.line_props
            )
        
        # update bar plots
        for j,col in enumerate(df.bar_cols):
            X, Y = self.get_snapshot(i, col)
            bar = self.ax2.bar(
                X+self.bar_shift[j], Y, label=' '.join(x.capitalize() for x in col.split('_')), 
                width=self.bar_width, 
                alpha=self.bar_alpha, 
                color=self.column_colors[col],
                hatch=self.bar_hatch.get(col, None)
            )
            for bc in bar:
                bc._hatch_color = self.column_colors[col]
                bc.stale = True

        self.fixed_xlim and self.ax.set_xlim(self.xlim)
        if self.fixed_ylim:
            self.ax.set_ylim(self.ylim['line'])
            self.ax2.set_ylim(self.ylim['bar'])
        self.xticks and self.ax.tick_params(**self.xtick_props)
        if self.yticks:
            self.ax.tick_params(**self.ytick_props)
            self.ax2.tick_params(**self.ytick_props)
        
        self.ax.set_axisbelow(self.grid_behind)
        self.post_update(self, i)
        
        for v in self.text_collection.values():
            callback, props_dict = v[0], v[1]
            if callback:
                self.ax.text(
                    s=callback(i, self.datafier),
                    transform=self.ax.transAxes,
                    **props_dict,
                )
            else:
                self.ax.text(
                    **props_dict,
                    transform=self.ax.transAxes,
                )


def post_update(self, i):
    def prepare_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)    
    for ax in (self.ax, self.ax2):
        prepare_ax(ax)
    # Set x-tick labels to dates
    self.ax.set_xticks(np.arange(len(self.dfr.data.index)))
    self.ax.set_xticklabels(self.dfr.data.index.strftime('%Y-%m'), rotation=45, ha='right')
    self.ax.legend(loc='upper left')
    self.ax2.legend(loc='upper right')
    
    # set y labels for ax and ax1
    common_kwargs = { "color": "#777777", "size": 12, "weight": "bold", "rotation": 90, "va": "center" }
    # self.text_collection["line_label"] = (None, {"s": "Price", **common_kwargs, "x": -0.07, "y": 0.5})
    # self.text_collection["bar_label"] = (None, {"s": "Volume", **common_kwargs, "x": 1.07, "y": 0.5})
        

if __name__ == '__main__':
    # Generate random dates within the specified range
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-12-31')
    num_dates = 24  # Number of random dates
    np.random.seed(42)
    random_dates = pd.to_datetime(np.random.randint(start_date.value, end_date.value, num_dates)).strftime('%Y%m%d').tolist()
    
    data = {
        'date': np.tile(random_dates, 2),  # Repeat dates for six groups
        'price': np.random.uniform(50, 150, len(random_dates) * 2),
        'volume': np.random.randint(100, 1000, len(random_dates) * 2),
        'group': ['A'] * len(random_dates) + ['B'] * len(random_dates) 
    }
    df = pd.DataFrame(data).set_index('date')
    cnv = MyCanvas(nrows=1, ncols=2)
    cnv.set_gs_properties(hspace=0.3)
    dfr = LineBarDatafier(df, '%Y%m%d', 'ME', ip_method='pad', line_cols=['price'], bar_cols=['volume'], group_cols=['group'], only_interpolate=True)
    
    plot = LineBarplot(dfr, bar_width=0.4, post_update=post_update) 
    plot.set_line_properties(col='A', linestyle='dashed')
    plot.set_bar_properties( col='A', hatch='//')
    plot.set_xticks(length=0, labelsize=10)
    plot.set_yticks(labelsize=12)
    from copy import deepcopy
    plot_ = deepcopy(plot)
    cnv.add_plot(plot_, None, 0)
    cnv.add_plot(plot, None, 1)
    
    # Create a tqdm progress bar
    pbar = tqdm(total=plot.length, desc='Saving frames')
    # Define the progress callback function
    def progress_callback(i, n): 
        pbar.update(1)
    cnv.animate(frames_callback=lambda x: plot.length)
    cnv.save('assets/linebar_light', fps=1, progress_callback=progress_callback)
    pbar.close()  # Close the progress bar