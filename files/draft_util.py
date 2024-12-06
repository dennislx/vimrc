__all__ = [
    'MyDatafier', 'MyCanvas',
    'get_file_or_dir', 'generate_synthetic', 'transform_data',
    'gather_history'
]

import pandas as pd
import numpy as np
import pynimate as nim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm
from typing import Literal, Dict, List, Callable
from enum import Enum
from pathlib import Path


MONTH_IDX   = '%Y-%m'
MyColumnProp = Literal['type', 'style', 'label']

def get_proj_dir():
    return Path(__file__).parents[1]

def get_data_dir():
    return get_proj_dir() / 'data_house'

def get_file_or_dir(name: str, folder=None, ext=".csv", parent=get_data_dir()):
    """
    Constructs a file or directory path based on the provided parameters.

    Args:
        name (str): The name of the file or directory.
        folder (str, optional): The subfolder within the parent directory. Defaults to None.
        ext (str, optional): The file extension. Set to an empty string to create a directory. Defaults to ".csv".
    """  
    parent = Path(parent)
    if folder:
        parent = parent / folder
        parent.mkdir(exist_ok=True, parents=True)
    return parent / "{}{}".format(name, ext)

def generate_synthetic():
    """
                Volume      Price  Federal Rate Stock
    2021-01-31     827  19.304650      2.052469     A
    2021-01-31     972  73.522370      2.052469     B
    2021-02-28     924  62.950356      2.139303     A
    2021-02-28     799  64.946417      2.139303     B
    """
    date_range = pd.date_range(start='2021-01-01', end='2024-12-31', freq='ME')
    volume = np.random.randint(100, 1000, size=len(date_range)*2) * 100_000
    price = np.random.uniform(10, 100, size=len(date_range)*2) * 100_000
    
    # Generate small random changes and take the cumulative sum to make the federal rate more stable
    federal_rate_changes = np.random.uniform(-0.01, 0.01, size=len(date_range))
    federal_rate = np.zeros(len(date_range))
    for i in range(1, len(federal_rate)):
        federal_rate[i] = federal_rate[i-1] + federal_rate_changes[i]
        if federal_rate[i] < 0:
            federal_rate[i] = -federal_rate[i]  # Bounce back if below 0
        federal_rate[i] = min(federal_rate[i], 0.05)  # Ensure values stay between 0 and 0.05
    
    stock = np.tile(['A', 'B'], len(date_range))
    data = {
        'Volume': volume,
        'Price': price,
        'Federal Rate': np.repeat(federal_rate, 2),
        'Stock': stock
    }
    df = pd.DataFrame(data, index=np.repeat(date_range, 2))
    return df

def transform_data(data, group_bys=[], group_vars=[], group_excludes=[]):
    """
    >>> transform_data(
        df, group_vars=['Price'], 
        group_bys=['Stock'], 
        group_excludes=['Federal Rate']
    )
    """
    index = data.index
    if index.name is None: index.name = 'index'
    if len(group_bys) > 0:
        index_vars = [data.index.name] + group_excludes
        data = data.pivot_table(index=index_vars, columns=group_bys, values=group_vars)
        data = data.reset_index()
        data.columns = ['_'.join(map(str, filter(None, col))).strip() for col in data.columns.values]
    if index.name in data.columns:
        data = data.set_index(index.name)
    return data

def calc_errorbar(dct_lst: Dict[str, List]):
    """
    Calculate the mean and standard deviation of the summed values from a dictionary of lists.
    """
    summed_values = np.nansum(list(dct_lst.values()), axis=0)
    return {
        'mean': np.mean(summed_values),
        'std': np.std(summed_values)
    }

def next_month_start(date: pd.Timestamp):
    """
    >>> next_month_start(pd.Timestamp('20231203'))
    2024-01-01 00:00:00
    """
    return (date+pd.DateOffset(months=1)).replace(day=1, hour=0, minute=0)


def get_slice(val: int | str):
    if val is None:
        return slice(None)
    elif isinstance(val, int):
        return val
    elif isinstance(val, str):
        start, end = map(int, val.split(':'))
        return slice(start, end)
    else:
        raise ValueError("Invalid value for slicing")

def convert(old_filename, new_filename = None, fps = 1):
    from PIL import Image, PngImagePlugin as PNG
    import io
    images = []
    with Image.open(old_filename) as img:
        for i in range(img.n_frames):
            img.seek(i)
            buf = io.BytesIO()
            img.save(buf, format="png")
            buf.seek(0)
            images.append(Image.open(buf))
    duration = 1 / fps * 1_000  # number of miliseconds between each frame
    new_filename = new_filename or old_filename
    img: PNG.PngImageFile = images[0]
    img.save(new_filename, save_all=True, append_images=images[1:], duration=duration, loop=0)
    
class MyCanvas(nim.Canvas):

    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[int, int] = (12.8, 7.2),
        post_update: Callable[[plt.Figure, list[list[plt.Axes]]], None] = None,
        tight_layout: bool = False,
        wspace = None,
        hspace = None,
        **kwargs,
    ) -> None:
        self.post_update = post_update or (lambda *args: None)
        self.plots = []
        self.length = 0
        self.fig = plt.figure(figsize=figsize, **kwargs)
        self.fig.set_tight_layout(tight_layout)
        self.ax = gridspec.GridSpec(nrows, ncols, figure=self.fig, wspace=wspace, hspace=hspace) 
    
    def set_space(self, wspace = None, hspace = None):
        self.ax.update(wspace = wspace, hspace = hspace)

    def set_mpl_properties(self, group: str, subgroup=False, **kwargs):
        """
        >>> set_mpl_properties('lines', linewidth=4)
        >>> set_mpl_properties('legend', fontsize=22)
        """
        if subgroup:
            for sub, sub_kwargs in kwargs.items():
                for k, v in sub_kwargs.items():
                    plt.rc('{}.{}'.format(group, sub), **{k: v})
        else:
            plt.rc(group, **kwargs)
    
    def add_plot(self, plot, row, col=None):
        row_slice, col_slice = map(get_slice, (row, col))
        ax = self.fig.add_subplot(self.ax[row_slice, col_slice])
        plot.set_axes(ax)
        self.length = max(self.length, plot.length)
        self.plots.append(plot)
        return self

    def save(self, filename, fps, extension = "gif", **kwargs):
        def update_pbar(i, total):
            if self.pbar.total is None:
                self.pbar.reset(total = total)
            self.pbar.update(1)
        self.pbar = tqdm(total=None, desc='Saving Frames')
        super().save(
            filename, fps, extension, progress_callback = update_pbar,
            **kwargs
        )
        if False and extension == 'gif' and self.pbar.total > 1:
            convert('{}.{}'.format(filename, extension), fps = fps)


def gather_history(df: pd.DataFrame):
    idx = df.index
    assert isinstance(idx, pd.DatetimeIndex)
    history_data = {}
    for date in idx.to_period('M').unique():
        first_day_of_month = date.start_time
        df_ = df[(df.index < first_day_of_month) & (idx.month == date.month)]
        if not df_.empty:
            history = df_.groupby(df_.index.month).agg(list).to_dict('records')[0]
            history_data.update({first_day_of_month.strftime(MONTH_IDX): history})
    return history_data

def gather_history(data, columns=[]):
    def std(x):
        return np.std(x, ddof=1) / np.sqrt(len(x))
    def helper(year):
        subdata = data[all_year_index < year]
        stats = subdata.groupby(subdata.index.month).agg(['mean', std])
        if isinstance(stats.columns, pd.MultiIndex):
            return {
                c: [
                    stats[(x, 'mean')].tolist(),  
                    stats[(x, 'std')].tolist() 
                ]
                for x, c in zip(stats.columns.get_level_values(0).unique(), columns)
            }
        elif isinstance(stats.columns, pd.Index):
            return {columns[0]: [stats['mean'].tolist(), stats['std'].tolist()]}
    all_year_index = data.index.year
    history_data = {}
    for year in all_year_index.unique():
        history_data[year] = helper(year)
    return history_data
    

class MyDatafier(nim.LineDatafier):
    def __init__(
        self, data, time_format, ip_freq, 
        ip_method = "linear", 
        term: Literal['year', 'none', 'month'] = 'none'
    ):
        super().__init__(data, time_format, ip_freq, ip_method)
        self.index_all = self.data.index
        self.start_term, self.end_term = 0, 0
        self.term = term
        self.init_props()

    def __repr__(self):
        data_repr = self.data.head(4).__repr__()
        extra_repr = self.show_props_table().__repr__()
        return f"\n{data_repr}\n\n{extra_repr}"

    def show_props_table(self):
        """
        Returns a DataFrame showing the properties of each column.
        """
        props_df = pd.DataFrame({
            'type': self.column_type, 
            'label': self.column_label,
            'format': self.column_format
        })
        style_df = pd.DataFrame(self.column_style).T
        props_df = pd.concat([props_df, style_df], axis=1)
        return props_df

    def __getitem__(self, i):
        return self.data.iloc[i]

    def init_props(self):
        self.column_style = {}
        self.column_label = {}
        self.column_type  = {}
        self.column_format = {}
        self.update_props('style', **{
            c: dict(color = "#{:06x}".format(np.random.randint(0, 0xFFFFFF))) for c in self.columns
        })
        self.update_props('label', **{c: c for c in self.columns})
        self.update_props('type', **{c: 'line' for c in self.columns})
        self.update_props('format', **{c: 'number' for c in self.columns})
        
    def update_props(
        self, 
        group: MyColumnProp, 
        action: Literal['update', 'replace'] = 'update',
        **kwargs
    ):
        this = getattr(self, f'column_{group}')
        if action == 'update': this.update(**kwargs)
        elif action == 'replace': setattr(self, f'column_{group}', kwargs)

    def resample_data(self, cols = [], method: Literal['linear', 'bfill'] = 'bfill'):
        match method:
            case 'bfill':
                new_cols = self.interpolate_even(self.raw_data[cols], freq=self.ip_freq, method='pad')
            case _:
                new_cols = self.interpolate_even(self.raw_data[cols], freq=self.ip_freq, method=method)
        self.data[cols] = new_cols

    @property
    def columns(self): return self.data.columns

    def get_index(self, i):
        
        if self.start_term <= i <= self.end_term:
            return slice(self.start_term, i+1), self.end_term - i

        today = self.index_all[i]
        match self.term:
            case 'year':
                self.start_term = self.index_all.searchsorted(today.replace(month=1,day=1, hour=0, minute=0))
                self.end_term = self.index_all.searchsorted(next_month_start(today.replace(month=12))) - 1
            case 'month':
                self.start_term = self.index_all.searchsorted(today.replace(day=1, hour=0, minute=0))
                self.end_term = self.index_all.searchsorted(next_month_start(today)) - 1
            case _:
                self.start_term = 0
                self.end_term = len(self.index_all) - 1
        return slice(self.start_term, i+1), self.end_term - i

