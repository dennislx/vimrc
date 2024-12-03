import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.legend import Legend
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os.path as op

import os
import bisect
import functools
import pynimate as nim
from enum import Enum
from collections import defaultdict
from pynimate.datafier import LineDatafier
from pynimate.lineplot import Lineplot
from pynimate.canvas import Canvas
from typing import Callable, Literal
from tqdm import tqdm
import seaborn as sns

DEFAULT_UPDATE = lambda self, i: None
BAR_WIDTH = 0.5
BAR_ALPHA = 0.5
YLIM_EPS = 0.05
DATA_HOUSE = '/home/dalab2/.proj/graph/pynimate_repo/data_house'

# typing hints
Series = str | pd.Series
Data = pd.DataFrame
MPL_GROUP = Literal["legend", "hist", ...]
Axis = Literal["xaxis", "yaxis"]
YlimEvent = Literal["fix_l1", "fix_l2", "equal_ax", "drop_l2"]
PlotType = Literal["line", "bar", "area"]


class Event(Enum):
    fix_l1 = "fix primary yaxis"
    fix_l2 = "fix secondary yaxis"
    equal_ax = "keep axis identical"
    drop_l2 = "hide secondary axis"
    switch_ax = "switch axis"


Kwarg = dict[Event, any]

# helper function
sum = lambda x: x.sum().reset_index()


def get_date(dt_str: Series, format="%Y%m", errors="coerce", eom=False):
    ret = pd.to_datetime(dt_str, format=format, errors=errors)
    if eom:
        return ret + pd.offsets.MonthEnd(0)
    return ret


def sub_date(dt_str: Series, format="%Y%m", errors="coerce", **subtract_kwargs):
    return (
        pd.to_datetime(dt_str, format=format, errors=errors)
        - pd.offsets.DateOffset(**subtract_kwargs)
    ).dt.strftime(format)


def get_file_or_dir(name: str, folder=None, ext=".csv", parent=DATA_HOUSE):
    if folder:
        parent = op.join(parent, exist_ok=True)
        os.makedirs(parent, exist_ok=True)
    return op.join(parent, "{}{}".format(name, ext))


def generate_all_colors(n_colors: int, palette=["viridis"]):
    all_colors = []
    for pal in palette:
        all_colors.extend(
            list(sns.color_palette(pal, int(np.ceil(n_colors / len(palette)))))
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
        raise ValueError("Invalid value for slicing")

def get_abbr(val: int | float, precision = 2, percentage = False):
    if val == np.nan: return ''
    if percentage: return '{1:,.{0}%}'.format(precision, val)
    magnitude = 0
    while abs(val) >= 1_000:
        magnitude += 1 
        val /= 1_000.0
    return f'{np.round(val, precision)}{["", "K", "M", "B", "T", "Q"][magnitude]}'

def get_balc_all() -> Data:
    fp_df = get_file_or_dir('data_all_together')
    df = pd.read_csv(fp_df, dtype={'YEAR_MONTH': str})
    df['EFF_DT'] = get_date(df['YEAR_MONTH'], format='%Y%m', eom=True)
    return df

def get_balc_upper() -> Data:
    df = get_balc_all()
    return df[['ACTUAL_BALANCE', 'EFF_DT', 'FF_RATE', 'CARD_BALANCE', 'PRDCT_GRP_CD']].set_index('EFF_DT')

def get_lower_left() -> Data:
    df = get_balc_all()
    df = df[['MOB3_BALANCE', 'EFF_DT', 'PRDCT_GRP_CD']].set_index('EFF_DT')
    df['LAST_MOB3'] = df.groupby('PRDCT_GRP_CD')['MOB3_BALANCE'].shift(12)
    return df

def get_lower_middle() -> Data:
    df = get_balc_all()
    df = df[['ATTRITION_RATE', 'EFF_DT', 'PRDCT_GRP_CD']].set_index('EFF_DT')
    df['LAST_MOB3'] = df.groupby('PRDCT_GRP_CD')['ATTRITION_RATE'].shift(12)
    return df

def get_lower_right() -> Data:
    df = get_balc_all()
    df = df[['THIS_RATE', 'EFF_DT', 'COMPETITIONS_RATE']].set_index('EFF_DT')
    return df.drop_duplicates()

def title_case(s: str):
    return "".join(map(min, zip(s, s.title())))

def get_label(col_str: str, date = None):
    kwargs = {}
    if date is not None:
        kwargs.update(
            year=date.year, month=date.strftime('%b'), day=date.strftime('%y%b%d'),
            last_year = date.year - 1
        )
    col_str = col_str.format(**kwargs)
    return ' '.join(map(title_case, col_str.split('_')))

def get_ylimit(data: Data, up_eps = .1, down_eps = .1):
    ylim_max = data.max().max() * (1 + up_eps)
    ylim_min = data.min().min() * (1 + up_eps)
    if np.isnan(ylim_max): ylim_max = None
    if np.isnan(ylim_min): ylim_min = None
    return ylim_min, ylim_max

def find_item(x, lst):
    _lst = [y if y <= x else 0 for y in sorted(lst)]
    return min(_lst, key = lambda y: x- y)

def convert(old_filename, new_filename = None, fps = 1):
    from PIL import Image, PngImagePlugin as PNG
    import io
    images = []
    with Image.open(old_filename) as img:
        for i in range(img.n_frames):
            img.seek(i)
            buf = io
            img.save(buf, format="png")
            buf.seek(0)
            images.append(Image.open(buf))
    duration = 1 / fps * 1_000  # number of miliseconds between each frame
    new_filename = new_filename or old_filename
    img: PNG.PngImageFile = images[0]
    img.save(new_filename, save_all=True, append_images=images[1:], duration=duration, loop=0)

class MyCanvas(Canvas):

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

    def set_mpl_properties(self, group: MPL_GROUP, **kwargs):
        """
        >>> set_mpl_properties('lines', linewidth=4)
        >>> set_mpl_properties('legend', fontsize=22)
        """
        plt.rc(group, **kwargs)
    
    def add_plot(self, plot, row, col=None):
        row_slice, col_slice = map(get_slice, (row, col))
        ax = self.fig.add_subplot(self.ax[row_slice, col_slice])
        plot.set_axes(ax)
        self.length = max(self.length, plot.length)
        self.plots.append(plot)
        return self

    def save(self, filename: str, fps: int, extension: str = 'gif', **kwargs):
        if extension == 'mp4':
            ... # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        super().save(filename, fps, extension, **kwargs)
        if extension == 'gif':
            convert(filename, fps = fps)

class LineBarDatafier(LineDatafier):
    def __init__(
        self, 
        df: Data,
        time_format: str,
        ip_freq: str,
        ip_method: str = 'linear',
        line_cols: list[str] = None,
        bar_cols: list[str] = None,
        group_cols: list[str] = None,
        only_interpolate: bool = False,
        group_excludes: list[str] = None,
        year_data = False,
        line_labels: dict = None,
        bar_labels: dict = None,
    ):
        self.only_interpolate = only_interpolate
        df_transformed = self.transform_data(df, group_cols, line_cols, bar_cols, group_excludes)  
        super(LineBarDatafier, self).__init__(df_transformed, time_format, ip_freq, ip_method)
        self.year_data = year_data
        self.new_added = None
        self.index_all = self.data.index
        self.line_label = {}; self.set_linelabel(line_labels)
        self.bar_label = {}; self.set_barlabel(bar_labels)
        self.event_tracking: dict[int, Kwarg] = defaultdict(dict)

    def transform_data(self, data, group_cols=None, line_cols=None, bar_cols=None, group_excludes=None):
        n = len(group_cols) if group_cols is not None else 0
        if n > 0:
            old_index_name = data.index.name
            group_vars = [x for x in line_cols+bar_cols if x not in group_cols]
            index_vars = [data.index, *group_excludes]
            data = data.pivot_table(index=index_vars, columns=group_vars, values=group_vars)
            data = data.reset_index()
            if old_index_name:
                data = data.set_index(old_index_name)
            data.columns = ['_'.join(col).rstrip('_') for col in data.columns]
            line_cols = [x for b in line_cols for x in data.columns if x.startswith(b)]
            bar_cols = [x for b in bar_cols for x in data.columns if x.startswith(b)]
        self.group_cols = group_cols
        self.line_cols =  line_cols if line_cols is not None else data.columns
        self.bar_cols =  bar_cols if bar_cols is not None else data.columns
        return data

    def write_event(self, event: Event, i: int = None, content: any = True):
        assert Event[event], f"{event} is not a valid event, please add it to the Event class"
        if isinstance(i, int):
            self.event_tracking[i][event] = content
            return
        for i in self.event_tracking:
            if event in self.event_tracking[i]:
                self.event_tracking[i][event] = content
    
    def delete_event(self, event: Event, i: int = None):
        assert Event[event], f"{event} is not a valid event, please add it to the Event class"
        if isinstance(i, int):
            self.event_tracking[i].pop(event, None)
            return
        for i in self.event_tracking:
            if event in self.event_tracking[i]:
                self.event_tracking[i].pop(event, None)

    def set_linelabel(self, line_labels: None | dict):
        if line_labels is None:
            self.line_label = {x: x for x in self.line_cols}
        else:
            self.line_label.update(**line_labels)

    def set_barlabel(self, bar_labels: None | dict):
        if bar_labels is None:
            self.bar_label = {x: x for x in self.bar_cols}
        else:
            self.bar_label.update(**bar_labels)

    def rename(self, **old_new: dict[str, str]):
        self.data.rename(columns=old_new)
    
    def get_ylimit(self, col_cat: PlotType, y_eps = (0, 0), data = None):
        cols = '{}_cols'.format(col_cat)
        if data is None: data= self.data
        assert hasattr(self, cols), f"{cols} not found in the datafier"
        return get_ylimit(data[getattr(self, cols)], *y_eps)
    
    def get_index(self, i):
        today = self.index_all[i]
        if self.year_data:
            start = self.index_all.searchsorted(today.replace(month=1, day=1))
            return slice(start, i+1)
        return slice(None, i+1)

    def interpolate_data(self):
        if not self.ip_method: return self.data
        data = self.data
        ncols = data.select_dtypes(include=['number']).columns
        num_data = data[ncols]
        add_new_flag = False
        if self.ip_freq != None:
            new_ind = pd.date_range(num_data.index.min(), num_data.index.max(), freq=self.ip_freq)
            num_ser = pd.Series([0] * len(new_ind), index=new_ind, name='new_ind').to_frame()
            num_data = num_ser.join(num_data, how='outer').drop('new_ind', axis=1).sort_index()
            self.expanded = (
                num_ser.join(self.expanded, how='outer')
                .drop('new_ind', axis=1)
                .sort_index()
            )
            add_new_flag = True
        num_data = num_data.interpolate(method=self.ip_method)
        obCols = data.select_dtypes(exclude=['number']).columns
        data = data[obCols].join(num_data, how='right')
        data[obCols] = data[obCols].bfill().ffill()
        if add_new_flag:
            new_data = data.loc[new_ind]
            if self.only_interpolate: return new_data
            self.new_added = new_data
        return data

class LineAreaDatafier(LineBarDatafier):
    def __init__(
        self,
        df: Data,
        time_format: str,
        ip_freq: str,
        ip_method: str = 'linear',
        line_cols: list[str] = None,
        area_cols: list[str] = None,
        group_cols: list[str] = None,
        only_interpolate: bool = False,
        group_excludes: list[str] = [],
        area_orders: dict = None,
        area_labels: dict = None,
        line_labels: dict = None,
        year_data = False,
    ):
        super().__init__(df, time_format, ip_freq, ip_method, line_cols, area_cols, group_cols, only_interpolate, group_excludes, year_data, line_labels)
        self.area_order = {}; self.set_areaorder(area_orders)
        self.area_label = {}; self.set_arealabel(area_labels)

    def set_areaorder(self, area_orders: None | dict):
        if area_orders is None:
            self.area_order = (
                self.data[self.area_cols]
                .mean(skipna=True)
                .rank(ascending=False)
                .to_dict()
            )
        else:
            self.area_order.update(**area_orders)

    def set_arealabel(self, area_labels: None | dict):
        if area_labels is None:
            self.area_label = {x: x for x in self.area_cols}
        else:
            self.area_label.update(**area_labels)

    @property
    def area_cols(self): return self.bar_cols

    @property
    def ys(self):
        d = self.area_order
        return sorted(d.keys(), key=lambda x: d[x])
    
    @property
    def labels(self):
        return [self.area_label[x] for x in self.ys]

def get_after(x: str, anchor = '_'):
    idx = x.rfind(anchor)
    if idx == -1: return ''
    return x[idx+1: ]

def truncate_tick(ax: Axes, every=4, axis: Axis = 'axis'):
    labels = getattr(ax, axis).get_ticklabels()
    for i, label in enumerate(labels):
        if i % every != 0:
            label.set_visible(False)

class LineBarplot(Lineplot):
    L1 = 'line'
    L2 = 'bar'

    def __init__(
        self,
        datafier: LineBarDatafier,
        post_update: Callable[[plt.Figure, list[plt.Axes]], None] = DEFAULT_UPDATE,
        bar_width: float = BAR_WIDTH,
        bar_alpha: float = BAR_ALPHA,
        ylim: list[YlimEvent] = ['fix_l1', 'fix_l2'],
        set_bar = True,
        disable_l2 = False
    ):
        fixed_ylim = len(set(ylim) & set(['fix_l1', 'fix_l2'])) > 0
        super().__init__(
            datafier, plattes=['viridis'], post_update=post_update,
            fixed_xlim=True, fixed_ylim=fixed_ylim, xticks=True, yticks=True, grid=True
        )
        self.bar_width = bar_width
        n = len(datafier.bar_cols)
        self.all_cols = self.dfr.data.columns
        self.write_schedule(ylim)
        self.now = 0
        if set_bar:
            self.bar_shift = np.linspace(-bar_width*(n-1)/2, bar_width*(n-1)/2, n)
            self.bar_alpha = bar_alpha
            self.bar_colors = self.column_colors.copy()
            self.bar_hatch = {col: None for col in self.dfr.bar_cols}
        self.enable_l2 = not disable_l2
        if disable_l2:
            self.datafier.bar_cols = []
            self.update = self.update_line
        self.set_legend_pos()

    def write_schedule(self, ylim_settings: list[YlimEvent]):
        def write_schedule_helper(i, data=None):
            for s in ylim_settings:
                match Event[s]:
                    case Event.fix_l1:
                        dfr.write_event('fix_l1', i, dfr.get_ylimit(self.L1, (0, 0.1), data))
                    case Event.fix_l2:
                        dfr.write_event('fix_l2', i, dfr.get_ylimit(self.L2, (0, 0.1), data))
                    case Event.equal_ax:
                        ylim_l1 = dfr.get_ylimit(self.L1, (0, 0.1), data)
                        ylim_l2 = dfr.get_ylimit(self.L2, (0, 0.1), data)
                        y_limit = (minimize(ylim_l1[0], ylim_l2[0]), maximize(ylim_l1[1], ylim_l2[1]))
                        dfr.write_event('fix_l1', i, y_limit)
                        dfr.write_event('fix_l2', i, y_limit)
                    case Event.drop_l2:
                        dfr.write_event('drop_l2', i)
                    case Event.switch_ax:
                        dfr.write_event('switch_ax', i)

        minimize = lambda *x: min(i for i in x if i is not None)
        maximize = lambda *x: max(i for i in x if i is not None)
        dfr: LineBarDatafier = self.datafier
        if not dfr.year_data:
            write_schedule_helper(0)
        else:
            i = 0
            for _, df in dfr.data.groupby(dfr.index_all.year):
                write_schedule_helper(i, df)
                i += len(df)

    def generate_column_colors(self):
        df: LineBarDatafier = self.dfr
        if not df.group_cols:
            colors = generate_all_colors(len(df.bar_cols+df.line_cols), self.palettes)
            return {c: color for c, color in zip(df.data.columns, colors)}
        postfixes = [get_after(x) for x in df.data.columns]
        unique_postfixes = list(set(postfixes))
        colors = generate_all_colors(len(unique_postfixes), self.palettes)
        postfix_color_map = {postfix: color for postfix, color in zip(unique_postfixes, colors)}
        return {x: postfix_color_map[get_after(x)] for x in df.data.columns}

    def set_fontsize(self, font_size=22):
        def get_legend(ax: Axes):
            handler = ax.get_legend()
            return [] if handler is None else handler.get_texts()
        axes = [self.ax, self.ax2] if self.enable_l2 else [self.ax]
        for ax in axes:
            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label] + 
                ax.get_xticklabels() + ax.get_yticklabels()
            ):
                item.set_fontsize(font_size)
    
    def clear_axes(self):
        self.ax.clear()
        self.enable_l2 and self.ax2.clear()
    
    def set_xticks(self):
        super().set_xticks(labelsize=plt.rcParams['axes.labelsize'])
    def set_yticks(self):
        super().set_yticks(labelsize=plt.rcParams['axes.labelsize'])
    
    def set_legend_pos(self, loc=(0.5, -0.05)):
        self.legend_loc = loc

    def set_tick_params(self, **tick_params):
        self.ax.tick_params(**tick_params)
        self.enable_l2 and self.ax2.tick_params(**tick_params)

    def set_axes(self, ax: Axes = None) -> None:
        if ax is not None:
            self.ax = ax
            if self.enable_l2:
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
            self.set_column_colors({c: color for c in col})
        if hatch:
            self.set_column_hatches({c: hatch for c in col})
        if width is not None:
            self.bar_width = width
        if alpha is not None:  
            self.bar_alpha = alpha

    def get_snapshot(self, i, col):
        selected = self.dfr.get_index(i)
        data = self.dfr.data.loc[selected]
        return np.arange(len(data)), data[col]

    def set_xylim(
        self,
        xlim: list[float] = [],
        ylim: dict[str, list[float]] = None,
        x_shift = BAR_WIDTH,
        y_shift = YLIM_EPS,
        fill_missing = True
    ):
        dfr: LineBarDatafier = self.datafier
        if fill_missing and xlim == []:
            xlim = [0-x_shift, len(dfr.data)-1+x_shift]
        if xlim != []:
            self.xlim = xlim
        ylim = ylim or {}
        if fill_missing and self.L1 not in ylim:
            ylim[self.L1] = dfr.get_ylimit(self.L1, (0, y_shift))
        if fill_missing and self.L2 not in ylim:
            ylim[self.L2] = dfr.get_ylimit(self.L2, (0, y_shift))
        if not hasattr(self, 'ylim'):
            self.ylim = {}
        if ylim != {}:
            self.ylim = {**self.ylim, **ylim}
        if ylim != {} and hasattr(self, 'all_cols'):
            dfr.delete_event('fix_l1')
            dfr.delete_event('fix_l2')
    
    def update_L1(self, dfr: LineBarDatafier, index: int):
        for _, col in enumerate(dfr.line_cols):
            x, y = self.get_snapshot(index, col)
            self.ax.plot(
                x+0,
                y,
                label=get_label(dfr.line_label.get(col), date=y.index[-1]),
                color=self.column_colors[col],
                linestyle=self.column_linestyles[col],
                **self.line_props
            )

    def update_L2(self, dfr: LineBarDatafier, index: int):
        for i, col in enumerate(dfr.bar_cols):
            x, y = self.get_snapshot(index, col)
            self.ax2.bar(
                x+self.bar_shift[i],
                y,
                width=self.bar_width,
                color=self.column_colors[col],
                alpha=self.bar_alpha,
                hatch=self.bar_hatch.get(col, None),
                label=get_label(dfr.bar_label.get(col), date=y.index[-1]),
            )   

    def run_schedule(self, events: dict[int, Kwarg], index: int):
        if index not in events: return
        for s, _ in events[index].items():
            match Event[s]:
                case Event.fix_l1:
                    self.ylim[self.L1] = _
                case Event.fix_l2:
                    self.ylim[self.L2] = _
                case Event.drop_l2:
                    self.ax2.yaxis.set_visible(False)
                case Event.switch_ax:
                    self.ax.yaxis.tick_right()
                    self.ax2.yaxis.tick_left()
    
    def update_line(self, i):
        self.ax.clear()
        self.update_L1(self.dfr, i)
        if self.row == 0:
            track = self.dfr.event_tracking
            j = find_item(i, track)
            track[i] = track[j].copy()
        self.run_schedule(self.dfr.event_tracking, i)
    
        self.fixed_xlim and self.ax.set_xlim(self.xlim)
        self.xticks and self.ax.tick_params(**self.xtick_props)
        if self.fixed_ylim and self.L1 in self.ylim:
            self.ax.set_ylim(self.ylim[self.L1])
        self.yticks and self.ax.tick_params(**self.ytick_props)
        self.ax.set_axisbelow(self.grid_behind)

        def prepare_ax(ax):
            for _ in ('top', 'left', 'right'):
                ax.spines[_].set_visible(False)
        prepare_ax(self.ax)    
        h, l = self.ax.get_legend_handles_labels()
        self.ax.set_ytickslabels([f'{get_abbr(x)}' for x in self.ax.get_yticks()])
        self.ax.yaxis.set_major_locator(plt.MaxnLocator(6))
        l: Legend = self.ax.legend(
            h, l, ncol=len(h), loc='upper center', bbox_to_anchor=self.legend_loc, fancybox=True
        )
        self.post_update(self, i)
        for v in self.text_collection.values():
            callback, props_dict = v[0], v[1]
            if callable:
                self.ax.text(
                    s= callback(i, self.datafier),
                    transform=self.ax.transAxes,
                    **props_dict
                )
            else:
                self.ax.text(
                    **props_dict,
                    transform=self.ax.transAxes
                )
        self.now = i

    def update(self, i):
        self.clear_axes()
        self.set_axes()
        self.update_L1(self.dfr, i)
        self.upadata_L2(self.dfr, i)

        if self.now == 0:
            track = self.dfr.event_tracking
            j = find_item(i, track)
            track[i] = track[j].copy()
        self.run_schedule(self.dfr.event_tracking, i)

        self.fixed_xlim and self.ax.set_xlim(self.xlim)
        self.xticks and self.ax.tick_params(**self.xtick_props)

        for l, ax in zip([self.L1, self.L2], [self.ax, self.ax2]):
            if self.fixed_ylim and l in self.ylim:
                ax.set_ylim(self.ylim[l])
        self.yticks and self.set_tick_params(**self.ytick_props)
        self.ax.set_axisbelow(self.grid_behind)

        def prepare_ax(ax):
            for _ in ('top', 'left', 'right'):
                ax.spines[_].set_visible(False)
        hs, ls = [], []
        for ax in (self.ax, self.ax2):
            prepare_ax(ax)
            h, l = ax.get_legend_handles_labels()
            hs.extend(h); ls.extend(l)
            ax.set_ytickslabels([f'{get_abbr(x)}' for x in ax.get_yticks()])
            ax.yaxis.set_major_locator(plt.MaxnLocator(6))
        l: Legend = self.ax.legend(
            hs, ls, ncol=len(hs), loc='upper center', bbox_to_anchor=self.legend_loc, fancybox=True
        )
        self.ax.set_zorder(self.ax2.get_zorder() + 1)
        self.ax.set_frames_on(False)
        self.post_update(self, i)

        for v in self.text_collection.values():
            callback, props_dict = v[0], v[1]
            if callable:
                self.ax.text(
                    s= callback(i, self.datafier),  
                    transform=self.ax.transAxes,
                    **props_dict
                )
            else:
                self.ax.text(
                    **props_dict,
                    transform=self.ax.transAxes
                )
        self.now = i

class LineAreaplot(LineBarplot):
    L1 = 'line'
    L2 = 'area'

    def __init__(
        self,
        datafier: LineAreaDatafier,
        post_update: Callable[[plt.Figure, list[plt.Axes]], None] = DEFAULT_UPDATE,
        area_colors = None,
        area_alpha=None,
        ylim: list[YlimEvent] = ['fix_l1', 'fix_l2'],
        nonstacked_cols=[]
    ):
        super().__init__(
            datafier, post_update=post_update,ylim=ylim, set_bar=False
        )
        self.area_alpha = area_alpha
        self.area_colors = self.set_area_color(area_colors)
        self.nonstacked_cols = nonstacked_cols

    def set_area_color(self, colors):
        dfr: LineAreaDatafier = self.dfr
        area_cols = dfr.ys
        if isinstance(colors, str):
            colors = sns.color_palette(colors)[::2]
            area_colors = {x: c for x,c in zip(area_cols, colors)}
            self.column_colors.update(**area_colors)
        elif isinstance(colors, list):
            area_colors = {x: c for x,c in zip(dfr.area_cols, colors)}
            self.column_colors.update(**area_colors)
        else:
            area_colors = {x: self.column_colors[x] for x in dfr.area_cols}
        return area_colors

    def set_area_properties(self, col, color=None, alpha=None):
        col = [x for x in self.all_cols if x.endswith(col)]
        if color:
            self.area_colors.update({c: color for c in col})
        if alpha is not None:
            self.area_alpha = alpha
    
    def update_L1(self, dfr: LineAreaDatafier, index: int):
        def pp_draw(*col):
            labels, colors = [], []
            for c in col:
                labels.append(get_label(dfr.area_label.get(c)))
                colors.append(self.area_colors.get(c))
            if len(labels) == 1:
                return dict(label=labels[0], color=colors[0], alpha=self.area_alpha)
            else:
                return dict(labels=labels, colors=colors, alpha=self.area_alpha)
        
        data = dfr.data.iloc[:index+1]
        x = np.arange(len(data))
        for y in self.nonstacked_cols:
            self.ax2.fill_between(x, data[y], 0, **pp_draw(y))
        cols = [y for y in dfr.ys if y not in self.nonstacked_cols]
        ys = [data.get(c) for c in cols]
        self.ax2.stackplot(x, *ys, **pp_draw(*cols))

def compare_difference(tbl: Data, *cmp_tuples):
    results = {}
    for name, c1, c2 in cmp_tuples:
        results[name] = (
            ((tbl[c1]-tbl[c2]) / tbl[c2])
            .astype(float)
            .describe()
            .apply(get_abbr, percentage=True)
        )
    return pd.DataFrame(results).T

def post_update(self, i):
    self.ax.set_xtickslabels(self.dfr.data.index.strftime('%Y-%m'), rotation=45, ha='right')
    kwargs = {'color': '#77777', 'size': 12, 'weight': 'bold', 'rotation': 90, 'va': 'center'}
    self.text_collection['line_label'] = (None, {'s': 'Price', **kwargs, 'x': -0.07, 'y': 0.5})
    self.text_collection['bar_label'] = (None, {'s': 'Volume', **kwargs, 'x': 1.07, 'y': 0.5})

def synthetic_data():
    start_date = pd.to_datetime('2020-01-01')
    end_date = pd.to_datetime('2021-01-01')
    num_dates = 24
    np.random.seed(42)
    random_dates = pd.to_datetime(
        np.random.randint(
            start_date.value, end_date.value, num_dates, dtype=np.int64
        )
    ).strftime("%Y%m%d").to_list()
    data = {
        'date': np.tile(random_dates),
        'price': np.random.uniform(50, 150, len(random_dates)*2),
        'volume': np.random.randint(100, 1000, len(random_dates)*2),
        'group': np.repeat(['A', 'B'], len(random_dates))
    }
    df = pd.DataFrame(data).set_index('date')
    return df

def synthetic_linebar(df:Data):
    dfr = LineBarDatafier(
        df, time_format='%Y%m%d', ip_freq='ME', ip_method='pad',
        line_cols=['price'], bar_cols=['volume'], group_cols=['group'],
        only_interpolate=True
    )
    plot = LineBarplot(dfr, post_update=post_update, bar_width=0.4)
    plot.set_line_properties(col='A', linestyle='dashed')
    plot.set_bar_properties(col='A', hatch='//')
    return plot

def synthetic_linearea(df:Data):
    df['ceil'] = df['volume'].max()
    dfr = LineAreaDatafier(
        df, time_format='%Y%m%d', ip_freq='ME', ip_method='pad',
        line_cols=['price'], area_cols=['volume', 'ceil'], group_cols=['group'],
        group_excludes=['ceil'], only_interpolate=True
    )
    plot = LineAreaplot(dfr, post_update=post_update, area_colors='Blues', area_alpha=0.3)
    plot.set_line_properties(col='A', linestyle='dashed')
    return plot

def setup_canvas(figsize=(30,15), nrows=1, ncols=1, **kwargs):
    cnv = MyCanvas(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    cnv.set_mpl_properties('lines', linewidth=4)
    cnv.set_mpl_properties('legend', fontsize=30)
    cnv.set_mpl_properties('axes', labelsize=30)
    return cnv

fn_save = lambda x: get_file_or_dir(x, folder='Testing', ext='', parent='../assets')

def setup(func, figsize=(30, 15), nframes=None, fp=5):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cnv: nim.Canvas = setup_canvas(figsize=figsize)
        saved_name = func.__name__
        plot = func(*args, **kwargs)
        if kwargs.pop('return_plot', False):
            cnv.fig.clear()
            return plot
        cnv.add_plot(plot, 0, 0)
        nframes = kwargs.get('nframes', _nframes)
        if nframes == 1:
            frames = range(plot.length - 1, plot.length)
        else:
            frames = range(plot.length-(nframes or plot.length), plot.length)
            saved_name = saved_name.lstrip('test')
        cnv.animate(frames_callback=lambda x: frames, repeat=False)
        cnv.save(fn_save(saved_name.title()), fps=fp)
    _nframes = nframes
    return wrapper

def pprocess_upper(self, i):
    index = self.dfr.data.index
    xticks = np.arrange(len(index))[::12]
    xlabels = index.strftime('%Y')[xticks]
    self.ax.set_xticks(xticks)
    self.ax.set_xticklabels(xlabels, ha='center')
    self.ax.set_ytickslabels([f'{get_abbr(x, percentage=True)}' for x in self.ax.get_yticks()])
    kwargs = {'color': '#777777', 'size': 35, 'weight': 'bold'}
    self.text_collection['line_label'] = (None, {'s': 'Actual Balance', **kwargs, 'x': -0.02, 'y': 1.01, 'ha': 'left'})

@setup
def testrun_upper(return_plot=False, nframes=1) -> nim.Baseplot:
    df = get_balc_upper()
    dfr = LineAreaDatafier(
        df, time_format='%Y%m', ip_freq='MS', ip_method='pad',
        line_cols=['ACTUAL_BALANCE'], area_cols=['CARD_BALANCE'],
        group_cols=['PRDCT_GRP_CD'], only_interpolate=True
    )
    plot = LineAreaplot(
        dfr, time_format='%Y%m', ip_freq='MS', ip_method='pad',
        line_cols=['ACTUAL_BALANCE'], area_cols=['CARD_BALANCE'],
        group_cols=['PRDCT_GRP_CD'], only_interpolate=True
    )
    plot.set_legend_pos(loc=(0.5, 1.1))
    plot.set_xylim(xlim=[-0.5, len(dfr.data)-0.5], ylim={plot.L1: [0, 1_000_000_000], plot.L2: [0, 1_000_000_000]})
    plot.set_line_properties(col='CARD_BALANCE', linestyle='dashed')
    return plot








        


        
        

        
        

    
