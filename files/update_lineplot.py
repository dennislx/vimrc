import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.patches as patches
import pandas as pd
import functools
import pynimate as nim

from pynimate.utils import human_readable
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from typing import Literal, Dict, List
from types import MethodType, SimpleNamespace
from tqdm import tqdm
from enum import Enum
from collections import defaultdict, namedtuple

for side in ["left", "right", "top", "bottom"]:
    mpl.rcParams[f"axes.spines.{side}"] = False

MONTH_IDX = '%Y-%m'
StackBar = namedtuple('StackBar', ['total', 'prev'])

BAR_COLS = {
    "A": "#2a9d8f",
    "B": "#e9c46a",
    "C": "#e76f51",
    "D": "#a7c957",
    "E": "#e5989b",
}

class PlotType:
    area = 'area'
    bar  = 'bar'
    line = 'line'

def dark_theme():
    mpl.rcParams["figure.facecolor"] = "#001219"
    mpl.rcParams["axes.facecolor"] = "#001219"
    mpl.rcParams["savefig.facecolor"] = "#001219"

def calc_errorbar(dct_lst: Dict[str, List]):
    summed_values = np.nansum(list(dct_lst.values()), axis=0)
    return {
        'mean': np.mean(summed_values),
        'std': np.std(summed_values)
    }


def setup_1(update_func):
    @functools.wraps(update_func)
    def wrapper(*args, **kwargs):
        global x, y, line, ax  # Define as global variables
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        fig, ax = plt.subplots()
        line, = ax.plot(x, y)
        ani = FuncAnimation(fig, update_func, frames=len(x), interval=50)
        ani.save('assets/expand_28=NOV.mp4', writer='ffmpeg', fps=30)
    return wrapper

@setup_1
def h1_animate(i):
    x_data = x[:i+1]
    y_data = y[:i+1]
    line.set_data(x_data, y_data)
    ax.relim()
    ax.autoscale_view()


def h2_post(self, i):
    self.ax.yaxis.set_major_formatter(
        tick.FuncFormatter(lambda x, pos: human_readable(x))
    )
    self.ax.xaxis.set_major_locator(mdates.MonthLocator())
    self.ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

def next_month_start(date: pd.Timestamp):
    return (date+pd.DateOffset(months=1)).replace(day=1, hour=0, minute=0)

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

class MyDatafier(nim.LineDatafier):
    def __init__(
        self, data, time_format, ip_freq, 
        ip_method = "linear", 
        term: Literal['year', 'none', 'month'] = 'none'
    ):
        super().__init__(data, time_format, ip_freq, ip_method)
        self.index_all = self.data.index
        self.start_term, self.end_term = 0, 0
        self.set_plot_type()
        self.term = term
        self.gather_history()

    def set_plot_type(self, p_type: Dict[str, PlotType] | str = {}):
        if isinstance(p_type, str):
            self.plot_type = {c: p_type for c in self.data.columns}
        else:
            self.plot_type = p_type or {c: 'line' for c in self.data.columns}

    def gather_history(self):
        match self.term:
            case 'year':
                self.history = gather_history(self.raw_data)
                self.history = {k: calc_errorbar(v) for k,v in self.history.items()}
            case _:
                self.history = {}

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
    


class MyLinePlot(nim.Lineplot):
    def __init__(self, datafier, post_update=...):
        super().__init__(
            datafier, ['viridis'], 
            post_update=post_update,
            scatter_markers=False, 
            legend=True, 
            fixed_xlim=False, 
            grid=False
        )
        self.lines = {}
        self.annots = {}
        self.text_objects = {}
        self.X = self.dfr.data.index
        self.Y = self.dfr.data
        self.set_legend_pos()

    def set_legend_pos(self, loc=(0.5, -0.05)):
        self.legend_loc = loc

    def get_snapshot(self, i, col):
        selected = self.dfr.get_index(i)
        data = self.dfr.data.iloc[selected]
        return data.index, data[col] 

    def update_lines(self, row, col, gap=None, ax=None):
        X, Y = self.X, self.Y[col]
        # Update the line data
        if col in self.lines:
            # Update existing line
            self.lines[col].set_data(X[row], Y[row])
        else:
            # Create a new line if it doesn't exist
            ax = ax or self.ax
            self.lines[col], = ax.plot(
                X[row],
                Y[row],
                color=self.column_colors[col],
                linestyle=self.column_linestyles[col],
                label=col,
                **self.line_props,
            )
    
    def update_annots(self, row, col, gap=None, ax=None, y_offset=0, x_offset=0):
        X, Y = self.X, self.Y[col]
        # Update the annotation
        i = row.stop - 1
        x = mdates.date2num(X[i]) + x_offset
        y = Y.iloc[i] + y_offset

        if i == row.start:
            del self.annots[col]
        
        if self.line_annots:
            if col in self.annots:
                # Update existing annotation
                self.annots[col].set_text(self.line_annot_props["callback"](col, y))
                self.annots[col].set_position((x, y))
            else:
                # Create a new annotation if it doesn't exist
                ax = ax or self.ax
                self.annots[col] = self.ax.annotate(
                    self.line_annot_props["callback"](col, y),
                    (x, y),
                    **self.line_annot_props["kwargs"],
                )
            

    def update_text_objects(self, i, ax=None):
        for key, v in self.text_collection.items():
            callback, props_dict = v[0], v[1]
            if key not in self.text_objects:
                ax = ax or self.ax
                # Create the text object only once and store it
                self.text_objects[key] = self.ax.text(
                    transform=self.ax.transAxes,  # Apply transformation once
                    s=callback(i, self.datafier) if callback else "", 
                    **props_dict,
                )
            else:
                # Update the existing text object
                if callback:
                    self.text_objects[key].set_text(callback(i, self.datafier))

    def draw_trend_arrow(self) -> None:
    # Calculate the start and end points of the arrow
        x_start, y_start = self.X[0], self.Y.iloc[0].mean()
        x_end, y_end = self.X[-1], self.Y.iloc[-1].mean()

        arrow = patches.FancyArrowPatch( 
            (x_start, y_start),
            (x_end, y_end),
            mutation_scale=15,
            color='red',
            arrowstyle='->',
            linewidth=2,
        )
        self.ax.add_patch(arrow)

    def update_legend_grid(self):
        if self.legend: 
            handles, labels = self.ax.get_legend_handles_labels()
            # Remove duplicate labels
            by_label = dict(zip(labels, handles))
            legend = self.ax.legend(
            by_label.values(), by_label.keys(),
            ncol=len(by_label),
            loc='upper center',
            bbox_to_anchor=self.legend_loc,
            fancybox=True,
            **self.legend_props
            )
        if self.grid: self.ax.grid(**self.grid_props)
        self.ax.set_axisbelow(self.grid_behind)

    def _update(self, i):
        row, gap = self.dfr.get_index(i)
        for col in self.dfr.data.columns:
            if self.validate_col(col):
                self.update_lines(row, col, gap)
                self.update_annots(row, col, gap)

    def update(self, i) -> None:
        # Loop through columns and update their corresponding line and annotation
        self._update(i)
        self.post_update(self, i)
        self.update_text_objects(i)
        self.update_graph_prop()
        self.ax.relim()
        self.ax.autoscale_view(scalex=True, scaley=True)

    def update_graph_prop(self):
        if self.fixed_xlim:
            self.ax.set_xlim(self.xlim)
        if self.fixed_ylim:
            self.ax.set_ylim(self.ylim)
        if self.xticks:
            self.ax.tick_params(**self.xtick_props)
        if self.yticks:
            self.ax.tick_params(**self.ytick_props)
        self.update_legend_grid()

    def validate_col(self, col):
        return self.dfr.plot_type.get(col, None) == 'line'

class StackBottom:
    def __init__(self):
        self.stack_x = defaultdict(dict)
        self.visited_vintage = set()
        self.stack_y = {}
    def reset(self):
        self.visited_vintage.clear()
    def clear(self):
        self.stack_x.clear()
    def visit(self, key):
        self.visited_vintage.add(key)
    def update_position(self, vintage, pos):
        self.stack_y[vintage] = pos
    def update(self, vintage, col, val, additive=False):
        this = self.stack_x[vintage]
        if additive:
            this[col] = this.get(vintage, 0) + val
        else:
            this[col] = val
        self.visit(col)
    def __getitem__(self, vintage):
        return sum(x for k, x in self.stack_x[vintage].items() if k in self.visited_vintage)
    def __repr__(self):
        repr_str = ""
        for period, subdict in self.stack_x.items():
            repr_str += f"{period}:\n"
            for key, val in subdict.items():
                repr_str += f"  {key} = {val}\n"
        return repr_str
        


class MyBarPlot(MyLinePlot):
    """
    >>> dfr = MyLineDatafier(df, "%Y-%m-%d", "12h", term='year')      # this is the covid case data
    >>> dfr.set_plot_type({ 'cases': 'bar', 'cured': 'bar' })
    >>> plot = MyBarPlot( dfr, post_update=h2_post)
    """
    def __init__(self, datafier, post_update=...):
        super().__init__(datafier, post_update)
        self.bars = {}
        self.now = 0
        self.box_plot = None
        self.now = None
        self.bar_bottoms = StackBottom()
    
    def _update(self, i):
        row, gap = self.dfr.get_index(i)
        self.bar_bottoms.reset()
        for col in self.dfr.data.columns:
            if self.validate_col(col):
                self.update_bars(row, col)

    def update_box(self):
        ...

    def update_bars(self, row, col, gap=None, ax=None):
        i = row.stop - 1
        X, Y = self.X, self.Y[col]

        if X[i].strftime(MONTH_IDX) != self.now:
            # remove all artisits of bar, a new month is coming 
            for _, bar in self.bars.items():
                bar.remove()
            self.bars.clear()
            self.now = X[i].strftime(MONTH_IDX)
            self.bar_bottoms.clear()
            self.update_box()
        
        if col in self.bars:
            bar = self.bars[col][-1]
            bar.set_height(Y.iloc[i])
            bar.set_y(self.bar_bottoms[self.now])
            self.bar_bottoms.update(self.now, col, Y.iloc[i])
        else:
            # this will restore bottom value for each previous months
            ax = ax or self.ax
            df_bar = Y[row].groupby(X[row].to_period('M')).last()
            bar_bottoms = [self.bar_bottoms[str(i)] for i in df_bar.index]
            self.bars[col] = ax.bar(
                df_bar.index.to_timestamp(), 
                df_bar.values, 
                width=20,
                bottom=bar_bottoms,
                color=self.column_colors[col],
                label=col, 
                alpha=0.8
            )
            for j, v in zip(df_bar.index, df_bar.values):
                self.bar_bottoms.update(str(j), col, v)
            
    def validate_col(self, col):
        return self.dfr.plot_type.get(col, None) == 'bar'

class MyBarhPlot(MyLinePlot, nim.Barhplot):
    def __init__(self, 
        datafier: MyDatafier,
        palettes: list[str] = ["viridis"],
        post_update = lambda self, i: None,
        bar_width=20
    ):
        nim.Barhplot.__init__(
            self, datafier, palettes, post_update, 
            annot_bars=True, 
            rounded_edges=True, 
            fixed_xlim=False, 
            xticks=True, 
            yticks=True, 
            grid=False
        )
        self.fixed_ylim, self.fixed_xlim = False, False
        self.legend = True
        self.set_legend()
        self.set_legend_pos()
        self.text_objects = {}
        self.annots = {}
        self.X, self.Y = self.dfr.index_all, self.dfr.data
        self.set_barh(bar_height=bar_width)
        self.set_bar_border_props(
            edge_color="black", pad=0.1, mutation_aspect=1, radius=0.2, mutation_scale=0.6
        )
        self.bars = {}
        self.now = None
        self.error_plot = None
        self.bar_left = StackBottom()

    def set_xylim(self,  xlim: list[float] = [], ylim: list[float] = [], xoffset = 5, yoffset = 0.6):
        super(nim.Barhplot, self).set_xylim(xlim, ylim)

    def _update(self, i):
        row, gap = self.dfr.get_index(i)
        self.bar_left.reset()
        self.get_ith_bar_attrs(i, row, gap)
        for col in self.dfr.data.columns:
            if self.validate_col(col):
                self.update_bars(row, col)
        self.update_annots()
        self.update_history()

    def update_history(self):
        if self.bar_attr.now != self.bar_attr.just_now:
            if self.error_plot: 
                self.error_plot.remove()
            hist = {k: v for k, v in self.dfr.history.items() if k in self.bar_left.stack_y}
            x = [v['mean'] for v in hist.values()]
            y = self.bar_left.stack_y.values()
            xerr = [v['std'] for v in hist.values()]
            self.error_plot = self.ax.errorbar(
                x, y, xerr=xerr, fmt='o', color='#FF5733', ecolor='#a75750', elinewidth=2, capsize=5,
                markersize=10, markeredgewidth=1, markeredgecolor='#a75750'
            )
            # for xi, yi in zip(x, y):
            #     self.text_collection[f'hist-{self.now}'] = (lambda *x: human_readable(xi), {
            #         'x': xi, 'y': yi, 'color': 'w', 'fontsize': 10, 'ha':'center', 'va':'center'
            #     })

    def get_ith_bar_attrs(self, i, row, gap) -> SimpleNamespace:
        bar_attr = SimpleNamespace(
            now      =  self.X[i].strftime(MONTH_IDX),
            is_start =  i==row.start,
            is_end   =  gap==0,
            value    =  {c: self.Y.iloc[i][c] for c in self.dfr.plot_type if self.validate_col(c)},
            i        =  i,
            just_now = self.now
        )
        if bar_attr.now != bar_attr.just_now:
            for _, bar in self.bars.items():
                bar.remove()
            self.bars.clear()
            self.bar_left.clear()
            self.now = bar_attr.now

        self.bar_attr = bar_attr

    def update_bars(self, row, col, ax=None):
        if col in self.bars:
            bar = self.bars[col][-1]
            bar.set_width(self.bar_attr.value[col])
            bar.set_x(self.bar_left[self.now])
            self.bar_left.update(self.now, col, self.bar_attr.value[col])
        else:
            ax = ax or self.ax
            df_bar = self.Y[row].groupby(self.X[row].to_period('M'))[col].last()
            df_key = df_bar.index.to_timestamp()
            self.bars[col] = ax.barh(
                y = df_key,
                width = df_bar.values,
                left  = [self.bar_left[str(i)] for i in df_bar.index],
                color = self.column_colors[col],
                label = col,
                **self.barh_props
            )
            self.use_fancybox(col)
            
            for j, v, y in zip(df_bar.index, df_bar.values, df_key):
                self.bar_left.update(str(j), col, v)
                self.bar_left.update_position(str(j), mdates.date2num(y))

    def use_fancybox(self, col):
        new_patches = []
        default_kwargs = dict(
            boxstyle='round,pad={pad},rounding_size={radius}'.format(**self.bar_border_props),
            mutation_aspect=self.bar_border_props['mutation_aspect'],
            **self.bar_border_props['kwargs']
        )
        for patch in reversed(self.bars[col]):
            bb    = patch.get_bbox()
            p_bbox = patches.FancyBboxPatch(
                (bb.xmin, bb.ymin), abs(bb.width), abs(bb.height),
                fc=patch.get_facecolor(),
                zorder=patch.zorder,
                **default_kwargs
            )
            patch.remove()
            new_patches.insert(0, p_bbox)
        for patch in new_patches:
            self.ax.add_patch(patch)
        self.bars[col] = mpl.container.BarContainer(new_patches, orientation='horizontal')
        if self.ax.legend_ is not None:
            self.ax.legend_.remove()  # Clear the legend before adding new bars
        self.ax.add_container(self.bars[col])

    def update_annots(self):
        now = self.bar_attr.now
        self.bar_attr.is_start and self.annots.pop(now, None)
        if self.annot_bars:
            for z, vintage in enumerate(self.bar_left.stack_x):
                text = self.bar_left[vintage]
                x  = self.bar_annot_props['xoffset'] + text
                y  = self.bar_annot_props['yoffset'] + self.bar_left.stack_y[vintage]
                if now in self.annots:
                    if vintage != now: continue
                    self.annots[now].set_text(human_readable(text))
                    self.annots[now].set_position((x, y))
                    break
                self.annots[vintage] = self.ax.text(
                    x, y, human_readable(text), 
                    ha=self.bar_annot_props["ha"],
                    **self.bar_annot_props["kwargs"],
                    zorder=z
                )

    def validate_col(self, col):
        return self.dfr.plot_type.get(col, None) == 'bar'

    
class MyAreaPlot(MyLinePlot):
    def __init__(self, datafier, post_update=..., stacked_cols=[], line_cols=[]):
        super().__init__(datafier, post_update)
        self.stacked_cols = stacked_cols
        self.stacked_plot = None
        self.areas = {}

    def set_axes(self, ax: Axes = None) -> None:
        if ax is not None:
            self.ax = ax
            self.ax2 = ax.twinx()

    def update(self, i):
        super().update(i)
        # add second graph -- line plot, fix xlimit only
        row, gap = self.dfr.get_index(i)
        self.update_lines(row, 'rate', ax=self.ax2)
        self.ax2.relim()
        self.ax2.autoscale_view(scalex=True, scaley=False)

    def _update(self, i):
        row, gap = self.dfr.get_index(i)

        X, Y = self.X[row], self.Y[row]
        if self.stacked_plot is not None:
            for poly in self.stacked_plot:
                poly.remove()
        self.stacked_plot = self.ax.stackplot(
            X, *[Y.get(c) for c in self.stacked_cols], 
            alpha=0.7, 
            labels=self.stacked_cols, 
            colors=[self.column_colors[c] for c in self.stacked_cols]
        )
        if 'aggregated' in self.annots:
            self.annots['aggregated'].remove()
        x, y = mdates.date2num(X[-1]), sum(self.Y.iloc[i][x] for x in self.stacked_cols)
        self.annots['aggregated'] = self.ax.annotate(
            self.line_annot_props["callback"](None, y),
            (x, y), **self.line_annot_props["kwargs"],
        )

        for col in self.dfr.data.columns:
            if self.dfr.plot_type.get(col, None) == 'area' and col not in self.stacked_cols:
                kwargs = dict(alpha=0.6, label=col, color=self.column_colors[col])
                if col in self.areas: self.areas[col].remove()
                self.areas[col] = self.ax.fill_between(X, Y[col], 0, **kwargs)
                self.update_lines(row, col, gap)
                self.update_annots(row, col, gap)
    
    def update_legend_grid(self):
        if self.legend: 
            hs, ls = [], []
            for ax in (self.ax, self.ax2):
                for h, l in zip(*ax.get_legend_handles_labels()):
                    if isinstance(h, Line2D) and l == 'total': continue
                    hs.append(h); ls.append(l)
            l: Legend = self.ax.legend(
                hs, ls, ncol=len(ls), 
                loc='upper center', 
                bbox_to_anchor=self.legend_loc, 
                fancybox=True,
                **self.legend_props 
            )
        self.ax.set_zorder(self.ax2.get_zorder() - 1)
        if self.grid: self.ax.grid(**self.grid_props)
        self.ax.set_axisbelow(self.grid_behind)



class MyCanvas(nim.Canvas):
    """
    event_source [TimerTk object]
    """
    def save(self, filename, fps, extension = "gif", **kwargs):
        def update_pbar(i, total):
            if self.pbar.total is None:
                self.pbar.reset(total = total)
            self.pbar.update(1)
        self.pbar = tqdm(total=None, desc='Saving Frames')
        return super().save(
            filename, fps, extension, progress_callback = update_pbar,
            **kwargs
        )
        

def h2_animate():
    df = pd.read_csv("data_house/covid_IN.csv").set_index("time") # 558, 2
    df['total'] = (df['cases'] + df['cured']) * np.random.uniform(1.25, 1.75, size=len(df))
    df['rate'] = np.random.uniform(0, 1, size=len(df))

    cnv = MyCanvas()
    dfr = MyDatafier(df, "%Y-%m-%d", "ME", term='year')                 # 1115, 2
    dfr.set_plot_type({
        'cases': 'area', 'cured': 'area', 'total': 'area', 'rate': 'line'
    })
    plot = MyAreaPlot( dfr, post_update=h2_post, stacked_cols=['cases', 'cured'])
    # plot.update = MethodType(update, plot)
    plot.set_line_annots(lambda col, val: f"({human_readable(val)})", color="k")
    plot.set_time(
        callback=lambda i, datafier: datafier.data.index[i].strftime("%d %b, %Y"),
        color="b", size=15,
    )
    cnv.add_plot(plot)
    cnv.animate(frames_callback=lambda x: range(plot.length - 100, plot.length))
    cnv.save("assets/areaplot_dark", 12, extension='gif')


def post_update(self, i):
    # annotates continents next to bars
    for ind, (bar, x, y) in enumerate(
        zip(self.bar_attr.top_cols, self.bar_attr.bar_length, self.bar_attr.bar_rank)
    ):
        self.ax.text(
            x - 0.3,
            y,
            self.dfr.col_var.loc[bar, "continent"],
            ha="right",
            color="k",
            size=12,
            zorder=ind,
        )

def h3_animate():
    mpl.rcParams["axes.facecolor"] = "#001219"
    df = pd.read_csv("data_house/sample.csv").set_index("time") # 558, 2
    col_var = pd.DataFrame(
        {
            "columns": ["Afghanistan", "Angola", "Albania", "USA", "Argentina"],
            "continent": ["Asia", "Africa", "Europe", "N America", "S America"],
        }
    ).set_index("columns")

    bar_cols = {
        "Afghanistan": "#2a9d8f",
        "Angola": "#e9c46a",
        "Albania": "#e76f51",
        "USA": "#a7c957",
        "Argentina": "#e5989b",
    }

    cnv = MyCanvas(figsize=(12.8, 7.2), facecolor="#001219")
    dfr = nim.BarDatafier(df, "%Y-%m-%d", "3d")
    dfr.add_var(col_var=col_var)
    bar = MyBarhPlot(dfr, post_update=post_update, bar_width=0.85)
    bar.set_column_colors(bar_cols)
    bar.set_bar_annots(color="w", size=13)
    bar.set_bar_border_props(
        edge_color="black", pad=0.1, mutation_aspect=1, radius=0.2, mutation_scale=0.6
    )
    cnv.add_plot(bar)
    cnv.animate(frames_callback=lambda: range(bar.length-100, bar.length))
    cnv.save("assets/barplot_work", 30, extension='gif')

def h4_post(self, i):
    self.ax.xaxis.set_major_formatter(
        tick.FuncFormatter(lambda x, pos: human_readable(x))
    )
    self.ax.yaxis.set_major_locator(mdates.MonthLocator())
    self.ax.yaxis.set_major_formatter(DateFormatter('%b'))

def h4_animate():
    mpl.rcParams["axes.facecolor"] = "#001219"
    cnv = MyCanvas(figsize=(12.8, 7.2), facecolor="#001219")
    df = pd.read_csv("data_house/covid_IN.csv").set_index("time") # 558, 2
    dfr = MyDatafier(df, "%Y-%m-%d", "1d", term='year')                 # 1115, 2
    dfr.set_plot_type({ 'cases': 'bar', 'cured': 'bar' })
    plot = MyBarhPlot( dfr, post_update=h4_post)
    plot.set_bar_annots(color="w", size=9, xoffset=3000)
    plot.set_time(
        callback=lambda i, datafier: datafier.data.index[i].strftime("%d %b, %Y"),
        color="b", size=15,
    )
    plot.set_column_colors({'cases': "#e5989b", 'cured': "#a7c957"})
    plot.set_bar_border_props(
        edge_color="black", pad=0.1, mutation_aspect=1, radius=0.2, mutation_scale=0.6
    )
    plot.set_legend(labelcolor='w')
    plot.set_xticks(colors="w", length=0, labelsize=13)
    plot.set_yticks(colors="w", labelsize=13)
    cnv.add_plot(plot)
    cnv.animate(frames_callback=lambda x: range(plot.length - 100, plot.length))
    cnv.save("assets/barplot_dark", 5, extension='gif')

if __name__ == '__main__':
    h4_animate()

