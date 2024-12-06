__all__ = [
    'MyLinePlot', 'get_abbr', 'MyAreaPlot', 'MyBarPlot'
]

import numpy as np
np.random.seed(42)
import pandas as pd
import pynimate as nim

import matplotlib.patches as patches
import matplotlib.dates as mdates
import matplotlib.text as mtext
import matplotlib.lines as mline
import matplotlib.collections as mcollect
import matplotlib.container as mcontainer
import matplotlib.axes as maxes

from typing import Dict, Literal, List
from collections import defaultdict
from types import MethodType, SimpleNamespace
from .draft_util import *

MONTH_IDX = '%Y-%m'
BAR_WIDTH = 20
MyPlotProp = Literal[
    'grid', 'annot', 'marker', 'head', 'xtick', 'ytick', 
    'legend', 'xlim', 'ylim', 'column'
]

def get_abbr(val: int | float, format: Literal['percentage', 'number', 'none'] = 'number', precision = 2):
    if val == np.nan: return ''
    match format:
        case 'percentage':
            return '{1:,.{0}%}'.format(precision, val)
        case 'number':
            magnitude = 0
            while abs(val) >= 1_000:
                magnitude += 1 
                val /= 1_000.0
            return f'{np.round(val, precision)}{["", "K", "M", "B", "T", "Q"][magnitude]}'
        case _:
            raise NotImplementedError()


def get_label(col_str: str, date = None) -> str:
    """
    Generate a formatted label from a column string and an optional date.
    Args:
        col_str (str): The column string to be formatted. It can contain placeholders for date components.
        date (datetime, optional): A datetime object to extract year, month, and day information. Defaults to None.
    """
    kwargs = {}
    title_case = lambda s: "".join(map(min, zip(s, s.title())))
    if date is not None:
        kwargs.update(
            year=date.year, month=date.strftime('%b'), day=date.strftime('%y%b%d'),
            last_year = date.year - 1
        )
    col_str = col_str.format(**kwargs)
    return ' '.join(map(title_case, col_str.split('_')))

def remove_from(obj: dict, key: str):
    x = obj.pop(key, None)
    if x is not None: x.remove()

def l2d(list_of_dicts):
    return {key: [d[key] for d in list_of_dicts] for key in list_of_dicts[0].keys()}

def d2d(**dict_of_dicts):
    all_keys = set().union(*dict_of_dicts.values())
    result = {key: {dict_name: d.get(key) for dict_name, d in dict_of_dicts.items()} for key in all_keys}
    return result

class MyLinePlot(nim.Lineplot):
    def __init__(self, datafier, post_update=lambda *x: None):
        super().__init__(
            datafier, ['viridis'], 
            post_update=post_update,
            scatter_markers=False, 
            legend=True, 
            fixed_xlim=False, 
            grid=False
        )
        self.set_prop_on(['annot', 'legend', 'xtick', 'ytick'])
        self.now = None
        self.init_props()

    @property
    def data(self) -> MyDatafier: 
        return self.dfr
    
    def get_label(self, col: str): 
        col_ = self.data.column_label.get(col, col)
        return get_label(col_)

    def get_format(self, col: str):
        return self.data.column_format.get(col, 'number')

    def set_prop_on(self, props: List[MyPlotProp] = []):
        all_props = MyPlotProp.__args__
        self.graph_switch: Dict[MyPlotProp, bool] = {
            c: True if c in props else False for c in all_props
        }
        
    def __repr__(self):
        # for each self.xxx_props that is not empty dict, print it as a newline attr1=va1, attr2=val2 ...
        # if self.now is not None, then also print self.ith_info as a new line 
        repr_str = f"<{self.__class__.__name__} now={self.now}>\n"
        for prop in self.graph_switch:
            prop_dict = getattr(self, f'{prop}_props')
            if prop_dict:
                status = 'on' if self.graph_switch[prop] else 'off'
                repr_str += f"{prop}({status}): ".ljust(13)
                repr_str += ", ".join(f"{k}={v}" for k, v in prop_dict.items()) + "\n"
        if self.now is not None:
            repr_str += f"current frame: ".ljust(13)
            repr_str += ", ".join(f"{k}={v}" for k, v in vars(self.ith_info).items()) + "\n"
        return repr_str
        
    def init_props(self):
        self.line_family: Dict[str, mline.Line2D] = {}
        self.annot_family: Dict[str, mtext.Annotation] = {}
        self.text_family: Dict[str, mtext.Text] = {}
        
        self.xlim_props, self.ylim_props, self.column_props = {}, {}, {}
        del self.line_annot_props, self.line_head_props
        self.annot_props, self.head_props = {}, {}

        self.update_props( 'legend', bbox_to_anchor=(0.5, -0.1), loc='upper center', fancybox=True)
        self.update_props( 'grid', which='major', axis='x', linestyle='-')
        self.update_props( 'xtick', color='#777777', action='replace')
        self.update_props( 'ytick', color='#777777', action='replace')
        self.update_props( 'xlim', left=self.xlim[0], right=self.xlim[1])
        self.update_props( 'ylim', bottom=self.ylim[0], top=self.ylim[1])
        self.update_props( 'annot', callback=lambda c, y, f: f"{c}({get_abbr(y, f)})")
        self.update_props( 'head', edgecolors='k' )

        column_style = d2d(color=self.column_colors, linestyle=self.column_linestyles)
        self.update_props( 'column', **column_style )
        
        
    def update_props(
        self, 
        group: MyPlotProp,
        action: Literal['update', 'replace'] = 'update',
        **kwargs
    ):
        def f(prop, **prop_kwargs):
            this = getattr(self, f'{prop}_props')
            if action == 'update': this.update(**prop_kwargs)
            elif action == 'replace': setattr(self, f'{prop}_props', prop_kwargs)
        match group:
            case 'xtick':
                f('xtick', axis='x', **kwargs)
            case 'ytick':
                f('ytick', axis='y', **kwargs)
            case 'column':
                f('column', **kwargs)
                self.data.column_style = self.column_props
            case _:
                f(group, **kwargs)

    def update_lines(self, row, col, gap=None, ax=None):
        Y = self.data[row][col] # data from history until now
        # Update the line data
        if col in self.line_family:
            # Update existing line
            self.line_family[col].set_data(Y.index, Y)
        else:
            # Create a new line if it doesn't exist
            ax = ax or self.ax
            self.line_family[col], = ax.plot(
                Y.index,
                Y,
                label=col,
                **self.column_props[col],
                **self.line_props,
            )
    
    def update_annots(self, col, ax=None, y_offset=0, x_offset=0, x=None, y=None):
        x = (x or mdates.date2num(self.ith_info.x)) + x_offset
        text = (y or self.ith_info.y[col])
        y = text + y_offset

        if not self.graph_switch['annot']: return 
        f = self.annot_props.get("callback", lambda c, y: f"{y}")
        kwargs = {k: v for k, v in self.annot_props.items() if k != 'callback'}
        if col in self.annot_family:
            self.annot_family[col].set_text( f(self.get_label(col), text, self.get_format(col)))
            self.annot_family[col].set_position((x, y))
        else:
            ax = ax or self.ax
            self.annot_family[col] = ax.annotate(
                f(self.get_label(col), text, self.get_format(col)), (x, y), **kwargs
            )

    def update_text(self, i, ax=None):
        for key, v in self.text_collection.items():
            callback, props_dict = v[0], v[1]
            if callback: 
                props_dict.update(s = callback(self.ith_info))
            if key not in self.text_family:
                ax = ax or self.ax
                # Create the text object only once and store it
                self.text_family[key] = ax.text(
                    transform=self.ax.transAxes, **props_dict,
                )
            else:
                callback and self.text_family[key].set_text(callback(self.ith_info))

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
        if self.graph_switch['legend']: 
            handles, labels = self.ax.get_legend_handles_labels()
            # Remove duplicate labels
            by_label = dict(zip(labels, handles))
            self.ax.legend(
                by_label.values(), by_label.keys(),
                ncol=len(by_label),
                **self.legend_props
            )
        if self.graph_switch['grid']: 
            self.ax.grid(**self.grid_props)
        self.ax.set_axisbelow(self.grid_behind)

    def get_ith_data_info(self, i, row, gap) -> SimpleNamespace:
        ith_data = self.data[i]
        ith_info = SimpleNamespace(
            now = ith_data.name.strftime(MONTH_IDX),
            is_start =  i==row.start,
            is_end   =  gap==0,
            i        =  i,
            x        =  ith_data.name,
            y        =  ith_data.to_dict(),
            range    =  row,
            gap      =  gap,
            just_now =  self.now
        )
        ith_info.update_now = ith_info.now != ith_info.just_now

        if self.now is None or ith_info.is_start:   
                                self.update_start(ith_info)
        if ith_info.update_now: self.update_now(ith_info)
        return ith_info


    def _update(self, i, ax=None):
        row, gap = self.data.get_index(i)
        self.ith_info = self.get_ith_data_info(i, row, gap)
        for col, _type in self.data.column_type.items():
            if _type == 'line':
                self.update_lines(row, col, gap, ax=ax)
                self.update_annots(col, ax=ax)

    def update(self, i) -> None:
        # Loop through columns and update their corresponding line and annotation
        self._update(i)
        self.post_update(self, i)
        self.update_text(i)
        self.fix_xy()


    def fix_xy(self):
        if self.fixed_xlim:
            self.ax.set_xlim(**self.xlim_props)
        if self.fixed_ylim:
            self.ax.set_ylim(**self.ylim_props)
        if self.graph_switch['xtick']:
            self.ax.tick_params(**self.xtick_props)
        if self.graph_switch['ytick']:
            self.ax.tick_params(**self.ytick_props)
        self.update_legend_grid()

    def update_start(self, ith_info):
        for col in list(self.annot_family):
            remove_from(self.annot_family, col)

        for col in list(self.line_family):
            remove_from(self.line_family, col)

    def update_now(self, ith_info):
        self.now = ith_info.now


class MyAreaPlot(MyLinePlot):
        
    def set_axes(self, ax: maxes.Axes = None) -> None:
        if ax is not None:
            self.ax = ax
            self.ax2 = ax.twinx()

    def update_legend_grid(self):
        if self.legend: 
            hs, ls = [], []
            for ax in [self.ax, self.ax2]:
                for h, l in zip(*ax.get_legend_handles_labels()):
                    if isinstance(h, mline.Line2D) and l == 'total': continue
                    hs.append(h); ls.append(l)
            self.ax.legend(
                hs, ls, ncol=len(ls), **self.legend_props 
            )
        self.ax.set_zorder(self.ax2.get_zorder() - 1)
        if self.grid: self.ax.grid(**self.grid_props)
        self.ax.set_axisbelow(self.grid_behind)
    
    def init_props(self):
        self.line_family: Dict[str, mline.Line2D] = {}
        self.area_family: Dict[str, mcollect.PolyCollection] = {}
        self.annot_family: Dict[str, mtext.Annotation] = {}
        self.text_family: Dict[str, mtext.Text] = {}
        
        self.xlim_props, self.ylim_props, self.column_props = {}, {}, {}
        del self.line_annot_props, self.line_head_props
        self.annot_props, self.head_props = {}, {}
        self.area_props, self.stack_props = {}, {}

        self.update_props( 'legend', bbox_to_anchor=(0.5, -0.05), loc='upper center', fancybox=True)
        self.update_props( 'xtick', color='#777777', action='replace')
        self.update_props( 'ytick', color='#777777', action='replace')
        self.update_props( 'annot', callback=lambda c, y, f: f"{c}({get_abbr(y, f)})", color='w')
        self.update_props( 'area', alpha = 0.6 )
        self.update_props( 'stack', alpha = 0.7 )

        for var in self.data.column_style:
            self.data.column_style[var].update(
                color=self.column_colors[var],
            )
        self.update_props( 'column', **self.data.column_style )

    def _update(self, i):
        row, gap = self.data.get_index(i)
        self.ith_info = self.get_ith_data_info(i, row, gap)
        Y = self.data[self.ith_info.range]

        for col, _type in self.data.column_type.items():
            if _type == 'area':
                remove_from(self.area_family, col)
                self.area_family[col] = self.ax.fill_between(
                    Y.index, 
                    Y[col],
                    0, 
                    label=col,
                    **self.column_props[col],
                    **self.area_props, 
                )
                self.update_annots(col)

        ordered_stacks = sorted(
            (k for k, v in self.data.column_type.items() if v.startswith("stack")),
            key=lambda k:  int(self.data.column_type[k][5:])
        )
        for col, plot in zip(ordered_stacks, self.ax.stackplot(
            Y.index, *[Y[c] for c in ordered_stacks],
            labels=ordered_stacks,
            colors = [self.data.column_style[c]['color'] for c in ordered_stacks],
            **self.stack_props, 
        )):
            remove_from(self.area_family, col)
            self.area_family[col] = plot
        y = sum(self.ith_info.y[c] for c in ordered_stacks)
        self.update_annots( col='aggregated_value', y=y)

        for col, _type in self.data.column_type.items():
            if _type == 'line':
                self.update_lines(row, col, gap, ax=self.ax2)
                self.update_annots(col, ax=self.ax2)
        self.ax2.relim()
        self.ax2.autoscale_view(scalex=True, scaley=True)


class StackBottom:
    def __init__(self, column_type={}):
        self.order = sorted(
            (k for k, v in column_type.items() if v.startswith("bar")),
            key=lambda k: int(column_type[k][3:])
        )
        self.reset()
    def reset(self):
        self.stack_y = defaultdict(dict)
        self.visited_vintage = set()
        self.visited_feature = set()
        self.stack_x = {}
    def clear(self):
        self.visited_feature.clear()
    def update_position(self, vintage, pos):
        self.stack_x[vintage] = pos
    def update(self, vintage, col, val, additive=False):
        this = self.stack_y[vintage]
        if additive:
            this[col] = this.get(vintage, 0) + val
        else:
            this[col] = val
        self.visited_vintage.add(vintage)
        self.visited_feature.add(col)
    def get(self, vintage, col):
        if vintage in self.stack_y:
            return self.stack_y[vintage].get(col, None)
    def __getitem__(self, vintage):
        return sum(x for k, x in self.stack_y[vintage].items() if k in self.visited_feature)
    def __repr__(self):
        repr_str = ""
        for period, subdict in self.stack_y.items():
            repr_str += f"{period}:\n"
            for key, val in subdict.items():
                repr_str += f"  {key} = {val}\n"
        return repr_str

def end_of_month(year, num_months):
    return pd.date_range(
        start=f"{year}-{12-num_months+1}-01",
        end=f"{year}-12-31",
        freq="ME"
    )
    
class MyBarPlot(MyLinePlot):
    def __init__(self, datafier, post_update=lambda *x: None):
        super().__init__(datafier, post_update)
        self.bar_bottoms = StackBottom(self.data.column_type)
        
    def init_props(self):
        self.bar_family: Dict[str, mcontainer.BarContainer] = defaultdict(list)
        self.annot_family: Dict[str, mtext.Annotation] = {}
        self.text_family: Dict[str, mtext.Text] = {}
        
        self.xlim_props, self.ylim_props, self.column_props = {}, {}, {}
        del self.line_annot_props, self.line_head_props
        self.annot_props, self.head_props = {}, {}
        self.area_props, self.stack_props, self.border_props = {}, {}, {}

        self.update_props( 'legend', bbox_to_anchor=(0.5, -0.05), loc='upper center', fancybox=True)
        self.update_props( 'xtick', color='#777777', action='replace')
        self.update_props( 'ytick', color='#777777', action='replace')
        self.update_props( 'annot', callback=lambda c, y, f: f"{get_abbr(y, f)}", color='w', ha='center')
        self.update_props( 'area', alpha = 0.2, color='#1b62cd', noise=0.02)
        self.update_props( 'stack', alpha = 0.8, edgecolor = "black")

        for var in self.data.column_style:
            self.data.column_style[var].update(
                color=self.column_colors[var],
            )
        self.update_props( 'column', **self.data.column_style )
        self.update_props( 'border', pad=0.1, mutation_aspect=1, radius=0.2, mutation_scale=0.6)

    def _update(self, i):
        row, gap = self.data.get_index(i)
        self.ith_info = self.get_ith_data_info(i, row, gap)

        self.bar_bottoms.clear()
        for col in self.bar_bottoms.order:
            self.update_bars(col)
            # self.update_annots(col)
            
    def update_start(self, ith_info):
        # here we clear the ax and draw history map (mean + error ban)
        self.ax.clear()             # erase everything
        self.bar_family.clear()     # make sure new month bar is drawn
        self.annot_family.clear()
        self.bar_bottoms.reset()

        # ith_info.data = self.data[ith_info.range].resample('ME').last()
        props = self.area_props.copy()
        noise = props.pop('noise', 0.01)
        year  = ith_info.x.year
        
        for label, (mean, std) in self.data.col_var[year].items():
            months = end_of_month(year, 12)
            if len(mean) == 0: 
                self.ax.plot(
                    months, [0] * len(months), color='none', label=self.get_label(label)
                )
            else:
                sampled_days = pd.date_range(months[0] - pd.Timedelta(days=BAR_WIDTH/2), end=months[-1] + pd.Timedelta(days=BAR_WIDTH/2), freq='1d')
                x_interp = months.union(sampled_days).sort_values()
                # mean = np.pad(mean, (len(months) - len(mean), 0), 'constant', constant_values=(0,))
                # std = np.pad(std, (len(months) - len(std), 0), 'constant', constant_values=(0,))
                new_mean = np.interp(x_interp.astype(int), months.astype(int), mean)
                new_std  = np.interp(x_interp.astype(int), months.astype(int), std)
                x_eps    = np.random.normal(0, noise * new_mean.max(), size=new_mean.shape)  
                x_eps[np.isin(x_interp, months)] = 0
                new_mean += x_eps
                x_interp = mdates.date2num(x_interp)

                self.ax.plot(
                    x_interp, new_mean, linestyle='dotted', 
                    label=self.get_label(label), color=props.get('color', 'b')
                )
                self.ax.fill_between(
                    x_interp, new_mean - new_std, new_mean + new_std, **props
                )
                self.ax.scatter( months, mean, color=props.get('color', 'b'))
        
        df_bar = self.data[ith_info.range].resample('ME').last()
        index  = df_bar.index
        for col in self.bar_bottoms.order:
            self.bar_family[col] = self.ax.bar(
                index,
                df_bar[col],
                width=BAR_WIDTH,
                bottom=[self.bar_bottoms[i.strftime(MONTH_IDX)] for i in index],
                label = col,
                **self.column_props[col],
                **self.stack_props, 
            )
            # self.use_fancybox(col)
            for j,v in df_bar[col].items():
                vintage = j.strftime(MONTH_IDX)
                self.bar_bottoms.update(vintage, col, v)
                self.bar_bottoms.update_position(vintage, mdates.date2num(j))
        
            for i, bar in enumerate(self.bar_family[col].patches):                
                self.update_annots(
                    col = f'{col}_{i+1}', 
                    x = bar.get_x() + bar.get_width() / 2,
                    y = bar.get_height(),
                    y_offset = bar.get_y() - bar.get_height() / 2,
                )
    
    def update_bars(self, col, ax=None):
        bar_attr = self.ith_info
        if self.bar_bottoms.get(bar_attr.now, col) is None:
            # we need to add new single bar to the plot
            eom = pd.to_datetime(bar_attr.now) + pd.offsets.MonthEnd(0)
            bar, = self.ax.bar(
                eom,
                bar_attr.y[col], 
                bottom=self.bar_bottoms[bar_attr.now],
                width=BAR_WIDTH, 
                **self.column_props[col],
                **self.stack_props, 
            )
            self.bar_family[col].patches.append(bar)
            # self.use_fancybox(col, patch=_)
            self.bar_bottoms.update_position(bar_attr.now, mdates.date2num(eom))
        else:
            bar = self.bar_family[col].patches[-1]
            bar.set_height(bar_attr.y[col])
            bar.set_y(self.bar_bottoms[self.now])
            
        self.update_annots(
            col = f'{col}_{int(bar_attr.now[-2:])}',
            x = bar.get_x() + bar.get_width() / 2,
            y = bar.get_height(),
            y_offset = bar.get_y() - bar.get_height() / 2,
        )
        self.bar_bottoms.update(bar_attr.now, col, bar_attr.y[col])

    def use_fancybox(self, col, patch=None):
        props = self.border_props.copy()
        pad, radius = props.pop('pad'), props.pop('radius')

        def create_fancybox(patch):
            bb = patch.get_bbox()
            p_bbox = patches.FancyBboxPatch(
                (bb.xmin, bb.ymin), abs(bb.width), abs(bb.height),
                fc=patch.get_facecolor(),
                zorder=patch.zorder,
                boxstyle='round,pad={},rounding_size={}'.format(pad, radius),
                **props
            )
            patch.remove()
            return p_bbox

        if patch is not None:
            patch = create_fancybox(patch)
            self.ax.add_patch(patch)
            self.bar_family[col].patches.append(patch)
        else:
            new_patches = []
            for patch in reversed(self.bar_family[col].patches):
                new_patches.insert(0, create_fancybox(patch))
            for patch in new_patches:
                self.ax.add_patch(patch)
            self.bar_family[col] = mcontainer.BarContainer(new_patches, orientation='vertical')