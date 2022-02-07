# -*- coding: utf-8 -*-
"""
pysteps.visualization.animations
================================

Functions to produce animations for pysteps.

.. autosummary::
    :toctree: ../generated/

    animate
"""

import os
import warnings

import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import pysteps as st
import numpy as np
from pysteps.visualization import get_colormap

PRECIP_VALID_TYPES = ("ensemble", "mean", "prob")
PRECIP_DEPRECATED_ARGUMENTS = (
    "units",
    "colorbar",
    "colorscale",
)  # TODO: remove in version >= 1.6
MOTION_VALID_METHODS = ("quiver", "streamplot")

class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs):
        self.i = 0
        self.min=mini
        self.max=maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self,self.fig, self.update, frames=self.play(), 
                                           init_func=init_func, fargs=fargs,
                                           save_count=save_count, **kwargs )    

    def play(self):
        while self.runs:
            self.i = self.i+self.forwards-(not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()
    def backward(self, event=None):
        self.forwards = False
        self.start()
    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()
    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i+=1
        elif self.i == self.max and not self.forwards:
            self.i-=1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0],pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="500%", pad=0.07)
        
        self.button_back = matplotlib.widgets.Button(playerax, label='b')
        self.button_oneback = matplotlib.widgets.Button(bax, label='1b')
        self.button_stop = matplotlib.widgets.Button(sax, label='stop')
        self.button_oneforward = matplotlib.widgets.Button(fax, label='1f')
        self.button_forward = matplotlib.widgets.Button(ofax, label='f')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(sliderax, '', 
                                                self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self,i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self,i):
        self.slider.set_val(i)



def animate_interactive(
        precip_obs,
        grid_on=False,
        colorbar_on=True,
        predefined_value_range=True,
        cmap=None,
):
    # figure axis setup 
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.15)

    # display initial image
    ntsteps = precip_obs.shape[0]
    ny = precip_obs.shape[1]
    nx = precip_obs.shape[2]
    idx = 0
    
    if cmap==None:
        cmap, norm, _, _ = get_colormap("intensity","dBZ","pysteps")
    if grid_on:
        ax.set_xticks(np.arange(-.5, nx, 1), minor=True)
        ax.set_yticks(np.arange(-.5, ny, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    if predefined_value_range:
        vmin = 0
        vmax = 55
        normin = norm
    else:
        vmin = None
        vmax = None
        normin = None
    im_h = ax.imshow(
        precip_obs[idx, :, :],
        cmap=cmap,
        interpolation='none',
        norm=normin,
        vmin = vmin,
        vmax = vmax,
    )

    # add colorbar
    if colorbar_on:
        # get colormap and color levels
        _, _, clevs, clevs_str = get_colormap("intensity","dBZ","pysteps")
        extend = "max"
        cbar = plt.colorbar(
            im_h, ticks=clevs, spacing="uniform", extend=extend, shrink=0.8, cax=None
        )
        if clevs_str is not None:
            cbar.ax.set_yticklabels(clevs_str)

        cbar.ax.set_title("dBZ", fontsize=10)
        cbar.set_label("Precipitation intensity")
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])

    def update(idx):
        im_h.set_data(precip_obs[idx, :, :])

    ani = Player(fig, update, maxi=ntsteps-1)
    #plt.show()
    return ani

def animate(
    precip_obs,
    precip_fct=None,
    timestamps_obs=None,
    timestep_min=None,
    motion_field=None,
    ptype="ensemble",
    motion_plot="quiver",
    geodata=None,
    title=None,
    prob_thr=None,
    display_animation=True,
    nloops=1,
    time_wait=0.2,
    savefig=False,
    fig_dpi=100,
    fig_format="png",
    path_outputs="",
    precip_kwargs=None,
    motion_kwargs=None,
    map_kwargs=None,
    **kwargs,
):
    """Function to animate observations and forecasts in pysteps.

    It also allows to export the individual frames as figures, which
    is useful for constructing animated GIFs or similar.

    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

    Parameters
    ----------
    precip_obs: array-like
        Three-dimensional array containing the time series of observed
        precipitation fields.
    precip_fct: array-like, optional
        The three or four-dimensional (for ensembles) array
        containing the time series of forecasted precipitation field.
    timestamps_obs: list of datetimes, optional
        List of datetime objects corresponding to the time stamps of
        the fields in precip_obs.
    timestep_min: float, optional
        The time resolution in minutes of the forecast.
    motion_field: array-like, optional
        Three-dimensional array containing the u and v components of
        the motion field.
    motion_plot: string, optional
        The method to plot the motion field. See plot methods in
        :py:mod:`pysteps.visualization.motionfields`.
    geodata: dictionary or None, optional
        Dictionary containing geographical information about
        the field.
        If geodata is not None, it must contain the following key-value pairs:

        .. tabularcolumns:: |p{1.5cm}|L|

        +----------------+----------------------------------------------------+
        |        Key     |                  Value                             |
        +================+====================================================+
        |   projection   | PROJ.4-compatible projection definition            |
        +----------------+----------------------------------------------------+
        |    x1          | x-coordinate of the lower-left corner of the data  |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    y1          | y-coordinate of the lower-left corner of the data  |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    x2          | x-coordinate of the upper-right corner of the data |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    y2          | y-coordinate of the upper-right corner of the data |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    yorigin     | a string specifying the location of the first      |
        |                | element in the data raster w.r.t. y-axis:          |
        |                | 'upper' = upper border, 'lower' = lower border     |
        +----------------+----------------------------------------------------+

    title: str or None, optional
        If not None, print the string as title on top of the plot.
    ptype: {'ensemble', 'mean', 'prob'}, str, optional
        Type of the plot to animate. 'ensemble' = ensemble members,
        'mean' = ensemble mean, 'prob' = exceedance probability
        (using threshold defined in prob_thrs).
    prob_thr: float, optional
        Intensity threshold for the exceedance probability maps. Applicable
        if ptype = 'prob'.
    display_animation: bool, optional
        If set to True, display the animation (set to False if only
        interested in saving the animation frames).
    nloops: int, optional
        The number of loops in the animation.
    time_wait: float, optional
        The time in seconds between one frame and the next. Applicable
        if display_animation is True.
    savefig: bool, optional
        If set to True, save the individual frames into path_outputs.
    fig_dpi: float, optional
        The resolution in dots per inch. Applicable if savefig is True.
    fig_format: str, optional
        Filename extension. Applicable if savefig is True.
    path_outputs: string, optional
        Path to folder where to save the frames. Applicable if savefig is True.
    precip_kwargs: dict, optional
        Optional parameters that are supplied to
        :py:func:`pysteps.visualization.precipfields.plot_precip_field`.
    motion_kwargs: dict, optional
        Optional parameters that are supplied to
        :py:func:`pysteps.visualization.motionfields.quiver` or
        :py:func:`pysteps.visualization.motionfields.streamplot`.
    map_kwargs: dict, optional
        Optional parameters that need to be passed to
        :py:func:`pysteps.visualization.basemaps.plot_geography`.

    Returns
    -------
    None
    """

    if precip_kwargs is None:
        precip_kwargs = {}

    if motion_kwargs is None:
        motion_kwargs = {}

    if map_kwargs is None:
        map_kwargs = {}

    if precip_fct is not None:
        if len(precip_fct.shape) == 3:
            precip_fct = precip_fct[None, ...]
        n_lead_times = precip_fct.shape[1]
        n_members = precip_fct.shape[0]
    else:
        n_lead_times = 0
        n_members = 1

    if title is not None and isinstance(title, str):
        title_first_line = title + "\n"
    else:
        title_first_line = ""

    if motion_plot not in MOTION_VALID_METHODS:
        raise ValueError(
            f"Invalid motion plot method '{motion_plot}'."
            f"Supported: {str(MOTION_VALID_METHODS)}"
        )

    if ptype not in PRECIP_VALID_TYPES:
        raise ValueError(
            f"Invalid precipitation type '{ptype}'."
            f"Supported: {str(PRECIP_VALID_TYPES)}"
        )

    # TODO: remove in version >= 1.6
    if "type" in kwargs:
        warnings.warn(
            "The 'type' keyword will be deprecated in version 1.6. "
            "Use 'ptype' instead."
        )
        ptype = kwargs.get("type")

    # TODO: remove in version >= 1.6
    if "timestamps" in kwargs:
        warnings.warn(
            "The 'timestamps' keyword will be deprecated in version 1.6. "
            "Use 'timestamps_obs' instead."
        )
        timestamps_obs = kwargs.get("timestamps")

    # TODO: remove in version >= 1.6
    if "plotanimation" in kwargs:
        warnings.warn(
            "The 'plotanimation' keyword will be deprecated in version 1.6. "
            "Use 'display_animation' instead."
        )
        display_animation = kwargs.get("timestamps")

    # TODO: remove in version >= 1.6
    for depr_key in PRECIP_DEPRECATED_ARGUMENTS:
        if depr_key in kwargs:
            warnings.warn(
                f"The {depr_key} argument will be deprecated in version 1.6. "
                "Add it to 'precip_kwargs' instead."
            )
            precip_kwargs[depr_key] = kwargs.get(depr_key)

    if timestamps_obs is not None:
        if len(timestamps_obs) != precip_obs.shape[0]:
            raise ValueError(
                f"The number of timestamps does not match the size of precip_obs: "
                f"{len(timestamps_obs)} != {precip_obs.shape[0]}"
            )
        if precip_fct is not None:
            reftime_str = timestamps_obs[-1].strftime("%Y%m%d%H%M")
        else:
            reftime_str = timestamps_obs[0].strftime("%Y%m%d%H%M")
    else:
        reftime_str = None

    if ptype == "prob" and prob_thr is None:
        raise ValueError("ptype 'prob' needs a prob_thr value")

    if ptype != "ensemble":
        n_members = 1

    n_obs = precip_obs.shape[0]

    loop = 0
    while loop < nloops:
        for n in range(n_members):
            for i in range(n_obs + n_lead_times):
                plt.clf()

                # Observations
                if i < n_obs and (display_animation or n == 0):

                    title = title_first_line + "Analysis"
                    if timestamps_obs is not None:
                        title += (
                            f" valid for {timestamps_obs[i].strftime('%Y-%m-%d %H:%M')}"
                        )

                    plt.clf()
                    if ptype == "prob":
                        prob_field = st.postprocessing.ensemblestats.excprob(
                            precip_obs[None, i, ...], prob_thr
                        )
                        ax = st.plt.plot_precip_field(
                            prob_field,
                            ptype="prob",
                            geodata=geodata,
                            probthr=prob_thr,
                            title=title,
                            map_kwargs=map_kwargs,
                            **precip_kwargs,
                        )
                    else:
                        ax = st.plt.plot_precip_field(
                            precip_obs[i, :, :],
                            geodata=geodata,
                            title=title,
                            map_kwargs=map_kwargs,
                            **precip_kwargs,
                        )

                    if motion_field is not None:
                        if motion_plot == "quiver":
                            st.plt.quiver(
                                motion_field, ax=ax, geodata=geodata, **motion_kwargs
                            )
                        elif motion_plot == "streamplot":
                            st.plt.streamplot(
                                motion_field, ax=ax, geodata=geodata, **motion_kwargs
                            )

                    if savefig & (loop == 0):
                        figtags = [reftime_str, ptype, f"f{i:02d}"]
                        figname = "_".join([tag for tag in figtags if tag])
                        filename = os.path.join(path_outputs, f"{figname}.{fig_format}")
                        plt.savefig(filename, bbox_inches="tight", dpi=fig_dpi)
                        print("saved: ", filename)

                # Forecasts
                elif i >= n_obs and precip_fct is not None:

                    title = title_first_line + "Forecast"
                    if timestamps_obs is not None:
                        title += f" valid for {timestamps_obs[-1].strftime('%Y-%m-%d %H:%M')}"
                    if timestep_min is not None:
                        title += " +%02d min" % ((1 + i - n_obs) * timestep_min)
                    else:
                        title += " +%02d" % (1 + i - n_obs)

                    plt.clf()
                    if ptype == "prob":
                        prob_field = st.postprocessing.ensemblestats.excprob(
                            precip_fct[:, i - n_obs, :, :], prob_thr
                        )
                        ax = st.plt.plot_precip_field(
                            prob_field,
                            ptype="prob",
                            geodata=geodata,
                            probthr=prob_thr,
                            title=title,
                            map_kwargs=map_kwargs,
                            **precip_kwargs,
                        )
                    elif ptype == "mean":
                        ens_mean = st.postprocessing.ensemblestats.mean(
                            precip_fct[:, i - n_obs, :, :]
                        )
                        ax = st.plt.plot_precip_field(
                            ens_mean,
                            geodata=geodata,
                            title=title,
                            map_kwargs=map_kwargs,
                            **precip_kwargs,
                        )
                    else:
                        ax = st.plt.plot_precip_field(
                            precip_fct[n, i - n_obs, ...],
                            geodata=geodata,
                            title=title,
                            map_kwargs=map_kwargs,
                            **precip_kwargs,
                        )

                    if motion_field is not None:
                        if motion_plot == "quiver":
                            st.plt.quiver(
                                motion_field, ax=ax, geodata=geodata, **motion_kwargs
                            )
                        elif motion_plot == "streamplot":
                            st.plt.streamplot(
                                motion_field, ax=ax, geodata=geodata, **motion_kwargs
                            )

                    if ptype == "ensemble" and n_members > 1:
                        plt.text(
                            0.01,
                            0.99,
                            "m %02d" % (n + 1),
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                        )

                    if savefig & (loop == 0):
                        figtags = [reftime_str, ptype, f"f{i:02d}", f"m{n + 1:02d}"]
                        figname = "_".join([tag for tag in figtags if tag])
                        filename = os.path.join(path_outputs, f"{figname}.{fig_format}")
                        plt.savefig(filename, bbox_inches="tight", dpi=fig_dpi)
                        print("saved: ", filename)

                if display_animation:
                    plt.pause(time_wait)

            if display_animation:
                plt.pause(2 * time_wait)

        loop += 1

    plt.close()
