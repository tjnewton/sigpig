"""
Functions to find low-frequency earthquakes in seismic time series data.
"""

import getpass
import logging
import obspy
from obspy import UTCDateTime, Stream, Trace, read, read_events, Inventory
from obspy.clients.fdsn import Client
from obspy.signal.cross_correlation import correlate, correlate_template, \
    xcorr_max, xcorr
from obspy.core.event import Catalog, Comment, CreationInfo
from obspy.core.event.event import Event
from obspy.core.event.magnitude import Magnitude
from obspy.core.event.origin import Pick, Origin
from obspy.core.event.base import WaveformStreamID
from obspy.signal import PPSD
from eqcorrscan import Tribe, Party, Family, Template
from eqcorrscan.utils.clustering import cluster
from eqcorrscan.utils.sac_util import sactoevent
from eqcorrscan.utils import pre_processing
from eqcorrscan.utils.stacking import PWS_stack, linstack, align_traces
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import calendar
from tqdm import tqdm
from figures import plot_stack, plot_stream_absolute, plot_stream_relative, \
                    plot_party_detections, plot_distribution, \
                    plot_template_and_stack, plot_Time_Series_And_Spectrogram
from scipy.signal import hilbert
import time
from data import max_amplitude, snr
import os
from celluloid import Camera
from math import ceil
import pandas as pd
import random

Logger = logging.getLogger(__name__)

# class to enable detection via stacks within EQcorrscan
class Stack(Tribe):
    """
    A class to use stacks as templates for matched-filtering.
    """

    def template_gen(self, method, lowcut, highcut, samp_rate, filt_order,
                     length, prepick, swin="all", process_len=86400,
                     all_horiz=False, delayed=True, plot=False, plotdir=None,
                     return_event=False, min_snr=None, parallel=False,
                     num_cores=False, save_progress=False,
                     skip_short_chans=False,
                     **kwargs):
        """
        Generate processed and cut waveforms for use as templates.

        :type method: str
        :param method:
            Template generation method, must be one of ('from_client',
            'from_seishub', 'from_sac', 'from_meta_file'). - Each method requires
            associated arguments, see note below.
        :type lowcut: float
        :param lowcut: Low cut (Hz), if set to None will not apply a lowcut.
        :type highcut: float
        :param highcut: High cut (Hz), if set to None will not apply a highcut.
        :type samp_rate: float
        :param samp_rate: New sampling rate in Hz.
        :type filt_order: int
        :param filt_order: Filter level (number of corners).
        :type length: float
        :param length: Length of template waveform in seconds.
        :type prepick: float
        :param prepick: Pre-pick time in seconds
        :type swin: str
        :param swin:
            P, S, P_all, S_all or all, defaults to all: see note in
            :func:`eqcorrscan.core.template_gen.template_gen`
        :type process_len: int
        :param process_len: Length of data in seconds to download and process.
        :type all_horiz: bool
        :param all_horiz:
            To use both horizontal channels even if there is only a pick on one of
            them.  Defaults to False.
        :type delayed: bool
        :param delayed: If True, each channel will begin relative to it's own \
            pick-time, if set to False, each channel will begin at the same time.
        :type plot: bool
        :param plot: Plot templates or not.
        :type plotdir: str
        :param plotdir:
            The path to save plots to. If `plotdir=None` (default) then the figure
            will be shown on screen.
        :type return_event: bool
        :param return_event: Whether to return the event and process length or not.
        :type min_snr: float
        :param min_snr:
            Minimum signal-to-noise ratio for a channel to be included in the
            template, where signal-to-noise ratio is calculated as the ratio of
            the maximum amplitude in the template window to the rms amplitude in
            the whole window given.
        :type parallel: bool
        :param parallel: Whether to process data in parallel or not.
        :type num_cores: int
        :param num_cores:
            Number of cores to try and use, if False and parallel=True, will use
            either all your cores, or as many traces as in the data (whichever is
            smaller).
        :type save_progress: bool
        :param save_progress:
            Whether to save the resulting templates at every data step or not.
            Useful for long-running processes.
        :type skip_short_chans: bool
        :param skip_short_chans:
            Whether to ignore channels that have insufficient length data or not.
            Useful when the quality of data is not known, e.g. when downloading
            old, possibly triggered data from a datacentre

        :returns: List of :class:`obspy.core.stream.Stream` Templates
        :rtype: list

        .. note::
            By convention templates are generated with P-phases on the
            vertical channel and S-phases on the horizontal channels, normal
            seismograph naming conventions are assumed, where Z denotes vertical
            and N, E, R, T, 1 and 2 denote horizontal channels, either oriented
            or not.  To this end we will **only** use Z channels if they have a
            P-pick, and will use one or other horizontal channels **only** if
            there is an S-pick on it.

        .. warning::
            If there is no phase_hint included in picks, and swin=all, all channels
            with picks will be used.

        .. note::
            If swin=all, then all picks will be used, not just phase-picks (e.g. it
            will use amplitude picks). If you do not want this then we suggest
            that you remove any picks you do not want to use in your templates
            before using the event.

        .. note::
            *Method specific arguments:*

            - `from_client` requires:
                :param str client_id:
                    string passable by obspy to generate Client, or any object
                    with a `get_waveforms` method, including a Client instance.
                :param `obspy.core.event.Catalog` catalog:
                    Catalog of events to generate template for
                :param float data_pad: Pad length for data-downloads in seconds
            - `from_seishub` requires:
                :param str url: url to seishub database
                :param `obspy.core.event.Catalog` catalog:
                    Catalog of events to generate template for
                :param float data_pad: Pad length for data-downloads in seconds
            - `from_sac` requires:
                :param list sac_files:
                    osbpy.core.stream.Stream of sac waveforms, or list of paths to
                    sac waveforms.
                .. note::
                    See `eqcorrscan.utils.sac_util.sactoevent` for details on
                    how pick information is collected.
            - `from_meta_file` requires:
                :param str meta_file:
                    Path to obspy-readable event file, or an obspy Catalog
                :param `obspy.core.stream.Stream` st:
                    Stream containing waveform data for template. Note that this
                    should be the same length of stream as you will use for the
                    continuous detection, e.g. if you detect in day-long files,
                    give this a day-long file!
                :param bool process:
                    Whether to process the data or not, defaults to True.

        .. note::
            process_len should be set to the same length as used when computing
            detections using match_filter.match_filter, e.g. if you read
            in day-long data for match_filter, process_len should be 86400.
        """
        client_map = {'from_client': 'fdsn', 'from_seishub': 'seishub'}
        assert method in ('from_client', 'from_seishub', 'from_meta_file',
                          'from_sac')
        if not isinstance(swin, list):
            swin = [swin]
        process = True
        if method == 'from_meta_file':
            if isinstance(kwargs.get('meta_file'), Catalog):
                catalog = kwargs.get('meta_file')
            elif kwargs.get('meta_file'):
                catalog = read_events(kwargs.get('meta_file'))
            else:
                catalog = kwargs.get('catalog')
            sub_catalogs = [catalog]
            st = kwargs.get('st', Stream())
            process = kwargs.get('process', True)
        elif method == 'from_sac':
            sac_files = kwargs.get('sac_files')
            if isinstance(sac_files, list):
                if isinstance(sac_files[0], (Stream, Trace)):
                    # This is a list of streams...
                    st = Stream(sac_files[0])
                    for sac_file in sac_files[1:]:
                        st += sac_file
                else:
                    sac_files = [read(sac_file)[0] for sac_file in sac_files]
                    st = Stream(sac_files)
            else:
                st = sac_files
            # Make an event object...
            catalog = Catalog([sactoevent(st)])
            sub_catalogs = [catalog]

        temp_list = []
        process_lengths = []
        catalog_out = Catalog()

        if "P_all" in swin or "S_all" in swin or all_horiz:
            all_channels = True
        else:
            all_channels = False
        for sub_catalog in sub_catalogs:
            Logger.info('Pre-processing data')
            st.merge()
            if len(st) == 0:
                Logger.info("No data")
                continue
            if process:
                data_len = max([len(tr.data) / tr.stats.sampling_rate
                                for tr in st])
                if 80000 < data_len < 90000:
                    daylong = True
                    starttime = min([tr.stats.starttime for tr in st])
                    min_delta = min([tr.stats.delta for tr in st])
                    # Cope with the common starttime less than 1 sample before the
                    #  start of day.
                    if (starttime + min_delta).date > starttime.date:
                        starttime = (starttime + min_delta)
                    # Check if this is stupid:
                    if abs(starttime - UTCDateTime(starttime.date)) > 600:
                        daylong = False
                    starttime = starttime.date
                else:
                    daylong = False
                # Check if the required amount of data have been downloaded - skip
                # channels if arg set.
                for tr in st:
                    tr.data = tr.data[0:int(process_len * tr.stats.sampling_rate)]
                if len(st) == 0:
                    Logger.info("No data")
                    continue
                if daylong:
                    st = pre_processing.dayproc(
                        st=st, lowcut=lowcut, highcut=highcut,
                        filt_order=filt_order, samp_rate=samp_rate,
                        parallel=parallel, starttime=UTCDateTime(starttime),
                        num_cores=num_cores, ignore_bad_data=True,
                        ignore_length=True)
                else:
                    st = pre_processing.shortproc(
                        st=st, lowcut=lowcut, highcut=highcut,
                        filt_order=filt_order, parallel=parallel,
                        samp_rate=samp_rate, num_cores=num_cores,
                        ignore_bad_data=True, ignore_length=True)

            # if len(st) == 0:
            #     Logger.info("No data")
            #     continue
            # # FIXME: I changed this. TJN 9/13/2021
            if len([tr.stats.starttime for tr in st]) > 0:
                data_start = min([tr.stats.starttime for tr in st])
                data_end = max([tr.stats.endtime for tr in st])
            else:
                continue

            for event in sub_catalog:
                stations, channels, st_stachans = ([], [], [])
                if len(event.picks) == 0:
                    Logger.warning(
                        'No picks for event {0}'.format(event.resource_id))
                    continue
                use_event = True
                # Check that the event is within the data
                for pick in event.picks:
                    if not data_start < pick.time < data_end:
                        Logger.warning(
                            "Pick outside of data span: Pick time {0} Start "
                            "time {1} End time: {2}".format(
                                str(pick.time), str(data_start),
                                str(data_end)))
                        use_event = False
                if not use_event:
                    Logger.error('Event is not within data time-span')
                    continue
                # Read in pick info
                Logger.debug("I have found the following picks")
                for pick in event.picks:
                    if not pick.waveform_id:
                        Logger.warning(
                            'Pick not associated with waveforms, will not use:'
                            ' {0}'.format(pick))
                        continue
                    Logger.debug(pick)
                    stations.append(pick.waveform_id.station_code)
                    channels.append(pick.waveform_id.channel_code)
                # Check to see if all picks have a corresponding waveform
                for tr in st:
                    st_stachans.append('.'.join([tr.stats.station,
                                                 tr.stats.channel]))
                # Cut and extract the templates
                template = self._template_gen(
                    event.picks, st, length, swin, prepick=prepick, plot=plot,
                    all_horiz=all_horiz, delayed=delayed, min_snr=min_snr,
                    plotdir=plotdir)
                process_lengths.append(len(st[0].data) / samp_rate)
                temp_list.append(template)
                catalog_out += event
            if save_progress:
                if not os.path.isdir("eqcorrscan_temporary_templates"):
                    os.makedirs("eqcorrscan_temporary_templates")
                for template in temp_list:
                    template.write(
                        "eqcorrscan_temporary_templates{0}{1}.ms".format(
                            os.path.sep, template[0].stats.starttime.strftime(
                                "%Y-%m-%dT%H%M%S")),
                        format="MSEED")
            del st
        if return_event:
            return temp_list, catalog_out, process_lengths
        return temp_list


    def _template_gen(self, picks, st, length, swin='all', prepick=0.05,
                      all_horiz=False, delayed=True, plot=False, min_snr=None,
                      plotdir=None):
        """
        Master function to generate a multiplexed template for a single event.

        Function to generate a cut template as :class:`obspy.core.stream.Stream`
        from a given set of picks and data.  Should be given pre-processed
        data (downsampled and filtered).

        :type picks: list
        :param picks: Picks to extract data around, where each pick in the \
            list is an obspy.core.event.origin.Pick object.
        :type st: obspy.core.stream.Stream
        :param st: Stream to extract templates from
        :type length: float
        :param length: Length of template in seconds
        :type swin: str
        :param swin:
            P, S, P_all, S_all or all, defaults to all: see note in
            :func:`eqcorrscan.core.template_gen.template_gen`
        :type prepick: float
        :param prepick:
            Length in seconds to extract before the pick time default is 0.05
            seconds.
        :type all_horiz: bool
        :param all_horiz:
            To use both horizontal channels even if there is only a pick on one
            of them.  Defaults to False.
        :type delayed: bool
        :param delayed:
            If True, each channel will begin relative to it's own pick-time, if
            set to False, each channel will begin at the same time.
        :type plot: bool
        :param plot:
            To plot the template or not, default is False. Plots are saved as
            `template-starttime_template.png` and `template-starttime_noise.png`,
            where `template-starttime` is the start-time of the template
        :type min_snr: float
        :param min_snr:
            Minimum signal-to-noise ratio for a channel to be included in the
            template, where signal-to-noise ratio is calculated as the ratio of
            the maximum amplitude in the template window to the rms amplitude in
            the whole window given.
        :type plotdir: str
        :param plotdir:
            The path to save plots to. If `plotdir=None` (default) then the figure
            will be shown on screen.

        :returns: Newly cut template.
        :rtype: :class:`obspy.core.stream.Stream`

        .. note::
            By convention templates are generated with P-phases on the
            vertical channel and S-phases on the horizontal channels, normal
            seismograph naming conventions are assumed, where Z denotes vertical
            and N, E, R, T, 1 and 2 denote horizontal channels, either oriented
            or not.  To this end we will **only** use Z channels if they have a
            P-pick, and will use one or other horizontal channels **only** if
            there is an S-pick on it.

        .. note::
            swin argument: Setting to `P` will return only data for channels
            with P picks, starting at the pick time (minus the prepick).
            Setting to `S` will return only data for channels with
            S picks, starting at the S-pick time (minus the prepick)
            (except if `all_horiz=True` when all horizontal channels will
            be returned if there is an S pick on one of them). Setting to `all`
            will return channels with either a P or S pick (including both
            horizontals if `all_horiz=True`) - with this option vertical channels
            will start at the P-pick (minus the prepick) and horizontal channels
            will start at the S-pick time (minus the prepick).
            `P_all` will return cut traces starting at the P-pick time for all
            channels. `S_all` will return cut traces starting at the S-pick
            time for all channels.

        .. warning::
            If there is no phase_hint included in picks, and swin=all, all
            channels with picks will be used.
        """
        from eqcorrscan.utils.plotting import pretty_template_plot as tplot
        from eqcorrscan.utils.plotting import noise_plot

        # the users picks intact.
        if not isinstance(swin, list):
            swin = [swin]
        for _swin in swin:
            assert _swin in ['P', 'all', 'S', 'P_all', 'S_all']
        picks_copy = []
        for pick in picks:
            if not pick.waveform_id:
                Logger.warning(
                    "Pick not associated with waveform, will not use it: "
                    "{0}".format(pick))
                continue
            if not pick.waveform_id.station_code or not \
                    pick.waveform_id.channel_code:
                Logger.warning(
                    "Pick not associated with a channel, will not use it:"
                    " {0}".format(pick))
                continue
            picks_copy.append(pick)
        if len(picks_copy) == 0:
            return Stream()
        st_copy = Stream()
        for tr in st:
            # Check that the data can be represented by float16, and check they
            # are not all zeros
            if np.all(tr.data.astype(np.float16) == 0):
                Logger.error(
                    "Trace is all zeros at float16 level, either gain or "
                    "check. Not using in template: {0}".format(tr))
                continue
            st_copy += tr
        st = st_copy
        if len(st) == 0:
            return st
        # Get the earliest pick-time and use that if we are not using delayed.
        picks_copy.sort(key=lambda p: p.time)
        first_pick = picks_copy[0]
        if plot:
            stplot = st.slice(first_pick.time - 20,
                              first_pick.time + length + 90).copy()
            noise = stplot.copy()
        # Work out starttimes
        starttimes = []
        for _swin in swin:
            for tr in st:
                starttime = {'station': tr.stats.station,
                             'channel': tr.stats.channel, 'picks': []}
                station_picks = [pick for pick in picks_copy
                                 if pick.waveform_id.station_code ==
                                 tr.stats.station]
                if _swin == 'P_all':
                    p_pick = [pick for pick in station_picks
                              if pick.phase_hint.upper()[0] == 'P']
                    if len(p_pick) == 0:
                        continue
                    starttime.update({'picks': p_pick})
                elif _swin == 'S_all':
                    s_pick = [pick for pick in station_picks
                              if pick.phase_hint.upper()[0] == 'S']
                    if len(s_pick) == 0:
                        continue
                    starttime.update({'picks': s_pick})
                elif _swin == 'all':
                    if all_horiz and tr.stats.channel[-1] in ['1', '2', '3',
                                                              'N', 'E']:
                        # Get all picks on horizontal channels
                        channel_pick = [
                            pick for pick in station_picks
                            if pick.waveform_id.channel_code[-1] in
                               ['1', '2', '3', 'N', 'E']]
                    else:
                        channel_pick = [
                            pick for pick in station_picks
                            if
                            pick.waveform_id.channel_code == tr.stats.channel]
                    if len(channel_pick) == 0:
                        continue
                    starttime.update({'picks': channel_pick})
                elif _swin == 'P':
                    p_pick = [pick for pick in station_picks
                              if pick.phase_hint.upper()[0] == 'P' and
                              pick.waveform_id.channel_code == tr.stats.channel]
                    if len(p_pick) == 0:
                        continue
                    starttime.update({'picks': p_pick})
                elif _swin == 'S':
                    if tr.stats.channel[-1] in ['Z', 'U']:
                        continue
                    s_pick = [pick for pick in station_picks
                              if pick.phase_hint.upper()[0] == 'S']
                    if not all_horiz:
                        s_pick = [pick for pick in s_pick
                                  if pick.waveform_id.channel_code ==
                                  tr.stats.channel]
                    starttime.update({'picks': s_pick})
                    if len(starttime['picks']) == 0:
                        continue
                if not delayed:
                    starttime.update({'picks': [first_pick]})
                starttimes.append(starttime)
        # Cut the data
        st1 = Stream()
        for _starttime in starttimes:
            Logger.info(f"Working on channel {_starttime['station']}."
                        f"{_starttime['channel']}")
            tr = st.select(
                station=_starttime['station'], channel=_starttime['channel'])[
                0]
            Logger.info(f"Found Trace {tr}")
            used_tr = False
            for pick in _starttime['picks']:
                if not pick.phase_hint:
                    Logger.warning(
                        "Pick for {0}.{1} has no phase hint given, you should not "
                        "use this template for cross-correlation"
                        " re-picking!".format(
                            pick.waveform_id.station_code,
                            pick.waveform_id.channel_code))
                starttime = pick.time - prepick
                Logger.debug("Cutting {0}".format(tr.id))
                noise_amp = self._rms(
                    tr.slice(starttime=starttime - 100,
                             endtime=starttime).data)
                tr_cut = tr.slice(
                    starttime=starttime, endtime=starttime + length,
                    nearest_sample=False).copy()
                if plot:
                    noise.select(
                        station=_starttime['station'],
                        channel=_starttime['channel']).trim(
                        noise[0].stats.starttime, starttime)
                if len(tr_cut.data) == 0:
                    Logger.warning(
                        "No data provided for {0}.{1} starting at {2}".format(
                            tr.stats.station, tr.stats.channel, starttime))
                    continue
                # Ensure that the template is the correct length
                if len(tr_cut.data) == (tr_cut.stats.sampling_rate *
                                        length) + 1:
                    tr_cut.data = tr_cut.data[0:-1]
                Logger.debug(
                    'Cut starttime = %s\nCut endtime %s' %
                    (str(tr_cut.stats.starttime), str(tr_cut.stats.endtime)))
                if min_snr is not None and \
                        max(tr_cut.data) / noise_amp < min_snr:
                    Logger.warning(
                        "Signal-to-noise ratio {0} below threshold for {1}.{2}, "
                        "not using".format(
                            max(tr_cut.data) / noise_amp, tr_cut.stats.station,
                            tr_cut.stats.channel))
                    continue
                st1 += tr_cut
                used_tr = True
            if not used_tr:
                Logger.warning('No pick for {0}'.format(tr.id))
        if plot and len(st1) > 0:
            plot_kwargs = dict(show=True)
            if plotdir is not None:
                if not os.path.isdir(plotdir):
                    os.makedirs(plotdir)
                plot_kwargs.update(dict(show=False, save=True))
            tplot(st1, background=stplot, picks=picks_copy,
                  title='Template for ' + str(st1[0].stats.starttime),
                  savefile="{0}/{1}_template.png".format(
                      plotdir, st1[0].stats.starttime.strftime(
                          "%Y-%m-%dT%H%M%S")),
                  **plot_kwargs)
            noise_plot(signal=st1, noise=noise,
                       savefile="{0}/{1}_noise.png".format(
                           plotdir, st1[0].stats.starttime.strftime(
                               "%Y-%m-%dT%H%M%S")),
                       **plot_kwargs)
            del stplot
        return st1


    def _rms(self, array):
        """
        Calculate RMS of array.

        :type array: numpy.ndarray
        :param array: Array to calculate the RMS for.

        :returns: RMS of array
        :rtype: float
        """
        return np.sqrt(np.mean(np.square(array)))


    def construct(self, method, lowcut, highcut, samp_rate, filt_order,
                  length, prepick, swin="all", process_len=86400,
                  all_horiz=False, delayed=True, plot=False, plotdir=None,
                  min_snr=None, parallel=False, num_cores=False,
                  skip_short_chans=False, save_progress=False, **kwargs):
        """
        Generate a Tribe of Templates.

        :type method: str
        :param method:
            Method of Tribe generation. Possible options are: `from_client`,
            `from_seishub`, `from_meta_file`.  See below on the additional
            required arguments for each method.
        :type lowcut: float
        :param lowcut:
            Low cut (Hz), if set to None will not apply a lowcut
        :type highcut: float
        :param highcut:
            High cut (Hz), if set to None will not apply a highcut.
        :type samp_rate: float
        :param samp_rate:
            New sampling rate in Hz.
        :type filt_order: int
        :param filt_order:
            Filter level (number of corners).
        :type length: float
        :param length: Length of template waveform in seconds.
        :type prepick: float
        :param prepick: Pre-pick time in seconds
        :type swin: str
        :param swin:
            P, S, P_all, S_all or all, defaults to all: see note in
            :func:`eqcorrscan.core.template_gen.template_gen`
        :type process_len: int
        :param process_len: Length of data in seconds to download and process.
        :type all_horiz: bool
        :param all_horiz:
            To use both horizontal channels even if there is only a pick on
            one of them.  Defaults to False.
        :type delayed: bool
        :param delayed: If True, each channel will begin relative to it's own
            pick-time, if set to False, each channel will begin at the same
            time.
        :type plot: bool
        :param plot: Plot templates or not.
        :type plotdir: str
        :param plotdir:
            The path to save plots to. If `plotdir=None` (default) then the
            figure will be shown on screen.
        :type min_snr: float
        :param min_snr:
            Minimum signal-to-noise ratio for a channel to be included in the
            template, where signal-to-noise ratio is calculated as the ratio
            of the maximum amplitude in the template window to the rms
            amplitude in the whole window given.
        :type parallel: bool
        :param parallel: Whether to process data in parallel or not.
        :type num_cores: int
        :param num_cores:
            Number of cores to try and use, if False and parallel=True,
            will use either all your cores, or as many traces as in the data
            (whichever is smaller).
        :type save_progress: bool
        :param save_progress:
            Whether to save the resulting template set at every data step or
            not. Useful for long-running processes.
        :type skip_short_chans: bool
        :param skip_short_chans:
            Whether to ignore channels that have insufficient length data or
            not. Useful when the quality of data is not known, e.g. when
            downloading old, possibly triggered data from a datacentre
        :type save_progress: bool
        :param save_progress:
            Whether to save the resulting party at every data step or not.
            Useful for long-running processes.

        .. note::
            *Method specific arguments:*

            - `from_client` requires:
                :param str client_id:
                    string passable by obspy to generate Client, or any object
                    with a `get_waveforms` method, including a Client instance.
                :param `obspy.core.event.Catalog` catalog:
                    Catalog of events to generate template for
                :param float data_pad: Pad length for data-downloads in seconds
            - `from_seishub` requires:
                :param str url: url to seishub database
                :param `obspy.core.event.Catalog` catalog:
                    Catalog of events to generate template for
                :param float data_pad: Pad length for data-downloads in seconds
            - `from_meta_file` requires:
                :param str meta_file:
                    Path to obspy-readable event file, or an obspy Catalog
                :param `obspy.core.stream.Stream` st:
                    Stream containing waveform data for template. Note that
                    this should be the same length of stream as you will use
                    for the continuous detection, e.g. if you detect in
                    day-long files, give this a day-long file!
                :param bool process:
                    Whether to process the data or not, defaults to True.

        .. Note::
            Method: `from_sac` is not supported by Tribe.construct and must
            use Template.construct.

        .. Note:: Templates will be named according to their start-time.
        """
        templates, catalog, process_lengths = self.template_gen(
            method=method, lowcut=lowcut, highcut=highcut, length=length,
            filt_order=filt_order, samp_rate=samp_rate, prepick=prepick,
            return_event=True, save_progress=save_progress, swin=swin,
            process_len=process_len, all_horiz=all_horiz, plotdir=plotdir,
            delayed=delayed, plot=plot, min_snr=min_snr, parallel=parallel,
            num_cores=num_cores, skip_short_chans=skip_short_chans,
            **kwargs)
        for template, event, process_len in zip(templates, catalog,
                                                process_lengths):
            t = Template()
            for tr in template:
                if not np.any(tr.data.astype(np.float16)):
                    Logger.warning('Data are zero in float16, missing data,'
                                   ' will not use: {0}'.format(tr.id))
                    template.remove(tr)
            if len(template) == 0:
                Logger.error('Empty Template')
                continue
            t.st = template
            t.name = template.sort(['starttime'])[0]. \
                stats.starttime.strftime('%Y_%m_%dt%H_%M_%S')
            t.lowcut = lowcut
            t.highcut = highcut
            t.filt_order = filt_order
            t.samp_rate = samp_rate
            t.process_length = process_len
            t.prepick = prepick
            event.comments.append(Comment(
                text="eqcorrscan_template_" + t.name,
                creation_info=CreationInfo(agency='eqcorrscan',
                                           author=getpass.getuser())))
            t.event = event
            self.templates.append(t)
        return self


# function to convert snuffler marker file to event template
def markers_to_template(marker_file_path, prepick_offset, time_markers=False):
    """
    Loads snuffler marker file (for a single event) and generates template
    objects required for signal detection via matched-filter analysis with
    EQcorrscan.

    Limitations: built to work with a single event.

    Example:
        # define the path to the snuffler marker file
        marker_file_path = "lfe_template.mrkr"

        # define the offset from the S wave pick to start of template
        prepick_offset = 11 # in seconds

        templates, station_dict, pick_offset = markers_to_template(marker_file_path, prepick_offset)

        # print each line in templates without newline character
        for line in templates:
            print(line[:-1])

        # print contents of station dict
        for station in station_dict.keys():
            print(f"{station} {station_dict[station]}")

        # print contents of pick offset dict
        for station in pick_offset.keys():
            print(f"{station} {pick_offset[station]}")

    """

    # build dict of picks from the marker file
    pick_dict = {}
    station_dict = {}
    template_windows = {}
    pick_offset = {}
    # keep track of earliest pick time
    earliest_pick_time = []

    # read the marker file line by line
    with open(marker_file_path, 'r') as file:
        for line_Contents in file:
            if len(line_Contents) > 52:  # avoid irrelevant short lines
                if (line_Contents[0:5] == 'phase') and (line_Contents[
                                                        -20:-19] == 'S'):
                    # avoids error from "None" contents
                    if len(line_Contents[35:51].split('.')) > 1:
                        # print(line_Contents[7:31])
                        pick_station = line_Contents[35:51].split('.')[1]
                        pick_channel = line_Contents[35:51].split('.')[3]
                        pick_channel = pick_channel.split(' ')[0]
                        pick_channel = pick_channel[:-1] + "Z"
                        pick_network = line_Contents[35:51].split('.')[0]
                        pick_time = UTCDateTime(line_Contents[7:31])
                        pick_dict[pick_station] = pick_time

                        # print(f"{pick_network} {pick_station} "
                        #       f"{pick_channel} {pick_time}")

                        # build station_dict
                        station_dict[pick_station] = {"network": pick_network,
                                                      "channel": pick_channel}

                        # build template window object for plotting, 16 s templ
                        template_windows[f"{pick_network}.{pick_station}"] =\
                            [pick_time - prepick_offset, (pick_time -
                                                          prepick_offset) + 16]

                        # check if this is the earliest pick time
                        if len(earliest_pick_time) == 0:
                            earliest_pick_time.append(pick_time)
                            earliest_pick_station = pick_station
                        else:
                            # redefine earliest if necessary
                            if pick_time < earliest_pick_time[0]:
                                earliest_pick_time[0] = pick_time
                                earliest_pick_station = pick_station

    # account for prepick offset in earliest pick
    earliest_pick_time = earliest_pick_time[0] - prepick_offset

    # build templates object header (location is made up)
    templates = [f"# {earliest_pick_time.year} {earliest_pick_time.month:02} "
                 f"{earliest_pick_time.day:02} {earliest_pick_time.hour:02} "
                 f"{earliest_pick_time.minute:02} "
                 f"{earliest_pick_time.second:02}."
                 f"{earliest_pick_time.microsecond:02}  61.8000 "
                 f"-144.0000  30.00  1.00  0.0  0.0  0.00  1\n"]

    # append earliest station to templates list
    templates.append(f"{earliest_pick_station}    0.000  1       P\n")

    # create pick offset dict for pick offset from main trace
    pick_offset[earliest_pick_station] = 0.0

    # append all other stations to templates list
    for station in pick_dict.keys():
        if station != earliest_pick_station:
            time_diff = pick_dict[station] - (earliest_pick_time + prepick_offset)
            microseconds = int((time_diff - int(time_diff)) * 1000)
            # print(f"time_diff: {int(time_diff)} microseconds:{microseconds}")
            templates.append(f"{station:4}   {int(time_diff):2}."
                             f"{microseconds:03}  1       P\n")
            pick_offset[station] = time_diff

    return_tuple = [templates, station_dict, pick_offset]

    if time_markers:
        return_tuple.append(template_windows)

    return tuple(return_tuple)


# helper function to shift trace times
def time_Shift(trace, time_offset):
    """
    Shifts a trace in time by the specified time offset (in seconds).

    Example:
        # shift trace times by -2 seconds
        shift = -2
        shifted_trace = time_Shift(trace, shift)
    """
    frequencies = np.fft.fftfreq(trace.stats.npts, d=trace.stats.delta)
    fourier_transform = np.fft.fft(trace.data)

    # Shift
    for index, freq in enumerate(frequencies):
        fourier_transform[index] = fourier_transform[index] * np.exp(2.0
                                                                     * np.pi * 1j * freq * time_offset)

    # back fourier transform
    trace.data = np.real(np.fft.ifft(fourier_transform))
    trace.stats.starttime += time_offset

    return trace


# function to build template for matched-filter
def make_Templates(templates, template_files, station_dict, template_length,
                   template_prepick):
    """
    Generates an EQcorrscan Tribe object which stores templates for
    matched-filter analysis. Template start times are specified by the
    templates object as shown in the example, and template data are loaded
    from the miniseed files stored in files in the template_files list.
    station_dict contains the network and channel code for each station (
    information needed by EQcorrscan),

    Example:
        # create HYPODD format file with template events to import them into EQcorrscan
        # entries are: #, YR, MO, DY, HR, MN, SC, LAT, LON, DEP, MAG, EH, EZ, RMS, ID
        #              followed by lines of observations: STA, TT, WGHT, PHA
        # as specified here: https://www.ldeo.columbia.edu/~felixw/papers/Waldhauser_OFR2001.pdf
        templates = ["# 2016  9 26  9 25 49.00  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
                     "WASW    0.000  1       P\n",
                     "MCR3    3.000  1       P\n",
                     "N25K    3.500  1       P\n"]
        template_files = [
        "/Users/human/Dropbox/Research/Alaska/build_templates/subset_stations/AV.WASW.SHZ.2016-09-26.ms",
        "/Users/human/Dropbox/Research/Alaska/build_templates/subset_stations/YG.MCR3.BHZ.2016-09-26.ms",
        "/Users/human/Dropbox/Research/Alaska/build_templates/subset_stations/TA.N25K.BHZ.2016-09-26.ms"]

        station_dict = {"WASW": {"network": "AV", "channel": "SHZ"},
                    "MCR3": {"network": "YG", "channel": "BHZ"},
                    "N25K": {"network": "TA", "channel": "BHZ"}}

        # make templates 14 seconds with a prepick of 0.5s, so 14.5 seconds
        # total
        template_length = 14
        template_prepick = 0.5

        tribe = make_Templates(templates, files, station_dict,
                               template_length, template_prepick)
    """

    # FIXME: make this more pythonic, e.g.

    '''
    # initialize an event to add to the Obspy catalog
    event = Event(
        # define the event origin location and time
        origins=[Origin(
            latitude=61.9833, longitude=-144.0437, depth=1700, 
            time=UTCDateTime(2016, 9, 26, 8, 52, 40))],
        # define the event magnitude
        magnitudes=[Magnitude(mag=1.1)],
        # define the arrival times of phases at different stations
        picks=[
            # three picks are defined here on the BHZ component of three stations in the YG network
            Pick(time=UTCDateTime(2016, 9, 26, 8, 52, 45, 180000), phase_hint="P",
                    waveform_id=WaveformStreamID(
                        network_code="YG", station_code="RH08", channel_code="BHZ")),
            Pick(time=UTCDateTime(2016, 9, 26, 8, 52, 45, 809000), phase_hint="P",
                    waveform_id=WaveformStreamID(
                        network_code="YG", station_code="NEB1", channel_code="BHZ")),
            Pick(time=UTCDateTime(2016, 9, 26, 8, 52, 45, 661000), phase_hint="P",
                    waveform_id=WaveformStreamID(
                        network_code="YG", station_code="NEB3", channel_code="BHZ"))])
    
    # generate the catalog from a list of events (in this case 1 event comprised of 3 picks)
    catalog = Catalog([event])  
    '''

    # now write to file
    with open("templates.pha", "w") as file:
        for line in templates:
            file.write(line)

    # read the file into an Obspy catalog
    catalog = read_events("templates.pha", format="HYPODDPHA")
    # complete missing catalog info (required by EQcorrscan)
    for event in catalog.events:
        picks = event.picks.copy()
        for index, pick in enumerate(event.picks):
            if pick.waveform_id.station_code in station_dict:
                picks[index].waveform_id.network_code = \
                    station_dict[pick.waveform_id.station_code]["network"]
                # # for single component templates
                # picks[index].waveform_id.channel_code = \
                #     station_dict[pick.waveform_id.station_code]["channel"]

                # for 3 component templates
                # force Z component first
                picks[index].waveform_id.channel_code = \
                    station_dict[pick.waveform_id.station_code]["channel"][
                    :-1] + "Z"
                # copy Z entry
                pick_copy1 = picks[index].copy()
                pick_copy2 = picks[index].copy()
                # make N and E entries
                pick_copy1.waveform_id.channel_code = \
                    pick_copy1.waveform_id.channel_code[:-1] + 'N'
                picks.append(pick_copy1)
                pick_copy2.waveform_id.channel_code = \
                    pick_copy2.waveform_id.channel_code[:-1] + 'E'
                picks.append(pick_copy2)

        event.picks = picks

    # fig = catalog.plot(projection="local", resolution="h")

    # build stream of data from local files (use same length files as desired for detection)
    st = Stream()
    for file in template_files:
        st += read(file)
    st.interpolate(sampling_rate=100.0)

    tribe = Tribe().construct(method="from_meta_file", meta_file=catalog,
                              st=st, lowcut=1.0, highcut=15.0, samp_rate=100.0,
                              length=template_length, filt_order=4,
                              prepick=template_prepick, swin='all',
                              process_len=86400, parallel=False)  # min_snr=5.0,
    # 46 detections for 2-8 Hz
    # 56 detections for 1-15 Hz

    # print(tribe)
    return tribe


# function to drive signal detection with the help of EQcorrscan
def detect_signals(templates, template_files, station_dict, template_length,
                template_prepick, detection_files_path, start_date,
                   end_date, thresh_type, detect_thresh):
    """
    Detects signals (LFEs in this case) in time series via matched-filtering
    with specified template(s).

    Example:
        # manually define templates from station TA.N25K (location is made up)
        templates = ["# 2016  9 26  9 28 41.34  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
                     "N25K    0.000  1       P\n"]

        # and define a station dict to add data needed by EQcorrscan
        station_dict = {"N25K": {"network": "TA", "channel": "BHZ"}}

        # define template length and prepick length (both in seconds)
        template_length = 16.0
        template_prepick = 0.5

        # build stream of all station files for templates
        files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/N25K"
        template_files = glob.glob(f"{files_path}/*.ms")

        # define path of files for detection
        detection_files_path = "/Users/human/ak_data/inner"
        # define dates of interest
        # # this takes 18h 13m for 1 template of N25K across inner stations
        # #          abs.25: 1218
        # # SNR > 3: abs.25: 445 (36.5%)
        start_date = UTCDateTime("2016-06-15T00:00:00.0Z")
        end_date = UTCDateTime("2018-08-11T23:59:59.9999999999999Z")
        # # MAD8: 195, MAD9: 55, MAD10: 25, MAD11: 11, abs.2: 425, abs.25: 45,
        # #       abs.23: 100
        # start_date = UTCDateTime("2016-09-26T00:00:00.0Z")
        # end_date = UTCDateTime("2016-10-01T23:59:59.9999999999999Z")

        # run detection and time it
        start = time.time()
        party = detect_signals(templates, template_files, station_dict,
                               template_length, template_prepick,
                               detection_files_path, start_date, end_date)
        end = time.time()
        hours = int((end - start) / 60 / 60)
        minutes = int(((end - start) / 60) - (hours * 60))
        seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
        print(f"Runtime: {hours} h {minutes} m {seconds} s")

        # load party object from file
        infile = open('party_06_15_2016_to_08_12_2018_abs.25.pkl', 'rb')
        party = pickle.load(infile)
        infile.close()

        # get the catalog
        catalog = party.get_catalog()

        # inspect the party growth over time
        detections_fig = party.plot(plot_grouped=True)
        rate_fig = party.plot(plot_grouped=True, rate=True)
        print(sorted(party.families, key=lambda f: len(f))[-1])

        # plot the party detections
        title = "abs_0.25_detections"
        plot_party_detections(party, detection_files_path, title=title,
                              save=False)

        # get the most productive family
        family = sorted(party.families, key=lambda f: len(f))[-1]
        print(family)

        # look at family template
        fig = family.template.st.plot(equal_scale=False, size=(800, 600))

        # look at family detection at index 0
        detection = family.detections[0]
        detection_time = detection.detect_time
        from figures import plot_Time_Series
        # define dates of interest
        print(f"Detection time: {detection_time}")
		doi = detection_time - 10
		doi_end = doi + 30
		# define time series files path
		files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/N25K"
		# bandpass filter from 1-15 Hz
		filter = True
		bandpass = [1, 15]
		fig = plot_Time_Series(doi, doi_end, files_path, filter=filter,
							   bandpass=bandpass)

		# TODO: plot all stations around detection times

		# TODO: stacking utilizing a small template for xcorr timeshift

	Another example spanning multiple days:
        # time the run
        import time
        start = time.time()

        # define template length and prepick length (both in seconds)
        template_length = 16.0
        template_prepick = 0.5

        # build stream of all station files for templates
        files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/2016-09-26"
        doi = UTCDateTime("2016-09-26T00:00:00.0Z")
        template_files = glob.glob(f"{files_path}/*.{doi.year}-{doi.month:02}"
                                      f"-{doi.day:02}.ms")

        # get templates and station_dict objects from picks in marker file
        marker_file_path = "lfe_templates_2.mrkr"
        prepick_offset = 11 # in seconds (was 11 for templates_2, 0 for templates_3)
        templates, station_dict, pick_offset = markers_to_template(marker_file_path, prepick_offset)

        # define path of files for detection
        detection_files_path = "/Volumes/DISK/alaska/data"
        # define period of interest for detection
        start_date = UTCDateTime("2016-06-15T00:00:00.0Z")
        end_date = UTCDateTime("2018-08-11T23:59:59.9999999999999Z")

        # # run detection
        # party = detect_signals(templates, template_files, station_dict,
        #                        template_length, template_prepick,
        #                        detection_files_path, start_date, end_date)
        # end = time.time()
        # hours = int((end - start) / 60 / 60)
        # minutes = int(((end - start) / 60) - (hours * 60))
        # seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
        # print(f"Runtime: {hours} h {minutes} m {seconds} s")

        # load party object from file
        infile = open('party_06_15_2016_to_08_12_2018.pkl', 'rb')
        party = pickle.load(infile)
        infile.close()

        # inspect the party object detections
        detections_fig = party.plot(plot_grouped=True)
        rate_fig = party.plot(plot_grouped=True, rate=True)
        print(sorted(party.families, key=lambda f: len(f))[-1])

        # load previous stream list?
        load_stream_list = False
        # get the stacks
        start = time.time()

        stack_list = stack_waveforms(party, pick_offset, detection_files_path,
                                     template_length, template_prepick,
                                     load_stream_list=load_stream_list,
                                     main_trace)

        end = time.time()
        hours = int((end - start) / 60 / 60)
        minutes = int(((end - start) / 60) - (hours * 60))
        seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
        print(f"Runtime: {hours} h {minutes} m {seconds} s")

        # loop over stack list and show the phase weighted stack and linear
        # stack for each group
        for group in stack_list:
            # get the stack
            stack_pw = group[0]
            # get the plot handle for the stack, add a title, show figure
            plot_stack(stack_pw, filter=filter, bandpass=bandpass,
                   	   title="Phase weighted stack")

            stack_lin = group[1]
            plot_stack(stack_lin, filter=filter, bandpass=bandpass,
            		   title="Linear stack")
    """
    # set up logging (levels: DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    tribe = make_Templates(templates, template_files, station_dict,
                           template_length, template_prepick)

    # loop over days and get detections
    iteration_date = start_date
    party_list = []
    while iteration_date < end_date:
        print(iteration_date) # print date for long runs
        # build stream of all files on day of interest
        day_file_list = glob.glob(f"{detection_files_path}/*."
                                  f"{iteration_date.year}"
                                  f"-{iteration_date.month:02}"
                                  f"-{iteration_date.day:02}.ms")

        # load files into stream
        st = Stream()
        for file in day_file_list:
            st += read(file)

        try:
            # detect
            party = tribe.detect(stream=st, threshold=detect_thresh, daylong=True,
                                 threshold_type=thresh_type, trig_int=8.0,
                                 plot=False,
                                 return_stream=False, parallel_process=False,
                                 ignore_bad_data=True)
        except Exception:
            party = Party(families=[Family(template=Template())])
            pass

        # append detections to party list if there are detections
        if len(party.families[0]) > 0:
            party_list.append(party)

        # update the next file start date
        iteration_year = iteration_date.year
        iteration_month = iteration_date.month
        iteration_day = iteration_date.day
        # get number of days in current month
        month_end_day = calendar.monthrange(iteration_year,
                                            iteration_month)[1]
        # check if year should increment
        if iteration_month == 12 and iteration_day == month_end_day:
            iteration_year += 1
            iteration_month = 1
            iteration_day = 1
        # check if month should increment
        elif iteration_day == month_end_day:
            iteration_month += 1
            iteration_day = 1
        # else only increment day
        else:
            iteration_day += 1

        iteration_date = UTCDateTime(f"{iteration_year}-"
                                     f"{iteration_month:02}-"
                                     f"{iteration_day:02}T00:00:00.0Z")

    # add all detections to a single party
    if len(party_list) > 1:
        for index, iteration_party in enumerate(party_list):
            # skip first party
            if index > 0:
                # add events of current party family to first party family
                party_list[0].families[0] = party_list[0].families[0].append(
                                                   iteration_party.families[0])

    # extract and return the first party or None if no party object
    if len(party_list) > 0:
        party = party_list[0]

        # save party to pickle
        filename = f'party_{start_date.month:02}_{start_date.day:02}_' \
                   f'{start_date.year}_to_{end_date.month:02}' \
                   f'_{end_date.day:02}_{end_date.year}.pkl'
        outfile = open(filename, 'wb')
        pickle.dump(party, outfile)
        outfile.close()

        return party

    else:
        return None

    # # # # .mrkr template # # # #
    # detection over entire period
    # 601 family detections: Runtime: 47 h 44 m 34 s

    # # # 14 second template w/ 0.5 prepick # # #
    # 17 detections: "MAD" @ 11.0  <---
    # 35 detections: "MAD" @ 9.0
    # 56 detections: "MAD" @ 8.0
    # ----------------------------
    # 24 detections: "abs" @ 0.1
    # 11 detections: "abs" @ 0.12  <---
    #  3 detections: "abs" @ 0.15
    #  0 detections: "abs" @ 0.2
    # ----------------------------
    # 24 detections: "av_chan_corr" @ 0.1
    # 11 detections: "av_chan_corr" @ 0.12

    # # # 25 second template w/ 0.5 prepick # # #
    # 15 detections: "MAD" @ 11.0
    # 36 detections: "MAD" @ 9.0
    # 65 detections: "MAD" @ 8.0

    # # # 10 second template w/ 0.5 prepick # # #
    # 13 detections: "MAD" @ 11.0  <---
    # 31 detections: "MAD" @ 9.0
    # 52 detections: "MAD" @ 8.0


def cull_detections(party, detection_files_path, snr_threshold, main_trace):
    """
    Removes detections in a party object that are below a specified signal
    to noise ratio (snr_threshold[0]), and above a specified signal to noise
    ratio (snr_threshold[1]). Also removes traces with mostly zeros and
    generates a histogram of the new and old SNR distributions.

    Args:
        party: EQcorrscan Party object
        detection_files_path: string containing path to waveforms
        snr_threshold: list of two floats of signal to noise ratio cut-offs

    Returns:
        new_party: EQcorrscan Party object containing only detections with
                   SNR >= snr_threshold

    Example:
        # load party object from file
        infile = open('party_06_15_2016_to_08_12_2018_abs.25.pkl', 'rb')
        party = pickle.load(infile)
        infile.close()

        # define path of files from detection
        detection_files_path = "/Users/human/ak_data/inner"

        # set snr threshold and cull the party detections
        snr_threshold = [3.0, 8.0]
        culled_party = cull_detections(party, detection_files_path,
                                       snr_threshold, main_trace)

    """
    deletion_indices = []
    old_snrs = []
    new_snrs = []
    detections = []
    # loop over detections
    for index, detection in enumerate(party.families[0].detections):
        # define the date of interest to get appropriate files
        doi = detection.detect_time

        # find all files for specified day
        day_file_list = sorted(glob.glob(f"{detection_files_path}/"
                                         f"{main_trace[0]}.{main_trace[1]}."
                                         f"{main_trace[2]}"
                                         f".{doi.year}-{doi.month:02}"
                                         f"-{doi.day:02}.ms"))

        # guard against missing file
        try:
            # should only be one file, guard against many
            file = day_file_list[0]

            # load file into stream
            st = Stream()
            st += read(file)

            # filter before trimming to avoid edge effects
            if filter:
                # bandpass filter specified frequencies
                st.filter('bandpass', freqmin=1, freqmax=15)

            # trim to 60 seconds total
            st.trim(doi - 20, doi + 40)
            trace = st[0]  # get the trace

            # round data array to nearest int to count zeros
            trace_data_zeros = np.rint(trace.data.copy())
            zero_count = trace_data_zeros[np.where(trace_data_zeros == 0)].size

            trace_snr = snr(trace)[0]
            # check the snr condition for deletion
            if trace_snr < snr_threshold[0] or trace_snr > snr_threshold[1]:
                deletion_indices.append(index)
                old_snrs.append(trace_snr)
            elif trace_snr == np.nan:
                deletion_indices.append(index)
                old_snrs.append(trace_snr)
            # check the zeros condition for deletion
            elif zero_count > 0.3 * trace.stats.npts:
                deletion_indices.append(index)
                old_snrs.append(trace_snr)
            else:
                new_snrs.append(trace_snr)
                old_snrs.append(trace_snr)
                # build list of detections to keep
                detections.append(detection)

        # if file isn't found, mark detection for deletion
        except Exception:
            deletion_indices.append(index)
            old_snrs.append(np.nan)
            pass

    # make a new party object
    template = party.families[0].template
    new_party = Party(families=Family(template, detections))

    # new_party = party.copy()
    # # delete detections below threshold
    # deleted_detections = []
    # for index in sorted(deletion_indices, reverse=True):
    #     deleted_detections.append(new_party.families[0].detections[
    #                                   index].copy())
    #     del new_party.families[0].detections[index]

    # plot_distribution(old_snrs, title="SNR distribution of all detections",
    #                   save=False)
    # plot_distribution(new_snrs, title="SNR distribution of culled detections",
    #                   save=False)

    return new_party

# function to generate linear and phase-weighted waveform stacks station by
# station (to avoid memory bottleneck) via EQcorrscan stacking routines
def stack_waveforms(party, pick_offset, streams_path, template_length,
                    template_prepick, station_dict, main_trace):
    """
    Generates stacks of waveforms from families specified within a Party
    object, using the miniseed files present in the specified path
    (streams_path). Building streams for party families is slow,
    so previously generated stream lists can be used by specifying
    load_stream_list=True.

    Limitations:
        - lowest sampling rate is currently statically defined
        - limited to single day
        - filter is statically defined

    Example:
        # time the run
        import time
        start = time.time()

        # define template length and prepick length (both in seconds)
        template_length = 16.0
        template_prepick = 0.5

        # get templates and station_dict objects from picks in marker file
        marker_file_path = "lfe_templates_3.mrkr"
        prepick_offset = 0 # in seconds (was 11 for templates_2, 0 for templates_3)
        templates, station_dict, pick_offset = markers_to_template(marker_file_path, prepick_offset)

        # define path of files for detection
        detection_files_path = "/Volumes/DISK/alaska/data"

        # load party object from file
        infile = open('party_06_15_2016_to_08_12_2018.pkl', 'rb')
        party = pickle.load(infile)
        infile.close()

        # inspect the party object detections
        detections_fig = party.plot(plot_grouped=True)
        rate_fig = party.plot(plot_grouped=True, rate=True)
        print(sorted(party.families, key=lambda f: len(f))[-1])

        # # load previous stream list?
        # load_stream_list = False
        # # get the stacks station by station to avoid memory error
        # stack_list = stack_waveforms(party, pick_offset,
        #                              detection_files_path, template_length,
        #                              template_prepick, station_dict,
        #                              main_trace)
        # end = time.time()
        # hours = int((end - start) / 60 / 60)
        # minutes = int(((end - start) / 60) - (hours * 60))
        # seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
        # print(f"Runtime: {hours} h {minutes} m {seconds} s")
        #
        # # save stacks as pickle file
        # outfile = open('stack_list.pkl', 'wb')
        # pickle.dump(stack_list, outfile)
        # outfile.close()

        # load stack list from file
        infile = open('stack_list.pkl', 'rb')
        stack_list = pickle.load(infile)
        infile.close()

        # loop over stack list and show the phase weighted stack and linear
        # stack for each group
        for group in stack_list:
            # get the stack
            stack_pw = group[0]
            # get the plot handle for the stack, add a title, show figure
            plot_stack(stack_pw, filter=filter, bandpass=bandpass,
                   	   title="Phase weighted stack")

            stack_lin = group[1]
            plot_stack(stack_lin, filter=filter, bandpass=bandpass,
            		   title="Linear stack")
    """
    # extract pick times for each event from party object
    # pick_times is a list of the pick times for the main trace (with
    # earliest pick time)
    pick_times = []
    for event in party.families[0].catalog.events:
        for pick in event.picks:
            # station with earliest pick defines "pick time"
            # if pick.waveform_id.station_code == "WASW" and \
            #         pick.waveform_id.network_code == "AV" and \
            #         pick.waveform_id.channel_code == "SHZ":
            #     pick_times.append(pick.time)
            # FIXME: this should be dynamic, not hard coded
            if pick.waveform_id.station_code == main_trace[1] and \
                    pick.waveform_id.network_code == main_trace[0] and \
                    pick.waveform_id.channel_code == main_trace[2]:
                pick_times.append(pick.time)

    # loop over stations and generate a stack for each station:channel pair
    stack_pw = Stream()
    stack_lin = Stream()
    for station in station_dict.keys():
        network = station_dict[station]["network"]
        channels = []
        # # for 1 component
        # channels.append(station_dict[station]["channel"]) # append main compnt

        # for 3 components
        # force Z component
        channels.append(station_dict[station]["channel"][:-1] + "Z")
        channels.append(f"{channels[0][:-1]}N") # append N component
        channels.append(f"{channels[0][:-1]}E")  # append E component

        for channel in channels:
            print(f"Assembling streams for {station}.{channel}")

            stream_list = []
            for index in tqdm(range(len(pick_times))):
                pick_time = pick_times[index]

                # build stream of detection file
                day_file_list = glob.glob(f"{streams_path}/{network}."
                                          f"{station}."
                                          f"{channel}.{pick_time.year}"
                                          f"-{pick_time.month:02}"
                                          f"-{pick_time.day:02}.ms")

                # guard against missing expected files
                if len(day_file_list) > 0:

                    # should only be one file, but safeguard against many
                    file = day_file_list[0]
                    # extract file info from file name
                    file_station = file.split("/")[-1]
                    file_station = file_station.split(".")[1]
                    # load day file into stream
                    day_st = Stream()
                    day_st += read(file)
                    # bandpass filter
                    day_st.filter('bandpass', freqmin=1, freqmax=15)
                    # interpolate to 100 Hz
                    day_st.interpolate(sampling_rate=100)

                    # match station with specified pick offset
                    station_pick_time = pick_time + pick_offset[file_station]

                    # trim trace before adding to stream from pick_offset spec
                    day_st.trim(station_pick_time, station_pick_time +
                                template_length + template_prepick)

                    stream_list.append((day_st, index))
                    # st.plot()

            # get group streams to stack (only a single group here)
            group_streams = [st_tuple[0] for st_tuple in stream_list]

            # loop over each detection in group
            for group_idx, group_stream in enumerate(group_streams):
                # align traces before stacking
                for trace_idx, trace in enumerate(group_stream):
                    # align traces from pick offset dict
                    shift = -1 * pick_offset[trace.stats.station]
                    group_streams[group_idx][trace_idx] = time_Shift(trace, shift)

            # guard against stacking error:
            try:
                # generate phase-weighted stack
                stack_pw += PWS_stack(streams=group_streams)

                # and generate linear stack
                stack_lin += linstack(streams=group_streams)

            except Exception:
                pass

    # if the stacks exist, plot them
    if len(stack_pw) > 0:
        plot_stack(stack_pw, filter=filter, bandpass=bandpass,
                   title='Phase weighted stack')

    if len(stack_lin) > 0:
        plot_stack(stack_lin, filter=filter, bandpass=bandpass,
                   title='Linear stack')

    return [stack_pw, stack_lin]

# function to get a stream of time series' of detections from a party object
def get_detections(party, streams_path, main_trace):
    # first extract pick times for each event from party object
    # pick_times is a list of the pick times for the main trace
    pick_times = []
    pick_network, pick_station, pick_channel = main_trace
    for event in party.families[0].catalog.events:
        for pick in event.picks:
            # trace with highest amplitude signal
            if pick.waveform_id.station_code == pick_station and \
                    pick.waveform_id.network_code == pick_network and \
                    pick.waveform_id.channel_code == pick_channel:
                pick_times.append(pick.time)

    # loop over stations and generate a stack for each station:channel pair

    print(f"Assembling streams for {pick_station}.{pick_channel}")

    sta_chan_stream = Stream()
    for index in tqdm(range(len(pick_times))):
        pick_time = pick_times[index]

        # find the local file corresponding to the station:channel pair
        day_file_list = glob.glob(f"{streams_path}/{pick_network}."
                                  f"{pick_station}."
                                  f"{pick_channel}.{pick_time.year}"
                                  f"-{pick_time.month:02}"
                                  f"-{pick_time.day:02}.ms")

        # guard against missing files
        if len(day_file_list) > 0:
            # should only be one file, but safeguard against many
            file = day_file_list[0]
            # load day file into stream
            day_st = read(file)
            # bandpass filter
            day_st.filter('bandpass', freqmin=1, freqmax=15)
            # interpolate to 100 Hz
            day_st.interpolate(sampling_rate=100)
            # trim trace to 60 seconds surrounding pick time
            day_st.trim(pick_time - 20, pick_time + 40, pad=True,
                        fill_value=np.nan, nearest_sample=True)

            sta_chan_stream += day_st

        # if no file, append blank trace to preserve stream length
        # equality with pick_times
        else:
            sta_chan_stream += Trace()

    return sta_chan_stream

# stacking routine to generate stacks from template detections (doesn't use
# EQcorrscan stacking routine)
def stack_template_detections(party, streams_path, main_trace,
                              template_times, align_type,
                              animate_stacks=False):
    """
    # FIXME: change to non-memory-limited implementation

    An implementation of phase-weighted and linear stacking that is
    independent of EQcorrscan routines, allowing more customization of the
    workflow.

    Types of trace shifting for stacking (align_type):
    'zero'  : no time shift for stacking.
    'self'  : cross-correlation of each detection trace with the template
              signal on each station to determine time shifts for stacking.
    'fixed' : time shifts for all stations are determined from the cross-
              correlation of each detection on the main trace with the template
              signal. This is equivalent to the 'self' method on the main
              trace, then applying those same shift times to each station.
    'med'   : cross-correlation of each detection trace with the detection
              trace with the median SNR signal on each station to determine
              time shifts for stacking.
    'max'   : cross-correlation of each detection trace with the detection
              trace with the maximum SNR signal on each station to determine
              time shifts for stacking.

    Example:
        # time the run
        start = time.time()

        # define template length and prepick length (both in seconds)
        template_length = 16.0
        template_prepick = 0.5

        # define the main trace to use for detections (best amplitude station)
        main_trace = ("TA", "N25K", "BHN")

        # define path of files for detection
        streams_path = "/Users/human/ak_data/inner"

        # load party object from file
        party_file = 'party_06_15_2016_to_08_12_2018_MAD8.pkl'
        infile = open(party_file, 'rb')
        party = pickle.load(infile)
        infile.close()

        # inspect the party object detections
        detections_fig = party.plot(plot_grouped=True)
        rate_fig = party.plot(plot_grouped=True, rate=True)
        print(sorted(party.families, key=lambda f: len(f))[-1])

        # get the stacks
        stack_list = stack_template_detections(party, streams_path, main_trace,
                                               align_type='zero')
        end = time.time()
        hours = int((end - start) / 60 / 60)
        minutes = int(((end - start) / 60) - (hours * 60))
        seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
        print(f"Runtime: {hours} h {minutes} m {seconds} s")

        # save stacks as pickle file
        outfile = open('inner_stack_list.pkl', 'wb')
        pickle.dump(stack_list, outfile)
        outfile.close()
    """
    # function to build a stream from which to generate time shifts for
    # a fixed-location event
    def build_main_stream(main_trace, streams_path, pick_times):
        main_stream = Stream()
        main_stream_snrs = []
        # loop over pick times to assemble stream & show tqdm progress bar
        for index in tqdm(range(len(pick_times))):
            pick_time = pick_times[index]
            # find the local file corresponding to the station:channel pair
            file_list = glob.glob(f"{streams_path}/{main_trace[0]}."
                                  f"{main_trace[1]}."
                                  f"{main_trace[2]}.{pick_time.year}"
                                  f"-{pick_time.month:02}"
                                  f"-{pick_time.day:02}.ms")

            # guard against missing files
            if len(file_list) > 0:
                # should only be one file, but safeguard against many
                file = file_list[0]
                # load day file into stream
                day_st = read(file)
                # bandpass filter
                day_st.filter('bandpass', freqmin=1, freqmax=15)
                # interpolate to 100 Hz
                day_st.interpolate(sampling_rate=100)
                # trim trace to 60 seconds surrounding pick time
                day_st.trim(pick_time - 27, pick_time + 33, pad=True,
                            fill_value=np.nan, nearest_sample=True)
                # append snr
                main_stream_snrs.append(snr(day_st)[0])

                # add trace to main_stream
                main_stream += day_st

            # otherwise append empty Trace to preserve equal length with
            # pick_times
            else:
                main_stream += Trace()
                main_stream_snrs.append(np.nan)

        return main_stream, main_stream_snrs

    # function to generate linear and phase-weighted stacks from a stream
    def generate_stacks(stream, normalize=True, animate=False):
        # guard against stacking zeros and create data array
        data = []
        reference_idx = 0
        for tr_idx ,trace in enumerate(stream):
            if len(trace.data) > 0:
                if trace.data.max() > 0:
                    data.append(trace.data)
                    reference_idx = tr_idx
        # put the data into a numpy array
        data = np.asarray(data, dtype='float64')

        # if there is no data, return zeros
        if data.size == 0:
            lin = stream[0].copy()
            lin.data = np.zeros_like(lin.data)
            pws = stream[0].copy()
            pws.data = np.zeros_like(lin.data)
            return lin, pws

        # remove traces with NaN data
        # data = data[~np.any(np.isnan(data), axis=1)]

        # normalize data or not
        if normalize:
            maxs = np.max(np.abs(data), axis=1)
            data = data / maxs[:, None]

        # make structures for stacks
        linear_stack = np.nanmean(data, axis=0)
        phas = np.zeros_like(data)

        # loop over each trace and get instantaneous phase of each sample
        for tr_idx in range(np.shape(phas)[0]):
            # hilbert transform of each time series
            tmp = hilbert(data[tr_idx, :])
            # get instantaneous phase using the hilbert transform
            phas[tr_idx, :] = np.arctan2(np.imag(tmp), np.real(tmp))

        # sum of phases for each sample across traces
        sump = np.abs(np.sum(np.exp(np.complex(0, 1) * phas), axis=0)) / \
               np.shape(phas)[0]

        # pws = traditional stack * phase weighting stack
        phase_weighted_stack = sump * linear_stack

        lin = stream[reference_idx].copy()
        lin.data = linear_stack
        pws = stream[reference_idx].copy()
        pws.data = phase_weighted_stack

        # generate an animation of the stack if specified
        if animate:
            # # check animation writers
            # import matplotlib.animation as manimation
            # manimation.writers.list()

            # initialize figure and camera to show trace and stack animation
            fig, axes = plt.subplots(2, figsize=(7,4))
            camera = Camera(fig)

            data_len = len(data)
            trace_len = data.shape[1]
            # get x values from trace shape and sampling rate
            x_vals = np.linspace(0, trace_len, trace_len, endpoint=False)
            x_vals = x_vals / stream[0].stats.sampling_rate

            # get y limits before plotting
            y_min = -1 if normalize else np.mean(data.min(axis=1))
            y_max = 1 if normalize else np.mean(data.max(axis=1))
            for trace_idx, stack_trace in enumerate(data):
                axes[0].plot(x_vals, stack_trace, color='blue', linewidth=1)
                axes[0].set_ylim(bottom=y_min, top=y_max)
                axes[0].set_xlim(0, ceil(x_vals.max()))
                axes[1].plot(x_vals, data[:trace_idx+1].mean(axis=0),
                                     color='blue', linewidth=1)
                axes[1].text(1, lin.data.max(), f"{trace_idx+1}/{data_len}")
                axes[1].set_ylim(bottom=lin.data.min() * 1.25,
                                 top=lin.data.max() * 1.25)
                axes[1].set_xlim(0, ceil(x_vals.max()))
                axes[1].set_xlabel(f"Time (seconds)")
                camera.snap()

            animation = camera.animate()
            animation.save(f'./stack_gifs/{stream[0].stats.network}'
                           f'.{stream[0].stats.station}.'
                           f'{stream[0].stats.channel}.gif',
                           writer='imagemagick')

        return lin, pws

    def load_trace(streams_path, pick_time, ref_ID):
        st = Stream()
        network = ref_ID.split('.')[0]
        station = ref_ID.split('.')[1]
        channel = ref_ID.split('.')[2]

        # find the local file corresponding to the station:channel pair
        day_file_list = glob.glob(f"{streams_path}/{network}."
                                  f"{station}."
                                  f"{channel}.{pick_time[0].year}"
                                  f"-{pick_time[0].month:02}"
                                  f"-{pick_time[0].day:02}.ms")

        # guard against missing files
        if len(day_file_list) > 0:
            # should only be one file, but safeguard against many
            file = day_file_list[0]
            # load day file into stream
            day_st = read(file)
            # interpolate to 100 Hz
            day_st.interpolate(sampling_rate=100.0)
            # bandpass filter
            day_st.filter('bandpass', freqmin=1, freqmax=15)
            # trim trace to 60 seconds surrounding pick time
            day_st.trim(pick_time[0], pick_time[1], pad=True,
                        fill_value=np.nan, nearest_sample=True)

            st += day_st

        return st[0]

    # helper function to determine time offset of each time series in a
    # stream with respect to a main trace via cross correlation
    def xcorr_time_shifts(stream, reference_signal, template_times,
                          streams_path):
        # get index of first non-empty trace
        for index, trace in enumerate(stream):
            if len(trace) > 0:
                ID_idx = index
                break
        # get ID of stream
        ID = f"{stream[ID_idx].stats.network}.{stream[ID_idx].stats.station}."\
             f"{stream[ID_idx].stats.channel}"

        # TODO: rename "maxs" to targets
        shifts = []
        indices = []
        ccs = []

        # find reference index with strongest signal or median snr, this serves
        # as a template
        max_snr = 0
        snrs = []
        for index, trace in enumerate(stream):
            if len(snr(trace)) > 0:
                if snr(trace)[0] > 0:
                    snrs.append(snr(trace)[0])
                else:
                    snrs.append(np.nan)
            else:
                snrs.append(np.nan)

        # define target signal as max, median, self, or similar
        reference_idx = -1 # flag to check for unset reference_idx
        if reference_signal == "max":
            reference_idx = np.nanargmax(snrs)
        elif reference_signal == "med":
            median_snr = np.nanmedian(snrs)
            # find index of SNR closest to median
            reference_idx = np.nanargmin(np.abs(snrs - median_snr))
        elif reference_signal == "self":
            # find index that corresponds with template event time
            if ID in template_times:
                for tr_idx, trace in enumerate(stream):
                    if trace.stats.starttime < template_times[ID][0] and \
                            trace.stats.endtime > template_times[ID][1]:
                        reference_idx = tr_idx
                        break

        # check for unset reference_idx, then set reference signal to a
        # signal on another station
        if reference_idx == -1:
            # if template event time wasn't found within stream, use a similar
            # signal. The first choice is a template signal on a different
            # station in the same network on the same line, and on the same
            # channel. The second choice is a template signal on a different
            # station in the same network on the same channel. The third
            # choice is a template signal on a different station in the same
            # network.
            ID_list = list(template_times.keys())
            same_network_ID_list = []
            same_channel_ID_list = []
            same_line_ID_list = []
            for item in ID_list:
                # check for common network
                if item.split('.')[0] == ID.split('.')[0]:
                    same_network_ID_list.append(item)
                # check for common channel and network
                if item.split('.')[0] == ID.split('.')[0] and item.split(\
                    '.')[2] == ID.split('.')[2]:
                    same_channel_ID_list.append(item)
                # check for common station line by first two characters of
                # station name and channel code
                if item.split('.')[1][:2] == ID.split('.')[1][:2] and \
                   item.split('.')[2] == ID.split('.')[2]:
                    same_line_ID_list.append(item)

            # check for most ideal condition to least ideal condition
            if len(same_line_ID_list) > 0:
                # if only one candidate, use it
                if len(same_line_ID_list) == 1:
                    ref_ID = same_line_ID_list[0]
                    pick_time = template_times[ref_ID]
                    reference_trace = load_trace(streams_path, pick_time, ref_ID)
                # if many candidates, select the closest one
                else:
                    # first decypher naming scheme to find digits for "closeness"
                    digit_start_idx = -1
                    for index, char in same_line_ID_list[0]:
                        if char.isdigit():
                            digit_start_idx = index
                            break
                    # if there are no digits, blindly choose a station
                    if digit_start_idx == -1:
                        ref_ID = same_line_ID_list[0]
                        pick_time = template_times[ref_ID]
                        reference_trace = load_trace(streams_path, pick_time,
                                                     ref_ID)
                    # otherwise, find the closest digit for the closest station
                    else:
                        # store distances
                        digit_dist = []
                        for entry in same_line_ID_list:
                            entry_digits = int(entry.split('.')[1][
                                           digit_start_idx:])
                            station_digits = int(ID.split('.')[1][
                                             digit_start_idx:])
                            digit_dist.append(abs(station_digits -
                                                  entry_digits))
                        # find minimum distance index and use that one
                        min_dist_idx = digit_dist.index(min(digit_dist))
                        ref_ID = same_line_ID_list[min_dist_idx]
                        pick_time = template_times[ref_ID]
                        reference_trace = load_trace(streams_path, pick_time,
                                                     ref_ID)

            # check for 2nd best condition, signal from same network and
            # common channel code
            elif len(same_channel_ID_list) > 0:
                # select candidate
                ref_ID = same_channel_ID_list[0]
                pick_time = template_times[ref_ID]
                reference_trace = load_trace(streams_path, pick_time, ref_ID)

            # check for 3nd best condition, signal from same network
            elif len(same_network_ID_list) > 0:
                # select candidate
                ref_ID = same_channel_ID_list[0]
                pick_time = template_times[ref_ID]
                reference_trace = load_trace(streams_path, pick_time, ref_ID)

            # else error
            else:
                print(f"ERROR: Things are about to break because there is no "
                      "suitable reference signal for {ID}")

        else:
            trace = stream[reference_idx]
            ref_snr = snrs[reference_idx]
            # trim the reference trace to the template length
            reference_trace = trace.copy().trim(template_times[ID][0],
                                                template_times[ID][1])

            # check centering of signals
            st = Stream()
            st += stream[0]
            st += reference_trace
            st0_middle = st[0].stats.starttime + ((st[0].stats.npts / 2) / st[
                                                        0].stats.sampling_rate)
            st1_middle = st[1].stats.starttime + ((st[1].stats.npts / 2) / st[
                1].stats.sampling_rate)
            difference = st1_middle - st0_middle



            # print a warning if SNR is bad
            if ref_snr == 0:
                print("- ! - ! - ! - ! - ! - ! -")
                print("- ! - ! - ! - ! - ! - ! -")
                print("- ! - ! - ! - ! - ! - ! -")
                print("ERROR: reference SNR is 0")
                print("- ! - ! - ! - ! - ! - ! -")
                print("- ! - ! - ! - ! - ! - ! -")
                print("- ! - ! - ! - ! - ! - ! -")
            # else:
            #     print(f"SNR in reference trace: {ref_snr}")

        # guard against empty trace
        if len(trace) > 0:

            # loop through each trace and get cross-correlation time delay
            for st_idx, trace in enumerate(stream):

                # length of trace must be greater than template for xcorr
                if len(trace.data) > len(reference_trace.data):
                    # # correlate the reference trace through the trace
                    # cc = correlate_template(trace, reference_trace, mode='valid',
                    #                         normalize='naive', demean=True,
                    #                         method='auto')
                    # # find the index with the max correlation coefficient
                    # max_idx = np.argmax(cc)

                    # # to visualize a trace, the template, and the max correlation
                    # stt = Stream()
                    # stt += trace # the trace
                    # # the section of the trace where max correlation coef. starts
                    # stt += trace.copy().trim(trace.stats.starttime + (max_idx /
                    #                          trace.stats.sampling_rate),
                    #                          trace.stats.endtime)
                    # # the template aligned with the max correlation section
                    # stt += reference_trace.copy()
                    # stt[2].stats.starttime = stt[1].stats.starttime
                    # stt.plot()

                    # correlate the reference trace through the trace
                    # FIXME: check centering
                    max_shift = 50 # maximum xcorr shift in samples
                    cc = correlate(trace, reference_trace, max_shift,
                                   demean=True, normalize='naive',
                                   method='auto')
                    # find the index with the max correlation coefficient
                    max_idx = np.argmax(cc) - max_shift # + (2 * max_shift)
                    ccs.append(cc.max())

                    # keep track of negative correlation coefficients
                    if cc.max() < 0:
                        indices.append(st_idx)

                    # append the cross correlation time shift for this trace
                    # referenced from trace.stats.starttime
                    shifts.append((max_idx / trace.stats.sampling_rate) + 20)

                else:
                    # keep track of bad traces
                    indices.append(st_idx)
                    # append a zero shift
                    shifts.append(0)

        # case for bad data
        else:
            shifts = None
            indices = None

        return shifts, indices, ccs

    # helper function to align all traces in a stream based on xcorr shifts
    # from the main_trace for each pick time. Stream is altered in place and
    # each trace is trimmed to be 40 seconds long.
    def align_stream(stream, shifts, indices):
        # first check if data are bad, if so zero shift instead
        if shifts == None:
            zero_shift_stream(stream)

        # data are good, so process them
        else:
            # shift each trace of stream in place to avoid memory issues
            for tr_idx, tr in enumerate(stream):
                # create false starttime to shift around
                tr.stats.starttime = UTCDateTime("2016-01-01T12:00:00.0Z") - \
                                     shifts[tr_idx]

            # consider 40 seconds total for stacks
            new_start_time = UTCDateTime("2016-01-01T12:00:00.0Z") - 10
            new_end_time = new_start_time + 40
            # all traces need to be same length for further processing
            stream.trim(new_start_time, new_end_time, pad=True,
                        fill_value=np.nan, nearest_sample=True)

        # remove bad indices
        if indices != None and len(indices) > 0:
            removal_list = []
            for tr_idx in indices:
                removal_list.append(stream[tr_idx])
            for trace in removal_list:
                stream.remove(trace)

        # remove traces with no data
        removal_list = []
        for trace in stream:
            if trace.stats.npts == 0:
                removal_list.append(trace)
        for trace in removal_list:
            stream.remove(trace)

        return None

    # align the start time of each trace in the stream to the same
    # arbitrary time
    def zero_shift_stream(stream):
        # shift each trace of stream in place to avoid memory issues
        for tr_idx, tr in enumerate(stream):
            # create false starttime to shift around
            tr.stats.starttime = UTCDateTime("2016-01-01T00:00:00.0Z")

        new_start_time = UTCDateTime("2016-01-01T00:00:00.0Z") + 15
        new_end_time = new_start_time + 40
        stream.trim(new_start_time, new_end_time, pad=True, fill_value=np.nan,
                    nearest_sample=True)

        return None

    # first extract pick times for each event from party object
    # pick_times is a list of the pick times for the main trace
    pick_times = []
    pick_network, pick_station, pick_channel = main_trace
    for event in party.families[0].catalog.events:
        for pick in event.picks:
            # trace with highest amplitude signal
            if pick.waveform_id.station_code == pick_station and \
                    pick.waveform_id.network_code == pick_network and \
                    pick.waveform_id.channel_code == pick_channel:
                pick_times.append(pick.time)

    # build dict of stations from contents of specified directory
    file_list = glob.glob(f"{streams_path}/*.ms")
    station_dict = {}
    for file in file_list:
        filename = file.split("/")[-1]
        file_station = filename.split(".")[1]
        if file_station not in station_dict:
            file_network = filename.split(".")[0]

            # # for 1 component
            # file_channel = filename.split(".")[2][:-1] + "N" # force N
            # for 3 components
            file_channel = filename.split(".")[2]

            # component
            station_dict[file_station] = {"network": file_network,
                                          "channel": file_channel}

    if align_type == 'fixed':
        # get the main trace detections in a stream
        print(f"Assembling main stream {pick_network}.{pick_station}"
              f".{pick_channel}")
        main_stream, _ = build_main_stream(main_trace, streams_path,
                                           pick_times)
        # get the fixed location time shifts from the main trace
        shifts, indices, ccs = xcorr_time_shifts(main_stream, 'self',
                                            template_times, streams_path)

        # free up memory
        del main_stream

    # loop over stations and generate a stack for each station:channel pair
    stack_pw = Stream()
    stack_lin = Stream()
    stations = list(station_dict.keys())
    stack_ccs = []
    # keep track of missing files
    # missing_files = open("missing_files", "w")
    for station_idx, station in enumerate(stations):
        network = station_dict[station]["network"]
        channels = []

        # # for 1 component
        # channels.append(station_dict[station]["channel"])  # append main component

        # for 3 components
        # force Z component
        channels.append(station_dict[station]["channel"][:-1] + "Z")
        channels.append(f"{channels[0][:-1]}N")  # append N component
        channels.append(f"{channels[0][:-1]}E")  # append E component

        for channel in channels:
            print(f"Assembling streams for {station}.{channel} ["
                  f"{station_idx+1}/{len(stations)}]")

            sta_chan_stream = Stream()
            for index in tqdm(range(len(pick_times))):
                pick_time = pick_times[index]

                # find the local file corresponding to the station:channel pair
                day_file_list = glob.glob(f"{streams_path}/{network}."
                                          f"{station}."
                                          f"{channel}.{pick_time.year}"
                                          f"-{pick_time.month:02}"
                                          f"-{pick_time.day:02}.ms")

                # guard against missing files
                if len(day_file_list) > 0:
                    # should only be one file, but safeguard against many
                    file = day_file_list[0]
                    # load day file into stream
                    day_st = read(file)
                    # interpolate to 100 Hz
                    day_st.interpolate(sampling_rate=100.0)
                    # bandpass filter
                    day_st.filter('bandpass', freqmin=1, freqmax=15)
                    # trim trace to 60 seconds surrounding pick time
                    day_st.trim(pick_time - 20, pick_time + 40, pad=True,
                                fill_value=np.nan, nearest_sample=True)

                    sta_chan_stream += day_st

                # if no file, append blank trace to preserve stream length
                # equality with pick_times
                else:
                    sta_chan_stream += Trace()
                    # missing_files.write(f"No file for {network}.{station}."
                    #                     f"{channel}.{pick_time.year}-"
                    #                     f"{pick_time.month:02}-"
                    #                     f" {pick_time.day:02}\n")

            # guard against empty stream
            if len(sta_chan_stream) > 0:
                # sta_chan_stream.write(f"{network}.{station}.{channel}.ms",
                #                      format="MSEED")

                # # FIXME: delete after testing
                # # Temp to plot traces and their shifts
                # st = Stream()
                # for tr_idx in range(0, 10):
                #     tr = sta_chan_stream[tr_idx].copy()
                #     tr.stats.starttime = UTCDateTime("2016-01-01T12:00:00.0Z")
                #     st += tr
                #     tr2 = sta_chan_stream[tr_idx].copy()
                #     tr2.stats.starttime = UTCDateTime(
                #         "2016-01-01T12:00:00.0Z") - shifts[tr_idx]
                #     tr2.trim(UTCDateTime("2016-01-01T12:00:00.0Z"),
                #             UTCDateTime("2016-01-01T12:01:00.0Z"), pad=True,
                #                 fill_value=0, nearest_sample=True)
                #     st += tr2
                # plot_stack(st)
                # # FIXME: delete above after testing

                # guard against stream of empty traces
                EMPTY_FLAG = True
                for trace in sta_chan_stream:
                    if len(trace) > 0:
                        EMPTY_FLAG = False

                # dont consider empty traces (case of no data present)
                if not EMPTY_FLAG:

                    # process according to specified method and alignment
                    if align_type == 'zero':
                        # align the start time of each trace in stream
                        zero_shift_stream(sta_chan_stream)
                    elif align_type == 'med' or align_type == 'max' or \
                            align_type == 'self':
                        # get xcorr time shift from reference signal
                        shifts, indices, ccs = xcorr_time_shifts(
                                                            sta_chan_stream,
                                                            align_type,
                                                            template_times,
                                                            streams_path)
                        # append cross-correlation coefficients to list
                        stack_ccs.append(ccs)

                        # align stream traces from time shifts
                        align_stream(sta_chan_stream, shifts, indices)
                    else:
                        # align stream traces from fixed location time shifts
                        align_stream(sta_chan_stream, shifts, indices)
                        # append cross-correlation coefficients to list
                        stack_ccs.append(ccs)

                    # plot aligned stream to verify align function works
                    # plot_stream_absolute(sta_chan_stream[:100])
                    # plot_stream_relative(aligned_sta_chan_stream)

                    # guard against stacking error:
                    try:
                        # # check if the right animation writers exist
                        # import matplotlib.animation as manimation
                        # manimation.writers.list()

                        # generate linear and phase-weighted stack
                        lin, pws = generate_stacks(sta_chan_stream,
                                                   normalize=True,
                                                   animate=animate_stacks)

                        # add phase-weighted stack to stream
                        stack_pw += pws
                        # and add linear stack to stream
                        stack_lin += lin

                        # # fill nan's with zeros so plot_stack doesn't go boom
                        # sta_chan_stream.trim(sta_chan_stream[0].stats.starttime,
                        #                      sta_chan_stream[0].stats.endtime,
                        #                      pad=True, fill_value=0,
                        #                      nearest_sample=True)
                        # # get rid of short traces for plotting
                        # trace_pop_list = []
                        # for tr in sta_chan_stream:
                        #     if tr.stats.npts < 4001:
                        #         trace_pop_list.append(tr)
                        # for tr in trace_pop_list:
                        #     sta_chan_stream.remove(tr)
                        # # inspect the stack contents
                        # plot_stack(sta_chan_stream, title=f'top_{n}_{station}'
                        #            f't5_stack_contents_7s_100Hz_0.5prepick',
                        #            save=True)

                    except Exception:
                        pass

    # missing_files.close()

    # # if the stacks exist, plot them and don't bandpass filter from 1-15 Hz
    # if len(stack_pw) > 0:
    #     plot_stack(stack_pw, title='phase_weighted_stack', save=True)
    #
    # if len(stack_lin) > 0:
    #     plot_stack(stack_lin, title='linear_stack', save=True)

    return [stack_pw, stack_lin, stack_ccs]


def detections_from_stacks(stack, detection_files_path, start_date,
                           end_date, main_trace):
    """ Transform stacks so they can be used as templates for matched-filter
    analysis via EQcorrscan, FIXME, then finds detections corresponding to
    stacks.

    Returns:

    Example:

    """

    # FIXME: stack should be denormalized for this? check if EQcorrscan
    #  normalizes before xcorr

    # build stream of day-long data from stack and compile picks list
    st = Stream()
    picks = []
    add_noise = False
    for index, trace in enumerate(stack):
        if add_noise:
            # take first 10 seconds of trace data as noise
            tr_data = trace.data[:int(10 * trace.stats.sampling_rate)]
            # generate a gaussian distribution and extract values from12 it to
            # generate synthetic noise to fill day-long stream
            tr_mean = tr_data.mean()
            tr_stdev = tr_data.std()
            # extend the stack to be a day long
            st += trace.trim(UTCDateTime("2016-01-01T00:00:00.0Z"), UTCDateTime(
                             "2016-01-01T23:59:59.99999999999Z"), pad=True,
                             fill_value=0, nearest_sample=True)

            # fill the data before the stack
            samples = int(43180 * trace.stats.sampling_rate)
            tr_data = np.random.normal(tr_mean, tr_stdev, samples)
            st[index].data[:samples] = tr_data
            # then fill the data after the stack
            tr_data = np.random.normal(tr_mean, tr_stdev, samples)
            st[index].data[-1 * samples:] = tr_data

            # set process length for EQcorrscan
            process_len = 86400
        else:
            st += trace
            process_len = 40

        # # for testing
        # new_start_time = UTCDateTime("2016-01-01T12:00:00.0Z") - 5
        # new_end_time = new_start_time + 6
        # # all traces need to be same length for further processing
        # st2 = st.copy()
        # st2.trim(new_start_time, new_end_time, pad=True,
        #             fill_value=np.nan, nearest_sample=True)
        # st2.plot()

        # t4: 2016  9 27  6 31 15.00
        if trace.stats.station == "RH08":
            picks.append(Pick(time=UTCDateTime(2016, 1, 1, 0, 0, 27, 0),
                              phase_hint="P", waveform_id=WaveformStreamID(
                    network_code=trace.stats.network,
                    station_code=trace.stats.station,
                    channel_code=trace.stats.channel)))
        elif trace.stats.station == "RH10" or trace.stats.station == "MCR1" \
                or trace.stats.station == "RH09":
            picks.append(Pick(time=UTCDateTime(2016, 1, 1, 0, 0, 26, 0),
                              phase_hint="P", waveform_id=WaveformStreamID(
                    network_code=trace.stats.network,
                    station_code=trace.stats.station,
                    channel_code=trace.stats.channel)))
        else:
            picks.append(Pick(time=UTCDateTime(2016, 1, 1, 0, 0, 22, 0),
                              phase_hint="P", waveform_id=WaveformStreamID(
                              network_code=trace.stats.network,
                              station_code=trace.stats.station,
                              channel_code=trace.stats.channel)))

    # build catalog object from picks list with made up origin and magnitude
    event = Event(origins=[Origin(latitude=61.9833, longitude=-144.0437,
                        depth=1700, time=UTCDateTime(2016, 1, 1, 0, 0, 22))],
                  magnitudes=[Magnitude(mag=1.1)], picks=picks)
    catalog = Catalog([event])

    # construct the EQcorrscan tribe object using my construct_stack method
    stack_template = Tribe().construct(method="from_meta_file",
                                       meta_file=catalog, st=st, lowcut=1.0,
                                       highcut=15.0, samp_rate=100.0,
                                       length=12.0, filt_order=4, prepick=2.0,
                                       swin='all', process_len=process_len,
                                       parallel=True, skip_short_chans=False)

    # loop over days and get detections
    iteration_date = start_date
    party_list = []
    while iteration_date < end_date:
        print(iteration_date)  # print date for long runs
        # build stream of all files on day of interest
        day_file_list = glob.glob(f"{detection_files_path}/*."
                                  f"{iteration_date.year}"
                                  f"-{iteration_date.month:02}"
                                  f"-{iteration_date.day:02}.ms")

        # load files into stream
        st = Stream()
        for file in day_file_list:
            st += read(file)

        try:
            # detect
            party = stack_template.detect(stream=st, threshold=8.0,
                                          daylong=True, threshold_type="MAD",
                                          trig_int=8.0, plot=False,
                                          return_stream=False,
                                          parallel_process=False,
                                          ignore_bad_data=True)
        except Exception:
            party = Party(families=[Family(template=Template())])
            pass

        # append detections to party list if there are detections
        if len(party.families[0]) > 0:
            party_list.append(party)

        # update the next file start date
        iteration_year = iteration_date.year
        iteration_month = iteration_date.month
        iteration_day = iteration_date.day
        # get number of days in current month
        month_end_day = calendar.monthrange(iteration_year,
                                            iteration_month)[1]
        # check if year should increment
        if iteration_month == 12 and iteration_day == month_end_day:
            iteration_year += 1
            iteration_month = 1
            iteration_day = 1
        # check if month should increment
        elif iteration_day == month_end_day:
            iteration_month += 1
            iteration_day = 1
        # else only increment day
        else:
            iteration_day += 1

        iteration_date = UTCDateTime(f"{iteration_year}-"
                                     f"{iteration_month:02}-"
                                     f"{iteration_day:02}T00:00:00.0Z")

    # add all detections to a single party
    if len(party_list) > 1:
        for index, iteration_party in enumerate(party_list):
            # skip first party
            if index > 0:
                # add events of current party family to first party family
                party_list[0].families[0] = party_list[0].families[0].append(
                    iteration_party.families[0])

    # extract and return the first party or None if no party object
    if len(party_list) > 0:
        party = party_list[0]

        # save party to pickle
        filename = f'party_{start_date.month:02}_{start_date.day:02}_' \
                   f'{start_date.year}_to_{end_date.month:02}' \
                   f'_{end_date.day:02}_' \
                   f'{end_date.year}_MAD8_14s_stackDetects.pkl'
        outfile = open(filename, 'wb')
        pickle.dump(party, outfile)
        outfile.close()

        return party

    else:
        return None


def inspect_template(template_date, main_trace, streams_path, filter):
    """
    Generates figures to visualize a template event and writes a file
    containing the template time series.

    Input:
        template_date: UTCDateTime object

    Output:
        returns: None, and writes files

    Example:
        # define start time of template
        # template_date = UTCDateTime("2016-09-26T09:28:41.34Z")
        # template_date = UTCDateTime("2016-09-27T06:31:15.00000Z")
        # template_date = UTCDateTime("2016-09-27T06:31:32.00000Z")
        # template_date = UTCDateTime("2017-12-01T11:51:00.00000Z")
        template_date = UTCDateTime("2016-09-26T09:25:48.50000Z")

        # define the main trace to use for template
        main_trace = ("AV", "WASW", "SHZ")

        # define path of files for template and detections
        streams_path = "/Users/human/ak_data/inner"
        # streams_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"

        # generate the figures and time series file
        inspect_template(template_date, main_trace, streams_path, filter=True)
    """

    # time offsets
    max_offset = 60*2  # 43200
    min_offset = 30

    # find the local file corresponding to the main trace station:channel pair
    file_list = glob.glob(f"{streams_path}/{main_trace[0]}."
                          f"{main_trace[1]}."
                          f"{main_trace[2]}.{template_date.year}"
                          f"-{template_date.month:02}"
                          f"-{template_date.day:02}.ms")

    # load time series into stream
    if len(file_list) > 0: # guard against missing files
        # should only be one file, but safeguard against many
        file = file_list[0]
        # load file into stream
        st = read(file)
        # bandpass filter
        if filter:
            st.filter('bandpass', freqmin=1, freqmax=15)
        # interpolate to 100 Hz
        st.interpolate(sampling_rate=100.0)
        # trim trace to 60 seconds surrounding pick time
        st.trim(template_date - max_offset, template_date + max_offset,
                pad=True,
                fill_value=np.nan, nearest_sample=True)

        st_min = st.copy().trim(template_date - min_offset, template_date +
                                 min_offset,
                                 pad=True, fill_value=np.nan,
                                 nearest_sample=True)

    else:
        return False

    # plot the time series
    # st.plot()
    # st_min.plot()

    time_markers = {"AV.WASW": [UTCDateTime("2016-09-26T09:25:46.0Z"),
                                UTCDateTime("2016-09-26T09:25:58.0Z")],
                    "TA.N25K": [UTCDateTime("2016-09-26T09:25:47.5Z"),
                                UTCDateTime("2016-09-26T09:25:59.5Z")],
                    "YG.MCR2": [UTCDateTime("2016-09-26T09:25:46.5Z"),
                                UTCDateTime("2016-09-26T09:25:58.5Z")],
                    "YG.MCR1": [UTCDateTime("2016-09-26T09:25:48.5Z"),
                                UTCDateTime("2016-09-26T09:26:00.5Z")],
                    "YG.RH09": [UTCDateTime("2016-09-26T09:25:55.0Z"),
                                UTCDateTime("2016-09-26T09:26:07.0Z")],
                    "YG.RH10": [UTCDateTime("2016-09-26T09:25:55.0Z"),
                                UTCDateTime("2016-09-26T09:26:07.0Z")],
                    "YG.RH08": [UTCDateTime("2016-09-26T09:25:55.0Z"),
                                UTCDateTime("2016-09-26T09:26:07.0Z")],
                    "AV.WACK": [UTCDateTime("2016-09-26T09:25:47.5Z"),
                                UTCDateTime("2016-09-26T09:25:59.5Z")],
                    "AK.GLB": [UTCDateTime("2016-09-26T09:25:48.5Z"),
                                UTCDateTime("2016-09-26T09:26:00.5Z")],
                    }

    # plot spectrograms and time series together
    fig_max = plot_Time_Series_And_Spectrogram(template_date - max_offset,
                                               template_date + max_offset,
                                               streams_path, filter=filter,
                                               bandpass=[1, 15],
                                               time_markers=time_markers)
    fig_min = plot_Time_Series_And_Spectrogram(template_date - min_offset,
                                               template_date + min_offset,
                                               streams_path, filter=filter,
                                               bandpass=[1, 15],
                                               time_markers=time_markers)

    # save template stream to file
    write = False
    if write:
        st_copy = st.copy()
        st_copy.trim(template_date - 15, template_date + 15, pad=True,
                     fill_value=np.nan, nearest_sample=True)
        st_copy.write(f"template_16.5_seconds_bandpass_1-15.ms",
                      format="MSEED")
        st_copy.plot()

        st_copy = st.copy()
        st_copy.trim(template_date - 150, template_date + 150, pad=True,
                     fill_value=np.nan, nearest_sample=True)
        st_copy.write(f"template_1_minute_bandpass_1-15.ms", format="MSEED")
        st_copy.plot()

        st_copy = st.copy()
        st_copy.write(f"template_whole_day_bandpass_1-15.ms", format="MSEED")
        st_copy.plot()

    # put all this in a shareable folder for Aaron
        # spectrograms
        # template

    return True

# function to calculate snrs of traces that correspond to party detections
def party_snrs(party, streams_path, main_trace):
    """ Takes in a party object containing detections and calculates the SNR
    (signal to noise ratio) of the trace associated with the detection station
    at the detection time.
    """
    # first extract pick times for each event from party object
    # pick_times is a list of the pick times for the main trace
    pick_times = []
    pick_network, pick_station, pick_channel = main_trace
    for event in party.families[0].catalog.events:
        PICK_APPENDED = False
        for pick in event.picks:
            # trace with highest amplitude signal
            if pick.waveform_id.station_code == pick_station and \
                    pick.waveform_id.network_code == pick_network and \
                    pick.waveform_id.channel_code == pick_channel:
                pick_times.append(pick.time)
                PICK_APPENDED = True

        # if no pick time has been assigned, assign a duplicate to avoid error
        if not PICK_APPENDED:
            pick_times.append(pick.time)

    snrs = []
    for index in range(len(pick_times)):
        pick_time = pick_times[index]

        # find the local file corresponding to the station:channel pair
        day_file_list = glob.glob(f"{streams_path}/{pick_network}."
                                  f"{pick_station}."
                                  f"{pick_channel}.{pick_time.year}"
                                  f"-{pick_time.month:02}"
                                  f"-{pick_time.day:02}.ms")

        # guard against missing files
        if len(day_file_list) > 0:
            # should only be one file, but safeguard against many
            file = day_file_list[0]
            # load day file into stream
            day_st = read(file)
            # bandpass filter
            day_st.filter('bandpass', freqmin=1, freqmax=15)
            # interpolate to 100 Hz
            day_st.interpolate(sampling_rate=100.0)
            # trim trace to 60 seconds surrounding pick time
            day_st.trim(pick_time - 20, pick_time + 40, pad=True,
                        fill_value=np.nan, nearest_sample=True)

            snrs.append(snr(day_st)[0])

        # if no file, append blank trace to preserve stream length
        # equality with pick_times
        else:
            snrs.append(np.nan)

    return snrs

# driver function to find LFEs from a template and time series files
def find_LFEs(templates, template_files, station_dict, template_length,
              template_prepick, detection_files_path, start_date, end_date,
              snr_threshold, main_trace, detect_thresh,
              thresh_type, shift_method='self', load_party=False,
              cull=False, load_stack=False, load_stack_detects=False,
              load_second_stack=False, plot=False):
    """
    Driver function to detect signals (LFEs in this case) in time series via
    matched-filtering with specified template(s), then stacking of signals
    found from that template, and finally template matching of the stacked
    waveform template.

    Inputs:

        shift_method: string identifying the cross-correlation time shift
                      method to use. 'zero', 'med', 'max', and 'fixed' are
                      options.

    Example:
        # manually define templates from station (location is made up)
        # templates = ["# 2016  9 26  9 28 41.34  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
        #              "N25K    0.000  1       P\n"]
        # templates = ["# 2016  9 26  9 28 41.34  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
        #              "MCR1    0.000  1       P\n"]

        # T4
        # templates = ["# 2016  9 27  6 31 15.00  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
        #              "N25K    0.000  1       P\n"]
        # templates = ["# 2016  9 27  6 31 15.00  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
        #              "MCR1    0.000  1       P\n"]
        # # define template length and prepick length (both in seconds)
        # template_length = 7.0
        # template_prepick = 0.0

        # # T5 - Aaron's template on WASW
        # templates = ["# 2016  9 26  9 25 48.50  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
        #              "WASW    0.000  1       P\n"]
        # template_length = 7.0
        # template_prepick = 0.5

        # T6 - Aaron's WASW template + TA.N25K + YG.MCR2
        # templates = ["# 2016  9 26  9 25 48.00  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
        #              "WASW    0.000  1       P\n",
        #              "N25K    4.500  1       P\n",
        #              "MCR2    0.500  1       P\n"]
        # templates = ["# 2016  9 26  9 25 48.00  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
        #                      "WASW    0.500  1       P\n"]
        # template_length = 7.0
        # template_prepick = 0.5
        templates = ["# 2016  9 26  9 25 48.00  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
                     "WASW    0.000  1       P\n",
                     "N25K    1.500  1       P\n",
                     "MCR2    0.500  1       P\n",
                     "MCR1    2.500  1       P\n",
                     "RH09    9.000  1       P\n",
                     "RH10    9.000  1       P\n",
                     "RH08    9.000  1       P\n",
                     "WACK    1.500  1       P\n",
                     "GLB     2.500  1       P\n"]
        template_length = 12.0
        template_prepick = 2.0


        # st = sta_chan_stream[195].copy()
        # st.filter('bandpass', freqmin=1, freqmax=15)
        # st.trim(UTCDateTime("2016-09-27T07:37:49.0Z"), UTCDateTime(
        #                      "2016-09-27T07:38:05.0Z"), pad=True,
        #                      fill_value=0, nearest_sample=True)
        # st.plot()

        # and define a station dict to add data needed by EQcorrscan
        # station_dict = {"WASW": {"network": "AV", "channel": "SHZ"}}
        # station_dict = {"N25K": {"network": "TA", "channel": "BHZ"}}
        # station_dict = {"MCR2": {"network": "YG", "channel": "BHZ"}}
        station_dict = {"WASW": {"network": "AV", "channel": "SHN"},
                        "N25K": {"network": "TA", "channel": "BHN"},
                        "MCR2": {"network": "YG", "channel": "BHN"},
                        "MCR1": {"network": "YG", "channel": "BHN"},
                        "RH09": {"network": "YG", "channel": "BHN"},
                        "RH08": {"network": "YG", "channel": "BHN"},
                        "RH10": {"network": "YG", "channel": "BHN"},
                        "WACK": {"network": "AV", "channel": "BHN"},
                        "GLB": {"network": "AK", "channel": "BHN"}}


        # build stream of all station files for templates
        # files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/WASW"
        # files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/N25K"
        # files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/MCR2"
        files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"
        template_files = glob.glob(f"{files_path}/*.ms")

        # define path of files for detection: TA.N25K, YG.MCR2, YG.MCR1, YG.RH09, AV.WASW
        detection_files_path = "/Users/human/ak_data/inner"

        # define dates of interest
        # start_date = UTCDateTime("2016-06-15T00:00:00.0Z")
        # RH08 2018-06-04
        end_date = UTCDateTime("2018-08-11T23:59:59.9999999999999Z")
        # when using WASW, start date is 07-18
        start_date = UTCDateTime("2016-07-18T00:00:00.0Z")

        # set snr threshold to cull the party detections
        # snr_threshold = [1.0, 25.0] # [1.0, 8.0]
        # snr_threshold = [1.0, 9.0]
        snr_threshold = [1.0, 15.0]
        # set detection threshold and type
        detect_thresh = 8.0
        thresh_type = "MAD"
        # detect_thresh = 0.65
        # thresh_type = "abs"

        # define the main trace to use for detections (best amplitude station)
        main_trace = ("AV", "WASW", "SHZ")
        # main_trace = ("TA", "N25K", "BHN")
        # main_trace = ("YG", "MCR2", "BHN")

        # run detection and time it
        start = time.time()
        # --------------------------------------------------------------------
        party = find_LFEs(templates, template_files, station_dict,
                          template_length, template_prepick,
                          detection_files_path, start_date, end_date,
                          snr_threshold, main_trace,
                          detect_thresh, thresh_type, shift_method='zero',
                          load_party=True, save_detections=False, cull=False,
                          load_stack=True, load_stack_detects=True,
                          load_second_stack=True, plot=True)
        # --------------------------------------------------------------------
        end = time.time()
        hours = int((end - start) / 60 / 60)
        minutes = int(((end - start) / 60) - (hours * 60))
        seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
        print(f"Runtime: {hours} h {minutes} m {seconds} s")
    """
    # # FIXME: delete after testing
    shift_method = 'zero'
    load_party = True
    save_detections = True

    top_n = True
    n = 100

    load_stack = False
    load_stack_detects = False
    load_second_stack = False
    cull = False
    plot = True

    # get main station template detections
    if load_party:
        # load party object from file

        ###########################  TEMPLATE 1  ##############################
        ###################### 2016  9 26  9 28 41.34 #########################
        #######################################################################
        # results in bad detections

        # abs 0.27 = 368 detections
        # infile = open('party_06_15_2016_to_08_12_2018_abs.27_16s.pkl', 'rb')

        # abs 0.25 = 1218 detections, 16 seconds
        # infile = open('party_06_15_2016_to_08_12_2018_abs.25_16s.pkl', 'rb')
        # abs 0.25 = 1218 detections, 8 seconds
        # infile = open('party_06_15_2016_to_08_12_2018_abs.25_8s.pkl', 'rb')

        # abs 0.24 = 2248 detections
        # infile = open('party_06_15_2016_to_08_12_2018_abs.24_16s.pkl', 'rb')

        # abs 0.23 = 4381 detections, require more memory than MBP has
        # infile = open('party_06_15_2016_to_08_12_2018_abs.23.pkl', 'rb')

        # MAD 9.0 = 1435 detections, fits in MBP memory
        # infile = open('party_06_15_2016_to_08_12_2018_MAD9.pkl', 'rb')

        # MAD 8.0 = 3857 detections, doesn't fit in MBP memory
        # infile = open('party_06_15_2016_to_08_12_2018_MAD8_16s.pkl', 'rb')
        # MAD 8.0 =  detections, 8 seconds
        # infile = open('party_06_15_2016_to_08_12_2018_MAD8_8s.pkl', 'rb')

        ###########################  TEMPLATE 2  ##############################
        ###################### 2016  9 27  7 37 49.00 #########################
        #######################################################################
        # results in bad detections

        # abs 0.25 = 289 detections
        # infile = open('party_06_15_2016_to_08_12_2018_t2_abs.25_16s.pkl', 'rb')

        # abs 0.23 = 1137 detections
        # infile = open('party_06_15_2016_to_08_12_2018_t2_abs.23_16s.pkl', 'rb')

        # MAD 10.0 = 460 detections
        # infile = open('party_06_15_2016_to_08_12_2018_t2_MAD10_16s.pkl','rb')

        # MAD 9.0 = 1187 detections
        # infile = open('party_06_15_2016_to_08_12_2018_t2_MAD9_16s.pkl','rb')

        # MAD 8.0 = 3388 detections
        # infile = open('party_06_15_2016_to_08_12_2018_t2_MAD8_16s.pkl', 'rb')

        ############################  TEMPLATE 3  #############################
        ###################### 2016  9 27  6 31 15.00 #########################
        #######################################################################

        # abs 0.25 = 96 detections, 16 seconds
        # infile = open('party_06_15_2016_to_08_12_2018_abs.25_16s_mcr1.pkl','rb')

        # abs 0.29 = 15 detections, 10.5 seconds
        # infile = open('party_06_15_2016_to_08_12_2018_abs.29_16s_t3_10.5.pkl',
        #               'rb')

        # abs 0.29 = 212 detections, 7.0 seconds
        # infile = open('party_06_15_2016_to_08_12_2018_abs.29_16s_t3_7.0.pkl',
        #               'rb')

        #########################  TEMPLATE 4 MCR1  ###########################
        ###################### 2016  9 27  6 31 15.00 #########################
        #######################################################################

        # if thresh_type == "MAD":
        #
        #     if detect_thresh == 9.0:
        #         # MAD 9.0 = 306 detections, 7.0 seconds, BHN only
        #         infile = open('party_06_15_2016_to_08_12_2018_MAD9_7s_t4_BHN.pkl',
        #                       'rb')
        #     elif detect_thresh == 8.5:
        #         # MAD 8.5 = 1127 detections, 7.0 seconds, BHN only
        #         infile = open('party_06_15_2016_to_08_12_2018_MAD8.5_7s_t4_BHN.pkl',
        #                       'rb')
        #     elif detect_thresh == 8.0:
        #         # MAD 8.0 = 3467 detections, 7.0 seconds, BHN only
        #         infile = open('party_06_15_2016_to_08_12_2018_MAD8_7s_t4_BHN.pkl',
        #                       'rb')

        #########################  TEMPLATE 5 N25K  ###########################
        ###################### 2016  9 26  9 25 49.50 #########################
        #######################################################################

        if thresh_type == "MAD":
            if detect_thresh == 15.0:
                # MAD 15.0 = 2547 detections, 7 seconds, BHN only
                infile = open(
                    'party_06_15_2016_to_08_12_2018_MAD15_7s_t5_SHN.pkl',
                    'rb')
            elif detect_thresh == 12.0:
                # MAD 10.0 =  detections, 7 seconds, BHN only
                infile = open(
                    'party_06_15_2016_to_08_12_2018_MAD12_7s_t5_SHN.pkl',
                    'rb')
            elif detect_thresh == 10.0:
                # MAD 10.0 =  detections, 7 seconds, BHN only
                infile = open(
                    'party_06_15_2016_to_08_12_2018_MAD10_7s_t5_SHN.pkl',
                    'rb')
            elif detect_thresh == 8.0:
                # MAD 8.0 = 49340 detections, 7 seconds, BHN only
                infile = open(
                    'party_06_15_2016_to_08_12_2018_MAD8_7s_t5_SHN.pkl',
                    'rb')
        elif thresh_type == "abs":
            if detect_thresh == 0.90:
                # abs 0.9 = 6 detections, 7 seconds, BHN only
                infile = open(
                    'party_06_15_2016_to_08_12_2018_abs.9_7s_t5_SHN.pkl',
                    'rb')
            elif detect_thresh == 0.70:
                # abs 0.7 = 98 detections, 7 seconds, BHN only
                infile = open(
                    'party_06_15_2016_to_08_12_2018_abs.7_7s_t5_SHN.pkl',
                    'rb')
            elif detect_thresh == 0.65:
                # abs 0.65 = 1573 detections, 7 seconds, SHN only
                # same threshold at 12 seconds only gets 3 detections
                infile = open(
                    'party_06_15_2016_to_08_12_2018_abs.65_7s_t5_SHN.pkl',
                    'rb')
            elif detect_thresh == 0.60:
                # abs 0.60 = 9973 detections, 7 seconds, BHN only
                infile = open(
                    'party_06_15_2016_to_08_12_2018_abs.6_7s_t5_SHN.pkl',
                    'rb')
            elif detect_thresh == 0.50:
                # abs 0.5 = 139552 detections, 7 seconds, BHN only
                infile = open(
                    'party_06_15_2016_to_08_12_2018_abs.5_7s_t5_SHN.pkl',
                    'rb')

        # # 100 Hz party for testing, 906 detections
        # infile = open('party_06_15_2016_to_08_12_2018_abs.65_7s_t5_SHN_100Hz'
        #               '.pkl', 'rb')

        # 100 Hz party w/ 0.5s prepick for testing, 3129 detections
        # infile = open('party_06_15_2016_to_08_12_2018_abs.65_7s_t5_SHN_100Hz_0.5prepick.pkl', 'rb')
        # culled & top 50 version
        # infile = open('top_50_WASW_t5_7.0s_0.5_prepick_abs0.65_culled_sorted_party.pkl','rb')

        # 100 Hz party, 3 component, w/ 0.5s prepick for testing
        # infile = open(
        # 'party_06_15_2016_to_08_12_2018_MAD8_7s_t5_SHZNE_100Hz_0.5prepick.pkl','rb')
        # culled & top 1000 version
        # infile = open('top_1000_3comp_WASW_3comp_t5_7.0s_0.5_prepick_MAD8.0_culled_sorted_party.pkl','rb')

        # # Culled 100 Hz party w/ 0.5s prepick for testing
        # infile = open('top_3048_WASW_t5_culled_snr1.0-15.0_selfShift_abs0'
        #               '.65_7s_100Hz_prepick_party.pkl', 'rb')

        # 100 Hz party w/ 1.5s prepick for testing
        # infile = open('WASW_t5_culled_snr1.0-15.0_selfShift_abs0'
        #               '.65_7s_100Hz_1.5prepick_party.pkl', 'rb')

        # 3 stations 100 Hz party, 3 component, w/ 0.5s prepick for testing,
        # 12380 detections
        # infile = open('party_07_18_2016_to_08_12_2018_t6_3sta_MAD8_7s_t6_100Hz_0.5prepick.pkl','rb')
        # culled & top 1000 version
        # infile = open('top_1000_3sta_3comp_t6_7.0s_0.5_prepick_MAD8.0_culled_sorted_party.pkl','rb')

        # MCR2, 1 station, 100 Hz party, 3 component, w/ 0.5s prepick
        # 125254 detections
        # infile = open('t6_MCR2_3comp_party_07_18_2016_to_08_12_2018.pkl','rb')
        # top 1000 culled and sorted version
        # infile = open('top_1000_MCR2_3comp_t6_7.0s_0.5_prepick_MAD8.0_culled_sorted_party.pkl','rb')

        # N25K, 1 station, 100 Hz party, 3 component, w/ 0.5s prepick
        # infile = open('t6_N25K_3comp_party_07_18_2016_to_08_12_2018.pkl', 'rb')
        # top 1000 culled and sorted version
        # infile = open('top_1000_N25K_3comp_t6_7.0s_0.5_prepick_MAD8.0_culled_sorted_party.pkl','rb')

        # 3 station, 3 component, 14 second template
        # infile = open('t6_3sta_3comp_14s_party_07_18_2016_to_08_12_2018.pkl', 'rb')
        # top 1000 culled and sorted version
        # infile = open('top_1000_3sta_3comp_t6_12.0s_2.0_prepick_MAD8.0_culled_sorted_party.pkl','rb')

        # 5 station, 3 component, 14 second template
        # infile = open('t6_5sta_3comp_14s_party_07_18_2016_to_08_12_2018.pkl', 'rb')
        # top 1000 culled and sorted version
        # infile = open('top_1000_5sta_3comp_t6_12.0s_2.0_prepick_MAD8.0_culled_sorted_party.pkl','rb')


        # 9 station, 3 component, 14 second template, raw detections
        infile = open('t6_9sta_3comp_14s_party_07_18_2016_to_08_12_2018.pkl', 'rb')

        # SNR 1-9
        # top 808 culled and sorted version
        infile = open('top_808_9sta_3comp_t6_12.0s_2.0_prepick_MAD8.0_culled_sorted_party.pkl', 'rb')
        # top 100 culled and sorted version SNR 1-9
        infile = open('top_100_9sta_3comp_t6_12.0s_2.0_prepick_MAD8.0_culled_sorted_party.pkl', 'rb')
        # top 500 culled and sorted version SNR 1-9
        infile = open('top_500_9sta_3comp_t6_12.0s_2.0_prepick_MAD8.0_culled_sorted_party.pkl','rb')

        # SNR 1-15
        # top N culled and sorted version SNR 1-15
        infile = open('top_50_9sta_3comp_t6_12.0s_2.0_prepick_MAD8.0_culled_sorted_party_1-15.pkl','rb')
        infile = open('top_100_9sta_3comp_t6_12.0s_2.0_prepick_MAD8.0_culled_sorted_party_1-15.pkl','rb')
        infile = open('top_250_9sta_3comp_t6_12.0s_2.0_prepick_MAD8.0_culled_sorted_party_1-15.pkl','rb')
        infile = open('top_500_9sta_3comp_t6_12.0s_2.0_prepick_MAD8.0_culled_sorted_party_1-15.pkl','rb')

        # StacksDetects SNR 1-15 from top_250 linear stack
        infile = open('stacksDetects_9sta_3comp_t6_12.0s_2.0_prepick_MAD8.0_culled_sorted_party_1-15.pkl','rb')
        infile = open('top_1000_stacksDetects_9sta_3comp_t6_12.0s_2.0_prepick_MAD8.0_culled_sorted_party_1-15.pkl','rb')

        party = pickle.load(infile)
        infile.close()
    else:
        party = detect_signals(templates, template_files, station_dict,
                               template_length, template_prepick,
                               detection_files_path, start_date, end_date,
                               thresh_type, detect_thresh)

    # cull the party detections below the specified signal to noise ratio
    if cull:
        # SNR based culling & removal of traces with > 0.3 *
        # trace.stats.npts NaNs.
        culled_party = cull_detections(party, detection_files_path,
                                       snr_threshold, main_trace)
        # cull based on spectral energy in a specific band?
        # TODO: add spectral function above?

        if plot:
            # inspect the culled party growth over time
            detections_fig = culled_party.plot(plot_grouped=True)
            rate_fig = culled_party.plot(plot_grouped=True, rate=True)
            print(sorted(culled_party.families, key=lambda f: len(f))[-1])

        print(f"Culled detections comprise "
              f"{(round(100 * (len(culled_party) / len(party)), 1))}% of all "
              f"detections ({len(culled_party)}/{len(party)}).")
        # allow old party to get trash collected from memory
        party = culled_party

        if plot:
            detection_stream = get_detections(party, detection_files_path,
                                              main_trace)
            plot_stack(detection_stream[:100],
                       title=f"t6_{template_length}s_"
                             f"{template_prepick}_prepick_{thresh_type}"
                             f"{detect_thresh}_culled_unsorted", save=True)
            # free up some memory
            del detection_stream

    # # calculate and plot snrs of detections, this call is memory-limited
    # # because get_detections loads all traces into a single stream
    # detection_stream = get_detections(party, detection_files_path, main_trace)
    # snrs = snr(detection_stream)
    # del detection_stream

    # this call is not memory-limited because party_snrs loads only one trace
    # into memory at a time
    snrs = party_snrs(party, detection_files_path, main_trace)

    if plot and not cull:
        plot_distribution(snrs, title=f"SNR_distribution_WASW_t5"
                          f"_{template_length}s_{template_prepick}_prepick"
                          f"_{thresh_type}{detect_thresh}", save=True)

    # consider only top n detections ranked by the cross-channel correlation sum
    if top_n:
        # FIXME: move this to a sort function, too complex for inline
        data = []
        for index, detection in enumerate(party.families[0].detections):
            data.append(
                [index, detection.detect_time, abs(detection.detect_val),
                 detection.threshold, detection.threshold_type,
                 detection.threshold_input, detection.template_name,
                 snrs[index]])

        df = pd.DataFrame(data,
                          columns=['index', 'time', 'abs_correlation_sum',
                                   'correlation_sum_threshold',
                                   'threshold_metric',
                                   'metric_input', 'template', 'snr'])

        # get top n indices of detections with largest correlation sum
        top_n_df = df.nlargest(n, 'abs_correlation_sum')
        keeper_indices = top_n_df['index'].tolist()

        # build list of sorted detections to keep
        detections = [party.families[0].detections[idx] for idx in keeper_indices]
        # build catalog for sorted party object by copying, clearing, rebuild
        catalog = party.families[0].catalog.copy()
        catalog.clear()
        for idx in keeper_indices:
            catalog.append(party.families[0].catalog.events[idx])

        # make a sorted top n party
        template = party.families[0].template
        top_n_party = Party(families=Family(template, detections))
        party = top_n_party

        if save_detections:
            # save party detections as text file
            df.to_csv(f"top_{n}_5sta_3comp_t6_{template_length}s_"
                      f"{template_prepick}_prepick_{thresh_type}"
                      f"{detect_thresh}_culled_sorted_detections_1-15.csv",
                      index=False)

            # save party to pickle file
            outfile = open(f"top_{n}_stacksDetects_9sta_3comp_t6_{template_length}s_"
                      f"{template_prepick}_prepick_{thresh_type}"
                      f"{detect_thresh}_culled_sorted_party_1-15.pkl", 'wb')

            pickle.dump(party, outfile)
            outfile.close()

    if plot:
        # inspect the party growth over time
        detections_fig = party.plot(plot_grouped=True)
        rate_fig = party.plot(plot_grouped=True, rate=True)
        print(sorted(party.families, key=lambda f: len(f))[-1])

        # look at family template
        family = sorted(party.families, key=lambda f: len(f))[-1]
        fig = family.template.st.plot(equal_scale=False, size=(800, 600))

        detection_stream = get_detections(party, detection_files_path, main_trace)
        plot_stack(detection_stream[:100],
                   title=f"top_{n}_stacksDetects_9sta_3comp_t6"
                         f"_{template_length}s_"
                         f"{template_prepick}_prepick_{thresh_type}"
                         f"{detect_thresh}_culled_sorted_1-15SNR", save=True)
        # free up some memory
        del detection_stream

    # generate or load a stack
    if load_stack:
        # load stack list from file
        # TEMPLATE 1
        # infile = open(f'inner_stack_0_snr{snr_threshold}_'
        #               f'{shift_method}Shift_abs.25_16s.pkl', 'rb')
        # infile = open(f'inner_stack_0_snr{snr_threshold}_'
        #               f'{shift_method}Shift_abs.27_16s.pkl', 'rb')
        # TEMPLATE 2
        # infile = open(f'inner_t2_stack_0_snr{snr_threshold}_'
        #               f'{shift_method}Shift_abs.23_16s.pkl', 'rb')
        # TEMPLATE 4
        infile = open(f'top_{n}_stack_9sta_t6_snr{snr_threshold[0]}-'
                      f'{snr_threshold[1]}_'
                      f'{shift_method}Shift_{thresh_type}'
                      f'{detect_thresh}_14s.pkl', 'rb')

        stack_list = pickle.load(infile)
        infile.close()
    else:
        # get the template start and end times
        family = sorted(party.families, key=lambda f: len(f))[-1]
        # build a dict of template times that are used for determining cross-
        # correlation time shifts for stacking (if align_type != 'zero')
        template_times = {}
        for trace in family.template.st:
            ID = f"{trace.stats.network}.{trace.stats.station}." \
                 f"{trace.stats.channel}"
            # this selects the entire template including prepick as ref. signal
            template_times[ID] = [trace.stats.starttime - template_prepick,
                                  trace.stats.endtime]

        # stack the culled party detections
        stack_list = stack_template_detections(party, detection_files_path,
                                               main_trace, template_times,
                                               align_type=shift_method,
                                               animate_stacks=False)
        # save stacks as pickle file
        outfile = open(f'top_{n}_stackDS_9sta_t6_snr{snr_threshold[0]}-'
                       f'{snr_threshold[1]}_'
                       f'{shift_method}Shift_{thresh_type}'
                       f'{detect_thresh}_14s.pkl', 'wb')

        pickle.dump(stack_list, outfile)
        outfile.close()

    # expand list into items
    stack_pw, stack_lin, stack_ccs = stack_list

    # # stack_ccs has uneven lengths of rows
    # for item in stack_ccs:
    #     print(len(item))
    # # pad stack_ccs rows to view as np.array
    # padded_stack_ccs = []
    # for row in range(len(stack_ccs)):
    #     row_ccs = []
    #     for column in range(0, n):
    #         try:
    #             row_ccs.append(stack_ccs[row][column])
    #         except Exception:
    #             row_ccs.append(0)
    #             pass
    #     padded_stack_ccs.append(row_ccs)
    # stack_ccs = np.array(padded_stack_ccs)
    # plt.hist(stack_ccs[0])
    # plt.show()
    # for trace in stack_pw:
    #     print(trace.stats.npts)

    # plot stacks
    if plot:
        if len(stack_pw) > 0:
            plot_stack(stack_pw, title=f'top_'
                                       f'{n}_9sta_phase_weighted_stackDS_snr'
                                       f'{snr_threshold[0]}-'
                                       f'{snr_threshold[1]}_{shift_method}'
                                       f'Shift_{thresh_type}'
                                       f'{detect_thresh}_14s',
                       save=True)

        if len(stack_lin) > 0:
            plot_stack(stack_lin, title=f'top_{n}_9sta_linear_stackDS_sn'
                                        f'r{snr_threshold[0]}-'
                                        f'{snr_threshold[1]}_'
                                        f'{shift_method}Shift_{thresh_type}'
                                        f'{detect_thresh}_14s',
                       save=True)

            # now plot template with the linear stack from same station for
            # comparison
            # plot_template_and_stack(party, stack_lin, stack_pw,
            #                         detection_files_path, save=False,
            #                         title=f'stacks_templates_sn'
            #                               f'r{snr_threshold}_'
            #                         f'{shift_method}Shift_abs.24_16s')

            plot_template_and_stack(party, stack_lin, stack_pw,
                                    detection_files_path, 6, save=True,
                                    title=f'top_{n}_9sta_stacks_templates_sn'
                                          f'r{snr_threshold[0]}-'
                                          f'{snr_threshold[1]}_'
                                          f'{shift_method}Shift_'
                                          f'{thresh_type}'
                                          f'{detect_thresh}_14s_UN')

        # # plot zoomed in
        # if len(stack_pw) > 0:
        #     stack_pw.trim(UTCDateTime("2016-01-01T11:59:50.0Z"), UTCDateTime(
        #         "2016-01-01T12:00:05.0Z"), pad=True,
        #                   fill_value=0, nearest_sample=True)
        #     plot_stack(stack_pw, title=f'phase_weighted_stack_snr'
        #                f'{snr_threshold}_{shift_method}'
        #                f'Shift_abs.25_16s_zoom', save=False)
        # if len(stack_lin) > 0:
        #     stack_lin.trim(UTCDateTime("2016-01-01T11:59:50.0Z"), UTCDateTime(
        #         "2016-01-01T12:00:05.0Z"), pad=True,
        #                    fill_value=0, nearest_sample=True)
        #     plot_stack(stack_lin, title=f'linear_stack_snr{snr_threshold}_'
        #                f'{shift_method}Shift_abs.25_16s_zoom',
        #                save=False)

    # use stacks as templates in matched-filter search to build catalog of
    # detections
    if load_stack_detects:
        # MAD 8 template: 1728 detections
        # culled MAD 8 detections: 1581
        # load stack list from file
        infile = open(f'party_{start_date.month:02}_{start_date.day:02}_' \
                   f'{start_date.year}_to_{end_date.month:02}' \
                   f'_{end_date.day:02}_' \
                   f'{end_date.year}_{thresh_type}'
                      f'{detect_thresh}_14s_stackDetects.pkl', 'rb')
        party = pickle.load(infile)
        infile.close()
    else:
        party = detections_from_stacks(stack_lin, detection_files_path,
                                       start_date, end_date, main_trace)
    if plot:
        if party != None and len(party) > 0:
            # inspect the party growth over time
            detections_fig = party.plot(plot_grouped=True)
            rate_fig = party.plot(plot_grouped=True, rate=True)
            print(sorted(party.families, key=lambda f: len(f))[-1])

            # look at family template
            family = sorted(party.families, key=lambda f: len(f))[-1]
            fig = family.template.st.plot(equal_scale=False, size=(800, 600))

    # TODO: Call to cull

    # TODO: Call to top_n

    # make another stack from the new detections
    if load_second_stack:
        # load stack list from file
        infile = open(f'inner_stack_2_snr{snr_threshold}_'
                      f'{shift_method}Shift_abs.25_16s.pkl', 'rb')
        stack_list = pickle.load(infile)
        infile.close()
    else:
        # TODO: add changed stack call

        # stack the culled party detections
        stack_list = stack_template_detections(party, detection_files_path,
                                               main_trace,
                                               align_type=shift_method)
        # save stacks as pickle file
        # abs 0.25: h m to stack
        outfile = open(f'inner_stack_t4_snr{snr_threshold}_'
                       f'{shift_method}Shift_abs.29_7s.pkl', 'wb')
        # MAD 8: 6h 35m to stack
        # outfile = open(f'inner_stack_0_snr{snr_threshold}_'
        #                f'{shift_method}Shift_MAD8_16s.pkl', 'wb')
        pickle.dump(stack_list, outfile)
        outfile.close()

    # plot second stacks
    stack_pw, stack_lin, stack_ccs = stack_list
    if plot:
        if len(stack_pw) > 0:
            plot_stack(stack_pw, title=f'phase_weighted_stack_t4_snr'
                       f'{snr_threshold}_{shift_method}'
                       f'Shift_abs.29_7s', save=True)
        if len(stack_lin) > 0:
            plot_stack(stack_lin, title=f'linear_stack_t4_snr{snr_threshold}_'
                       f'{shift_method}Shift_abs.29_7s',
                       save=True)


    # TODO: what should the ultimate return(s) be after generating second
    #  stack? Need to guard memory from large stacks.

    return party

# NEXT DO MAX & FIXED, ANIMATE IF BROKEN
if False:
    # TODO: develop time-frequency filter based on template signal to
    #  exclude shorter-period signal that xcorr captures
    trace = detection_stream[0].copy()
    trace.spectrogram()

    # download instrument response inventory
    datacenter = "IRIS"
    client = Client(datacenter)
    inv = Inventory()
    inv += client.get_stations(network=trace.stats.network,
                               station=trace.stats.station,
                               starttime=trace.stats.starttime,
                               endtime=trace.stats.endtime, level="response")

    # get PSD
    # FIXME: this doesn't always successfully add trace. Guard against
    #  empty case
    ppsd = PPSD(trace.stats, metadata=inv)
    ppsd.add(trace)
    print("number of psd segments:", len(ppsd.times_processed))
    ppsd.plot()

    # better spectrogram
    # https://krischer.github.io/seismo_live_build/html/Signal%20Processing/spectral_analysis+preprocessing_solution_wrapper.html

    # for fft npts
    # https://docs.obspy.org/packages/autogen/obspy.signal.util._npts2nfft.html?highlight=fft#obspy.signal.util._npts2nfft

# get locations from detection times and stacks via NLL
# TODO:
#     - build grid via stingray
#     - build NLL files from stingray files

# generate focal mechanisms from phase weighted stacks and locations
# TODO


# STATUS: detections contain events that are not similar period and spectrum
# TODO: culling function based on spectral energy & period
#     : notes below find_LFEs have structure for spect.