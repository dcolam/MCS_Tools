import spikeinterface.full as si
from spikeinterface.preprocessing import scale, common_reference
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from pathlib import Path
import os
import pandas as pd
from spikeinterface.extractors import read_mcsh5
from scipy.stats import norm
from math import ceil, floor


def compute_peaks(filtered, wholeTrace=True, startTime=0, endTime=60, pretrigger=2, posttrigger=5, thresh_factor=5):
    fs = filtered.get_sampling_frequency()
    channels = filtered.channel_ids

    # in seconds
    if wholeTrace:
        startTime = 0
        endTime = filtered.get_duration()

    filtered_sine = filtered.get_traces(start_frame=int(fs * startTime), end_frame=int(fs * endTime))
    pretrigger = (pretrigger * fs) / 1000  # Pre-event time window in ms
    posttrigger = (posttrigger * fs) / 1000  # Post-event time window in ms

    thresh_min_width = 0.1 * (fs / 1000)

    # Empty list to store the tables for each trace
    table_list = []

    # Loop through the current columns
    for i, peaks_signal in enumerate(filtered_sine.T):
        #print("Channel {}".format(i))
        rms = np.sqrt(np.mean(peaks_signal ** 2))
        peaks, peaks_dict = signal.find_peaks(-peaks_signal,
                                              height=thresh_factor * rms,
                                              rel_height=1,
                                              width=thresh_min_width)

        # Create table with results
        if len(peaks) > 0:

            table = pd.DataFrame(columns=["File", 'trace_index', "channel", 'event', 'peak_index',
                                          'peak_time_s',
                                          'event_window_start', 'event_window_end',
                                          "Amplitude_mV", 'width_ms',
                                          'inst_freq', 'isi_s',
                                          'area'])

            table.event = np.arange(1, len(peaks) + 1)

            table.trace_index = i  # Add index for each sweep
            table.channel = channels[i]
            table.peak_index = peaks
            table.peak_time_s = peaks / fs  # Divided by fs to get s
            table.event_window_start = peaks_dict['left_ips'] - pretrigger
            table.event_window_end = peaks_dict['right_ips'] + posttrigger
            #table.peak_amp = peaks_dict['peak_heights']
            table.Amplitude_mV = peaks_signal[peaks]  # height parameter is needed
            table.width_ms = peaks_dict['widths'] / (fs / 1000)  # Width (ms) at full-height

            # Calculations based on the parameters above
            table.inst_freq = np.insert((1 / (np.array(table.peak_index[1:]) -
                                              np.array(peaks_dict['left_ips'][:-1])) * fs),
                                        0, np.nan)

            table.isi_s = np.diff(peaks, axis=0, prepend=peaks[0]) / fs

            # Area
            for i, event in table.iterrows():
                individual_event = peaks_signal[int(event.peak_index - pretrigger): int(event.peak_index + posttrigger)]
                table.loc[i, 'area'] = np.round(individual_event.sum(), 1) / (fs / 1000)

                # Append the table to the list
            table_list.append(table)

    # Concatenate the tables in the list
    return pd.concat(table_list, ignore_index=True)

def plot_traces(recording, channels, startTime=0, endTime=60):
    fs = recording.get_sampling_frequency()
    filtered_sine = recording.get_traces(start_frame=int(fs * startTime),
                                         end_frame=int(fs * endTime),
                                         channel_ids=np.array(channels))
    numFrames = filtered_sine.shape[0]
    fs = numFrames / (endTime - startTime)
    #print(fs)
    time_range = np.arange(startTime, endTime, 1/fs)
    #print(time_range.shape)
    #channels = recording.channel_ids[channels]
    for i, _ in enumerate(channels):
        # plt.scatter(time_range, filtered_sine[:,i])
        plt.plot(time_range, filtered_sine[:, i])
        rms = np.sqrt(np.mean(filtered_sine[:, i] ** 2))
        peaks, _ = signal.find_peaks(-filtered_sine[:, i], height=6 * rms)
        plt.scatter(time_range[peaks], filtered_sine[:, i][peaks], c="r")

def plot_spikes(recording, channels, startTime=0, endTime=60, pretrigger=0.001, posttrigger=0.001):
    fs = recording.get_sampling_frequency()
    # channels = recording.channel_ids[channels]
    filtered_sine = recording.get_traces(start_frame=int(fs * startTime), end_frame=int(fs * endTime),
                                         channel_ids=np.array(channels))

    #print(filtered_sine.shape)
    #print(filtered_sine.shape[0]/fs)
    numFrames = filtered_sine.shape[0]
    fs = numFrames / (endTime - startTime)
    #print(fs)
    #time_range = np.arange(-pretrigger, posttrigger, 1/fs)
    pretrigger = int((pretrigger * fs))
    posttrigger = int((posttrigger * fs))
    time_range = np.arange(-pretrigger, posttrigger, 1)
    #print(pretrigger, posttrigger)
    #print(time_range.shape)
    n_rows = 3
    n_cols = round((len(channels) + 1) / n_rows)

    fig, axs = plt.subplots(figsize=(3*n_cols, 3.5 * n_rows), nrows=n_rows, ncols=n_cols, sharex=True)

    for i, c in enumerate(channels):
        rms = np.sqrt(np.mean(filtered_sine[:, i] ** 2))
        peaks, _ = signal.find_peaks(-filtered_sine[:, i], height=7 * rms)
        col = ceil((i + 1) / n_rows)-1
        row = i - (floor(i/n_rows)*n_rows)

        if n_cols == 1:
            pos = i
        else:
            pos = (row, col)

        if len(peaks):
            length = len(filtered_sine[:, i][:(pretrigger + posttrigger)])

            peak_traces = np.ones(((pretrigger + posttrigger), len(peaks)))

            for j, p in enumerate(peaks):
                temp = filtered_sine[:, i][(p - pretrigger):(p + posttrigger)]
                peak_traces[:, j] = temp
                axs[pos].plot(time_range, temp)
            mean_peak = np.mean(peak_traces, axis=1)
            std_peak = np.std(peak_traces, axis=1)
            axs[pos].plot(time_range, mean_peak, c="red", linewidth=4, label="Mean Trace")
            axs[pos].legend(loc="upper right")
            axs[pos].fill_between(time_range, mean_peak - std_peak, mean_peak + std_peak, alpha=0.6)
            axs[pos].set_title("Channel {}".format(c))
        else:
            axs[pos].hlines(0, time_range[0], time_range[-1])
            axs[pos].set_title("Channel {}".format(c))
    fig.supylabel('Peak Traces (mV)')
    fig.supxlabel('Time (s)')
    #plt.show()

def compute_burst_activity(spikes,
                           recording,
                           plotRaster=False,
                           fullTrace=True,
                           startTime=0,
                           endTime=60,
                           sigma=0.03,
                           binSize=0.01,
                           threshold=3,
                           pretrigger=10,
                           posttrigger=10):

    fs = recording.get_sampling_frequency()
    channels = recording.channel_ids
    pretrigger = (pretrigger * fs) / 1000  # Pre-event time window in ms
    posttrigger = (posttrigger * fs) / 1000

    if fullTrace:
        startTime = 0
        endTime = recording.get_duration()


    histc, _ = np.histogram(spikes["peak_time_s"], bins=np.arange(startTime, endTime, binSize))

    kernel_matrix = norm.pdf(np.arange(-3 * sigma, 3 * sigma, binSize), 0, sigma)
    freq = np.convolve(histc, kernel_matrix * binSize, 'same')
    freq = (freq / binSize) / len(channels)
    time_range = np.arange(startTime, endTime, binSize)
    histc = (histc / binSize) / len(channels)

    rms = np.sqrt(np.mean(freq ** 2))
    thresh_min_width = 0.2 * (fs / 1000)
    peaks, peaks_dict = signal.find_peaks(freq, height=rms * threshold, width=thresh_min_width)

    burst_table = pd.DataFrame(columns=["File", 'burst_index', 'burst_frame',
                                        'burst_time_s',
                                        'event_window_start', 'event_window_end', "Burst_Peak_frequency", 'width_ms',
                                        'inst_freq', 'isi_s',
                                        'area'])

    burst_table.burst_index = np.arange(1, len(peaks) + 1)
    # burst_table["File"] = spikes["File"]
    burst_table.burst_frame = peaks
    burst_table.burst_time_s = peaks / fs  # Divided by fs to get s
    burst_table.event_window_start = peaks_dict['left_ips'] - pretrigger
    burst_table.event_window_end = peaks_dict['right_ips'] + posttrigger
    # burst_table.burst_amp = peaks_dict['peak_heights']
    burst_table.Burst_Peak_frequency = freq[peaks]  # height parameter is needed
    burst_table.width_ms = peaks_dict['widths'] / (fs / 1000)  # Width (ms) at half-height

    # Calculations based on the parameters above
    burst_table.inst_freq = np.insert((1 / (np.array(burst_table.burst_frame[1:]) -
                                            np.array(peaks_dict['left_ips'][:-1])) * fs),
                                      0, np.nan)

    burst_table.isi_s = np.diff(peaks, axis=0, prepend=peaks[0]) / fs

    for i, event in burst_table.iterrows():
        # print(event)
        individual_event = freq[int(event.burst_frame - pretrigger): int(event.burst_frame + posttrigger)]
        # peak_trace = np.append(peak_trace, individual_event, axis=0)
        # print(individual_event.shape)
        # peak_trace =np.concatenate(([peak_trace],[individual_event]),axis=0)
        burst_table.loc[i, 'area'] = np.round(individual_event.sum(), 1) / (fs / 1000)

    if plotRaster:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 8))
        spikes.plot.scatter(x="peak_time_s", y="trace_index", ax=axes[0], c=spikes["trace_index"], colormap='viridis',
                            s=1, colorbar=False, alpha=0.5)
        # axes[1].plot(time_range[1:], histc)
        axes[1].plot(np.arange(startTime, endTime, binSize)[1:], freq)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency [Hz]")

        plt.axhline(y=rms * threshold, color='r', linestyle='-')
        plt.scatter(time_range[1:][peaks], peaks_dict["peak_heights"], c="r")
        #plt.show()

    return burst_table

def get_most_active_channels(table, n_channels = 6):
    table_aggregate = table.groupby('trace_index').agg(
        {"peak_index": ["count"], "Amplitude_mV": ['mean', "median"]}).reset_index()

    return table_aggregate["peak_index", "count"].nlargest(n_channels)
