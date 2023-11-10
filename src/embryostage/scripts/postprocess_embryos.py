import os
import pathlib
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from scipy.signal import medfilt

# kernel size for the median filter used to smooth the embryo states
MEDFILT_KERNEL = 7

# assign a numerical index to each embryo state, ordered by developmental progression
STATE_TO_INDEX = {
    'unfertilized': -1,
    'proliferation': 0,
    'bean': 1,
    'comma': 2,
    'fold': 3,
    'hatch': 4,
    'death': 5,
}

INDEX_TO_STATE = {v: k for k, v in STATE_TO_INDEX.items()}


def _cleanup_predicted_states(states: list[str]) -> [list[str], str, dict]:
    """
    Use temporal information to correct the raw state predictions
    for a given embryo

    Specific rules are:
        - Median filter to remove sporadic state transitions
        - If embryo ends in the 'hatch'' or 'fold' states, remove death transitions
        - Embryo states cannot go backwards in development

    Inputs:
        states: list[str]
            list of embryo states in chronological order

    Returns:
        states_out: list[str]
            list of cleaned-up embryo states
        final_state_postfilt: str
            final state of embryo after filtering
        state_durations: dict
            dictionary of each state and its duration in frames

    """

    # convert states to numerical values
    states_numeric = [STATE_TO_INDEX[state] for state in states]

    # median filter
    states_numeric = medfilt(states_numeric, kernel_size=MEDFILT_KERNEL)

    # final state (pre filtering)
    final_state_prefilt_numeric = scipy.stats.mode(states_numeric[-MEDFILT_KERNEL:]).mode
    final_state_prefilt = INDEX_TO_STATE[final_state_prefilt_numeric]

    # eliminate all transitions to 'death' if the embryo's final state
    # is either 'hatch' or 'fold'
    if final_state_prefilt in ['hatch', 'fold']:
        for ind in range(len(states_numeric) - 1, 0, -1):
            if states_numeric[ind] == STATE_TO_INDEX['death']:
                states_numeric[ind] = states_numeric[ind + 1]

    # embryo states cannot go backwards in development
    for ind in range(len(states_numeric) - 1):
        if states_numeric[ind + 1] < states_numeric[ind]:
            states_numeric[ind + 1] = states_numeric[ind]

    # convert numeric states back to strings
    states_out = [INDEX_TO_STATE[state] for state in states_numeric]

    # calculate the duration of each state
    state_durations = {state: 0 for state in set(states_out)}
    for state in states_out:
        state_durations[state] += 1

    # final state (post filtering)
    final_state_postfilt_numeric = scipy.stats.mode(states_numeric[-MEDFILT_KERNEL:]).mode
    final_state_postfilt = INDEX_TO_STATE[final_state_postfilt_numeric]

    return states_out, final_state_postfilt, state_durations


def _visualize_embryo_states(state_lists: list, ax=None):
    """
    Plot the progression of the embryo states over time.

    Inputs:
        state_lists: list
            list of embryo states in chronological order,
            or a list of such lists for multiple embryos
        ax: matpltlib.axes.Axes
            axes to plot on, if None, create new figure
    Outputs:
        None
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    # check if `states_list` is not a list of lists
    # but instead a list of states for a single embryo
    if isinstance(state_lists[0], str):
        state_lists = [state_lists]

    for state_list in state_lists:
        # convert states to numerical values
        states_numeric = [STATE_TO_INDEX[state] for state in state_list]
        ax.plot(states_numeric, marker='.', linewidth=2)

    # the legend is just the index of each embryo in the list of state lists
    ax.legend([str(ind) for ind in range(len(state_lists))])

    # set y labels to embryo states
    ax.set_yticks(list(STATE_TO_INDEX.values()))
    ax.set_yticklabels(list(STATE_TO_INDEX.keys()))

    ax.set_xlabel('Frame')
    ax.set_ylabel('State')


@click.option(
    '--predictions-filepath',
    type=pathlib.Path,
    help='path to the JSON file of predicted embryo states',
)
@click.option(
    '--output-dirpath',
    type=pathlib.Path,
    help='path to directory to which to save output files',
)
@click.command()
def main(predictions_filepath, output_dirpath):
    '''
    Use temporal information to correct the raw embryo state predictions
    generated by batch_classify_embryos.py, plot the corrected states, and save the results

    predictions_filepath: pathlib.Path
        path to the JSON file of predicted embryo states
        (generated by batch_classify_embryos.py)

    output_dirpath: pathlib.Path
        path to directory to which to save the plots of embryo state and the updated JSON file
    '''
    filename = predictions_filepath.stem
    data = pd.read_json(predictions_filepath)

    # if condition not specified, assume control
    if 'condition' not in data.columns:
        data['condition'] = 'control'

    # number of embryos to plot per batch
    batch_size = 20

    # number of columns in each plot
    num_cols = 2

    num_embryos = data.shape[0]
    num_batches = int(np.floor(num_embryos / batch_size))

    plots_dirpath = output_dirpath / 'plots' / f'plots_{filename}'
    os.makedirs(plots_dirpath, exist_ok=True)

    # initialize columns for the cleaned-up labels and final state
    data['denoised_labels'] = None
    data['embryo_end_state'] = None

    for batch_ind in np.arange(num_batches + 1):
        print(f'Plotting batch {batch_ind} of {num_batches}')

        embryos_inds_to_plot = np.arange(
            batch_ind * batch_size, min(num_embryos, batch_size * (batch_ind + 1))
        )

        _, axs = plt.subplots(
            int(np.ceil(len(embryos_inds_to_plot) / num_cols)), num_cols, figsize=(14, 20)
        )
        axs = axs.ravel()

        for ind, embryo_ind in enumerate(embryos_inds_to_plot):
            ax = axs[ind]

            data_row = data.loc[embryo_ind]

            predicted_states = data_row.labels
            cleaned_predicted_states, final_state, _ = _cleanup_predicted_states(
                predicted_states
            )

            data.at[embryo_ind, 'denoised_labels'] = cleaned_predicted_states
            data.at[embryo_ind, 'embryo_end_state'] = final_state

            _visualize_embryo_states(
                state_lists=[predicted_states, cleaned_predicted_states], ax=ax
            )

            ids_for_title = (
                data_row[['dataset_id', 'fov_id', 'embryo_id', 'condition']]
                .apply(str)
                .tolist()
            )
            ax.set_title(f'{" | ".join(ids_for_title)} (final: {final_state})')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dirpath, f'embryo-batch-{batch_ind}.png'))
        plt.close()

    data.to_json(output_dirpath / f'{filename}-cleaned.json')


if __name__ == '__main__':
    main()
