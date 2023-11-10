import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats as st
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


def fix_states(states: list[str]) -> [list[str], str, dict]:
    """
    Use time information to correct embryo state predictions. Specific rules are:
        - Median filter to remove sporadic state transitions
        - If embryo has hatched or finished in fold state, remove death transitions
        - Embryo states cannot go backwards in development

    Inputs:
        states: list[str]
            list of embryo states in chronological order

    Returns:
        states_out: list[str]
            list of fixed embryo states
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
    final_state_prefilt_numeric = st.mode(states_numeric[-MEDFILT_KERNEL:]).mode
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

    # summary dict of each state and its duration
    state_durations = {state: 0 for state in set(states_out)}
    for state in states_out:
        state_durations[state] += 1

    # final state (post filtering)
    final_state_postfilt_numeric = st.mode(states_numeric[-MEDFILT_KERNEL:]).mode
    final_state_postfilt = INDEX_TO_STATE[final_state_postfilt_numeric]

    return states_out, final_state_postfilt, state_durations


def visualize_progression(state_lists: list, ax=None):
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


if __name__ == '__main__':
    '''
    demo script of postprocessing step
    '''
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Postprocess embryo states')
    parser.add_argument(
        '--data_json', type=str, help='path to json file containing embryo states'
    )
    args = parser.parse_args()

    # parse filename from path
    filename = os.path.basename(args.data_json).split('.')[0]

    data_json = pd.read_json(args.data_json)

    # if condition not specified, assume control
    if 'condition' not in data_json.columns:
        data_json['condition'] = 'control'

    # number of embryos to plot per batch
    batch_size = 20
    num_embryos = data_json.shape[0]
    num_batches = int(np.floor(num_embryos / batch_size))

    plots_dir = os.path.join('post-processing-results', 'plots', f'plots_{filename}')
    os.makedirs(plots_dir, exist_ok=True)

    denoised_labels = []
    embryo_end_states = []
    for batch_ind in np.arange(num_batches + 1):
        print(f'Plotting batch {batch_ind} of {num_batches}')

        embryos_to_plot = np.arange(
            batch_ind * batch_size, min(num_embryos, batch_size * (batch_ind + 1))
        )
        _, axs = plt.subplots(int(np.ceil(len(embryos_to_plot) / 2)), 2, figsize=(14, 20))
        axs = axs.ravel()

        for ind, embryo_ind in enumerate(embryos_to_plot):
            data_embryo = data_json.loc[embryo_ind]
            sample_embryo = data_embryo.labels
            fixed_embryo, final_state, _ = fix_states(sample_embryo)
            denoised_labels.append(fixed_embryo)
            embryo_end_states.append(final_state)

            visualize_progression(state_lists=[sample_embryo, fixed_embryo], ax=axs[ind])

            ids_for_title = (
                data_embryo[['dataset_id', 'fov_id', 'embryo_id', 'condition']]
                .apply(str)
                .tolist()
            )
            axs[ind].set_title(f'{" | ".join(ids_for_title)} (final: {final_state})')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'embryo-batch-{batch_ind}.png'))
        plt.close()

    data_json['denoised_labels'] = denoised_labels
    data_json['embryo_end_states'] = embryo_end_states

    out_dir = os.path.join('post-processing-results', 'out_json')
    os.makedirs(out_dir, exist_ok=True)

    data_json.to_json(os.path.join(out_dir, f'{filename}-fixed.json'))
