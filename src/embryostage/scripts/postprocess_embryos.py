# conda activate sklearn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy import stats as st
import numpy as np
import os
import argparse

MEDFILT_KERNEL = 7 # median filter kernel size

# encoding embryo states as integers
STATE_CODES = {
    'unfertilized': -1,
    'proliferation': 0,
    'bean': 1,
    'comma': 2,
    'fold': 3,
    'hatch': 4,
    'death' : 5
}

STATE_CODES_INV = {v: k for k, v in STATE_CODES.items()} # inverse

def fix_states(states: list[str]) -> [list[str], 
                                      str, 
                                      dict]:
    """

    Use time information to correct embryo state predictions. Specific rules are:
        - Median filter to remove sporadic state transitions
        - If embryo has hatched or finished in fold state, remove death transitions
        - Embryo states cannot go backwards in development
    
    Saves embryo progression plots

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

    states_out = []

    # convert states to numerical values
    states_numeric = [STATE_CODES[s] for s in states]
    
    # median filter
    states_numeric = medfilt(states_numeric, kernel_size=MEDFILT_KERNEL)


    # final state (pre filtering)
    final_state_prefilt_numeric = st.mode(states_numeric[-1*MEDFILT_KERNEL:]).mode
    final_state_prefilt = STATE_CODES_INV[final_state_prefilt_numeric]

    # eliminate death transitions if the embryo has hatched or finished in fold state
    if final_state_prefilt in ['hatch', 'fold']:
        '''
        for i in range(len(states_numeric)-1):
            if states_numeric[i] == 3 and states_numeric[i+1] == 5:
                states_numeric[i+1] = 3
        '''
        # go backwards in time and fix death states
        for i in range(len(states_numeric)-1, 0, -1):
            if states_numeric[i] == STATE_CODES['death']:
                states_numeric[i] = states_numeric[i+1]
    
    # embryo states cannot go backwards in development
    for i in range(len(states_numeric)-1):
        if states_numeric[i+1] < states_numeric[i]:
            states_numeric[i+1] = states_numeric[i]


    # find any remaining transitions of magnitude > 1
    transitions = np.abs(np.diff(states_numeric))
    if any(transitions > 1):
        pass
        # print('Warning: Impossible transition detected')
    
    # convert numeric states back to strings
    states_out = [STATE_CODES_INV[s] for s in states_numeric]
    

    # summary dict of each state and its duration
    state_durations = {}
    for s in states_out:
        if s in state_durations:
            state_durations[s] += 1
        else:
            state_durations[s] = 1
    
    # final state (post filtering)
    final_state_postfilt_numeric = st.mode(states_numeric[-1*MEDFILT_KERNEL:]).mode
    final_state_postfilt = STATE_CODES_INV[final_state_postfilt_numeric]

    return states_out, final_state_postfilt, state_durations

def visualize_progression(states: list, ax = None):
    """
    Plot the progression of the embryo states over time.

    Inputs: 
        states: list 
            list of embryo states in chronological order (or list of lists for multiple embryos)
        ax: matpltlib.axes.Axes
            axes to plot on, if None, create new figure
    Outputs:
        None
    """

    if ax is None:
        f,ax = plt.subplots(figsize=(8,3))
    
    if type(states[0]) == str:
        # single embryo, convert to list
        states = [states]

    for state in states:
        # convert states to numerical values
        states_numeric = [STATE_CODES[s] for s in state]
        ax.plot(states_numeric, marker='.', linewidth=2)
    ax.legend(['' + str(i) for i in range(len(states))])

    # legend
    # set y labels to embryo states
    ax.set_yticks(list(STATE_CODES.values()))
    ax.set_yticklabels(list(STATE_CODES.keys()))

    ax.set_xlabel('Frame')
    ax.set_ylabel('State')


if __name__ == '__main__':

    '''
    demo script of postprocessing step

    '''
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Postprocess embryo states')
    parser.add_argument('--data_json', type=str, help='path to json file containing embryo states') # 
    args = parser.parse_args()

    # parse filename from path
    filename = os.path.basename(args.data_json).split('.')[0]

    data_json = pd.read_json(args.data_json)

    # if condition not specified, 
    if 'condition' not in data_json.columns:
        data_json['condition'] = 'control' + filename
    
    batch_size = 20 # number of embryos to plot per batch
    n_embryos = data_json.shape[0]
    num_batches = int(np.floor(n_embryos/batch_size))

    plots_dir = os.path.join('post', 'plots', 'plots_{}'.format(filename))
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    denoised_labels = []
    embryo_end_states = []
    for b in np.arange(num_batches+1):
        embryos_to_plot = np.arange(b*batch_size, min(n_embryos, batch_size * (b+1)))
        f,axs = plt.subplots(int(np.ceil(len(embryos_to_plot)/2)),2,figsize=(14,20))
        axs = axs.ravel()
        print(embryos_to_plot)
        for i,e in enumerate(embryos_to_plot):
            data_embryo = data_json.loc[e]
            sample_embryo = data_embryo.labels
            fixed_embryo, final_state, _ = fix_states(sample_embryo)
            denoised_labels.append(fixed_embryo)
            embryo_end_states.append(final_state)
            visualize_progression([sample_embryo, fixed_embryo], axs[i])
            axs[i].set_title('d_id: {} | fov_id: {} | e_id: {} | c: {}'.format(*data_embryo[['dataset_id', 'fov_id', 'embryo_id', 'condition']].values) +  ' (final: ' + final_state + ')')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'embryo-batch-' + str(b) + '.png'))
        plt.close()

    data_json['denoised_labels'] = denoised_labels
    data_json['embryo_end_states'] = embryo_end_states

    out_dir = os.path.join('post', 'out_json')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_json.to_json(os.path.join(out_dir, filename + '-fixed.json'))