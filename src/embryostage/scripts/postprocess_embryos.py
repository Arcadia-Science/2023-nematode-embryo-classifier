# conda activate sklearn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import numpy as np
import os

# encoding embryo states as integers
STATE_CODES = {
    'proliferation': 0,
    'bean': 1,
    'comma': 2,
    'fold': 3,
    'hatch': 4,
    'death' : 5
}
STATE_CODES_INV = {v: k for k, v in STATE_CODES.items()} # inverse

def fix_states(states: list):
    """
    Use time information to correct embryo state predictions. Specific rules are:
        - Median filter to remove sporadic state transitions
        - Eliminate fold->death transitions
        - Embryo states cannot go backwards in development
    
    Inputs: 
        states: list[str] or list[list[str]]
            list of embryo states in chronological order (or list of lists for multiple embryos)
    
    Outptuts:
        states_out: list[list[str]]
            list of corrected embryo states
    """
    if type(states[0]) == str:
        # single embryo, convert to list
        states = [states]
    
    states_out = []
    for state in states:
        # convert states to numerical values
        states_numeric = [STATE_CODES[s] for s in state]
        
        # median filter
        states_numeric = medfilt(states_numeric, kernel_size=7)

        # eliminate fold->death transitions
        for i in range(len(states_numeric)-1):
            if states_numeric[i] == 3 and states_numeric[i+1] == 5:
                states_numeric[i+1] = 3

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
        states_out.append([STATE_CODES_INV[s] for s in states_numeric])
        # print(states_out)
    
    return states_out

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

    1. load csv of embryo states
    for every embryo:
        2. fix states
        3. visualize embryo (blue: raw, orange: fixed)
        4. save figure
    
    '''
    # load csv of embryo states
    data = pd.read_csv('/Users/ilya_arcadia/Code/lolscripts/data/sample-embryo-classification.csv').T
    

    batch_size = 20 # number of embryos to plot per batch
    n_embryos = data.shape[1]
    num_batches = int(np.floor(n_embryos/batch_size))

    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    for b in range(num_batches):
        embryos_to_plot = np.arange(b*batch_size, b*batch_size + batch_size)
        f,axs = plt.subplots(int(np.floor(len(embryos_to_plot)/2)),2,figsize=(14,20))
        axs = axs.ravel()
        for i,e in enumerate(embryos_to_plot):
            sample_embryo = data[[e]].T.values.tolist()
            visualize_progression([sample_embryo[0], fix_states(sample_embryo)[0]], axs[i])
            axs[i].set_title('Embryo ' + str(i))
        plt.tight_layout()
        plt.savefig('plots/embryo-batch-' + str(b) + '.png')
