#!/usr/bin/env python

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# See the LICENSE file for more details.

import experiment
import vis

import argparse

def main():
    parser = argparse.ArgumentParser(description='Bounded logit attention.')
        
    parser.add_argument('-d','--dataset', required=False, default='cats_vs_dogs',
                       choices=['cats_vs_dogs', 'stanford_dogs', 'caltech_birds2011'])
    
    parser.add_argument('-f', '--fixed-size', action='store_true')
    parser.add_argument('-t', '--threshold', action='store_true')
    parser.add_argument('-p', '--post-hoc', action='store_true')
    parser.add_argument('-r', '--force-retraining', action='store_true')
        
    parser.add_argument('-s','--preset', required=False,
                       choices=['L2X-F', 'BLA', 'BLA-T', 'BLA-PH'])
    
    args = vars(parser.parse_args())
    
    fixed_size = args['fixed_size']
    threshold = args['threshold']
    train_head = not args['post_hoc']
    force_retrain = args['force_retraining']
    
    ds = args['dataset']
    
    preset = args['preset']
    if preset=='L2X-F':
        fixed_size = True
        threshold = False
        train_head = True
    elif preset=='BLA':
        fixed_size = False
        threshold = False
        train_head = True
    elif preset=='BLA-T':
        fixed_size = False
        threshold = True
        train_head = True
    elif preset=='BLA-PH':
        fixed_size = False
        threshold = True
        train_head = False
    
    wrapper = experiment.make_wrapper(ds,
                 fixed_size=fixed_size,
                 threshold=threshold,
                 train_head=train_head,
                 force_retrain=force_retrain
    )
    
    vis.visualize(wrapper)

if __name__ == "__main__":
    main()