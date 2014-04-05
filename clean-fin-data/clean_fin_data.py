#!/usr/bin/env python
# encoding: utf-8
"""
<script_name>.py

Created by Benjamin Gross on <insert date here>.

INPUTS:
--------

RETURNS:
--------

TESTING:
--------


"""

import argparse
import pandas
import numpy
import pandas.io.data

def save_data_to_hdf5(ticker_list, loc, start = '01/01/1990'):
    """
    Initialization to pull down data, and store in ``HDF5`` file format

    :ARGS:

        ticker_list: iterable for which you'd like to pull down financial data for

        loc: :class:`string` location to store the ``HDF5`` data type

        start: :class:`string` when you would like the price data to begin

    :RETURNS:

        saves a ``HDF5`` file format with tickers as key values
    """
    import pdb
    pdb.set_trace()
    reader = pandas.io.data.DataReader
    d = {}
    for ticker in ticker_list:
        try:
            d[ticker] = reader(ticker, 'yahoo', start = start)
            print "worked for " + ticker
        except:
            print "failed for " + ticker

    store = pandas.HDFStore(path = loc, mode = 'w')
    map(lambda x: store.put(x, d[x]), ticker_list)
    store.close()
    return None

def scipt_function(arg_1, arg_2):
	return None

if __name__ == '__main__':
	
	usage = sys.argv[0] + "usage instructions"
	description = "describe the function"
	parser = argparse.ArgumentParser(description = description, usage = usage)
	parser.add_argument('name_1', nargs = 1, type = str, help = 'describe input 1')
	parser.add_argument('name_2', nargs = '+', type = int, help = "describe input 2")

	args = parser.parse_args()
	
	script_function(input_1 = args.name_1[0], input_2 = args.name_2)
