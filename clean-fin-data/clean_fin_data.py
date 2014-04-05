#!/usr/bin/env python
# encoding: utf-8
"""
.. module:: clean_fin_data.py
   :synopsis: Download, save, append, and scrub financial data

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>
"""

import argparse
import datetime
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

def clean_existing_data(loc):
    hdf_store = pandas.HDFStore(path = loc, mode = 'r+')

    return None

def append_existing_data(loc):
    """
    Take an existing :class:`pandas.HDFStore` located at ``loc`` and loop
    through the key values, intelligently updating the price data if a more
    recent price exists from ``Yahoo!``

    :ARGS:

        loc: :class:`string` the location of the ``HDFStore`` file

    :RETURNS:

        :class:`NoneType` but updates the ``HDF5`` file, and prints to screen
        which values would not update

    """
    reader = pandas.io.data.DataReader
    strftime = datetime.datetime.strftime
    
    today_str = strftime(datetime.datetime.today(), format = '%m/%d/%Y')
    try:
        store = pandas.HDFStore(path = loc, mode = 'r+')
    except IOError:
        print  loc + " is not a valid path to an HDFStore Object"
        return

    for key in store.keys():
        stored_data = store.get(key)
        last_stored_date = stored_data.dropna().index.max()
        today = datetime.datetime.date(datetime.datetime.today())
        if last_stored_date < pandas.Timestamp(today):
            try:
                tmp = reader(key.strip('/'), 'yahoo', start = strftime(
                    last_stored_date, format = '%m/%d/%Y'))
                store.put(key, stored_data.append(tmp).drop_duplicates())
            except IOError:
                print "could not update " + key

    store.close()
    return None


def prep_append_test_data():
    tickers = ['AGG', 'LQD', 'IYR', 'EEM', 'EFA', 'IWV']
    path = '../data/test.h5'
    save_data_to_hdf5(tickers, loc = path , start = '01/01/1990')
    store = pandas.HDFStore(path = path, mode = 'r+')

    #take off the last 10 days of each key store
    for key in store.keys():
        tmp = store.get(key)
        store.put(key, tmp.iloc[:-10, :])

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
