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

def __tickers_to_dict(ticker_list, start = '01/01/1990'):
    """
    Utility function to return a dictionary of (ticker, price_df) mappings
    """
    reader = pandas.io.data.DataReader
    d = {}
    for ticker in ticker_list:
        try:
            d[ticker] = reader(ticker, 'yahoo', start = start)
            print "worked for " + ticker
        except:
            print "failed for " + ticker
    return d

def append_store_prices(ticker_list, loc, start = '01/01/1990'):
    """
    Given an existing store located at ``loc``, check to make sure the
    tickers in ``ticker_list`` are not already in the data set, and then
    insert the tickers into the store.

    :ARGS:

        ticker_list: :class:`list` of tickers to add to the :class:`pandas.HDStore`

        loc: :class:`string` of the path to the :class:`pandas.HDStore`

        start: :class:`string` of the date to begin the price data

    :RETURNS:

        :class:`NoneType` but appends the store and comments the successes
        ands failures
    """
    try:
        store = pandas.HDFStore(path = loc, mode = 'r')
    except IOError:
        print  loc + " is not a valid path to an HDFStore Object"
        return
    store_keys = map(lambda x: x.split('/'), store.keys())
    not_in_store = numpy.setdiff1d(ticker_list, store_keys )

    new_prices = __tickers_to_dict(not_in_store, start = start)
    map(lambda x: store.put(x, new_prices[x]), not_in_store)
    store.close()
    return None     
    
def save_data_to_store(ticker_list, loc, start = '01/01/1990'):
    """
    Initialization to pull down data, and store in ``HDF5`` file format.  This
    function should be used "the first time" a ``store`` is created, later using
    different update functionality.

    :ARGS:

        ticker_list: iterable for which you'd like to pull down financial data for

        loc: :class:`string` location to store the ``HDF5`` data type

        start: :class:`string` when you would like the price data to begin

    :RETURNS:

        saves a ``HDF5`` file format with tickers as key values
    """
    d = __tickers_to_dict(ticker_list)
    store = pandas.HDFStore(path = loc, mode = 'w')
    map(lambda x: store.put(x, d[x]), ticker_list)
    store.close()
    return None

def clean_existing_data(loc):
    hdf_store = pandas.HDFStore(path = loc, mode = 'r+')
    return None
    
def update_store_prices(loc):
    """
    Take an existing store at path ``loc`` and loop through the key values,
    intelligently updating the price data if a more recent price exists according to
    prices from ``Yahoo!``

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

                #need to drop duplicates because there's 1 row of overlap
                store.put(key, stored_data.append(tmp).drop_duplicates())
            except IOError:
                print "could not update " + key

    store.close()
    return None

def check_store_for_key(loc, key):
    """
    A quick check to determine whether the :class:`pandas.HDFStore` has data
    for ``key``

    :ARGS:

        loc: :class:`string` of path to :class:`pandas.HDFStore`

        key: :class:`string` of the ticker to check if currently available

    :RETURNS:

        whether ``key`` is currently a part of the data set
    """
    try:
        store = pandas.HDFStore(path = loc, mode = 'r')
    except IOError:
        print  loc + " is not a valid path to an HDFStore Object"
        return
    
    store_keys = store.keys()
    store.close()
    return key in map(lambda x: x.strip('/'), store_keys )
    
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
