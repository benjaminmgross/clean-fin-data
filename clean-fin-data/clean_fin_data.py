#!/usr/bin/env python
# encoding: utf-8

"""
.. module:: clean_fin_data.py
   :synopsis: Download, save, append, and scrub financial data

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>
"""

import argparse
import datetime
import numpy
import pandas
import os
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

def __gen_master_index(ticker_dict, n_min):
    """
    Because many tickers have missing data in one or two spots, this function
    aggregates ``n_min`` indexes to determine a Master Index ("MI").

    :ARGS:

        ticker_dict: :class:`dictionary` with tickers for keys and price data for
        values

        n_min: :class:integer of the minimum number of indexes to use before "all
        dates" have been determined

    :RETURNS:

        :class:`pandas.DatetimeIndex` of a Master Index
    """
    
    #create a sorted Series of the tickers and the dates they begin
    dt_starts = pandas.Series( map(lambda x: ticker_dict[x].dropna().index.min(),
        ticker_dict.keys()), index = ticker_dict.keys() )
    dt_starts.sort()

    #check to make sure the tickers end on the same date (or MI doesn't work)
    end_dates = pandas.Series( map( lambda x: ticker_dict[x].dropna().index.max(),
                     ticker_dict.keys()), index = ticker_dict.keys() )

    end_dates = end_dates[dt_starts.index]

    if (end_dates[:n_min][1:] != end_dates.shift(1)).all():
        print "Terminal Values are not the same and Master Index won't work"
        return

    else:
    #find the union of the first n_min indexes
    mi = reduce(lambda x, y: x & y, map(lambda x: ticker_dict[x].dropna().index,
                                          dt_starts[:n_min].index) )
    return mi
    

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
    
def initialize_data_to_store(ticker_list, loc, start = '01/01/1990'):
    """
    Initialization to pull down data, and store in ``HDF5`` file format.  This
    function should be used "the first time" a ``store`` is created, later using
    different update functionality.  Function will exist if a file with that name
    already exists.

    :ARGS:

        ticker_list: iterable for which you'd like to pull down financial data for

        loc: :class:`string` location to store the ``HDF5`` data type

        start: :class:`string` when you would like the price data to begin

    :RETURNS:

        saves a ``HDF5`` file format with tickers as key values

    .. note:: Will Not Overwite Existing File

        If a file already exists with that path, the function will not overwite
        that file -- and therefore must be deleted manually outside of the program
        in order for the function to run.

    """
    if os.path.isfile(loc) == False:
        d = __tickers_to_dict(ticker_list)
        store = pandas.HDFStore(path = loc, mode = 'w')
        map(lambda x: store.put(x, d[x]), ticker_list)
        store.close()
    else:
        print "A file already exists in that location"
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
