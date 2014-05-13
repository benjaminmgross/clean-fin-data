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
from statsmodels.api import Logit
import matplotlib.pyplot as plt

def append_store_prices(ticker_list, loc, start = '01/01/1990'):
    """
    Given an existing store located at ``loc``, check to make sure
    the tickers in ``ticker_list`` are not already in the data
    set, and then insert the tickers into the store.

    :ARGS:

        ticker_list: :class:`list` of tickers to add to the
        :class:`pandas.HDStore`

        loc: :class:`string` of the path to the     
        :class:`pandas.HDStore`

        start: :class:`string` of the date to begin the price data

    :RETURNS:

        :class:`NoneType` but appends the store and comments the
         successes ands failures
    """
    try:
        store = pandas.HDFStore(path = loc, mode = 'r')
    except IOError:
        print  loc + " is not a valid path to an HDFStore Object"
        return
    store_keys = map(lambda x: x.split('/'), store.keys())
    not_in_store = numpy.setdiff1d(ticker_list, store_keys )

    new_prices = tickers_to_dict(not_in_store, start = start)
    map(lambda x: store.put(x, new_prices[x]), not_in_store)
    store.close()
    return None  

def find_jump_logit_method(logit_model, price_df, threshold = .95):
    """
    Use the calibrated logit model to determine the threshold for 
    dividends an splits

    :ARGS:

        logit_model: :class:`statsmodels.api.Logit` that has already been 
        fit to dividends and splits.

        .. seealso: `get_trained_logit_model()`

        price_df: :class:`pandas.DataFrame` that have at least the columns 
        'Close' and 'Adj Close'

        threshold: :class:`float` for the which, when a given value is  
        greater than that probability, is considered "non-white noise"

    :RETURNS:

        :class:`float` of the threshold for the ``ln_chg`` for the ratio 
        of that ``price_df`` ('Close'/'Adj Close') is considered a dividend 
        or split

    """
    
    ln_chg = price_df['Close'].div(price_df['Adj Close']).apply(
        numpy.log).diff()

    data = pandas.DataFrame({'ln_chg':ln_chg, 'intercept':1.0})    
    prob = logit_model.predict(data[['intercept', 'ln_chg']])

    # the "smallest" is the max because all ln chg are negative
    
    return ln_chg[prob > threshold].max()

def find_jump_time_interval(price_df, jump_method, logit_model = None):
    """
    Returns the median interval (in days) between jumps, using any of the 
    three ``find_jump_*`` methods

    :ARGS:

        price_df: :class:`pandas.DataFrame` with at least columns of 'Close' 
        and 'Adj Close'

        jump_method: :class:`string` of which jump method to use to find 
        the dividends and splits. Options are 'logit', 'vol', or 'wn'.

        logit_model: :class:`statsmodels.api.Logit` already fitted for 
        the data, in the case that ``jump_method = 'logit'`` is used as 
        the parameter

    :RETURNS:

        :class:`float` of the number of days between dividends and splits
    """
    thresh_dict = {'logit': lambda m, df: find_jump_vol_logit_method(m, df),
                   'vol':lambda m, df: find_jump_vol_method(df),
                   'wn': lambda m, df: find_jump_wn_method(df) }
    threshold = threshold_dict[jump_method](
        model = logit_model, df = price_df)

    ln_chg = price_df['Close'].div(price_df['Adj Close']).apply(
        numpy.log).diff()

    jumps = ln_chg[ ln_chg < threshold ]

    #find the distances between the jumps
    deltas = map(lambda x, y: x - y, jumps.index[1:], jumps.index[:-1] )
    day_deltas = map(lambda x: x.days, deltas)
    return numpy.median(day_deltas)

def find_jump_vol_method(price_df):
    """
    Determines the appropriate vol band to determine when dividends and 
    splits have occured, in the case of missing data

    :ARGS:

        price_df: :class:`pandas.DataFrame` that has columns of (at least) 
        'Close' and 'Adj Close'

    :RETURNS:

        :class:`float` of the minimum volatility threshold for which 
        jumps have occurred
    
    """
    def __get_jump_stats(price_df):
        """
        Helper function to determine the number of ratio changes that 
        are outside a given volatility band.  Brute force algorithm

        :ARGS:

            price_df: :class:`pandas.DataFrame` that has columns of (at 
            least) 'Close' and 'Adj Close'

        :RETURNS:

            :class:`pandas.DataFrame` with columns 'num_outside' and 
            'vol_band' showing the number of ratio changes that were 
            outside of a given volatility band
    
        """
        ratio = price_df['Close'].div(price_df['Adj Close']).apply(
            numpy.log).diff()
        vol_bands = numpy.linspace(.001, 2, 1000)
        bands = map(lambda x: len(ratio[(
            ratio.abs() > ratio.mean() + x*ratio.std() )]), vol_bands)
        return pandas.DataFrame({'num_outside': bands, 
                                 'vol_band':vol_bands})

    if price_df['Close'].equals(price_df['Adj Close']):
        print "No Dividends or Splits, Adj Close and Close are all Equal"
        return 0.0
    else:
        out_df = __get_jump_stats(price_df)
        #the max frequency for each of the vol bands is our bogey
        band_cnt = out_df['num_outside'].value_counts()
        band_cnt.sort(ascending = False)
    
        #determine the band, in vol units, where the jumps have occurred
        max_out = out_df['num_outside'].max()
        min_out =   out_df['num_outside'].min()
        threshold = band_cnt[(band_cnt.index != max_out) & (
            band_cnt.index != min_out)].max()
        thr_ind = band_cnt[band_cnt == threshold].index
        agg = out_df.loc[out_df.num_outside == thr_ind[0],  :]
        thresh = agg['vol_band'].max()
        ln_chg = price_df['Close'].div(price_df['Adj Close']).apply(
            numpy.log).diff()

        return ln_chg.mean() - thresh * ln_chg.std()

def find_jump_wn_method(price_df, threshold = .0001):
    """
    Another way to approach the dividend / split recognition problem 
    (instead of incrementing the volatility band) is to sort ``ln_chg`` 
    of the :math:`\\frac{\\textrm{Close}}{\\textrm{Adj Close}}` and 
    look for the first jump that occurs. That is the end of the white 
    noise and beginning of the data we're looking for

    :ARGS:

        price_df: :class:`pandas.DataFrame` with 'Close' and 'Adj Close'

    :RETURNS:

        the distance for which dividends or / and splits begin to occur
    """
    ln_chg = price_df['Close'].div(price_df['Adj Close']).apply(
        numpy.log).diff()

    abs_sorted = ln_chg.abs()
    abs_sorted.sort(ascending = True)
    jump_size = abs_sorted.diff()
    
    #the first jump over the threshold is our bogey, but need to 
    #transform back to the original ln_chg value
    try:
        return ln_chg[jump_size[jump_size > threshold].argmin()]
    except ValueError:
        print "No values within that threshold"
        return 0.0

def first_valid_date(prices):
    """
    Helper function to determine the first valid date from a set of 
    different prices Can take either a :class:`dict` of 
    :class:`pandas.DataFrame`s where each key is a ticker's 'Open', 
    'High', 'Low', 'Close', 'Adj Close' or a single 
    :class:`pandas.DataFrame` where each column is a different ticker

    :ARGS:

        prices: either :class:`dictionary` or :class:`pandas.DataFrame`

    :RETURNS:

        :class:`pandas.Timestamp` 
   """
    iter_dict = { pandas.DataFrame: lambda x: x.columns,
                  dict: lambda x: x.keys() } 

    try:
        each_first = map(lambda x: prices[x].dropna().index.min(),
                         iter_dict[ type(prices) ](prices) )
        return max(each_first)
    except KeyError:
        print "prices must be a DataFrame or dictionary"
        return

def gen_master_index(ticker_dict, n_min):
    """
    Because many tickers have missing data in one or two spots, this 
    function aggregates ``n_min`` indexes to determine a Master Index.

    :ARGS:

        ticker_dict: :class:`dictionary` with tickers for keys and price 
        data for values

        n_min: :class:integer of the minimum number of indexes to use 
        before "all dates" have been determined

    :RETURNS:

        :class:`pandas.DatetimeIndex` of a Master Index
    """
    
    #create a sorted Series of the tickers and the dates they begin
    dt_starts = pandas.Series( 
        map(lambda x: ticker_dict[x].dropna().index.min(),
        ticker_dict.keys()), index = ticker_dict.keys() )
    dt_starts.sort()

    #check to make sure the tickers end on the same date
    end_dates = pandas.Series(
        map( lambda x: ticker_dict[x].dropna().index.max(),
        ticker_dict.keys()), index = ticker_dict.keys() )

    end_dates = end_dates[dt_starts.index]
    
    if (end_dates[:n_min] == end_dates[0]).all() == False:
        print "final dates aren't the same, Master Index won't work"
        return

    else:
        #find the union of the first n_min indexes
        mi = reduce(lambda x, y: x | y, map(
            lambda x: ticker_dict[x].dropna().index,
            dt_starts[:n_min].index) )
    return mi


def get_divs_and_splits(price_df, jump_method, logit_model = None):

    thresh_dict = {
        'logit': lambda model, df: find_jump_vol_logit_method(model, df),
        'vol':lambda model, df: find_jump_vol_method(df),
        'wn': lambda model, df: find_jump_wn_method(df)
                   }

    threshold = threshold_dict[jump_method](model = logit_model, 
                                            df = price_df)

    ln_chg = price_df['Close'].div(price_df['Adj Close']).apply(
        numpy.log).diff()

    return ln_chg[ln_chg < thresh]

def get_trained_logit_model():
    """
    In 'data/traning_data/' specific ETFs were visually inspected and 
    white noise (0) and not white noise (1) were assigned. This data is 
    loaded here to train the logistic parameters, but you can use this 
    functionality as a template to train your own

    :ARGS:

        :class:`NoneType`

    :RETURNS:

        a fitted :class:`statsmodels.Logit` Logistic regression that 
        has been fit to the trained data
    """
    f = pandas.ExcelFile('../data/training_data/Trained Data.xlsx')
    data = reduce(lambda a, b: numpy.vstack([ a, b]), map(
        lambda x: f.parse(x, index_col = 0)[['ln_chg', 'Y']], f.sheet_names))
    data = pandas.DataFrame(data, columns = ['ln_chg', 'Y'])

    #add an intercept for the model (required by statsmodels.api.Logit
    data['intercept'] = 1.0

    #fit the model
    logit_model = Logit(endog = data['Y'], 
                        exog = data[['intercept', 'ln_chg']])
    return logit_model.fit()

def initialize_data_to_store(ticker_list, loc, 
                             start = '01/01/1990'):
    """
    Initialization to pull down data, and store in ``HDF5`` file 
    format.  This function should be used "the first time" a 
    ``store`` is created, later using different update 
    functionality.  Function will exist if a file with that name 
    already exists.

    :ARGS:

        ticker_list: iterable for which you'd like to pull down 
        financial data for

        loc: :class:`string` location to store the ``HDF5``data  
        type

        start: :class:`string` when you would like the price data 
        to begin

    :RETURNS:

        saves a ``HDF5`` file format with tickers as key values

    .. note:: Will Not Overwite Existing File

        If a file already exists with that path, the function will
        not overwite that file -- and therefore must be deleted 
        manually outside of the program in order for the function
        to run.

    .. note:: Master Index

        Each store that is created gets a key of "Master Index" 
        that indicates
        
    """
    if os.path.isfile(loc) == False:
        d = tickers_to_dict(ticker_list)
        master_index = gen_master_index(d, n_min = 5)
        store = pandas.HDFStore(path = loc, mode = 'w')
        map(lambda x: store.put(x, d[x] ), ticker_list)
        master_index = pandas.Series(index = master_index)
        store.put("master_index", master_index)
        store.close()
    else:
        print "A file already exists in that location"
    return None

def is_key_in_store(loc, key):
    """
    A quick check to determine whether the :class:`pandas.HDFStore` 
    has datA for ``key``

    :ARGS:

        loc: :class:`string` of path to :class:`pandas.HDFStore`

        key: :class:`string` of the ticker to check if currently 
        available

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

def load_logit_model(path = '../data/training_data/logit_model'):
    """
    Use the :module:`pickle` module to load the saved logit file
    """
    return pickle.load(open(path, 'r') )

def save_logit_model(logit_model, path):
    """
    Use the :module"`pickle` module to save the logit regression model 
    (to prevent needing to constantly reload and recalculate the model)
    """
    return pickle.dump(logit_model, open(path, 'w') )

def test_jump_detection_methods(ticker_list):
    """
    To determine the efficacy of the algorithms by looking at ``ln_chg``
    graphs of several different tickers and see where each threshold is
    drawn for the two different algorithms.

    This function iterates over a givenlist of tickers, ``ticker_list`` 
    while asking the user for input whether or not the algorithm 
    "appeared to work," and returns the yes / no results for each 
    ticker into a :class:`pandas.DataFrame`.

    :ARGS:

        tick_list: :class:`list` or iterable of tickers to test the 
        dividend and split identification algorithm

    :RETURNS:

        :class:`pandas.DataFrame` of tickers and whether or not the 
        algorithm worked for each ticker based on the user responses.
    """
    d = {}
    logit_model = get_trained_logit_model()
    for ticker in ticker_list:
        incomplete = True
        price_df = tickers_to_dict(ticker)
        vol_height = find_jump_vol_method(price_df)
        wn_height = find_jump_wn_method(price_df)
        logit_height = find_jump_logit_method(logit_model, price_df)
        lims = (-.01, .01)

        while incomplete:
            try:
                fig = plt.plot()
                ln_chg.plot(label = 'ln_chg')
                plt.axhline(y = vol_height, color = 'r', ls = '--',
                            label = "vol band method")
                plt.axhline(y = wn_height, color = 'c', ls = '--',
                            label = "white noise method")
                plt.axhline(y = logit_height, color = 'g', ls = '--',
                            label = "trained logit method")
                plt.title(ticker, fontsize = 16)
                plt.legend(frameon = False)
                plt.ylim(lims)
                plt.show()
                resp = raw_input(
                    "Do the limits need to be changed? y/n ")
                
                if resp == 'y':
                    lims = raw_input(
                        "What are the new limits? ymin, ymax ")
                    lims = map(lambda x: float(x), lims.split(',') )
                else:
                    didwork = raw_input("Did the algorithm work? y/n " )
                    d[ticker] = didwork
                    incomplete = False
            
            except TypeError:
                print "There are no Dividends or Splits to Illustrate"
                d[ticker] = "No Div or Splits"
                incomplete = False
    return pandas.DataFrame(d)

def tickers_to_dict(ticker_list, api = 'yahoo', start = '01/01/1990'):
    """
    Utility function to return ticker data where the input is either a 
    ticker, or a list of tickers.

    :ARGS:

        ticker_list: :class:`list` in the case of multiple tickers or 
        :class:`str` in the case of one ticker

        api: :class:`string` identifying which api to call the data 
        from.  Either 'yahoo' or 'google'

        start: :class:`string` of the desired start date
                
    :RETURNS:

        :class:`dictionary` of (ticker, price_df) mappings or a
        :class:`pandas.DataFrame` when the ``ticker_list`` is 
        :class:`str`
    """
    def __get_data(ticker, api, start):
        reader = pandas.io.data.DataReader
        try:
            data = reader(ticker, api, start = start)
            print "worked for " + ticker
            return data
        except:
            print "failed for " + ticker
            return
    if isinstance(ticker_list, (str, unicode)):
        return __get_data(ticker_list, api = api, start = start)
    else:
        d = {}
        for ticker in ticker_list:
            d[ticker] = __get_data(ticker, api = api, start = start)
    return d

def update_store_prices(loc):
    """
    Update to the most recent prices for all keys of an existing store, 
    located at path ``loc``.

    :ARGS:

        loc: :class:`string` the location of the ``HDFStore`` file

    :RETURNS:

        :class:`NoneType` but updates the ``HDF5`` file, and prints to 
        screen which values would not update

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

                #need to drop duplicates because there's 1 row of 
                #overlap
                store.put(key, stored_data.append(tmp).drop_duplicates())
            except IOError:
                print "could not update " + key

    store.close()
    return None

def clean_existing_data(loc):
    hdf_store = pandas.HDFStore(path = loc, mode = 'r+')
    return None
    
def clean_df_data_gaps(price_frame, master_index, ticker):
    
    #the starting date is the more recent of the master index start or 
    #stock start
    d_o = max([price_frame.dropna().index.min(), master_index.min()])

    #if there are no gaps, somply return and don't alter the data
    if price_frame.loc[d_o:].index.equals(
            master_index[master_index.get_loc(d_o):]):
        print "No data gaps found for " + ticker
        return
    #otherwise, fill the gaps with Google data
    else:
        plug_data = tickers_to_dict(ticker, api = 'google', start = d_o)

        #find the gaps

        #fill the gaps
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

if __name__ == '__main__':
    
    usage = sys.argv[0] + "usage instructions"
    description = "describe the function"
    parser = argparse.ArgumentParser(description = description, 
                                     usage = usage)
    parser.add_argument('name_1', nargs = 1, type = str, 
                        help = 'describe input 1')

    parser.add_argument('name_2', nargs = '+', type = int,
                        help = "describe input 2")
    args = parser.parse_args()
    
    script_function(input_1 = args.name_1[0], input_2 = args.name_2)
