#clean-fin-data

A module to help you store, append, and scrub financial data using `HDF5` data structures
to ensure speed and efficiency


##Installation

###Clone the Repository

	$ git clone https://github.com/benjaminmgross/clean-fin-data
	$ cd clean-fin-data
	$ python setup.py install

###Test the install

	$ ipython
	Python 2.7.6 (default, Nov 12 2013, 10:54:02) 
	Type "copyright", "credits" or "license" for more information.

	IPython 2.0.0 -- An enhanced Interactive Python.
	?         -> Introduction and overview of IPython's features.
	%quickref -> Quick reference.
	help      -> Python's own help system.
	object?   -> Details about 'object', use 'object??' for extra details.

	In [1]: import clean_fin_data
	

##Dependencies

- `pandas` (obviously)
- `tables` (as in `PyTables`)

To install `tables` on Mac OS X Mavericks, you need [`HDF5`](http://www.hdfgroup.org/). I did the following to get `HDF5` functionality up and running (if you have Windows, it `PyTables` comes pre-installed with [`Anaconda`](https://store.continuum.io/cshop/anaconda/) and [`Enthought`](https://www.enthought.com/products/epd/), so you can skip this step):

1. You can't just `$ brew install hdf5` using [`homebrew`](http://brew.sh/), you actually need to go to [`Homebrew Science`](https://github.com/Homebrew/homebrew-science/) to get it.  So execute the following commands:

		$ brew tap homebrew/science
		
		#Then you can brew install it
		
		$ brew install hdf5
		
2. Now you've got the necessary components to install `PyTables` with `pip`:

		$ pip install tables
		
**NOTE:** Just because you have `pandas` installed doesn't mean you have `tables`, so if you try to execute any of this code and get:

	.. ERROR:: Could not find a local HDF5 installation.

You probably need to execute the steps just covered.


##To Do 
- Write the scrub functionality
- Write the append functionality
- Create some sort of testing mechanism to compare performance of read / write capacity of `cPickle` and `HDF5`.