Stephen Reilly 201527474

CA2 is an implementation of the kmeans and kmedians algorithms from scratch. CA2.py was created and run with Python 3.9. on MacOS 11 BigSur.

To run CA2.py there 2 libraries that need be installed.

Numpy will need to be installed on your system, if you do not have numpy installed it can be installed by running in the terminal(depending on OS and python version):
	pip3 install numpy 
		or
	pip install numpy

When running the program numpy has been coded to use seed 42 within each class if you wish to change this value you can pass it as a parameter when creating an instance of each class.

Some code that was used for visualisation of the data have been commented out so they do not run when you run the program. If you wish to have these run on your system you will need to have matplotlib installed on your system, if you do not have matplotlib installed it can be installed in the terminal:
	pip3 install matplotlib
		or
	pip install matplotlib

The test and train data should be kept in the same directory as the .py file and the CA2data folder should be unzipped but the files should be left inside. If you wish to run the files from a different directory you will need to update the location in lines 320-323 of the .py file.

The Program is built using 2 classes with the following method:
	
	KMeans.fit - Used to cluster data with given k using kmeans algorithm

	KMedians.fit - Used to cluster data with given k using kmedians algorithm

The program also has helper functions:

	load_data - takes raw data and converts it into useable forms for the algorithms

	make_data - splits the data into raw data, label classifications and normalised data.

	l2_norm - used to normalise the data

	bscores -  used to compute the BCUBED scores for each model
