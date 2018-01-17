# PQ Allocation Neural Net #

## neuralnet.py ##

This application trains the neural net, using the Tensor flow library.

The training data is contained within the application folder, in a file named json2.text

json2.txt is a dictionary of lists, with each Unit in DfT as a key, and each PQ they have answered in the last 2 years in the list contained within

Lines 12-88 of neuralnet.py are getting the data into a format tensorflow can use, ie converting each question into a 1 dimensional matrix of 0s and 1s

Lines 89-99 are designing the neural net

Lines 104 - 109 instantiate the model, provides a location to save logs, runs it and save the state of the model upon completion (in the current directory)

Lines 111-137 perform some tests immediately after the model has been run

## allocator.py ##

This appliation reinstatiates the model and allows the user to input questions, and outputs the suggested unit
