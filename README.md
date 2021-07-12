# MRI_cycGAN
UW-Madison ECE697 Sumer Project Codebase
The jupyter notbook contains my progress in model development in Tensorflow.
It has been ran in google colabs with the dataset stored in google drive.

To run the peer review script in bin/python_scripts make sure you have access to 
the follow libraries. tensorflow, numpy, nibabel, os, PIL. If you do not have access,
simply used the "pip3 'ex_lib'" command in Linux to install the library.

I have provided a small data subset to run the script on under the /data/ folder. The 
set contains multiple scans of a brain. 

TO RUN SCRIPT 

Navigate to /bin/python_script directory then run the pythong script using the following command
in Linix "python3 data_preprocess.py"

This script converts MRI and CT data to usable tensorflow datasets. It also implements
some preprocessing and displays 2 example images of a MRI and CT layer. 
