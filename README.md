The main file is catenator.py

Get the required modules by running pip3 install -r requirements.txt

You can get all the help information by running python3 catenator.py --help

You run in the follwoing order: 

1) python3 catenator.py configure # This gives you 2 options test and v1.  
2) python3 catenator.py download # This can take time depending on number of files that need to be downloaded
3) python3 catenator.py process # Runs PyBDSF image processing
4) python3 catenator.py vlad # create vlad subtile-based catalogues
5) python3 catenator.py diagnaostics # create/overwrite diagnostic QA files
6) python3 catenator.py qa # Adds QA flags to VLAD abd deploys BDPs
7) python3 catenator.py upload # Uploads BDPs to CIRADA database
8) python3 catenator.py flush # pipeline cleanup utility
9) python3 catenator.py logger # log file management utility
10) python3 catenator.py monitor # Pipeline monitoring utility

The logs are in data/logs directory

You can python3 catenator.py monitor to check status after running any of the above commands.


