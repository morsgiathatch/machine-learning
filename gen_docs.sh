#!/usr/bin/env bash
rm -r ./documentation/
mkdir ./documentation/
sphinx-apidoc -F -o ./documentation . *Test*
cd ./documentation/
sed -i "s/# import os/import os/" ./conf.py ./conf.py
sed -i "s/# import sys/import sys/" ./conf.py ./conf.py
sed -i "s/# sys.path.insert/sys.path.insert/" ./conf.py ./conf.py
make html
cd ../