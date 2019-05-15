#!/usr/bin/env bash

cd ./DecisionTree/
python3 setup.py build
sudo python3 setup.py install
cd ../
python3 Main.py
