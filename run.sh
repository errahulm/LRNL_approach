#!/usr/bin/env bash
echo "Epochs variations"
python weighted_avg.py 5 > noise30/results_5_epochs.txt
echo "5 epochs Complete"
python weighted_avg.py 10 > noise30/results_10_epochs.txt
echo "10 epochs Complete"
python weighted_avg.py 15 > noise30/results_15_epochs.txt
echo "15 epochs Complete"
python weighted_avg.py 20 > noise30/results_20_epochs.txt
echo "20 epochs Complete"
python weighted_avg.py 25 > noise30/results_25_epochs.txt
echo "25 epochs Complete"
python weighted_avg.py 30 > noise30/results_30_epochs.txt
echo "30 epochs Complete"
python weighted_avg.py 35 > noise30/results_35_epochs.txt
echo "35 epochs Complete"
python weighted_avg.py 40 > noise30/results_40_epochs.txt
echo "40 epochs Complete"
python weighted_avg.py 45 > noise30/results_45_epochs.txt
echo "45 epochs Complete"
python weighted_avg.py 50 > noise30/results_50_epochs.txt
echo "50 epochs Complete"
python weighted_avg.py 100 > noise30/results_100_epochs.txt
echo "100 epochs Complete"


