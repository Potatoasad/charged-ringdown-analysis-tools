#!/bin/bash

events_folder="./Config_files/*.ini"

for file in ./Config_files/*ini 
do
	echo ${file}
	python -Wignore analysis.py -c ${file} --diagnosticsonly
	python -Wignore analysis.py -c ${file} --charged --exactcharge
done
