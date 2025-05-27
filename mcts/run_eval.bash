#!/bin/bash

result_dir="../output/asqa"
dataset="asqa"
at_most_citations="3"

python eval.py --result_dir ${result_dir} --dataset ${dataset} --at_most_citations ${at_most_citations}