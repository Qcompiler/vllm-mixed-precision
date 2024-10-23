
mkdir data; wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
tar --no-same-owner  -xf data/mmlu.tar -C data && mv data/data data/mmlu
