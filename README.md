# process_cuwb_data

Tools for reading, processing, and writing CUWB data

### Steps

1. Copy `.env.template` to `.env` and update variables


2. Install packages

    `just build`

    You may need to install pybind11 `brew install pybind11`
    And you may need to manually install cython and numpy `pip install numpy cython pythran`


3. (Optional) Train Tray Detection Model
    1. Download/create [ground_truth_tray_carry.csv](https://docs.google.com/spreadsheets/d/1NLQ_7Cj432T1AXFcbKLX3P6LGGGRAklxMTjtBYS_xhA/edit?usp=sharing) to `./downloads/ground_truth_tray_carry.csv`
```
    curl -L 'https://docs.google.com/spreadsheets/d/1NLQ_7Cj432T1AXFcbKLX3P6LGGGRAklxMTjtBYS_xhA/export?exportFormat=csv' --output ./downloads/ground_truth_tray_carry.csv
```
3.
   2. Generate pickled groundtruth features dataframe from ground_truth_tray_carry.csv
```
    process_cuwb_data \
        generate-tray-carry-groundtruth \
        --groundtruth-csv ./downloads/ground_truth_tray_carry.csv
```

3.
    3. Train and pickle Tray Carry Detection Model using pickled groundtruth features

```
    process_cuwb_data \
        train-tray-carry-model \
        --groundtruth-features ./output/groundtruth/2021-05-13T12:53:26_tray_carry_groundtruth_features.pkl
```

4. Infer Tray Interactions using pickled Tray Carry Detection Model
    1. Use the model you've trained by following step 3
    2. Or, download the latest model:
```
    curl -L 'https://drive.google.com/uc?export=download&id=1_veyjLdAa8Fq7eYeT9GLdkcS6_VY0FLX' --output ./output/models/tray_carry_model_v1.pkl
```   

Then use the model to infer tray interactions:
```
    process_cuwb_data \
      infer-tray-interactions \
      --environment greenbrier \
      --start 2021-04-20T9:00:00-0500 \
      --end 2021-04-20T9:05:00-0500 \
      --tray-carry-model ./output/models/tray_carry_model_v1.pkl
```

### Other CLI Commands/Options

#### Export pickled UWB data

Working with Honeycomb's UWB endpoint can be painfully slow. For that reason there is an option to export pickled UWB data and provide that to subsequent inference commands.

        process_cuwb_data \
            fetch-cuwb-data \
            --environment greenbrier \
            --start 2021-04-20T9:00:00-0500 \
            --end 2021-04-20T9:05:00-0500


#### Use UWB export to run Tray Interaction Inference

        process_cuwb_data \
            infer-tray-interactions \
            --environment greenbrier \
            --start 2021-04-20T9:00:00-0500 \
            --end 2021-04-20T9:05:00-0500 \
            --tray-carry-model ./output/models/2021-05-13T14:49:32_tray_carry_model.pkl \
            --cuwb-data ./output/uwb_data/uwb-greenbrier-20210420-140000-20210420-140500.pkl

#### Supply Pose Track Inference to Tray Interaction Inference

Use Pose Tracks when determining nearest person to tray carry events.

Pose Inferences need to be sourced in a local directory. The pose directory can be supplied via CLI options.

        process_cuwb_data \
            infer-tray-interactions \
            --environment greenbrier \
            --start 2021-04-20T9:00:00-0500 \
            --end 2021-04-20T9:05:00-0500 \
            --tray-carry-model ./output/models/2021-05-13T14:49:32_tray_carry_model.pkl \
            --cuwb-data ./output/uwb_data/uwb-greenbrier-20210420-140000-20210420-140500.pkl \
            --pose-inference-id 3c2cca86ceac4ab1b13f9f7bfed7834e

### Development

#### MacOS (Monterey)

1) Install **pyenv**:


    brew install pyenv

2) Create a 3.x venv:


    pyenv virtualenv 3.x.x wf-process-cuwb-data

3) Set ENV


This was helpful after the transition from macOS X to macOS 11. As packages evolve, this may not be needed depending on exact Mac version and python version. e.g. macOS 12 with Python 3.10 doesn't require the env var


    export SYSTEM_VERSION_COMPAT=1

4) Install scipy dependencies:

    
    brew install openblas lapack pythran pybind11
    
    export OPENBLAS=$(brew --prefix openblas)
    export CFLAGS="-falign-functions=8 ${CFLAGS}"

5) Install python libraries


    pip install numpy cython pythran

6) Install add'l packages:


    just install-dev

