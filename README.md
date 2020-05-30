- Step 1
  Modify the config.json file, and specify the required paths correctly.

- Step 2
  Run `extract_features.py` to generate the `feature vector`. Run `extract_features --help` to know about the cli arguments.

- Step 3
  Run `classify.py` to run ML models like xgboost and random forest from the generated `input vector`s. Run `classify.py --help` to know about the cli arguments.
