# PhaseGNN

## Description:

More details regarding the network will be added here.

## Installation:
1. First, clone this repository to a local directory.

2. Next, install the conda environment required to use this model.

```
 conda env create --name chem_ml --file=environment.yml
```

3. Activate the conda environment.
```
conda activate chem_ml
```

4. To train a model, run the train_phasegnn.py file and give it the critical property (P_c or T_c) using the -c flag and number of epochs (default: 1000) using the -e flag. For example:
```
python train_phasegnn.py -c T_c -e 1000
```

5. To evaluate a trained model on a specified molecule, run the evaluate_phasegnn.py file. Give it a critical property (P_c or T_c) using the -c flag and molecule name (common name or IUPAC name, spaces are okay in name) using the -n flag. For example:
```
python evaluate_phasegnn.py -c T_c -e aniline
```

