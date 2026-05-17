# [Re] Optimal Play of the Dice Game Pig
## STOR609 Coursework 3

This project aims to replicate the results of `Optimal Play of the Dice Game Pig' by Tod Neller and Clifton Presser [1](#ref1). The scope of this project involves reproducing the optimal policy and plots thereof which appear in the original paper for the games Piglet and Pig.

## Using This Repository

### Installation as Editor
This installation allows full exploration of the reproducibility project including the ability to run unit tests and further develop the results. To exclusively install and use the package, please see the next section.

```
git clone https://github.com/C-Rigby/Game_of_Pig_Assessment_3.git
cd Game_of_Pig_Assessment_3
```
On Unix with pyenv running, proceed by creating a virtual environment
```
python -m venv env
source env/bin/activate
```
Install package & dependencies
```
python -m pip install -r requirements.txt
```
Run unit tests to confirm everything is working as intended
```
python -m pytest
```
Now feel free to explore and edit the repository. When finished, the virtual environment should be deactivated
```
deactivate
```

### Installing the Package

To simply use the package without the ability to edit or explore its functionality, install it for use in python by
```
python -m pip install 'https://github.com/C-Rigby/Game_of_Pig_Assessment_3.git'
```

## Repository Structure

The `examples/` folder contains notebooks illustrating how to use the code and play both Pig and Piglet. These should be followed as a guide for anyone attempting to use the package themselves.

The `experiments/` folder contains files used to replicate each of the results featured in the original paper. Due to the nature of the replication objectives, there is some overlap here with the examples.

The `src/` folder contains the package and license. which can be used by future readers to explore the optimal policy under various parameterisations.

The `test/` folder contains the unit tests to verify the package is functioning as intended.

The `article/` folder contains the files needed to create our report alongside the original article.



## Computation Times

Computation time will vary depending on computer performance however run times found in the example notebooks are indicative of relative workload. All files were run under the following specifications.

Operating System: Ubuntu 22.04.1
CPU: Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz x4
RAM: 16Gb


## Creators
Rick Cheng, James Marriner, Charlotte Rigby, Maria-Louiza Van den Bergh

## Version
This package was created using Python 3.10.8. Dependency requirements can be found in the file `requirements.txt`.

## References
[1] <a id="ref1"></a>
[Neller, Todd W. and Clifton G.M. Presser. "Optimal Play of the Dice Game Pig," The UMAP Journal 25.1 (2004), 25-47](https://cupola.gettysburg.edu/csfac/4/)

