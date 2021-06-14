# VL.MLNet

_Allows to run pre-trained ML.NET as nodes in vvvv gamma._

## Usage

For now, please clone/fork this repo in your [`package-repositories`](https://thegraybook.vvvv.org/reference/libraries/contributing.html#source-package-repositories) folder

##  Supported scenarios

- Regression
- Text classification

## Dataset preparation

The only thing you should do to your dataset is to change the name of the column holding the label to predict to `Label`. This way, the node factory knows which columns it should exclude from the input pins.