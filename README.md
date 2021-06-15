# VL.MLNet

_Allows to run pre-trained ML.NET models as nodes in vvvv gamma._

## Get it

For now, please clone/fork this repo in your [`package-repositories`](https://thegraybook.vvvv.org/reference/libraries/contributing.html#source-package-repositories) folder

##  Supported scenarios

- Regression
- Text classification
- Image classification

## Usage

This plugin allows to run pre-trained ML.NET models as nodes. For each model, consisting of a `zip` file, you get a node with the correct input and output pins.

- The first thing you should do is rename the column that will serve as a label in your dataset to `Label`. For instance, if you have a CSV that lists house prices depending on their size and number of floors, the column that actually contains the house price should be renamed `Label`
- Then, train your model using Visual Studio's [Model Builder plugin](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet/model-builder) (please note that only regression, text and image classification are supported for now)
- Once the training is complete, retrieve the `zip` file that's been created (you can look for its path in the code that has been automagically generated)
- Rename that file to a friendly name : this will be your node's name
- Append either one of those to the filename, depending on the scenario you picked :

  - `_Regression` for a regression model
  - `_TextClassification` for a text classification model
  - `_ImageClassification` for an image classification model
- Put this zip file in a folder named `ml-models` next to your VL document
- Start vvvv, you should see an `ML.MLNet` category in the node browser containing one node per `zip` file

## Known issues

For some reason, the factory does not retrigger when a new model is added to the `ml-models` subdir. For now, you have to kill and restart vvvv for new models to be taken into account.

## Credits

- Massive thanks to [azeno](https://github.com/azeno/) for his help on refactoring/optimizing the code
- Michael Hompus for [his voodoo code snippet](https://blog.hompus.nl/2020/09/14/get-all-prediction-scores-from-your-ml-net-model/) that allows to retrieve score labels for classification models
- Jonathan Crozier for his [DynamicTypeFactory](https://github.com/jonathancrozier/jc-samples-dynamic-properties/blob/master/JC.Samples.DynamicProperties/Factories/DynamicTypeFactory.cs) that gracefully spawns new classes at runtime

