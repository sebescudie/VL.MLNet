# VL.MLNet

[![Nuget](https://img.shields.io/nuget/vpre/VL.MLNet?style=flat-square)](https://www.nuget.org/packages/VL.MLNet)

_Allows to run pre-trained ML.NET models as nodes in vvvv gamma._

## Get it

Go to VL's command line and type

```
nuget install VL.MLNet -pre
```

For more information on how to install nugets in vvvv, please refere to [this section](https://thegraybook.vvvv.org/reference/libraries/referencing.html#manage-nugets) of the Gray Book.

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

## Pre-trained models

If you want to test the plugin right away without training your own models, you can try one of those I've trained and used during the development of the plugin :

| Model name     | Type                 | Description                                                                                                                                                                                                                                                     | Download link                                                              |
|----------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| CatDogOtter    | Image Classification | Trained to recognize either cats, dogs or otters                                                                                                                                                                                                                | [Here](http://sebescudie.fr/sharing/CatDogOtter_ImageClassification.zip)   |
| FakeNews       | Text Classification  | Trained over the [fake and real news](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) dataset. Will tell if an article, based on its title and content, appears to be fake news                                                              | [Here](http://sebescudie.fr/sharing/FakeNews_TextClassification.zip)       |
| OffensiveOrNot | Text Classification  | Trained over a subset of the [WikiDetoxAnnotated](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/BinaryClassification_SentimentAnalysis/SentimentAnalysis/Data) dataset. Will tell if a comment is offensive or not | [Here](http://sebescudie.fr/sharing/OffensiveOrNot_TextClassification.zip) |
| RedWineQuality | Regression           | Trained over the [Red wine quality](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) dataset. Will predict the quality of a red wine based on various inputs such as acidity, alcohol, etc                                                      | [Here](http://sebescudie.fr/sharing/RedWineQuality_Regression.zip)         |

Please note that when running the text classification models, you will get extra prediction labels that appear to be random excerpts of the input dataset. I'm not sure why this happens but I'm pretty confident it's because I made some mistakes when cleaning/merging CSV files before training.

## Known issues

For some reason, the factory does not retrigger when a new model is added to the `ml-models` subdir. For now, you have to kill and restart vvvv for new models to be taken into account.

## Credits

- Massive thanks to [azeno](https://github.com/azeno/) for his help on refactoring/optimizing the code
- Michael Hompus for [his voodoo code snippet](https://blog.hompus.nl/2020/09/14/get-all-prediction-scores-from-your-ml-net-model/) that allows to retrieve score labels for classification models
- Jonathan Crozier for his [DynamicTypeFactory](https://github.com/jonathancrozier/jc-samples-dynamic-properties/blob/master/JC.Samples.DynamicProperties/Factories/DynamicTypeFactory.cs) that gracefully spawns new classes at runtime

