using System;
using System.IO;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Collections.Immutable;
using VL.Core;
using Microsoft.ML;
using System.Linq;
using System.Reflection;

namespace VL.ML
{
    public class MLNetRunnerNodeFactory : IVLNodeDescriptionFactory
    {
        const string mlSubDir = "ml-models";
        const string identifier = "VL.MLNet-Factory";

        public readonly string Dir;
        public readonly string DirToWatch;

        private readonly NodeFactoryCache factoryCache = new NodeFactoryCache();
        
        public MLNetRunnerNodeFactory(string directory = default, string directoryToWatch = default)
        {
            Dir = directory;
            DirToWatch = directoryToWatch;

            var builder = ImmutableArray.CreateBuilder<IVLNodeDescription>();
            
            if(directory != null)
            {
                if(Directory.Exists(directory))
                {
                    var preTrainedModelFiles = Directory.GetFiles(directory, "*.zip");

                    foreach(string preTrainedModelFile in preTrainedModelFiles)
                    {
                        builder.Add(new MLNetRunnerNodeDescription(this, preTrainedModelFile));
                    }
                }
                else
                {
                    Console.WriteLine("ML subdirectory does not exist");
                }
            }
            
            NodeDescriptions = builder.ToImmutable();
        }

        public ImmutableArray<IVLNodeDescription> NodeDescriptions { get; }

        public string Identifier
        {
            get
            {
                if (Dir != null)
                    return GetIdentifierForPath(Dir);
                else
                    return identifier;
            }
        }

        // Fires when something has been added to the watched dir
        public IObservable<object> Invalidated
        {
            get
            {
                if(Dir != null)
                {
                    return NodeBuilding.WatchDir(Dir).Where(e => string.Equals(e.Name, mlSubDir, StringComparison.OrdinalIgnoreCase));
                }
                else if(DirToWatch != null)
                {
                    return NodeBuilding.WatchDir(DirToWatch).Where(e => e.Name == mlSubDir);
                }
                else
                {
                    return Observable.Empty<object>();
                }
            }
        }

        public void Export(ExportContext exportContext)
        {

        }

        string GetIdentifierForPath(string path) => $"{identifier} ({path})";

        public IVLNodeDescriptionFactory ForPath(string path)
        {
            var identifier = GetIdentifierForPath(path);
            return factoryCache.GetOrAdd(identifier, () =>
            {
                var mldir = Path.Combine(path, mlSubDir);
                if (System.IO.Directory.Exists(mldir))
                {
                    return new MLNetRunnerNodeFactory(mldir);
                }
                return new MLNetRunnerNodeFactory(directoryToWatch: path);
            });
        }
    }

    class MyNode : VLObject, IVLNode
    {
        class MyPin : IVLPin
        {
            public object Value { get; set; }
            public Type Type { get; set; }
            public string Name { get; set; }
        }

        readonly MLNetRunnerNodeDescription description;

        // Type factory stuff
        Type inputType;
        Type outputType;
        List<DynamicProperty> inputTypeProperties = new List<DynamicProperty>();
        List<DynamicProperty> outputTypeProperties = new List<DynamicProperty>();
        
        // Prediction engine
        dynamic predictionEngine;

        public MyNode(MLNetRunnerNodeDescription description, NodeContext nodeContext) : base(nodeContext)
        {
            this.description = description;
            Inputs = description.Inputs.Select(p => new MyPin() { Name = p.Name, Type = p.Type, Value = p.DefaultValue }).ToArray();
            Outputs = description.Outputs.Select(p => new MyPin() { Name = p.Name, Type = p.Type, Value = p.DefaultValue }).ToArray();

            MLContext = description.MLContext;

            #region Type Generation
            var factory = new DynamicTypeFactory();

            var inputTypeProperties = new List<DynamicProperty>();
            var outputTypeProperties = new List<DynamicProperty>();

            Type type = typeof(object);

            // Look at the pre-trained model and generate a type from it
            // Do not generate it based on the input pins
            // because they won't match what ML.NET expects
            if (description.FModelType == "Classification")
            {
                // Look at the model and retrieve all inputs that are not Label
                // Use hardcoded class for output
                foreach (var inputColumn in description.PredictionPipeline)
                {
                    GetType(inputColumn, ref type);

                    inputTypeProperties.Add(new DynamicProperty
                    {
                        PropertyName = inputColumn.Name,
                        DisplayName = inputColumn.Name,
                        SystemTypeName = type.ToString()
                    }) ;
                }

                foreach (var outputColumn in description.TrainedModel.GetOutputSchema(description.PredictionPipeline))
                {
                    GetType(outputColumn, ref type);

                    outputTypeProperties.Add(new DynamicProperty
                    {
                        PropertyName = outputColumn.Name,
                        DisplayName = outputColumn.Name,
                        SystemTypeName = type.ToString()
                    });
                }

                inputType = factory.CreateNewTypeWithDynamicProperties(typeof(object), inputTypeProperties);
                outputType = typeof(TextClassificationOutput);
            }
            else if(description.FModelType == "Regression")
            {
                foreach (var inputColumn in description.PredictionPipeline)
                {
                    GetType(inputColumn, ref type);

                    inputTypeProperties.Add(new DynamicProperty
                    {
                        PropertyName = inputColumn.Name,
                        DisplayName = inputColumn.Name,
                        SystemTypeName = type.ToString()
                    }) ;
                }

                var scoreColumn = description.TrainedModel.GetOutputSchema(description.PredictionPipeline).FirstOrDefault(o => o.Name == "Score");
                
                GetType(scoreColumn, ref type);
                
                outputTypeProperties.Add(new DynamicProperty
                {
                    PropertyName = scoreColumn.Name,
                    DisplayName = scoreColumn.Name,
                    SystemTypeName = type.ToString()
                });

                inputType = factory.CreateNewTypeWithDynamicProperties(typeof(object), inputTypeProperties);
                outputType = factory.CreateNewTypeWithDynamicProperties(typeof(object), outputTypeProperties);

            }
            else if(description.FModelType == "ImageClassification")
            {
                return;
            }
            else
            {
                return;
            }

            // Spawn instances of those
            // We actually just use those to spawn the dynamic prediction engine
            // Could we re-use them instead?
            var inputObject = Activator.CreateInstance(inputType);
            var outputObject = Activator.CreateInstance(outputType);
            
            #endregion TypeGeneration

            #region Prediction Engine
            // Create prediction engine
            var genericPredictionMethod = description.MLContext.Model.GetType().GetMethod("CreatePredictionEngine", new[] { typeof(ITransformer), typeof(DataViewSchema) });
            var predictionMethod = genericPredictionMethod.MakeGenericMethod(inputObject.GetType(), outputObject.GetType());
            predictionEngine = predictionMethod.Invoke(description.MLContext.Model, new object[] { description.TrainedModel, description.PredictionPipeline });
            #endregion Prediction Engine
        }

        private static void GetType(dynamic input, ref Type type)
        {
            if (input.Type.ToString() == "String")
            {
                type = typeof(string);
            }
            else if (input.Type.ToString() == "Single")
            {
                type = typeof(float);
            }
        }

        public IVLNodeDescription NodeDescription => description;

        public IVLPin[] Inputs { get; }
        public IVLPin[] Outputs { get; }
        public MLContext MLContext { get; }

        public void Update()
        {
            if (!Inputs.Any())
                return;

            if ((bool)Inputs.Last().Value)
            {
                // Create an input object that will hold our pin's data
                var inputObject = Activator.CreateInstance(inputType);

                if(description.FModelType == "Classification")
                {
                    // We know all input pins that are not named "Predict" should be taken into account
                    foreach(var dataPin in Inputs.Cast<MyPin>().Where(i => i.Name != "Predict"))
                    {
                        inputType.InvokeMember(dataPin.Name, BindingFlags.SetProperty, null, inputObject, new object[] { dataPin.Value });
                    }

                    // Invoke the predict method
                    var predictMethod = predictionEngine.GetType().GetMethod("Predict", new[] { inputType });
                    var result = predictMethod.Invoke(predictionEngine, new[] { inputObject });

                    // Look for the "Predicted Label" output pin, and assign it the value of the "PredictedLabel" field of the output type
                    var outputPin = Outputs.Cast<MyPin>().FirstOrDefault(o => o.Name == "Predicted Label");
                    outputPin.Value = outputType.InvokeMember("PredictedLabel", BindingFlags.GetProperty, null, result, new object[] { });

                    // Look for the "Score" output pin, and assign it the value of the "Score" field of the output type
                    var scorePin = Outputs.Cast<MyPin>().FirstOrDefault(o => o.Name == "Score");
                    scorePin.Value = outputType.InvokeMember("Score", BindingFlags.GetProperty, null, result, new object[] { });
                }
                else if(description.FModelType == "Regression")
                {
                    // We're looking at a regression, for now we just have a one to one mapping
                    // We'll need to find a way to get rid if the Label input later
                    foreach (var input in Inputs.Cast<MyPin>().SkipLast(1))
                    {
                        inputType.InvokeMember(input.Name, BindingFlags.SetProperty, null, inputObject, new object[] { input.Value });
                    }

                    // Invoke the predict method
                    var predictMethod = predictionEngine.GetType().GetMethod("Predict", new[] { inputType });
                    var result = predictMethod.Invoke(predictionEngine, new[] { inputObject });

                    // Look for the "Score" output pin and assign it the value return by the prediction
                    var outputPin = Outputs.Cast<MyPin>().FirstOrDefault(o => o.Name == "Score");
                    outputPin.Value = outputType.InvokeMember("Score", BindingFlags.GetProperty, null, result, new object[] { });
                }
                else if(description.FModelType == "ImageRecognition")
                {
                    return;
                }
                else
                {
                    return;
                }
            }
        }
        public void Dispose()
        {
            Console.WriteLine("Ok bye");
        }
    }
}
