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

            // --------------------------------------------------
            // INPUT TYPE CREATION
            // --------------------------------------------------

            var inputTypeProperties = new List<DynamicProperty>();

            // Look at the pre-trained model and generate a type
            foreach (var input in Inputs.Cast<MyPin>().SkipLast(1))
            {
                inputTypeProperties.Add(new DynamicProperty
                {
                    PropertyName = input.Name,
                    DisplayName = input.Name,
                    SystemTypeName = input.Type.ToString()
                });
            }

            inputType = factory.CreateNewTypeWithDynamicProperties(typeof(object), inputTypeProperties);

            // Create type for output data
            var outputTypeProperties = new List<DynamicProperty>();
            
            foreach (var output in Outputs.Cast<MyPin>())
            {
                outputTypeProperties.Add(new DynamicProperty
                {
                    PropertyName = output.Name,
                    DisplayName = output.Name,
                    SystemTypeName = output.Type.ToString()
                });
            }

            outputType = factory.CreateNewTypeWithDynamicProperties(typeof(object), outputTypeProperties);

            // Create instances of those
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

                // Stuff our input object with input pin's data
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
        }
        public void Dispose()
        {
            Console.WriteLine("Ok bye");
        }
    }
}
