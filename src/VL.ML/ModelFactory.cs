using System;
using System.IO;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Collections.Immutable;
using VL.Core;
using VL.Core.Diagnostics;
using Microsoft.ML;
using System.Linq;
using System.Reflection;

namespace VL.ML
{
    public class MLFactory : IVLNodeDescriptionFactory
    {
        const string mlSubDir = "ml-models";
        const string identifier = "VL.MLNet-Factory";

        public readonly string Dir;
        public readonly string DirToWatch;

        private readonly NodeFactoryCache factoryCache = new NodeFactoryCache();
        
        public MLFactory(string directory = default, string directoryToWatch = default)
        {
            Dir = directory;
            DirToWatch = directoryToWatch;

            var builder = ImmutableArray.CreateBuilder<IVLNodeDescription>();
            
            if(directory != null)
            {
                if(Directory.Exists(directory))
                {
                    var models = Directory.GetFiles(directory, "*.zip");

                    foreach(string model in models)
                    {
                        builder.Add(new ModelDescription(this, model));
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
                    return NodeBuilding.WatchDir(DirToWatch)
                        .Where(e => e.Name == mlSubDir);
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
                    return new MLFactory(mldir);
                return new MLFactory(directoryToWatch: path);
            });
        }
    }

    /// <summary>
    /// Defines a pin's model
    /// </summary>
    class ModelPinDescription : IVLPinDescription, IInfo
    {
        public string Name { get; }
        public Type Type { get; }
        public object DefaultValue { get; }

        public string Summary { get; }
        public string Remarks => "";

        public ModelPinDescription(string name, Type type, object defaultValue, string summary)
        {
            Name = name;
            Type = type;
            DefaultValue = defaultValue;
            Summary = summary;
        }
    }

    /// <summary>
    /// Defines the model of our ML nodes
    /// </summary>
    class ModelDescription : IVLNodeDescription, IInfo
    {
        // Fields
        bool FInitialized;
        bool FError;
        string FSummary;
        string FFullName;
        string FPath;

        DataViewSchema predictionPipeline;
        ITransformer trainedModel;

        // I/O
        List<ModelPinDescription> inputs = new List<ModelPinDescription>();
        List<ModelPinDescription> outputs = new List<ModelPinDescription>();

        public ModelDescription(IVLNodeDescriptionFactory factory, string path)
        {
            Factory = factory;
            FFullName = path;
            FPath = path;
            Name = Path.GetFileNameWithoutExtension(path);

            mlContext = new MLContext();
            trainedModel = mlContext.Model.Load(FPath, out predictionPipeline);
        }

        /// <summary>
        /// Gets input and outputs pin from the ML model
        /// </summary>
        void Init()
        {
            if (FInitialized)
                return;

            // Load the model and create input pins
            try
            {
                #region Create inputs and outputs

                Type type = typeof(object);
                object dflt = "";
                string descr = "";

                // Retrieve the inputs by looking at the prediction pipeline
                foreach (var input in predictionPipeline)
                {
                    GetTypeDefaultAndDescription(input, ref type, ref dflt, ref descr);
                    inputs.Add(new ModelPinDescription(input.Name, type, dflt, descr));
                }

                // Retrieve the output by looking for a "Score" output column
                var scoreColumn = trainedModel.GetOutputSchema(predictionPipeline).FirstOrDefault(o => o.Name == "Score");
                GetTypeDefaultAndDescription(scoreColumn, ref type, ref dflt, ref descr);
                outputs.Add(new ModelPinDescription(scoreColumn.Name, type, dflt, descr));

                // After we've added our inputs from the ML model, we add the Predict bool that will run the prediction
                inputs.Add(new ModelPinDescription("Predict", typeof(bool), false, "Runs a prediction every frame as long as enabled"));

                #endregion Create inputs and outputs

                FSummary = "Runs an ML.NET model";
                FInitialized = true;
            }
            catch(Exception e)
            {
                FError = true;
                Console.WriteLine("Error loading ML Model");
                Console.WriteLine(e.Message);
            }
        }

        public IVLNodeDescriptionFactory Factory { get; }


        public Type InputType { get; set; }
        public Type OutputType { get; set; }

        public string Name { get; }
        public string Category => "ML.Runners";
        public bool Fragmented => false;

        /// <summary>
        /// Returns the MLContext
        /// </summary>
        public MLContext mlContext { get; set; }

        /// <summary>
        /// Returns the prediction pipeline
        /// </summary>
        public DataViewSchema PredictionPipeline
        {
            get
            {
                return predictionPipeline;
            }
        }

        /// <summary>
        /// Returns the trained model
        /// </summary>
        public ITransformer TrainedModel
        {
            get
            {
                return trainedModel;
            }
        }

        /// <summary>
        /// Returns the emitted input type
        /// </summary>
        public Type inputType { get; set; }

        /// <summary>
        /// Returns the emitted output type
        /// </summary>
        public Type outputType { get; set; }

        /// <summary>
        /// Returns the input pins
        /// </summary>
        public IReadOnlyList<IVLPinDescription> Inputs
        {
            get
            {
                Init();
                return inputs;
            }
        }

        /// <summary>
        /// Returns the output pins
        /// </summary>
        public IReadOnlyList<IVLPinDescription> Outputs
        {
            get
            {
                Init();
                return outputs;
            }
        }

        /// <summary>
        /// Displays a warning on the node if something goes wrong
        /// </summary>
        public IEnumerable<Core.Diagnostics.Message> Messages
        {
            get
            {
                if (FError)
                    yield return new Message(MessageType.Warning, "Brrrrr");
                else
                    yield break;
            }
        }

        void GetTypeDefaultAndDescription(dynamic pin, ref Type type, ref object dflt, ref string descr)
        {
            descr = pin.Name;

            if(pin.Type.ToString() == "String")
            {
                type = typeof(string);
                dflt = "";
            }
            else if(pin.Type.ToString() == "Single")
            {
                type = typeof(float);
                dflt = 0.0f;
            }
        }

        public string Summary => FSummary;
        public string Remarks => "";
        public IObservable<object> Invalidated => Observable.Empty<object>();

        public IVLNode CreateInstance(NodeContext context)
        {
            return new MyNode(this, context);
        }

        /// <summary>
        /// Opens the node's editor
        /// </summary>
        /// <returns></returns>
        public bool OpenEditor()
        {
            // nope
            return true;
        }
    }

    /// <summary>
    /// Actual node
    /// </summary>
    class MyNode : VLObject, IVLNode
    {
        class MyPin : IVLPin
        {
            public object Value { get; set; }
            public Type Type { get; set; }
            public string Name { get; set; }
        }

        readonly ModelDescription description;

        // Type factory stuff
        Type inputType;
        Type outputType;
        List<DynamicProperty> inputTypeProperties = new List<DynamicProperty>();
        List<DynamicProperty> outputTypeProperties = new List<DynamicProperty>();
        
        // Prediction engine
        dynamic predictionEngine;

        public MyNode(ModelDescription description, NodeContext nodeContext) : base(nodeContext)
        {
            this.description = description;
            Inputs = description.Inputs.Select(p => new MyPin() { Name = p.Name, Type = p.Type, Value = p.DefaultValue }).ToArray();
            Outputs = description.Outputs.Select(p => new MyPin() { Name = p.Name, Type = p.Type, Value = p.DefaultValue }).ToArray();

            MLContext = description.mlContext;

            #region Type Generation
            var factory = new DynamicTypeFactory();

            // Create type for input data
            var inputTypeProperties = new List<DynamicProperty>();

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
            var genericPredictionMethod = description.mlContext.Model.GetType().GetMethod("CreatePredictionEngine", new[] { typeof(ITransformer), typeof(DataViewSchema) });
            var predictionMethod = genericPredictionMethod.MakeGenericMethod(inputObject.GetType(), outputObject.GetType());
            predictionEngine = predictionMethod.Invoke(description.mlContext.Model, new object[] { description.TrainedModel, description.PredictionPipeline });
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

            if((bool)Inputs.Last().Value)
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
            Console.Write("Ok bye");
        }
    }
}
