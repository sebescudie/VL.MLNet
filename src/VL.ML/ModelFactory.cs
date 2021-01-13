using System;
using System.IO;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Collections.Immutable;
using VL.Core;
using VL.Core.Diagnostics;
using Microsoft.ML;
using System.Linq;
using System.Reflection.Emit;

namespace VL.ML
{
    public class MLFactory : IVLNodeDescriptionFactory
    {
        const string mlSubDir = "ml-models";

        public readonly string Dir;
        public readonly string DirToWatch;
        
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
                var i = "VL.MLNet-Factory";
                if (Dir != null)
                    return $"{i} ({Dir})";
                return i;
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
            // Required by the interface
        }

        public IVLNodeDescriptionFactory ForPath(string path)
        {
            var modelsDir = Path.Combine(path, mlSubDir);
            if (Directory.Exists(modelsDir))
                return new MLFactory(modelsDir);
            return new MLFactory(directoryToWatch: path);
        }
    }

    /// <summary>
    /// Defines a pin
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

        // I/O
        List<ModelPinDescription> inputs = new List<ModelPinDescription>();
        List<ModelPinDescription> outputs = new List<ModelPinDescription>();

        // Type factory stuff
        List<DynamicProperty> inputTypeProperties = new List<DynamicProperty>();
        List<DynamicProperty> outputTypeProperties = new List<DynamicProperty>();

        public ModelDescription(IVLNodeDescriptionFactory factory, string path)
        {
            Factory = factory;
            FFullName = path;
            FPath = path;
            Name = Path.GetFileNameWithoutExtension(path);
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
                mlContext = new MLContext();
                DataViewSchema predictionPipeline;
                ITransformer trainedModel = mlContext.Model.Load(FPath, out predictionPipeline);

                FSummary = "An ML.NET model";

                Type type = typeof(object);
                object dflt = "";
                string descr = "";

                // Retrieve the inputs by looking at the prediction pipeline
                foreach(var input in predictionPipeline)
                {
                    GetTypeDefaultAndDescription(input, ref type, ref dflt, ref descr);
                    inputs.Add(new ModelPinDescription(input.Name, type, dflt, descr));
                }

                // Retrieve the output by looking for a "Score" output column
                // Might only work for regression models for now
                var scoreColumn = trainedModel.GetOutputSchema(predictionPipeline).FirstOrDefault(o => o.Name == "Score");
                GetTypeDefaultAndDescription(scoreColumn, ref type, ref dflt, ref descr);
                outputs.Add(new ModelPinDescription(scoreColumn.Name, type, dflt, descr));


                // After we've added our inputs from the ML model, we add the Predict bool that will run the prediction
                inputs.Add(new ModelPinDescription("Predict", typeof(bool), false, "Runs a prediction every frame as long as enabled"));

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
        public string Category => "ML.MLNet";
        public bool Fragmented => false;

        /// <summary>
        /// Returns the MLContext
        /// </summary>
        public MLContext mlContext { get; set; }

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
        public string Remarks => "Everything you know is wrong";
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

    class MyNode : VLObject, IVLNode
    {
        class MyPin : IVLPin
        {
            public object Value { get; set; }
            public Type Type { get; set; }
            public string Name { get; set; }
        }

        readonly ModelDescription description;

        public MyNode(ModelDescription description, NodeContext nodeContext) : base(nodeContext)
        {
            this.description = description;
            Inputs = description.Inputs.Select(p => new MyPin() { Name = p.Name, Type = p.Type, Value = p.DefaultValue }).ToArray();
            Outputs = description.Outputs.Select(p => new MyPin() { Name = p.Name, Type = p.Type, Value = p.DefaultValue }).ToArray();

            #region Retrieve ML Context

            MLContext = description.mlContext;

            #endregion Retrieve ML Context

            #region Type Generation
            var factory = new DynamicTypeFactory();

            // Create type for input data
            var inputTypeProperties = new List<DynamicProperty>();
            foreach (var input in description.Inputs.Where(i => i.Name != "Predict"))
            {
                inputTypeProperties.Add(new DynamicProperty
                {
                    PropertyName = input.Name,
                    DisplayName = input.Name,
                    SystemTypeName = input.Type.ToString()
                });
            }

            var inputType = factory.CreateNewTypeWithDynamicProperties(typeof(object), inputTypeProperties);

            // Create type for output data
            var outputTypeProperties = new List<DynamicProperty>();
            foreach (var output in description.Outputs)
            {
                outputTypeProperties.Add(new DynamicProperty
                {
                    PropertyName = output.Name,
                    DisplayName = output.Name,
                    SystemTypeName = output.Type.ToString()
                });
            }

            var outputType = factory.CreateNewTypeWithDynamicProperties(typeof(object), outputTypeProperties);
            #endregion TypeGeneration

            // Create the PredictionEngine
            description.mlContext.Model.CreatePredictionEngine<>
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
                // Stuff an object with data from our output pins

                // Create an object holding our result

                // Run the prediction engine

                // Retrieve the result of the prediction engine and assign it to the output pin

                // Debug stuff
            }
        }

        public void Dispose()
        {
            Console.Write("Ok bye");
        }
    }

    class InputData
    {
        // Will this work?
    }

    class OutputData
    {

    }
}
