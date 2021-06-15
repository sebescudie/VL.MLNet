using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using VL.Core;
using System.IO;
using VL.Core.Diagnostics;
using System.Reactive.Linq;

namespace VL.ML
{
    /// <summary>
    /// Abstract ML.NET runner node
    /// </summary>
    class MLNetRunnerNodeDescription : IVLNodeDescription, IInfo
    {
        // Fields
        bool FInitialized;
        bool FError;
        string FSummary;
        string FPath;
        string FFriendlyName;

        public Enums.ModelType FModelType { get; set; }
        private DataViewSchema predictionPipeline;
        
        public ITransformer TrainedModel { get; set; }

        // I/O
        List<PinDescription> inputs = new List<PinDescription>();
        List<PinDescription> outputs = new List<PinDescription>();
        private Type nodeType;

        public MLNetRunnerNodeDescription(IVLNodeDescriptionFactory factory, string path, string friendlyName, Enums.ModelType modelType)
        {
            Factory = factory;
            FPath = path;

            FFriendlyName = friendlyName;
            FModelType = modelType;
            Name = FFriendlyName;

            MLContext = new MLContext();
            TrainedModel = MLContext.Model.Load(FPath, out predictionPipeline);
        }

        /// <summary>
        /// Retrieves relevant input and output from the trained model
        /// and creates corresponding pins
        /// </summary>
        /// <remarks>
        /// This does not map straight to the dynamic types
        /// </remarks>
        void Init()
        {
            if (FInitialized)
                return;

            // Look at the trained model and create input pins
            try
            {
                #region Create inputs and outputs
                Type type = typeof(object);
                object dflt = "";
                string descr = "";

                // Retrieve all columns that are not named "Label"
                foreach (var inputCol in predictionPipeline.Where(i => i.Name != "Label"))
                {
                    GetTypeDefaultAndDescription(inputCol, ref type, ref dflt, ref descr);
                    inputs.Add(new PinDescription(inputCol.Name, type, dflt, descr));
                }

                if (FModelType == Enums.ModelType.TextClassification || FModelType == Enums.ModelType.ImageClassification)
                {
                    // Retrieve outputs
                    var predictedLabelColumn = TrainedModel.GetOutputSchema(predictionPipeline).FirstOrDefault(o => o.Name == "PredictedLabel");
                    GetTypeDefaultAndDescription(predictedLabelColumn, ref type, ref dflt, ref descr);
                    outputs.Add(new PinDescription("Predicted Label", type, dflt, descr));

                    var scoresColumn = TrainedModel.GetOutputSchema(predictionPipeline).FirstOrDefault(o => o.Name == "Score");
                    GetTypeDefaultAndDescription(scoresColumn, ref type, ref dflt, ref descr);
                    outputs.Add(new PinDescription("Score", type, dflt, descr));

                    // Add an extra output for labels
                    outputs.Add(new PinDescription("Labels", typeof(IEnumerable<string>), new string[0], "Score labels"));
                }
                else if (FModelType == Enums.ModelType.Regression)
                {
                    // Retrieve outputs
                    var scoreColumn = TrainedModel.GetOutputSchema(predictionPipeline).FirstOrDefault(o => o.Name == "Score");
                    GetTypeDefaultAndDescription(scoreColumn, ref type, ref dflt, ref descr);
                    outputs.Add(new PinDescription(scoreColumn.Name, type, dflt, descr));
                }
                else
                {
                    // Unknown model type
                }

                // Add the Trigger input that will allow to trigger the node
                inputs.Add(new PinDescription("Run", typeof(bool), false, "Runs a prediction every frame as long as enabled"));
                #endregion Create inputs and outputs

                FSummary = String.Format("Runs the ML.NET {0} {1} pre-trained model",FFriendlyName, FModelType);
                FInitialized = true;
            }
            catch (Exception e)
            {
                FError = true;
                Console.WriteLine("Error loading ML Model");
                Console.WriteLine(e.Message);
            }
        }

        public IVLNodeDescriptionFactory Factory { get; }

        public string Name { get; }
        public string Category => "ML.MLNet";
        public bool Fragmented => false;

        /// <summary>
        /// Returns the MLContext
        /// </summary>
        public MLContext MLContext { get; set; }

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

        private void GetTypeDefaultAndDescription(dynamic pin, ref Type type, ref object dflt, ref string descr)
        {
            descr = pin.Name;

            if (pin.Type.ToString() == "String")
            {
                type = typeof(string);
                dflt = "";
            }
            else if (pin.Type.ToString() == "Single")
            {
                type = typeof(float);
                dflt = 0.0f;
            }    
            else if (pin.Type.RawType.GenericTypeArguments.Length == 1 && pin.Type.RawType.GenericTypeArguments[0] == typeof(float))
                {
                type = typeof(IEnumerable<float>);
                dflt = Enumerable.Repeat<float>(0, 0).ToArray();
            }
        }

        public string Summary => FSummary;
        public string Remarks => "";
        public IObservable<object> Invalidated => Observable.Empty<object>();

        public IVLNode CreateInstance(NodeContext context)
        {
            InitNodeType();

            return (IVLNode)Activator.CreateInstance(nodeType, new object[] { this, context });
        }

        public bool OpenEditor()
        {
            // nope
            return true;
        }

        void InitNodeType()
        {
            if (nodeType != null)
                return;

            var factory = new DynamicTypeFactory();

            var inputTypeProperties = new List<DynamicProperty>();
            var outputTypeProperties = new List<DynamicProperty>();

            Type type = typeof(object);

            // Start by loading all inputs from the model and spawn a dynamic type with it
            foreach (var inputColumn in PredictionPipeline)
            {
                GetType(inputColumn, ref type);

                inputTypeProperties.Add(new DynamicProperty
                {
                    PropertyName = inputColumn.Name,
                    DisplayName = inputColumn.Name,
                    SystemTypeName = type.ToString()
                });
            }

            var inputType = factory.CreateNewTypeWithDynamicProperties(typeof(object), inputTypeProperties);

            switch (FModelType)
            {
                case Enums.ModelType.TextClassification:
                case Enums.ModelType.ImageClassification:
                    nodeType = typeof(ClassificationNode<>).MakeGenericType(inputType);
                    break;
                case Enums.ModelType.Regression:
                    nodeType = typeof(RegressionNode<>).MakeGenericType(inputType);
                    break;
                default:
                    throw new NotImplementedException();
            }
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
    }
}
