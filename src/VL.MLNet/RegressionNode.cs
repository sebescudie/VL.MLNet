using System;
using System.Reactive.Linq;
using System.Linq;
using System.Reflection;
using VL.Core;
using Microsoft.ML;

namespace VL.MLNet
{
    partial class RegressionNode<TInput> : VLObject, IVLNode
        where TInput : class, new()
    {
        readonly MLNetRunnerNodeDescription description;
        readonly Pin runPin;
        readonly Pin scorePin;

        // Prediction engine
        readonly PredictionEngine<TInput, RegressionOutput> predictionEngine;

        public RegressionNode(MLNetRunnerNodeDescription description, NodeContext nodeContext) : base(nodeContext)
        {
            this.description = description;
            Inputs = description.Inputs.Select(p => new Pin() { Name = p.Name, Type = p.Type, Value = p.DefaultValue }).ToArray();
            Outputs = description.Outputs.Select(p => new Pin() { Name = p.Name, Type = p.Type, Value = p.DefaultValue }).ToArray();

            scorePin = Outputs.FirstOrDefault(o => o.Name == "Score");
            runPin = Inputs.LastOrDefault();

            // Create prediction engine
            predictionEngine = description.MLContext.Model.CreatePredictionEngine<TInput, RegressionOutput>(description.TrainedModel, description.PredictionPipeline);
        }

        public IVLNodeDescription NodeDescription => description;

        public Pin[] Inputs { get; }
        public Pin[] Outputs { get; }

        public void Update()
        {
            if (runPin is null || !(bool)runPin.Value)
                return;

            // Create an input object that will hold our pin's data
            var inputObject = new TInput();

            // We're looking at a regression, for now we just have a one to one mapping
            // We'll need to find a way to get rid if the Label input later
            foreach (var dataPin in Inputs.Where(i => i != runPin))
            {
                typeof(TInput).InvokeMember(dataPin.Name, BindingFlags.SetProperty, null, inputObject, new object[] { dataPin.Value });
            }

            // Invoke the predict method
            var result = predictionEngine.Predict(inputObject);

            // Look for the "Score" output pin and assign it the value return by the prediction
            scorePin.Value = result.Score;
        }

        public void Dispose()
        {
            Console.WriteLine("Ok bye");
        }

        IVLPin[] IVLNode.Inputs => Inputs;
        IVLPin[] IVLNode.Outputs => Outputs;
    }
}