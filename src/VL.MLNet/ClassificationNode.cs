using System;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using VL.Core;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace VL.MLNet
{
    partial class ClassificationNode<TInput> : VLObject, IVLNode
        where TInput : class, new()
    {
        readonly MLNetRunnerNodeDescription description;
        readonly Pin runPin;
        readonly Pin outputPin;
        readonly Pin scorePin;
        
        // Prediction engine
        readonly PredictionEngine<TInput, ClassificationOutput> predictionEngine;

        public ClassificationNode(MLNetRunnerNodeDescription description, NodeContext nodeContext) : base(nodeContext)
        {
            this.description = description;
            Inputs = description.Inputs.Select(p => new Pin() { Name = p.Name, Type = p.Type, Value = p.DefaultValue }).ToArray();
            Outputs = description.Outputs.Select(p => new Pin() { Name = p.Name, Type = p.Type, Value = p.DefaultValue }).ToArray();

            outputPin = Outputs.FirstOrDefault(o => o.Name == "Predicted Label");
            scorePin = Outputs.FirstOrDefault(o => o.Name == "Score");
            runPin = Inputs.LastOrDefault();

            // Create prediction engine
            predictionEngine = description.MLContext.Model.CreatePredictionEngine<TInput, ClassificationOutput>(description.TrainedModel, description.PredictionPipeline);
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

            // We know all input pins that are not named "Predict" should be taken into account
            foreach (var dataPin in Inputs.Where(i => i != runPin))
            {
                typeof(TInput).InvokeMember(dataPin.Name, BindingFlags.SetProperty, null, inputObject, new object[] { dataPin.Value });
            }

            // Invoke the predict method
            var result = predictionEngine.Predict(inputObject);

            // Look for the "Predicted Label" output pin, and assign it the value of the "PredictedLabel" field of the output type
            if (outputPin != null)
                outputPin.Value = result.PredictedLabel;

            // Look for the "Score" output pin, and assign it the value of the "Score" field of the output type
            if (scorePin != null)
                scorePin.Value = result.Score;

            // Do some voodoo to retrieve the score labels from the pipeline
            // Credits goes to https://blog.hompus.nl/2020/09/14/get-all-prediction-scores-from-your-ml-net-model/
            var labelBuffer = new VBuffer<ReadOnlyMemory<Char>>();
            predictionEngine.OutputSchema["Score"].Annotations.GetValue("SlotNames", ref labelBuffer);
            var labelsPin = Outputs.FirstOrDefault(o => o.Name == "Labels");
            labelsPin.Value = labelBuffer.DenseValues().Select(l => l.ToString()).ToArray();
        }

        public void Dispose()
        {
            Console.WriteLine("Ok bye");
        }

        IVLPin[] IVLNode.Inputs => Inputs;
        IVLPin[] IVLNode.Outputs => Outputs;
    }
}