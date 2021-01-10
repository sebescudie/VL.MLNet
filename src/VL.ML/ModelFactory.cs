using System;
using System.IO;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Collections.Immutable;
using VL.Core;
using VL.Core.Diagnostics;
using VL.Lang.PublicAPI;

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
                if(Directory.Exists(mlSubDir))
                {
                    // Search for all *.zip files in the models directory
                    var models = Directory.GetFiles(mlSubDir, "*.zip");

                    foreach(string model in models)
                    {
                        builder.Add(new ModelD)
                    }
                }
            }
        }
    }

    // Description of a pin
    class ModelPinDescription : IVLPinDescription, IInfo
    {
        public string Name { get; }
        public Type Type { get; }
        public object DefaultValue { get; }

        public string Summary { get; }
        public string Remarks => "";

        public ModelPinDescription(string name, Type type, object defaultValue, string description)
        {
            Name = name;
            Type = type;
            DefaultValue = defaultValue;
            Summary = description;
        }
    }

    class ModelDescription : IVLNodeDescription, IInfo
    {
        string FSummary;
        bool FInitialized;

        public IVLNodeDescriptionFactory Factory { get; }
        public string Name;
        public string Category => "ML.MLNet";
        public bool Fragmented = false;

        public IReadOnlyList<IVLPinDescription> Inputs
        {
            get
            {
                Init();
            }
        }

        public IReadOnlyList<IVLPinDescription> Outputs
        {
            get
            {
                Init();
                return outputs;
            }
        }

        //public IEnumerable<Core.Diagnostics.Message> Messages
        //{
        //    get
        //    {
        //        if (FNotFound)
        //            yield return new Message(MessageType.Warning, "Model inactive: " + FUrl + "\r\nActivate it in your RunwayML dashboard and then restart vvvv.");
        //        else
        //            yield break;
        //    }
        //}

        string FFullName;
        string FPath;

        List<ModelPinDescription> inputs = new List<ModelPinDescription>();
        List<ModelPinDescription> outputs = new List<ModelPinDescription>();

        public ModelDescription(IVLNodeDescriptionFactory factory, string path)
        {
            Factory = factory;
            FFullName = path;
            Name = path;
        }

        void Init()
        {
            if (FInitialized)
                return;

            // Here we're gonna load the model and retrieve its inputs and outputs
            try
            {

            }
        }

        public string Summary => FSummary;
        public string Remarks => "";
        public IObservable<object> Invalidated => Observable.Empty<object>();

        public IVLNode CreateInstance(NodeContext context)
        {
            return new MyNode(this, context);
        }

        public bool OpenEditor()
        {
            // nope
            return true;
        }
    }
}
