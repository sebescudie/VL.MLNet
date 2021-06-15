using System;
using System.IO;
using System.Reactive.Linq;
using System.Collections.Immutable;
using System.Linq;
using VL.Core;

namespace VL.MLNet
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
                        var fullName = Path.GetFileNameWithoutExtension(preTrainedModelFile);
                        var friendlyName = fullName.Split('_')[0];
                        var modelTypeString = fullName.Split('_')[1];
                        if (Enum.TryParse<Enums.ModelType>(modelTypeString, out var modelType))
                            builder.Add(new MLNetRunnerNodeDescription(this, preTrainedModelFile, friendlyName, modelType));
                        else
                            Console.WriteLine($"Unsupported model type {modelTypeString} in '{preTrainedModelFile}'");
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
}