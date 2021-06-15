using VL.Core;
using VL.Core.CompilerServices;

// Tell VL where to find the initializer
[assembly: AssemblyInitializer(typeof(VL.MLNet.Initialization))]

namespace VL.MLNet
{
    public class Initialization : AssemblyInitializer<Initialization>
    {
        protected override void RegisterServices(IVLFactory factory)
        {
            factory.RegisterNodeFactory(mlFactory);
        }

        static IVLNodeDescriptionFactory mlFactory = new MLNetRunnerNodeFactory();
    }
}
