using VL.Core;
using VL.Core.CompilerServices;

// Tell VL where to find the initializer
[assembly: AssemblyInitializer(typeof(VL.ML.Initialization))]

namespace VL.ML
{
    public class Initialization : AssemblyInitializer<Initialization>
    {
        protected override void RegisterServices(IVLFactory factory)
        {
            factory.RegisterNodeFactory(mlFactory);
        }
    }
}
