using System;
using VL.Core;

namespace VL.MLNet
{
    class PinDescription : IVLPinDescription, IInfo
    {
            public string Name { get; }
            public Type Type { get; }
            public object DefaultValue { get; }

            public string Summary { get; }
            public string Remarks => "";

            public PinDescription(string name, Type type, object defaultValue, string summary)
            {
                Name = name;
                Type = type;
                DefaultValue = defaultValue;
                Summary = summary;
            }
    }
}