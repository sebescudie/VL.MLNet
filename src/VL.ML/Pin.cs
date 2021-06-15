using System;
using VL.Core;

namespace VL.ML
{
    class Pin : IVLPin
    {
        public object Value { get; set; }
        public Type Type { get; set; }
        public string Name { get; set; }
    }
}