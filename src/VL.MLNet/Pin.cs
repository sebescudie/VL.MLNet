using System;
using VL.Core;

namespace VL.MLNet
{
    class Pin : IVLPin
    {
        public object Value { get; set; }
        public Type Type { get; set; }
        public string Name { get; set; }
    }
}