using System;
using Microsoft.ML.Data;

namespace VL.ML
{
    public class TextClassificationInput
    {
        public string Label { get; set; }
        public string Input { get; set; }
    }
}