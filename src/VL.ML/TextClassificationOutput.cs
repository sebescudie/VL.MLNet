using Microsoft.ML.Data;
using System;

namespace VL.ML
{
    class TextClassificationOutput
    {
        [ColumnName("PredictedLabel")]
        public String PredictedLabel { get; set; }
    }
}
