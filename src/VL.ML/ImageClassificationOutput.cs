﻿using Microsoft.ML.Data;
using System;

namespace VL.ML
{
    class ImageClassificationOutput
    {
        [ColumnName("PredictedLabel")]
        public String PredictedLabel { get; set; }
        public float[] Score { get; set; }
    }
}