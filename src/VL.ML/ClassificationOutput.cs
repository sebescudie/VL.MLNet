using Microsoft.ML.Data;

namespace VL.ML
{
    class ClassificationOutput
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
        public float[] Score { get; set; }
    }
}