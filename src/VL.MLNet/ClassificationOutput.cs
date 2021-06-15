using Microsoft.ML.Data;

namespace VL.MLNet
{
    class ClassificationOutput
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
        public float[] Score { get; set; }
    }
}