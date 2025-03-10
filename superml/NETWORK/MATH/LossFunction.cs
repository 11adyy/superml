using superml.NETWORK.ACTIVATION.INTERFACES;
using superml.NETWORK.OBJECTS;

namespace superml.NETWORK.MATH {
    public static class LossFunction {
        public static Tensor GetErrorTensor(Tensor outputTensor, int expectedClass, IFunction activateFunction) {
            var prediction = outputTensor.Channels[0].GetAsList().ToArray();
            var error = new List<double>();

            for (var i = 0; i < prediction.Length; i++)
                if (i != expectedClass) error.Add(-activateFunction.Derivation(prediction[i]));
                else error.Add(1.0 - activateFunction.Derivation(prediction[i]));

            return new Vector(error.ToArray()).AsTensor(1, error.Count, 1);
        }
    }
}