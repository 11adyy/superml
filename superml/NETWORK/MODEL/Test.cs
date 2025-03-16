using superml.NETWORK.OBJECTS.DATA_OBJECTS;

namespace superml.NETWORK.MODEL;

public static class Test {
    public static double TestModel(Network network, List<IData> dataSet) =>
        dataSet.Count / (double)(from data in dataSet let prediction = 
            network.ForwardFeed(data.AsTensor()) where prediction == 
                                                       data.GetRight().GetMaxIndex() select data).Count();
}