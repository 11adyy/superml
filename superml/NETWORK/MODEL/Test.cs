using superml.DATA.DATA_OBJECTS;

namespace superml.NETWORK.MODEL;

public static class Test {
    public static double TestModel(Network network, List<IData> dataSet) =>
        dataSet.Count / (double)(from data in dataSet let prediction = 
            network.ForwardFeed(data.AsTensor(), AnswerType.Class) where (int)prediction == 
                                                                            data.GetRight().GetMaxIndex() select data).Count();
}