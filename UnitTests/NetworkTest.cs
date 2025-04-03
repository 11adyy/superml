using System.Drawing;
using System.Drawing.Imaging;
using superml.MODELS.NUMBER_CLASSIFICATION;
using superml.MODELS.SCRIPTS.COMPUTER_VISION;

namespace UnitTests;

public class NetworkTest {
    [Test]
    public void RCnnTest() {
        var bitmap = (Bitmap)Bitmap.FromFile(@"C://Users//11adyy//Desktop//fight.jpg");
        ComputerVision.Calculate(bitmap, CnnClassification.SimpleConvolutionNetwork, .3, 28, 28)
            .Save(@$"D:\загрузки\{Guid.NewGuid()}.png", ImageFormat.Png);
    }
}