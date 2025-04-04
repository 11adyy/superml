using System.Drawing;
using System.Drawing.Imaging;
using superml.SCRIPTS.REGION_CONVOLUTION.SCRIPTS;

namespace UnitTests;

public class RegionsTest {
    [Test]
    public void RegionsCreation() {
        var bitmap = (Bitmap)Bitmap.FromFile(@"C://Users//11adyy//Desktop//RCNN_TEST//test1.jpg");
        var subBitmaps = RegionsMaker.GetRegions(bitmap, 64, 9);

        var graphics = Graphics.FromImage(bitmap);
        foreach (var image in subBitmaps) {
            graphics.DrawRectangle(Pens.Black, image);
        }
        
        bitmap.Save(@$"D:\загрузки\{Guid.NewGuid()}.png", ImageFormat.Png);
    }
}