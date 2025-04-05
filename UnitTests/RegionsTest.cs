using System.Drawing;
using System.Drawing.Imaging;
using superml.SCRIPTS.REGION_CONVOLUTION.SCRIPTS;

namespace UnitTests;

public class RegionsTest {
    [Test]
    public void RegionsCreation() {
        var bitmap = (Bitmap)Bitmap.FromFile(@"C://Users//11adyy//Desktop//RCNN_TEST//test.jpg");
        var subBitmaps = RegionsMaker.GetRegions(bitmap, 100, 1);

        var graphics = Graphics.FromImage(bitmap);
        foreach (var image in subBitmaps) 
            graphics.DrawRectangle(new Pen(Color.Black), image);
        
        bitmap.Save(@$"C://Users//11adyy//Desktop//RCNN_TEST//answers//{Guid.NewGuid()}.png", ImageFormat.Png);
        foreach (var image in subBitmaps) 
            bitmap.Clone(image, bitmap.PixelFormat).Save(@$"C://Users//11adyy//Desktop//RCNN_TEST//answers//{Guid.NewGuid()}.png", ImageFormat.Png);
    }
}