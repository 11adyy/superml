using superml.NETWORK.MATH.OBJECTS;

namespace superml.NETWORK.LAYERS.NOISE.SCRIPTS;

public interface INoise {
    public Vector GenerateNoise(int size);

    public Matrix GenerateNoise((int Rows, int Columns) shape);

    public Tensor GenerateNoise((int Rows, int Columns, int Depth) shape);
}