using System;
using System.Globalization;
using System.Text.Json;
using Xunit;

namespace BenScr.NeuralNetworks.Tests;

public sealed class ModelTests
{
    private const string Valid = "{\"formatVersion\":1,\"inputSize\":1,\"layers\":[{\"units\":1,\"activation\":\"Linear\"}],\"parameters\":[2,3]}";

    [Fact]
    public void KnownVersionOneModelHasExactMeaning()
    {
        DenseNetwork network = DenseNetwork.FromJson(Valid);
        Assert.Equal(11, network.Predict([4])[0]);
        Assert.Equal(Valid, network.ToJson());
    }

    [Fact]
    public void RoundtripRetainsEveryFiniteParameterAndArchitecture()
    {
        var network = new DenseNetwork(2, [new(2, Activation.Tanh), new(1, Activation.Sigmoid)], -1);
        double[] parameters = network.GetParameters();
        parameters[0] = double.Epsilon;
        parameters[1] = double.MaxValue;
        parameters[2] = -double.MaxValue;
        network.SetParameters(parameters);
        DenseNetwork restored = DenseNetwork.FromJson(network.ToJson());
        Assert.Equal(network.InputSize, restored.InputSize);
        Assert.Equal(network.Layers, restored.Layers);
        Assert.Equal(parameters, restored.GetParameters());
    }

    [Fact]
    public void ModelFormatIsCultureIndependent()
    {
        CultureInfo previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("de-DE");
            DenseNetwork network = DenseNetwork.FromJson(Valid);
            network.SetParameters([1.5, 0.25]);
            Assert.Contains("[1.5,0.25]", network.ToJson(), StringComparison.Ordinal);
            Assert.Equal(3.25, DenseNetwork.FromJson(network.ToJson()).Predict([2])[0]);
        }
        finally
        {
            CultureInfo.CurrentCulture = previous;
        }
    }

    [Theory]
    [InlineData("")]
    [InlineData("null")]
    [InlineData("[]")]
    [InlineData("{}")]
    [InlineData("{\"formatVersion\":1}")]
    [InlineData("{\"formatVersion\":1,\"formatVersion\":1,\"inputSize\":1,\"layers\":[{\"units\":1,\"activation\":\"Linear\"}],\"parameters\":[2,3]}")]
    public void MalformedOrIncompleteJsonIsRejected(string json)
    {
        Assert.ThrowsAny<JsonException>(() => DenseNetwork.FromJson(json));
    }

    [Theory]
    [InlineData("\"formatVersion\":1", "\"formatVersion\":2")]
    [InlineData("\"inputSize\":1", "\"inputSize\":0")]
    [InlineData("\"inputSize\":1", "\"inputSize\":1.2")]
    [InlineData("\"units\":1", "\"units\":0")]
    [InlineData("\"units\":1", "\"units\":4097")]
    [InlineData("\"activation\":\"Linear\"", "\"activation\":\"Unknown\"")]
    [InlineData("\"activation\":\"Linear\"", "\"activation\":\"0\"")]
    [InlineData("\"activation\":\"Linear\"", "\"activation\":0")]
    [InlineData("\"activation\":\"Linear\"", "\"activation\":\"linear\"")]
    [InlineData("\"units\":1", "\"units\":1,\"units\":1")]
    [InlineData("\"units\":1", "\"units\":1,\"unknown\":0")]
    [InlineData("[2,3]", "[2]")]
    [InlineData("[2,3]", "[2,3,4]")]
    [InlineData("[2,3]", "[2,1e9999]")]
    [InlineData("[2,3]", "[2,\"NaN\"]")]
    [InlineData("[2,3]", "[2,null]")]
    [InlineData("[2,3]", "{}")]
    public void UnsupportedModelsAreRejected(string from, string to)
    {
        Assert.ThrowsAny<JsonException>(() => DenseNetwork.FromJson(Valid.Replace(from, to, StringComparison.Ordinal)));
    }

    [Fact]
    public void InputSizeAndDepthLimitsAreEnforced()
    {
        Assert.Throws<ArgumentNullException>(() => DenseNetwork.FromJson(null!));
        Assert.ThrowsAny<JsonException>(() => DenseNetwork.FromJson(new string(' ', DenseNetwork.MaxJsonLength + 1)));
        Assert.ThrowsAny<JsonException>(() => DenseNetwork.FromJson("[[[[[[[[[0]]]]]]]]]"));
    }
}
