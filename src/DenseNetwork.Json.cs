using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;

namespace BenScr.NeuralNetworks;

public sealed partial class DenseNetwork
{
    /// <summary>The inclusive limit for a model JSON string, in UTF-16 code units: 33,554,432.</summary>
    public const int MaxJsonLength = 32 * 1_024 * 1_024;

    /// <summary>Serializes architecture and parameters to compact, version-one JSON.</summary>
    /// <returns>An invariant-culture JSON string suitable for file storage.</returns>
    /// <remarks>Optimizer state and training samples are not part of the model format.</remarks>
    public string ToJson()
    {
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteNumber("formatVersion", 1);
            writer.WriteNumber("inputSize", InputSize);
            writer.WriteStartArray("layers");
            foreach (LayerDefinition layer in definitions)
            {
                writer.WriteStartObject();
                writer.WriteNumber("units", layer.Units);
                writer.WriteString("activation", layer.Activation.ToString());
                writer.WriteEndObject();
            }
            writer.WriteEndArray();
            writer.WriteStartArray("parameters");
            foreach (double parameter in parameters)
            {
                writer.WriteNumberValue(parameter);
            }
            writer.WriteEndArray();
            writer.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buffer.GetBuffer(), 0, (int)buffer.Length);
    }

    /// <summary>Loads a network from version-one JSON, validating the entire model.</summary>
    /// <param name="json">At most <see cref="MaxJsonLength"/> UTF-16 code units of model JSON.</param>
    /// <returns>A new network owning the parsed architecture and parameters.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="json"/> is null.</exception>
    /// <exception cref="JsonException">Malformed JSON, unsupported version, unknown/duplicate members,
    /// invalid activation/shape, excessive size, or nonfinite/mismatched parameters.</exception>
    public static DenseNetwork FromJson(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        if (json.Length > MaxJsonLength)
        {
            throw new JsonException("The model exceeds the JSON size limit.");
        }
        using JsonDocument document = JsonDocument.Parse(json, new JsonDocumentOptions { MaxDepth = 8 });
        JsonElement root = document.RootElement;
        RequireObject(root, "formatVersion", "inputSize", "layers", "parameters");
        int version = ReadInteger(root.GetProperty("formatVersion"));
        if (version != 1)
        {
            throw new JsonException("Unsupported model format version.");
        }
        int inputSize = ReadInteger(root.GetProperty("inputSize"));
        JsonElement layerArray = root.GetProperty("layers");
        if (layerArray.ValueKind != JsonValueKind.Array || layerArray.GetArrayLength() < 1 ||
            layerArray.GetArrayLength() > MaxLayers)
        {
            throw new JsonException("The layer array has an invalid length.");
        }
        var definitions = new LayerDefinition[layerArray.GetArrayLength()];
        int index = 0;
        foreach (JsonElement layer in layerArray.EnumerateArray())
        {
            RequireObject(layer, "units", "activation");
            int units = ReadInteger(layer.GetProperty("units"));
            JsonElement activationElement = layer.GetProperty("activation");
            if (activationElement.ValueKind != JsonValueKind.String)
            {
                throw new JsonException("Activation must be a supported name.");
            }
            string? name = activationElement.GetString();
            if (!Enum.TryParse(name, out Activation activation) || !Enum.IsDefined(activation) ||
                activation.ToString() != name)
            {
                throw new JsonException("Activation must be a supported name.");
            }
            definitions[index++] = new LayerDefinition(units, activation);
        }
        DenseNetwork network;
        try
        {
            network = new DenseNetwork(inputSize, definitions);
        }
        catch (ArgumentException exception)
        {
            throw new JsonException("Invalid network architecture.", exception);
        }
        JsonElement parameterArray = root.GetProperty("parameters");
        if (parameterArray.ValueKind != JsonValueKind.Array || parameterArray.GetArrayLength() != network.ParameterCount)
        {
            throw new JsonException("Parameter count does not match the architecture.");
        }
        index = 0;
        foreach (JsonElement parameter in parameterArray.EnumerateArray())
        {
            if (parameter.ValueKind != JsonValueKind.Number || !parameter.TryGetDouble(out double value) || !double.IsFinite(value))
            {
                throw new JsonException("All model parameters must be finite numbers.");
            }
            network.parameters[index++] = value;
        }
        return network;
    }

    private static int ReadInteger(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Number || !element.TryGetInt32(out int result))
        {
            throw new JsonException("Expected a 32-bit integer.");
        }
        return result;
    }

    private static void RequireObject(JsonElement element, params string[] members)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new JsonException("Expected an object.");
        }
        var remaining = new HashSet<string>(members, StringComparer.Ordinal);
        foreach (JsonProperty property in element.EnumerateObject())
        {
            if (!remaining.Remove(property.Name))
            {
                throw new JsonException("Unknown or duplicate model property: " + property.Name);
            }
        }
        if (remaining.Count != 0)
        {
            throw new JsonException("The model is missing required properties.");
        }
    }
}
