# Tree Species Classification API Documentation

This document provides information about the API endpoints available in the Tree Species Classification application.

## Base URL

When running locally, the base URL is:

```
http://localhost:5000/api
```

## Endpoints

### Classify Tree Species

**Endpoint:** `/api/classify`

**Method:** POST

**Description:** Classify a tree species from an uploaded image.

**Request:**

The request should be a `multipart/form-data` request with the following parameter:

- `image`: The image file to classify (JPEG, PNG, or GIF format)

**Example using cURL:**

```bash
curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:5000/api/classify
```

**Example using Python requests:**

```python
import requests

url = "http://localhost:5000/api/classify"
files = {"image": open("/path/to/your/image.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

**Response:**

The response is a JSON object with the following structure:

```json
{
  "success": true,
  "predictions": [
    {
      "species": "Oak",
      "confidence": 0.85
    },
    {
      "species": "Maple",
      "confidence": 0.10
    },
    {
      "species": "Pine",
      "confidence": 0.03
    },
    {
      "species": "Birch",
      "confidence": 0.01
    },
    {
      "species": "Palm",
      "confidence": 0.01
    }
  ],
  "processing_time": 0.235
}
```

In case of an error, the response will have the following structure:

```json
{
  "success": false,
  "error": "Error message"
}
```

### Get Supported Species

**Endpoint:** `/api/species`

**Method:** GET

**Description:** Get a list of all tree species that the model can classify.

**Example using cURL:**

```bash
curl http://localhost:5000/api/species
```

**Example using Python requests:**

```python
import requests

url = "http://localhost:5000/api/species"
response = requests.get(url)
print(response.json())
```

**Response:**

```json
{
  "success": true,
  "species": [
    "Oak",
    "Maple",
    "Pine",
    "Birch",
    "Palm",
    "Spruce",
    "Willow",
    "Cedar",
    "Cypress",
    "Fir"
  ]
}
```

### Get Model Information

**Endpoint:** `/api/model-info`

**Method:** GET

**Description:** Get information about the currently loaded model.

**Example using cURL:**

```bash
curl http://localhost:5000/api/model-info
```

**Example using Python requests:**

```python
import requests

url = "http://localhost:5000/api/model-info"
response = requests.get(url)
print(response.json())
```

**Response:**

```json
{
  "success": true,
  "model_name": "EfficientNetB0",
  "num_classes": 10,
  "input_shape": [224, 224, 3],
  "version": "1.0"
}
```

## Error Codes

The API may return the following error codes:

- `400 Bad Request`: The request was invalid or missing required parameters.
- `404 Not Found`: The requested resource was not found.
- `415 Unsupported Media Type`: The uploaded file format is not supported.
- `500 Internal Server Error`: An error occurred on the server.

## Rate Limiting

There are currently no rate limits implemented for the API when running locally. However, please be considerate with your request frequency.

## Image Requirements

- Supported formats: JPEG, PNG, GIF
- Maximum file size: 5MB
- Recommended image dimensions: At least 224x224 pixels

## Best Practices

1. Ensure the tree is clearly visible in the image.
2. Try to capture the tree's distinctive features (leaves, bark, overall shape).
3. Avoid images with multiple tree species if possible.
4. For better results, use images with good lighting and minimal blur.