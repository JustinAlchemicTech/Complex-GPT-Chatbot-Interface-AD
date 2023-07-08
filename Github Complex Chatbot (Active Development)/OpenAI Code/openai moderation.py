#To obtain a classification for a piece of text, make a request to the moderation endpoint as demonstrated in the following code snippets:
response = openai.Moderation.create(
    input="Sample text goes here"
)
output = response["results"][0]

#Below is an example output of the endpoint. It returns the following fields:
{
  "id": "modr-XXXXX",
  "model": "text-moderation-001",
  "results": [
    {
      "categories": {
        "hate": false,
        "hate/threatening": false,
        "self-harm": false,
        "sexual": false,
        "sexual/minors": false,
        "violence": false,
        "violence/graphic": false
      },
      "category_scores": {
        "hate": 0.18805529177188873,
        "hate/threatening": 0.0001250059431185946,
        "self-harm": 0.0003706029092427343,
        "sexual": 0.0008735615410842001,
        "sexual/minors": 0.0007470346172340214,
        "violence": 0.0041268812492489815,
        "violence/graphic": 0.00023186142789199948
      },
      "flagged": false
    }
  ]
}