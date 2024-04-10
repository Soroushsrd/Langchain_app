# PubMed Question Answering App

This application allows users to input medical questions and receive answers based on information gathered from PubMed. The answers are generated using the GPT-4 Turbo model, which provides responses in the form of an assay.
The app utilizes the Streamlit framework for the user interface and requires access to the OpenAI API for model inference.

## Installation

To run the application locally, follow these steps:

- Clone this repository to your local machine.
- Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
- Set up your OpenAI API key by creating a secrets.toml file in the root directory of the project and adding your key:
```toml
[streamlit]
OPENAI_API_KEY = "your_openai_api_key"
```
- Run the Streamlit app:
```bash
streamlit run app.py

```

## Usage
Once the application is running, users can enter their medical questions into the provided text box and click the "Answer" button to receive a response. The app will use the GPT-4 Turbo model to generate answers based on information retrieved from PubMed.


## Contributing

- This application relies on the LangChain library for natural language processing tasks.
- The GPT-4 Turbo model is provided by OpenAI.
- PubMed is used as a source of medical information.

## License

This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/)
