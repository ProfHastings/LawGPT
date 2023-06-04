# Legal Question-Answering

🤖Ask questions on the OGH database in natural language🤖

💪 Built with [LangChain](https://github.com/hwchase17/langchain)

# 🌲 Environment Setup

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

Then set your OpenAI API key (if you don't have one, get one [here](https://beta.openai.com/playground))

```shell
export OPENAI_API_KEY=....
```

## 💬 Ask a question
In order to ask a question, run a command like:

```shell
python qa.py "is there food in the office?"
```

You can switch out `Was sind die Voraussetzungen für das von der Rechtsprechung verlangte rechtliche Interesse für eine Feststellungsklage nach § 228 ZPO?` for any question of your liking!

This exposes a chat interface for interacting with the database

## 🚀 Code to deploy on StreamLit

The code to run the StreamLit app is in `main.py`. 
Note that when setting up your StreamLit app you should make sure to add `OPENAI_API_KEY` as a secret environment variable.
To use StreamLit run 'streamlit run main.py'
To stop StreamLit press ctrl+C in terminal (with website still open)