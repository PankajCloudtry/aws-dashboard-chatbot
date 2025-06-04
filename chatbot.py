import os
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

df = pd.read_csv("usage.csv")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    df,
    verbose=True,
    allow_dangerous_code=True
)

while True:
    question = input("Ask a question about your AWS usage (or type 'exit'): ")
    if question.lower() == "exit":
        break
    response = agent.run(question)
    print(f"Answer: {response}")
