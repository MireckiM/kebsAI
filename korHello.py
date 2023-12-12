# Kor!
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

# LangChain Models
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# Standard Helpers
import pandas as pd
import requests
import time
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Text Helpers
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# For token counting
from langchain.callbacks import get_openai_callback

def configure():
    load_dotenv()

configure()

def printOutput(output):
    print(json.dumps(output,sort_keys=True, indent=3))

openai_api_key = os.getenv("OPENAI_API_KEY") 

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", # Cheaper but less reliable
#    model_name="gpt-4",
    temperature=0,
    max_tokens=2000,
    openai_api_key=openai_api_key
)

schema = Object(
  id="kebab_zamowienie",
  description=(
      "Uzytkownik uzywa aplikacji, ktora przyjmuje zamowienia na kebaba z dodatkami."
      "Uzytkownik wprowadza zamowienie opisujace zamawiany kebab, oraz jego parametry i dodatki,"
  ),
  attributes=[
      Text(
          id="ciasto",
          description="Ciasto, ktore ma zostac uzyte do produkcji kebaba.",
          examples=[("Poprosze kebab w bulce z baranina, sosem czosnkowym i smazona cebulka.", "bulka")],
          many=True,
      ),
      Text(
          id="mieso",
          description="Rodzaj miesa, ktorym bedzie nadziany kebab.",
          examples=[("Poprosze kebab w bulce z baranina, sosem czosnkowym i smazona cebulka.", "baranina")],
          many=True,
      ),
      Text(
          id="sos",
          description="Sos, ktorym bedzie polany kebab.",
          examples=[("Poprosze kebab w bulce z baranina, sosem czosnkowym i smazona cebulka.", "czosnkowy")],
          many=True,
      ),
      Text(
          id="dodatki",
          description="Dodatki do kebabu.",
          examples=[("Poprosze kebab w bulce z baranina, sosem czosnkowym i smazona cebulka.", "smazona cebulka")],
          many=True,
      ),
    ],
  many=False,
)

chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')
output = chain.predict_and_parse(text="dwa z ketchupem i frytami, jeden ostry z surówką i 3 cole")['data']

printOutput(output)








