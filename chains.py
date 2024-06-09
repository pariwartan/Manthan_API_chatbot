from langchain_openai import ChatOpenAI
from prompts import initiate_onboarding_prompt, validate_onboarding_prompt
from langchain.chains import LLMChain, APIChain

llm = ChatOpenAI()

initiate_onboarding_chain = initiate_onboarding_prompt | llm

validate_onboarding_chain = validate_onboarding_prompt | llm