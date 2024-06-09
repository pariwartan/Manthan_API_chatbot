import streamlit as st
import langchain
from langgraph.graph import StateGraph, END
from pydantic import (BaseModel, Field)
from langchain.tools import BaseTool
from typing import Type
import time
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
import os
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain import hub
from langchain.agents import create_openai_functions_agent, initialize_agent, AgentType, AgentExecutor
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.agents import AgentFinish
import requests
import json
import base64
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    SystemMessagePromptTemplate
)
from langchain.chains import LLMChain, APIChain
from mongoConnect import get_required_files

# Local Stuffs

from langchain.memory.buffer import ConversationBufferMemory

openai_key = ''

os.environ["OPENAI_API_KEY"] = openai_key

llm = ChatOpenAI(temperature=0)

onboarding_required_info = "target hcm,source hcm,salesforce id"

project_dir = os.path.dirname(os.path.abspath(__file__))
temp_folder = os.path.join(project_dir, 'temp')

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    finalResponse: str


class StartOnboardingBase(BaseModel):
    user_question: str = Field(description="user_question")


class StartOnboardingClient(BaseTool):
    name = "start_onboard_client"
    description = """"
             To onboard a client or a customer.
            """
    args_schema: Type[BaseModel] = StartOnboardingBase

    def _run(self, user_question: str):
        response = startOnboarding(user_question)
        return response

    def _arun(self, user_question: str):
        raise NotImplementedError("error")


def startOnboarding(user_question):
    validate_onboarding_prompt = PromptTemplate(
        input_variables=['onboarding_required_info', 'user_question'],
        template=""""Validate whether these required information {onboarding_required_info} are present in user input or not.
        If any required keys are not present then return all the missing information else say nothing is missing .

        user input: {user_question} 
        """
    )

    chain = LLMChain(llm=llm, prompt=validate_onboarding_prompt, verbose=True)
    response = chain.run({"onboarding_required_info": onboarding_required_info, "user_question": user_question})
    print("@@@@@@@@@@ missing onboarding_required_info are  @@@@@@@@@" + response)

    if ((not response) or "nothing" in response.lower() or "are present" in response.lower() or "is present" in response.lower() or
            "no keys" in response.lower() or "no information" in response.lower()):
        # Set required onboarding data present to true
        st.session_state.localStore['onboarding_required_info'] = True
        if 'previous_run_fail' in st.session_state.localStore:
            st.session_state.localStore.pop('previous_run_fail')

        # check required files for this onboarding
        response = getRequiredFiles(user_question)
        st.session_state.upload = 'X'

    else:
        response = "required information are missing " + response
        st.session_state.localStore['previous_run_fail'] = response

    return response

def getrequiredfile(user_question):
    response = ["direct_deposit.pdf", "employee_earning.pdf"]
    return response

class GetFileStatus(BaseModel):
    user_question: str = Field(description="user_question")


class GetFileStatuss(BaseTool):
    name = "get_file_status"
    description = """"
            Return the status of the uploaded files required to onboard the client.
            """
    args_schema: Type[BaseModel] = GetFileStatus

    def _run(self, user_question: str):
        response = getstatusoffile(user_question)
        return response

    def _arun(self, user_question: str):
        raise NotImplementedError("error")


def getstatusoffile(user_question):
    if len(st.session_state.localStore) > 0:
        response = st.session_state.localStore.pop(-1)
    else:
        response = "processing is inprogress"
    return response


def graph(user_question):
    inputs = {"input": user_question,
              "chat_history": [],
              "intermediate_steps": []}

    def run_agent(data):
        agent_outcome = getrequiredfile(data)
        print("----------------------++++++++++++++++++++++++++________________________")
        print(data)
        return {"agent_outcome": agent_outcome}

    def validaterequiredfile(data):
        agent_action = data['agent_outcome']
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        allset = True
        base_path = temp_folder
        for file in data['agent_outcome']:
            path = os.path.join(base_path, file)
            if os.path.exists(path):
                pass
            else:
                allset = False
        if allset:
            response = "validation successfull"
        else:
            response = "some files are missing please upload missing files"
        print(response)
        return {"intermediate_steps": [(agent_action, str(response))]}

    def should_continue(data):
        status = data['intermediate_steps'][0][-1]
        print("______________--------------------------------")
        if "validation successfull" in status:
            return "continue"
        else:
            return "end"

    def file_extractor(data):
        print("----------------------file_extractor+++++++++++++++++++++++++++++")
        #st.session_state.localStore.append(data['intermediate_steps'][0][-1])
        # local_session_adder(data)
        print(data['agent_outcome'])
        base_path = temp_folder
        base64File = ""
        for file in data['agent_outcome']:
            path = os.path.join(base_path, file)
            if("direct" in file):
                base64File = base64.b64encode(open(path, "rb").read())
                break
        json_data = {'pdf': base64File.decode('utf-8')}
        print(json_data)
        response = requests.post("https://us-central1-image-processing-415012.cloudfunctions.net/extract_dd_azure", json=json_data)

        # Check the response
        if response.status_code == 200:
            print("Request was successful!")
            print("Response:", response.json())
        else:
            print("Request failed with status code:", response.status_code)
            print("Response:", response.text)
        response = response.text
        #response = "successful"
        return {"finalResponse": response}
        #return response

    workflow = StateGraph(AgentState)
    workflow.add_node("extraction_agent", run_agent)
    workflow.set_entry_point("extraction_agent")
    workflow.add_node("file_validator", validaterequiredfile)
    workflow.add_edge("extraction_agent", "file_validator")
    workflow.add_node("file_extractor", file_extractor)
    workflow.add_conditional_edges("file_validator", should_continue, {"continue": "file_extractor", "end": END})
    workflow.add_edge("file_extractor", END)
    app = workflow.compile()
    print("Find the visual representation of the graph...................")
    print(app.get_graph().draw_mermaid())
    response = app.invoke(inputs)
    return response


def local_session_adder(data):
    st.session_state.intermediate_steps.append(data['intermediate_steps'][0][-1])
    return data

def main():
    try:
        langchain.debug = True
        load_dotenv()
        logo = 'https://images.sftcdn.net/images/t_app-icon-m/p/69c53218-c7a9-11e6-8ce5-5a990553aaad/3424445367/adp-mobile-solutions-logo'
        logo2 = 'https://i.pinimg.com/736x/8b/16/7a/8b167af653c2399dd93b952a48740620.jpg'
        st.set_page_config(page_title="Manthan API Chatbot", page_icon=logo)
        st.header("Manthan API Testing")
        if 'area_key' not in st.session_state:
            st.session_state.area_key = 1
        if 'requests' not in st.session_state:
            st.session_state.requests = []
        if 'responses' not in st.session_state:
            st.session_state['responses'] = ["Hi, I am Project Manthan Testing Bot, How may i assist you?"]
        if 'localStore' not in st.session_state:
            st.session_state.localStore = {}
        if 'upload' not in st.session_state:
            st.session_state.upload = []
        if 'intermediate_steps' not in st.session_state:
            st.session_state.intermediate_steps = []
            # system_message = SystemMessagePromptTemplate.from_template(template=manthan_assistant_bot_template)
        system_message = SystemMessagePromptTemplate.from_template(
            template="""Simply return the response as it is , do not change anything""")
        response_container = st.container()
        # container for text box
        textcontainer = st.container()
        with textcontainer:
            user_question = st.empty()
            user_question = st.chat_input("I am here to help you!", key="input")
            if user_question:
                with st.spinner("Processing"):
                    prev_response = st.session_state.localStore.get('previous_run_fail')
                    missing_input = missing_input_status(prev_response)
                    if missing_input:
                        prev_question = st.session_state.requests[-1] if st.session_state.requests else ""
                        print(st.session_state.localStore.get('previous_chain_responseJson'))
                        user_question = prepare_input(prev_question, prev_response, user_question)

                    tools = [StartOnboardingClient(), GetFileStatuss()]
                    agent_kwargs = {
                        # "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
                        "system_message": system_message
                    }
                    if "memory" not in st.session_state:
                        st.session_state.memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
                    agent = initialize_agent(tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True,
                                             agent_kwargs=agent_kwargs)
                    response = agent.run(user_question)
                    print(response)
                    st.session_state.requests.append(user_question)
                    st.session_state.responses.append(response)

        if st.session_state.upload == 'X':
            with st.sidebar:
                st.subheader("Your documents")
                pdf_docs = st.file_uploader(
                    "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
                if st.button("Process"):
                    with st.spinner("Processing"):
                        uploaded_file_path = temp_folder
                        for uploaded_file in pdf_docs:
                            if uploaded_file is not None:
                                with open(os.path.join(uploaded_file_path, uploaded_file.name),
                                          'wb') as output_temporary_file:
                                    output_temporary_file.write(uploaded_file.read())
                        user_question = st.session_state.requests.pop(-1)
                        response = graph(user_question)

                        message("Data has been extracted from file successfully : ")
                        st.session_state.upload = ' '
                        # st.rerun()
                        st.markdown("""                                
                                     <style>                           
                                     [data-testid="stSidebar"] {       
                                         display: none                 
                                     }                                 
                                     [data-testid="collapsedControl"] {
                                         display: none                 
                                     }                                 
                                     </style>                          
                                     """, unsafe_allow_html=True)
                        # st.error(response['finalResponse'])
                        print("_________________________________________________")
                        st.session_state.intermediate_steps.append(response['intermediate_steps'][0][-1])
                        st.session_state.requests.append(user_question)
                        response = response['finalResponse']
                        st.session_state.responses.append(response)
                        print("test",response)
        with response_container:
            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])):
                    message(st.session_state['responses'][i], key=str(i), logo=logo)
                    if i < len(st.session_state['requests']):
                        message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user', logo=logo2)

    except Exception as e:
        st.error("Something went wrong. Please restart the application.")
        print(f"Error details: {str(e)}")


def prepare_input(previous_question, previous_response, curr_question):
    api_required_params = onboarding_required_info.split(",")
    print("========================prepare_input==================================")
    print(api_required_params)
    missing_params_list = []
    for param in api_required_params:
        if param.lower() in previous_response.lower():
            missing_params_list.append(param)
    pay_load_dict = {}
    user_passed_params = curr_question.split(',')
    print(user_passed_params)
    if len(user_passed_params) <= len(missing_params_list):
        for i in range(len(user_passed_params)):
            pay_load_dict[missing_params_list[i]] = user_passed_params[i]
    json_string = json.dumps(pay_load_dict)
    formatted_string = json_string.replace('{', '').replace('}', '').replace('"','')
    print("===================formatted_input==================")
    print(str(formatted_string))
    return previous_question + "," + formatted_string


def missing_input_status(previous_response):
    if previous_response is None or "nothing" in previous_response:
        return False
    return True


def getRequiredFiles(user_question):
    get_onboarding_details_prompt = PromptTemplate(
        input_variables=["user_question", "onboarding_required_info"],
        template=""""You are an AI assistant tasked with generating a json based on user input and a predefined set of keys. 
                     user input: {user_question}
                     key set: {onboarding_required_info}
                     """
    )

    chain = LLMChain(llm=llm, prompt=get_onboarding_details_prompt)
    onboarding_details = chain.run({"user_question": user_question, "onboarding_required_info": onboarding_required_info})
    print("@@@@@@@@@@ missing onboarding_required_info are  @@@@@@@@@" + onboarding_details)
    response = json.loads(onboarding_details)
    print(response)
    if 'source_hcm' in response or 'target_hcm' in response:
        return [get_required_files(response['source_hcm'], response['target_hcm'])]
    else:
        return ["direct_deposit.pdf", "employee_earning.pdf"]

if __name__ == '__main__':
    main()
