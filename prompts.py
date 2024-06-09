from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



initiate_onboarding_prompt = ChatPromptTemplate.from_messages(
    [
        (
            ""
        ),
        MessagesPlaceholder(variable_name="message")
    ]
)

validate_onboarding_prompt = ChatPromptTemplate(
    input_variables=['user_question'],
    template=""""Validate whether these required information {onboarding_required_info} are present in user input or not.
    If any required keys are not present then return all the missing keys else say nothing is missing .

    user input: {user_question} 
    """
)

