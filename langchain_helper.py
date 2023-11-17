from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv

load_dotenv()

def improve_text(raw_text):
    llm = OpenAI(temperature=0.2)

    prompt_template_name = PromptTemplate(
        input_variables=['raw_text'],
        template= "Please enhance the following text, making it clearer and more fluent without making it overly formal and keeping the text in the original language. Don't join the text, keep the same spacing and Don't replace the names of people involved: {raw_text} "
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="raw_text")
    
    response = name_chain({'raw_text': raw_text})
    return response

def improve_text_hf(raw_text):
    prompt_template_name = PromptTemplate(
        input_variables=['raw_text'],
        template="respond this text: {raw_text}"
    )
     
    repo_id = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
    )
    llm_chain = LLMChain(prompt=prompt_template_name, llm=llm, output_key="raw_text")
  
    response = llm_chain.run({'raw_text': raw_text})
    return response

# def langchain_agent():
#     llm = OpenAI(temperature=0.5)

#     tools = load_tools(["wikipedia", "llm-math"], llm = llm)
#     agent = initialize_agent(
#         tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
#     )
#     result = agent.run("what is the average age of a dog? Multiply the age by 3")
#     print(result)

if __name__ == "__main__":
    print(improve_text("This is my birday! Thansk!"))
    # langchain_agent()
