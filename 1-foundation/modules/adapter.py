from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic 

class Adapter:
    def __init__(self, env):
        #! We pull in the env variable and set defaults
        self.provider = env("LLM_PROVIDER", default="ollama") or "ollama"
        self.model = env("MODEL", default="llama3.2:3b") or "llama3.2:3b"
        self.api_key = env("API_KEY") or None
        #! Set up the basic prompt template
        self.prompt = ChatPromptTemplate.from_template(
            "answer the following request: {topic}"
        )
        #! Initialize the appropriate LLM based on the provider
        if self.provider.lower() == "ollama":       
            self.llm_chat = ChatOllama(
                model=self.model
            )
        #! If OpenAI
        elif self.provider.lower() == "openai":
            if self.api_key is None:
                raise ValueError("API key is required for OpenAI provider")
            self.llm_chat = ChatOpenAI(
                model_name=self.model,
                openai_api_key=self.api_key
            )
        #! If Anthropic
        elif self.provider.lower() == "anthropic":
            if self.api_key is None:
                raise ValueError("API key is required for Anthropic provider")
            self.llm_chat = ChatAnthropic(
                model_name=self.model,
                anthropic_api_key=self.api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def chat(self, query):
        from langchain_core.output_parsers import StrOutputParser
        #! We create a chain of the prompt, the llm, and the output parser
        chain = self.prompt | self.llm_chat | StrOutputParser()
        #! We invoke the chain with the query
        response = chain.invoke({"topic": query})
        #! Return the response
        return response
    