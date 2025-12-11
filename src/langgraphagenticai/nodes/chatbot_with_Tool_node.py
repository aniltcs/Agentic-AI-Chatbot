from src.langgraphagenticai.state.state import State
from langchain_core.messages import HumanMessage,SystemMessage

class ChatbotWithToolNode:
    """
    Chatbot logic enhanced with tool integration.
    """
    def __init__(self,model):
        self.llm = model
    
    def create_chatbot(self, tools):
        """
        Returns a chatbot node function.
        """
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State):
            """
            Chatbot logic for processing the input state and returning a response.
            """
            prompt="""Only use the TavilySearchResults tool. Do NOT call 'brave_search'."""
            return {"messages": [llm_with_tools.invoke([prompt] + state["messages"])]}

        return chatbot_node

