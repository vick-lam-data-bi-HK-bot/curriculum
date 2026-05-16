import key_param
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from typing import List
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.mongodb import MongoDBSaver
import voyageai

def init_mongodb():
    """
    Initialize MongoDB client and collections.

    Returns:
        tuple: MongoDB client, vector search collection, full documents collection.
    """
    
    mongodb_client = MongoClient(key_param.mongodb_uri)
    
    DB_NAME = "ai_agents"
    
    vs_collection = mongodb_client[DB_NAME]["chunked_docs"]
    
    full_collection = mongodb_client[DB_NAME]["full_docs"]
    
    return mongodb_client, vs_collection, full_collection

# Define the graph state type with messages that can accumulate
class GraphState(TypedDict):
    # Define a messages field that keeps track of conversation history
    messages: Annotated[list, add_messages]

def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a piece of text.

    Args:
        text (str): The text to embed.
        embedding_model (voyage-3-lite): The embedding model.

    Returns:
        List[float]: The embedding of the text.
    """

    embedding_model = voyageai.Client(api_key=key_param.voyage_api_key)

    embedding = embedding_model.embed(text, model="voyage-3-lite", input_type="query").embeddings[0]
    
    return embedding


@tool 
def get_information_for_question_answering(user_query: str) -> str:
    """
    Retrieve relevant documents for a user query using vector search.

    Args:
        user_query (str): The user's query.

    Returns:
        str: The retrieved documents as a string.
    """

    query_embedding = generate_embedding(user_query)

    vs_collection = init_mongodb()[1]
    
    pipeline = [
        {
            # Use vector search to find similar documents
            "$vectorSearch": {
                "index": "vector_index",  # Name of the vector index
                "path": "embedding",       # Field containing the embeddings
                "queryVector": query_embedding,  # The query embedding to compare against
                "numCandidates": 150,      # Consider 150 candidates (wider search)
                "limit": 5,                # Return only top 5 matches
            }
        },
        {
            # Project only the fields we need
            "$project": {
                "_id": 0,                  # Exclude document ID
                "body": 1,                 # Include the document body
                "score": {"$meta": "vectorSearchScore"},  # Include the similarity score
            }
        },
    ]
    
    results = vs_collection.aggregate(pipeline)
    
    context = "\n\n".join([doc.get("body") for doc in results])
    
    return context

@tool  # Decorator marks this function as a tool the agent can use
def get_page_content_for_summarization(user_query: str) -> str:
    """
    Retrieve the content of a documentation page for summarization.

    Args:
        user_query (str): The user's query (title of the documentation page).

    Returns:
        str: The content of the documentation page.
    """
    full_collection = init_mongodb()[2]

    query = {"title": user_query}
    
    projection = {"_id": 0, "body": 1}
    
    document = full_collection.find_one(query, projection)
    
    if document:
        return document["body"]
    else:
        return "Document not found"

def agent(state: GraphState, llm_with_tools) -> GraphState:
    """
    Agent node.

    Args:
        state (GraphState): The graph state.
        llm_with_tools: The LLM with tools.

    Returns:
        GraphState: The updated messages.
    """

    messages = state["messages"]
    
    result = llm_with_tools.invoke(messages)
    
    return {"messages": [result]}

def tool_node(state: GraphState, tools_by_name) -> GraphState:
    """
    Tool node.

    Args:
        state (GraphState): The graph state.
        tools_by_name (Dict[str, Callable]): The tools by name.

    Returns:
        GraphState: The updated messages.
    """
    result = []
    
    tool_calls = state["messages"][-1].tool_calls
    
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        
        observation = tool.invoke(tool_call["args"])
        
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    
    return {"messages": result}

def route_tools(state: GraphState):
    """
    Route to the tool node if the last message has tool calls. Otherwise, route to the end.

    Args:
        state (GraphState): The graph state.

    Returns:
        str: The next node to route to.
    """
    messages = state.get("messages", [])
    
    if len(messages) > 0:
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    
    return END

def init_graph(llm_with_tools, tools_by_name, mongodb_client):
    """
    Initialize the graph.

    Args:
        llm_with_tools: The LLM with tools.
        tools_by_name (Dict[str, Callable]): The tools by name.
        mongodb_client (MongoClient): The MongoDB client.

    Returns:
        StateGraph: The compiled graph.
    """
    graph = StateGraph(GraphState)
    
    graph.add_node("agent", lambda state: agent(state, llm_with_tools))
    
    graph.add_node("tools", lambda state: tool_node(state, tools_by_name))
    
    graph.add_edge(START, "agent")
    
    graph.add_edge("tools", "agent")
    
    graph.add_conditional_edges("agent", route_tools, {"tools": "tools", END: END})
    
    checkpointer = MongoDBSaver(mongodb_client)
    
    return graph.compile(checkpointer=checkpointer)

def execute_graph(app, thread_id: str, user_input: str) -> None:
    """
    Stream outputs from the graph.

    Args:
        app: The compiled graph application.
        thread_id (str): The thread ID.
        user_input (str): The user's input.
    """
    input = {"messages": [("user", user_input)]}
    
    config = {"configurable": {"thread_id": thread_id}}
    
    for output in app.stream(input, config):
        for key, value in output.items():
            print(f"Node {key}:")
            print(value)
    
    print("---FINAL ANSWER---")
    
    print(value["messages"][-1].content)

def main():
    """
    Main function to initialize and execute the graph.
    """
    mongodb_client, vs_collection, full_collection = init_mongodb()
    
    tools = [
        get_information_for_question_answering,
        get_page_content_for_summarization
    ]
    
    llm = ChatOpenAI(openai_api_key=key_param.openai_api_key, temperature=0, model="gpt-4o")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "You are a helpful AI assistant."
                " You are provided with tools to answer questions and summarize technical documentation related to MongoDB."
                " Think step-by-step and use these tools to get the information required to answer the user query."
                " Do not re-run tools unless absolutely necessary."
                " If you are not able to get enough information using the tools, reply with I DON'T KNOW."
                " You have access to the following tools: {tool_names}."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    
    bind_tools = llm.bind_tools(tools)
    
    llm_with_tools = prompt | bind_tools
    
    tools_by_name = {tool.name: tool for tool in tools}
    
    app = init_graph(llm_with_tools, tools_by_name, mongodb_client)
    
    # Ask a specific question about MongoDB
    execute_graph(app, "1", "What are some best practices for data backups in MongoDB?")
   
    # Test the agent's memory capabilities
    execute_graph(app, "1", "What did I just ask?")

main()
