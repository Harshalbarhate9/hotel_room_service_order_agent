import os
import time
from datetime import datetime
from pymongo import MongoClient

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory


GOOGLE_API_KEY = "ABC" #dummy key
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "hotel_db"

#-------------------------------------------------------------------------------------------------------------------------------

# --- 1. MONGODB SETUP ---
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Reset Menu
db["menu"].drop() 

menu_collection = db["menu"]
orders_collection = db["orders"]

#Adding menu to db
print("Adding Menu Data...")
menu_collection.insert_many([
    {"item": "Club Sandwich", "price": 15, "tags": ["contains_gluten", "meat"], "stock": 10},
    {"item": "Vegan Buddha Bowl", "price": 18, "tags": ["vegan", "gluten_free", "nuts"], "stock": 5},
    {"item": "Caesar Salad", "price": 12, "tags": ["vegetarian", "contains_dairy"], "stock": 0}, # Out of stock
    {"item": "Fruit Platter", "price": 10, "tags": ["vegan", "gluten_free"], "stock": 20}
])

#--------------------------------------------------------------------------------------------------------------------------------

# --- 2. DEFINE TOOLS ---

def check_menu(query: str):
    """Searches menu items using a simple text match. Returns availability."""
    print(f"   [Tool Log] Searching for: '{query}'") # Debug print
    
    # Search if 'item' OR 'tags' matches the query
    search_criteria = {
        "$or": [
            {"item": {"$regex": query, "$options": "i"}},
            {"tags": {"$regex": query, "$options": "i"}}
        ]
    }
    
    # If query is empty/generic, just fetch everything
    if not query or query.lower() == "menu":
        cursor = menu_collection.find({})
    else:
        cursor = menu_collection.find(search_criteria)

    response = []
    for doc in cursor:
        item = doc.get('item', 'Unknown')
        price = doc.get('price', 0)
        stock = doc.get('stock', 0)
        tags = doc.get('tags', [])
        
        status = f"Available (${price})" if stock > 0 else "OUT OF STOCK"
        response.append(f"- {item}: {status} (Tags: {', '.join(tags)})")
    
    if not response:
        return "No matching items found on the menu."
    
    return "\n".join(response)

def place_order(item_name: str):
    """Places an order for a specific item name."""
    # Exact match search
    item = menu_collection.find_one({"item": {"$regex": f"^{item_name}$", "$options": "i"}})
    
    if not item:
        return f"Error: '{item_name}' is not on the menu. Please check the menu first."
    if item['stock'] <= 0:
        return f"Sorry, {item['item']} is currently out of stock."
    
    # Deduct Stock 
    menu_collection.update_one({"_id": item["_id"]}, {"$inc": {"stock": -1}})

    # Create order
    order = {
        "item": item['item'],
        "price": item['price'],
        "status": "kitchen_preparing",
        "timestamp": datetime.now()
    }
    orders_collection.insert_one(order)
    return f"SUCCESS: Ordered {item['item']}. It will arrive in 30 mins."

#tools
tools = [
    Tool(
        name="MenuSearch",
        func=check_menu,
        description="Use this to search for food. Input can be an item name (e.g. 'burger') or a dietary preference (e.g. 'vegan')."
    ),
    Tool(
        name="PlaceOrder",
        func=place_order,
        description="Use this to place an order. Input must be the exact name of the item from the menu."
    )
]

#--------------------------------------------------------------------------------------------------------------------------------

# --- 3. LLM & AGENT SETUP ---

template = """
You are a helpful Hotel Room Service Agent.
Answer the user's questions. You have access to the following tools:

{tools}

**INSTRUCTIONS:**
1. ALWAYS use 'MenuSearch' to check for items or dietary needs. Do not guess.
2. If the user asks for "vegan", "vegetarian", etc., search the menu for those terms.
3. If an item is found, tell the user the price and if it is in stock.
4. To place an order, use 'PlaceOrder'.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

History: {chat_history}
Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# Using Google Gemini 2.5 Flash 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

# --- 4. EXECUTION LOOP ---

def run_chat_session():
    session_id = f"guest_{int(time.time())}"
    print(f"--- Starting Chat Session ({session_id}) ---")
    
    # Simple Memory Setup
    message_history = MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_URI,
        database_name=DB_NAME,
        collection_name="chat_history"
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        chat_memory=message_history
    )

    while True:
        try:
            user_input = input("\nGuest: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Load history
            history_text = memory.load_memory_variables({})['chat_history']
            
            # Run Agent
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": history_text
            })
            
            # Save interactions
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response['output'])
            
            print(f"Agent: {response['output']}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":

    run_chat_session()
